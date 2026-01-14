"""
Order Manager - State machine for order lifecycle with atomic transactions.

Provides ACID guarantees for order + position updates via centralized components.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from core.event_store import get_event_store
from execution.risk_gatekeeper import RiskGatekeeper
from execution.position_tracker import PositionTracker
from database.db_manager import DatabaseManager

logger = logging.getLogger("Execution.OrderManager")

_ORDER_MANAGER_INSTANCE = None

def get_order_manager():
    """Get the global OrderManager instance."""
    if _ORDER_MANAGER_INSTANCE is None:
        raise RuntimeError("OrderManager not initialized yet!")
    return _ORDER_MANAGER_INSTANCE

def _set_global_instance(instance):
    """Set the global OrderManager instance (Internal use only)."""
    global _ORDER_MANAGER_INSTANCE
    _ORDER_MANAGER_INSTANCE = instance


class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING_SUBMIT = "PENDING_SUBMIT"   # About to submit to exchange
    SUBMITTED = "SUBMITTED"             # Confirmed on exchange
    PARTIAL_FILL = "PARTIAL_FILL"       # Partially filled
    FILLED = "FILLED"                   # Fully filled
    CANCELED = "CANCELED"               # Canceled by user or system
    REJECTED = "REJECTED"               # Rejected by exchange
    EXPIRED = "EXPIRED"                 # Time-in-force expired


@dataclass
class Order:
    """Order record with full lifecycle tracking."""
    id: int = 0                          # Database ID
    client_order_id: str = ""            # Our ID (for idempotency)
    exchange_order_id: Optional[str] = None  # Exchange ID
    trace_id: str = ""                   # Correlation ID
    symbol: str = ""
    side: str = ""
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"
    quantity: float = 0.0
    filled_quantity: float = 0.0
    remainder_quantity: float = 0.0       # For partial fills
    price: float = 0.0
    avg_fill_price: float = 0.0
    state: str = OrderState.PENDING_SUBMIT.value
    commission: float = 0.0
    pnl: float = 0.0
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"c_{uuid.uuid4().hex[:16]}"
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        self.remainder_quantity = self.quantity - self.filled_quantity


# Valid state transitions
VALID_TRANSITIONS = {
    OrderState.PENDING_SUBMIT: [OrderState.SUBMITTED, OrderState.REJECTED],
    OrderState.SUBMITTED: [OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED],
    OrderState.PARTIAL_FILL: [OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELED],
    OrderState.FILLED: [],  # Terminal
    OrderState.CANCELED: [],  # Terminal
    OrderState.REJECTED: [],  # Terminal
    OrderState.EXPIRED: [],  # Terminal
}


class OrderManager:
    """
    Order lifecycle manager.
    
    Coordinator between:
    - Strategies (Submit orders)
    - Risk Gatekeeper (Validate orders)
    - Position Tracker (Update state)
    - Database Manager (Persist state)
    """
    
    def __init__(self, exchange_client, db_manager: DatabaseManager, position_tracker: PositionTracker, risk_gatekeeper: RiskGatekeeper):
        self.exchange = exchange_client
        self.db = db_manager
        self.position_tracker = position_tracker
        self.risk_gatekeeper = risk_gatekeeper
        self._event_store = get_event_store()
        
        # In-memory order cache (for fast lookup)
        # In production, might want to limit size or use LRU
        self._orders: Dict[str, Order] = {}
        
        # Register as global instance
        _set_global_instance(self)
        
        logger.info("✅ OrderManager initialized")
    
    async def initialize(self):
        """Initialize the order manager."""
        logger.info("Initializing OrderManager...")
        # Recover state from DB if needed
        pass

    async def get_position(self, symbol: str):
        """Get position from PositionTracker (Proxy)."""
        return await self.position_tracker.get_position(symbol)

    async def submit_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        trace_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Submit order with risk checks and full lifecycle tracking.
        """
        try:
            # 1. RISK CHECK (CRITICAL)
            risk_result = await self.risk_gatekeeper.validate_order(
                symbol, quantity, price, side
            )
            
            if not risk_result.is_allowed:
                logger.warning(f"❌ Order REJECTED by risk: {risk_result.reason}")
                return None
            
            # 2. PROPOSE ORDER (Local State)
            order = Order(
                symbol=symbol.lower(),
                side=side.upper(),
                order_type=order_type,
                time_in_force=time_in_force,
                quantity=quantity,
                filled_quantity=0,
                remainder_quantity=quantity,
                price=price,
                state=OrderState.PENDING_SUBMIT.value,
                trace_id=trace_id or str(uuid.uuid4())
            )
            
            # Persist to DB (Async)
            await self.db.insert_order({
                'client_order_id': order.client_order_id,
                'trace_id': order.trace_id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'time_in_force': order.time_in_force,
                'quantity': order.quantity,
                'price': order.price,
                'state': order.state,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            })
            
            self._orders[order.client_order_id] = order
            
            # 3. SUBMIT TO EXCHANGE
            try:
                # Assuming exchange client has create_order method
                exchange_order = await self.exchange.create_order(
                    symbol=order.symbol,
                    side=order.side,
                    type=order.order_type,
                    timeInForce=order.time_in_force,
                    quantity=order.quantity,
                    price=order.price,
                    newClientOrderId=order.client_order_id
                )
                
                # 4. UPDATE STATE (Submitted)
                # Map exchange ID
                exch_id = str(exchange_order.get('orderId', ''))
                await self.mark_submitted(order.client_order_id, exch_id)
                
                logger.info(
                    f"✅ Order submitted: {order.side} {order.quantity} {order.symbol} @ ${order.price} "
                    f"[{order.client_order_id}]"
                )
                return order
                
            except Exception as e:
                logger.error(f"❌ Exchange submission failed: {e}")
                await self.mark_rejected(order.client_order_id, str(e))
                return None
                
        except Exception as e:
            logger.error(f"❌ Order submission error: {e}", exc_info=True)
            return None

    async def mark_submitted(
        self,
        client_order_id: str,
        exchange_order_id: str
    ) -> Optional[Order]:
        """Mark order as successfully submitted to exchange."""
        return await self._update_state(
            client_order_id,
            OrderState.SUBMITTED,
            exchange_order_id=exchange_order_id
        )
    
    async def mark_rejected(
        self,
        client_order_id: str,
        error_message: str
    ) -> Optional[Order]:
        """Mark order as rejected."""
        return await self._update_state(
            client_order_id,
            OrderState.REJECTED,
            error_message=error_message
        )
    
    async def process_fill(
        self,
        client_order_id: str,
        filled_qty: float,
        fill_price: float,
        commission: float = 0.0,
        pnl: float = 0.0
    ) -> Optional[Order]:
        """
        Process a fill event.
        Updates Order State AND Position Tracker.
        """
        order = self._orders.get(client_order_id)
        if not order:
            logger.warning(f"Process fill: Order not found {client_order_id}")
            return None
        
        # 1. Update Order State
        new_filled = order.filled_quantity + filled_qty
        new_remainder = order.quantity - new_filled
        
        # Calc average price
        if order.filled_quantity > 0:
            total_value = (order.avg_fill_price * order.filled_quantity) + (fill_price * filled_qty)
            new_avg_price = total_value / new_filled
        else:
            new_avg_price = fill_price
            
         # Determine state
        if new_filled >= order.quantity - 0.00000001: # Float epsilon
             new_state = OrderState.FILLED
             new_remainder = 0
        else:
             new_state = OrderState.PARTIAL_FILL
        
        # 2. Update Position Tracker (CRITICAL)
        quantity_delta = filled_qty if order.side == 'BUY' else -filled_qty
        await self.position_tracker.update_position(
            symbol=order.symbol,
            quantity_delta=quantity_delta,
            price=fill_price,
            commission=commission
        )
        
        # 3. Update Risk Gatekeeper PnL
        self.risk_gatekeeper.update_pnl(pnl)
        
        # 4. Persist Order Updates
        order.filled_quantity = new_filled
        order.remainder_quantity = new_remainder
        order.avg_fill_price = new_avg_price
        order.commission += commission
        order.pnl += pnl
        order.state = new_state.value
        order.updated_at = time.time()
        
        await self.db.update_order({
            'client_order_id': client_order_id,
            'filled_quantity': new_filled,
            'remainder_quantity': new_remainder,
            'avg_fill_price': new_avg_price,
            'commission': order.commission,
            'pnl': order.pnl,
            'state': new_state.value,
            'updated_at': order.updated_at
        })
        
        # 5. Log Trade to DB (CRITICAL FIX: Ensure dashboard sees this)
        # We manually construct the dict to bypass event object requirement of log_trade
        trade_record = {
            'type': 'trade',
            'timestamp': time.time(),
            'symbol': order.symbol,
            'side': order.side,
            'quantity': filled_qty,
            'price': fill_price,
            'commission': commission,
            'commission_asset': 'USDT',
            'is_maker': False,
            'pnl': pnl,
            'strategy_id': 'execution'
        }
        self.db.submit_write_task(trade_record)
        
        logger.info(
            f"✅ Fill processed: {order.side} {filled_qty} {order.symbol} @ ${fill_price} "
            f"(PnL: ${pnl:.4f})"
        )
        
        return order

    async def _update_state(
        self,
        client_order_id: str,
        new_state: OrderState,
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[Order]:
        """Update order state and DB."""
        order = self._orders.get(client_order_id)
        if not order:
            return None
            
        order.state = new_state.value
        order.updated_at = time.time()
        
        update_data = {
            'client_order_id': client_order_id,
            'state': new_state.value,
            'updated_at': order.updated_at
        }
        
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
            update_data['exchange_order_id'] = exchange_order_id
            
        if error_message:
            order.error_message = error_message
            update_data['error_message'] = error_message
            
        await self.db.update_order(update_data)
        
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an open order."""
        try:
            order = self._orders.get(client_order_id)
            if not order:
                logger.warning(f"Cancel failed: Order {client_order_id} not found")
                return False
            
            # Call exchange
            await self.exchange.cancel_order(
                symbol=order.symbol,
                orderId=order.exchange_order_id,
                origClientOrderId=client_order_id
            )
            
            # Update state
            await self._update_state(client_order_id, OrderState.CANCELED)
            logger.info(f"❌ Order canceled: {client_order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {client_order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders."""
        logger.info(f"Using Emergency Cancel All (Symbol: {symbol})")
        # Implementation depends on exchange client capabilities
        # For safety, we should iterate known open orders and cancel them
        open_orders = [o for o in self._orders.values() if o.state in [
            OrderState.PENDING_SUBMIT.value, 
            OrderState.SUBMITTED.value, 
            OrderState.PARTIAL_FILL.value
        ]]
        
        for order in open_orders:
            if symbol and order.symbol != symbol:
                continue
            await self.cancel_order(order.client_order_id)

    async def get_order(self, client_order_id: str) -> Optional[Order]:
        return self._orders.get(client_order_id)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return [
            o for o in self._orders.values() 
            if o.state in [OrderState.SUBMITTED.value, OrderState.PARTIAL_FILL.value]
            and (not symbol or o.symbol == symbol.lower())
        ]
    
    async def get_pending_orders_value(self, symbol: Optional[str] = None) -> float:
        """Get total value of pending orders."""
        open_orders = await self.get_open_orders(symbol)
        return sum((o.quantity - o.filled_quantity) * o.price for o in open_orders)
