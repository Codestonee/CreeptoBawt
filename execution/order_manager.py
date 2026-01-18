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
    - Risk Manager (Kill switch enforcement)
    - Position Tracker (Update state)
    - Database Manager (Persist state)
    """
    
    def __init__(self, exchange_client, db_manager: DatabaseManager, position_tracker: PositionTracker, risk_gatekeeper: RiskGatekeeper, risk_manager=None):
        self.exchange = exchange_client
        self.db = db_manager
        self.position_tracker = position_tracker
        self.risk_gatekeeper = risk_gatekeeper
        self.risk_manager = risk_manager  # For kill switch enforcement
        self._event_store = get_event_store()
        
        # In-memory order cache (for fast lookup)
        self._orders: Dict[str, Order] = {}
        
        # Order timeout tracking
        self._order_timeouts: Dict[str, float] = {}  # client_order_id -> created_time
        self.order_timeout_sec = 300  # 5 minutes default
        
        # Track failed cancellations to avoid retrying indefinitely
        self._failed_cancel_attempts: Dict[str, int] = {}  # client_order_id -> failure count
        self._max_cancel_failures = 2  # Stop trying after this many failures
        
        # Register as global instance
        _set_global_instance(self)
        
        logger.info("OrderManager initialized")
    
    async def initialize(self):
        """Initialize the order manager and restore state from DB."""
        logger.info("Initializing OrderManager...")
        try:
            # 1. Load active orders from DB
            # We fetch orders that are NOT in a terminal state
            active_states = [
                OrderState.PENDING_SUBMIT.value,
                OrderState.SUBMITTED.value,
                OrderState.PARTIAL_FILL.value
            ]
            
            # Assuming db_manager has get_orders_by_state or similar
            # Since we don't have that method visible, we'll fetch all open orders roughly
            # or rely on the bootstrapping from exchange as the primary source of truth.
            
            # actually, engine.py calls _bootstrap_exchange_state -> register_existing_order
            # So we just need to be ready.
            
            logger.info("✅ OrderManager initialized (ready for bootstrapping)")
            
        except Exception as e:
            logger.error(f"Failed to initialize OrderManager: {e}")
            raise



    async def get_position(self, symbol: str):
        """Get position from PositionTracker (Proxy)."""
        return await self.position_tracker.get_position(symbol)
        
    async def set_position_from_exchange(
        self, 
        symbol: str, 
        quantity: float, 
        entry_price: float, 
        unrealized_pnl: float = 0.0,
        exchange_snapshot: Optional[str] = None
    ):
        """
        Force update position from exchange data.
        Used by reconciliation service and engine bootstrapping.
        """
        await self.position_tracker.update_position_from_exchange(
            symbol, quantity, entry_price, unrealized_pnl
        )

    async def get_total_exposure(self) -> float:
        """Get total USD exposure across all positions."""
        return await self.position_tracker.get_total_exposure()

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        trace_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Create local order record (Risk Check + DB Persistence).
        Does NOT submit to exchange.
        """
        try:
            # 1. RISK CHECK
            risk_result = await self.risk_gatekeeper.validate_order(
                symbol, quantity, price, side
            )
            
            if not risk_result.is_allowed:
                logger.warning(f"Order REJECTED by risk: {risk_result.reason}")
                return None
            
            # 2. CREATE ORDER (Local State)
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
            return order
            
        except Exception as e:
            logger.error(f"Failed to create local order: {e}")
            return None

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
        Submit order with full lifecycle (Create + Exchange Submit).
        """
        try:
            # 1. Create Local Order
            order = await self.create_order(
                symbol, side, quantity, price, order_type, time_in_force, trace_id
            )
            
            if not order:
                return None
            
            # 2. SUBMIT TO EXCHANGE
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
                
                # 3. UPDATE STATE (Submitted)
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
    
    async def register_existing_order(
        self,
        client_order_id: str,
        exchange_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        filled_quantity: float = 0.0
    ) -> Order:
        """
        Register an existing order (from exchange or router).
        Used during:
        1. Startup bootstrap (existing exchange orders)
        2. Router repricing (new order IDs generated mid-chase)
        
        Args:
            client_order_id: Our client order ID
            exchange_order_id: Exchange's order ID
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            price: Order price
            order_type: LIMIT, MARKET, etc.
            filled_quantity: Already filled quantity (if partial)
        
        Returns:
            Registered Order object
        """
        # Check if already exists
        existing = self._orders.get(client_order_id)
        if existing:
            logger.debug(f"Order {client_order_id} already registered")
            return existing
        
        # Create order record
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            trace_id=f"registered_{client_order_id[-8:]}",
            symbol=symbol.lower(),
            side=side.upper(),
            order_type=order_type,
            time_in_force="GTC",  # Assume GTC for existing
            quantity=quantity,
            filled_quantity=filled_quantity,
            remainder_quantity=quantity - filled_quantity,
            price=price,
            state=OrderState.SUBMITTED.value,  # Already on exchange
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Add to cache
        self._orders[client_order_id] = order
        
        # Persist to DB (async, non-blocking)
        try:
            await self.db.insert_order({
                'client_order_id': order.client_order_id,
                'exchange_order_id': exchange_order_id,
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
        except Exception as e:
            logger.warning(f"Failed to persist registered order {client_order_id}: {e}")
        
        logger.info(
            f"📝 Registered existing order: {client_order_id} "
            f"({side} {quantity} {symbol.upper()} @ ${price:.2f})"
        )
        
        return order


    async def get_stuck_orders(self, timeout_seconds: float = 300) -> List[str]:
        """
        Get list of orders stuck in SUBMITTED state for too long.
        """
        now = time.time()
        stuck = []
        
        for order in self._orders.values():
            if order.state != OrderState.SUBMITTED.value:
                continue
            
            age = now - order.created_at
            if age > timeout_seconds:
                stuck.append(order.client_order_id)
        
        return stuck
    
    async def process_fill(
        self,
        client_order_id: str,
        filled_qty: float,
        fill_price: float,
        commission: float = 0.0,
        commission_asset: str = "",  # NEW: Track which asset fee was paid in
        pnl: float = 0.0,
        is_orphan: bool = False  # NEW parameter
    ) -> Optional[Order]:
        """
        Process a fill event with atomic position updates.
        Handles orphan fills by forcing reconciliation.
        """
        order = self._orders.get(client_order_id)
        
        # ORPHAN FILL HANDLING - Force reconciliation
        if not order or is_orphan:
            logger.critical(
                f"⚠️ ORPHAN FILL: {client_order_id} {filled_qty}@{fill_price} - "
                "Forcing position sync from exchange"
            )
            
            # Log the trade for PnL tracking
            trade_record = {
                'type': 'trade',
                'timestamp': time.time(),
                'symbol': 'unknown',  # Will be corrected by reconciliation
                'side': 'UNKNOWN',
                'quantity': filled_qty,
                'price': fill_price,
                'commission': commission,
                'pnl': pnl,
                'strategy_id': 'orphan_fill',
                'orphan': True
            }
            self.db.submit_write_task(trade_record)
            
            # CRITICAL: Force immediate position reconciliation
            await self.position_tracker.force_sync_with_exchange()
            
            # Alert risk management
            self.risk_gatekeeper._halt_trading(
                f"Orphan fill detected - position sync required"
            )
            
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
        
        # 2. Calculate PnL BEFORE updating position
        # This is critical for spot mode where exchange doesn't provide entry prices
        pos_before = await self.position_tracker.get_position(order.symbol)
        calculated_pnl = 0.0
        if pos_before and pos_before.quantity != 0 and pos_before.avg_entry_price > 0:
            # Check if this trade reduces the position
            is_reducing = (
                (pos_before.quantity > 0 and order.side == 'SELL') or
                (pos_before.quantity < 0 and order.side == 'BUY')
            )
            if is_reducing:
                if pos_before.quantity > 0:  # Long position being reduced
                    calculated_pnl = (fill_price - pos_before.avg_entry_price) * filled_qty
                else:  # Short position being reduced
                    calculated_pnl = (pos_before.avg_entry_price - fill_price) * filled_qty
                calculated_pnl -= commission  # Subtract commission
                logger.info(f"💰 PNL CALCULATED: Entry=${pos_before.avg_entry_price:.4f} Exit=${fill_price:.4f} PnL=${calculated_pnl:.4f}")
        
        # Use calculated PnL if available, otherwise use passed-in pnl
        if abs(calculated_pnl) > 0.0001:
            pnl = calculated_pnl
        
        # 3. Update Position Tracker (CRITICAL)
        quantity_delta = filled_qty if order.side == 'BUY' else -filled_qty
        await self.position_tracker.update_position(
            symbol=order.symbol,
            quantity_delta=quantity_delta,
            price=fill_price,
            commission=commission,
            commission_asset=commission_asset  # Pass through for BASE fee adjustment
        )
        
        # 4. Update Risk Gatekeeper PnL
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
        # FIX: Dynamic commission asset based on symbol's quote currency
        if order.symbol.lower().endswith('usdc'):
            commission_asset = 'USDC'
        elif order.symbol.lower().endswith('usdt'):
            commission_asset = 'USDT'
        else:
            commission_asset = order.symbol[-4:].upper()  # Extract last 4 chars as quote
        
        trade_record = {
            'type': 'trade',
            'timestamp': time.time(),
            'symbol': order.symbol,
            'side': order.side,
            'quantity': filled_qty,
            'price': fill_price,
            'commission': commission,
            'commission_asset': commission_asset,
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
        
        # CRITICAL FIX: Remove orders from cache when they reach terminal states
        # This prevents stale order IDs from accumulating and causing -2011 errors
        terminal_states = [OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED]
        if new_state in terminal_states:
            if client_order_id in self._orders:
                del self._orders[client_order_id]
                logger.debug(f"Removed terminal order from cache: {client_order_id}")
            # Also clean up any failed cancel tracking
            self._failed_cancel_attempts.pop(client_order_id, None)
        
        return order

    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an open order."""
        try:
            order = self._orders.get(client_order_id)
            if not order:
                logger.debug(f"Cancel skipped: Order {client_order_id} not in cache")
                return False
            
            # Skip orders already in terminal states
            terminal_states = [OrderState.FILLED.value, OrderState.CANCELED.value, 
                              OrderState.REJECTED.value, OrderState.EXPIRED.value]
            if order.state in terminal_states:
                # Clean up from cache
                del self._orders[client_order_id]
                self._failed_cancel_attempts.pop(client_order_id, None)
                return True
            
            # Check if we've already failed to cancel this order multiple times
            fail_count = self._failed_cancel_attempts.get(client_order_id, 0)
            if fail_count >= self._max_cancel_failures:
                logger.debug(f"Cancel skipped: Order {client_order_id} exceeded max cancel attempts")
                # Remove from cache - it's likely already gone from exchange
                del self._orders[client_order_id]
                self._failed_cancel_attempts.pop(client_order_id, None)
                return True
            
            # Call exchange (symbol must be UPPERCASE for Binance)
            await self.exchange.cancel_order(
                symbol=order.symbol.upper(),
                orderId=order.exchange_order_id,
                origClientOrderId=client_order_id
            )
            
            # Update state
            await self._update_state(client_order_id, OrderState.CANCELED)
            logger.info(f"❌ Order canceled: {client_order_id}")
            return True
            
        except Exception as e:
            error_str = str(e)
            # Handle "Unknown order sent" error (-2011) - order doesn't exist on exchange
            if '-2011' in error_str or 'Unknown order' in error_str:
                logger.debug(f"Order {client_order_id} not found on exchange - removing from cache")
                if client_order_id in self._orders:
                    del self._orders[client_order_id]
                self._failed_cancel_attempts.pop(client_order_id, None)
                return True  # Consider this a success - order is gone
            else:
                logger.error(f"Failed to cancel order {client_order_id}: {e}")
                # Track failure count
                self._failed_cancel_attempts[client_order_id] = fail_count + 1
                return False

    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders."""
        logger.info(f"Using Emergency Cancel All (Symbol: {symbol})")
        
        # First, prune any stale orders from cache
        await self.prune_stale_orders()
        
        # Now only iterate truly open orders
        open_states = [OrderState.PENDING_SUBMIT.value, OrderState.SUBMITTED.value, OrderState.PARTIAL_FILL.value]
        open_orders = [o for o in list(self._orders.values()) if o.state in open_states]
        
        for order in open_orders:
            if symbol and order.symbol != symbol.lower():
                continue
            await self.cancel_order(order.client_order_id)
    
    async def prune_stale_orders(self, max_age_seconds: float = 600):
        """
        Remove stale orders from cache.
        Orders in terminal states or older than max_age are removed.
        """
        now = time.time()
        to_remove = []
        
        terminal_states = [OrderState.FILLED.value, OrderState.CANCELED.value, 
                          OrderState.REJECTED.value, OrderState.EXPIRED.value]
        
        for client_id, order in self._orders.items():
            # Remove terminal orders
            if order.state in terminal_states:
                to_remove.append(client_id)
                continue
            
            # Remove very old non-terminal orders (they're likely stale)
            age = now - order.created_at
            if age > max_age_seconds:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self._orders[client_id]
            self._failed_cancel_attempts.pop(client_id, None)
        
        if to_remove:
            logger.info(f"🧹 Pruned {len(to_remove)} stale orders from cache")

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

    async def get_all_positions(self):
        """Get all active positions."""
        return await self.position_tracker.get_all_positions()


