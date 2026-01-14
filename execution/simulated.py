import logging
import uuid
import asyncio
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from core.events import SignalEvent, OrderEvent, FillEvent, MarketEvent
from database.db_manager import DatabaseManager
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.risk_gatekeeper import RiskGatekeeper
from config.settings import settings

logger = logging.getLogger("Execution.Simulated")

class MockExchangeClient:
    """
    Simulates Binance AsyncClient for paper trading.
    Keeps track of 'exchange' state (orders, positions, balances).
    """
    def __init__(self):
        self.orders = {}  # exchange_order_id -> order_dict
        self.positions = defaultdict(lambda: {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnl': 0.0})
        self.current_prices = {}
        self.balances = {'USDT': settings.INITIAL_CAPITAL}
        self.order_id_counter = 100000

    # Alias for compatibility
    def create_order(self, **kwargs):
        return self.futures_create_order(**kwargs)

    async def futures_create_order(self, **kwargs):
        """Simulate creating an order."""
        symbol = kwargs.get('symbol').lower()
        side = kwargs.get('side').upper()
        quantity = float(kwargs.get('quantity'))
        price = float(kwargs.get('price', 0))
        order_type = kwargs.get('type', 'LIMIT').upper()
        client_order_id = kwargs.get('newClientOrderId', str(uuid.uuid4()))
        
        # Market order price estimation
        if order_type == 'MARKET':
            price = self.current_prices.get(symbol, 0)
            if price == 0:
                # If no price yet, assume slightly above 0 to avoid crashes, 
                # but in reality we should wait for a tick.
                price = 10000.0 
                logger.warning(f"Simulating fill at fallback price {price} for {symbol}")

        # Generate Exchange ID
        exchange_id = str(self.order_id_counter)
        self.order_id_counter += 1
        
        # Store order
        order = {
            'orderId': exchange_id,
            'clientOrderId': client_order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'origQty': quantity,
            'price': price,
            'executedQty': 0.0,
            'status': 'NEW',
            'avgPrice': 0.0
        }
        self.orders[exchange_id] = order
        
        # Auto-fill immediately for simulation simplicity
        # (In a real sim we might wait for price cross, but for now we fill immediately)
        asyncio.create_task(self._simulate_fill(exchange_id, price, quantity))
        
        return order

    async def futures_cancel_order(self, **kwargs):
        """Simulate cancelling an order."""
        # For this simple sim, orders fill immediately, so cancel is rarely hit
        pass

    async def get_positions(self) -> List[Dict]:
        """Return positions in Binance format."""
        pos_list = []
        for symbol, data in self.positions.items():
            pos_list.append({
                'symbol': symbol.upper(),
                'quantity': data['amount'],
                'avgPrice': data['entryPrice'],
                'unrealizedPnl': data['unrealizedPnl'],
                'positionAmt': data['amount'],
                'entryPrice': data['entryPrice']
            })
        return pos_list

    async def futures_exchange_info(self):
        """Return dummy exchange info."""
        return {'symbols': []} # Not strictly used in current mock logic

    async def _simulate_fill(self, exchange_order_id, price, quantity):
        """Helper to process a fill and update 'exchange' state."""
        order = self.orders.get(exchange_order_id)
        if not order:
            return

        symbol = order['symbol']
        side = order['side']
        commission = (quantity * price) * 0.0004 # 0.04% fee
        
        # Update order status
        order['status'] = 'FILLED'
        order['executedQty'] = quantity
        order['avgPrice'] = price
        
        # Calculate PnL if closing/reducing
        realized_pnl = 0.0
        
        # 1. Get current position state (BEFORE update)
        current_pos = self.positions[symbol]
        old_amt = current_pos['amount']
        
        signed_qty = quantity if side == 'BUY' else -quantity
        new_amt = old_amt + signed_qty
        
        # Check if we are reducing (signed_qty and old_amt carry opposite signs)
        # OR if we flipped (old_amt * new_amt < 0)
        
        # Simple Logic: 
        # If we held Long (old_amt > 0) and Sold (signed_qty < 0): PnL = (Price - Entry) * QtySold
        # If we held Short (old_amt < 0) and Bought (signed_qty > 0): PnL = (Entry - Price) * QtyBought
        
        qty_closed = 0.0
        
        if old_amt > 0 and signed_qty < 0:
            # Closing Long
            qty_closed = min(abs(old_amt), abs(signed_qty))
            realized_pnl = (price - current_pos['entryPrice']) * qty_closed
            
        elif old_amt < 0 and signed_qty > 0:
            # Closing Short
            qty_closed = min(abs(old_amt), abs(signed_qty))
            realized_pnl = (current_pos['entryPrice'] - price) * qty_closed
            
        # Subtract commission from PnL? Usually PnL is Gross or Net. 
        # Let's make it Net PnL for dashboard clarity.
        realized_pnl -= commission
        
        # Update Average Entry Price (if flipping, entry becomes fill price)
        # Update Average Entry Price (if flipping, entry becomes fill price)
        if abs(new_amt) > 0.000001:
            # Case 0: Opening from flat (old_amt is near 0)
            if abs(old_amt) < 0.000001:
                current_pos['entryPrice'] = price
                
            # Case 1: Increasing position (Same signs)
            elif (old_amt > 0 and signed_qty > 0) or (old_amt < 0 and signed_qty < 0):
                total_cost = (abs(old_amt) * current_pos['entryPrice']) + (quantity * price)
                current_pos['entryPrice'] = total_cost / abs(new_amt)
            
            # Case 2: Reducing (Opposite signs, no flip)
            elif (abs(signed_qty) <= abs(old_amt)):
                # Entry price doesn't change when reducing
                pass
                
            # Case 3: Flipping (Opposite signs, flip)
            else:
                 current_pos['entryPrice'] = price
        else:
            current_pos['entryPrice'] = 0.0
            
        current_pos['amount'] = new_amt
        
        # Callback hook (to be set by Handler)
        if hasattr(self, 'on_fill_callback'):
            await self.on_fill_callback({
                'e': 'ORDER_TRADE_UPDATE',
                'X': 'FILLED',
                'x': 'TRADE',
                'c': order['clientOrderId'],
                'i': exchange_order_id,
                's': symbol.upper(),
                'S': side,
                'l': quantity,
                'L': price,
                'n': commission,
                'N': 'USDT',
                'q': realized_pnl # Custom field for PnL
            })


class MockExecutionHandler:
    """
    Paper Trading Handler that uses the REAL OrderManager.
    """
    def __init__(self, event_queue, risk_manager):
        self.queue = event_queue
        self.risk_manager = risk_manager
        
        self.db = DatabaseManager()
        self.client = MockExchangeClient()
        
        # Initialize Core Components
        self.position_tracker = PositionTracker(self.client, self.db)
        self.gatekeeper = RiskGatekeeper(self.position_tracker, self.client)
        self.order_manager = OrderManager(self.client, self.db, self.position_tracker, self.gatekeeper)
        
        # Bind client callback to self
        self.client.on_fill_callback = self._handle_exchange_update

    async def connect(self):
        """Initialize components."""
        logger.info("🧻 Initializing PAPER TRADING environment...")
        
        # Sync positions (initially empty)
        await self.position_tracker.initialize()
        await self.order_manager.initialize()
        
        logger.info("✅ Paper Trading Ready")

    async def on_tick(self, event: MarketEvent):
        """Update mock exchange prices."""
        self.client.current_prices[event.symbol.lower()] = event.price

    async def execute(self, signal: SignalEvent):
        """Submit order via OrderManager."""
        try:
            # Create order via OrderManager (handles Risk, DB, State)
            # Default fallback price if not found
            price = signal.price
            if not price:
                price = self.client.current_prices.get(signal.symbol.lower(), 0.0)

            await self.order_manager.submit_order(
                symbol=signal.symbol,
                quantity=signal.quantity,
                price=price,
                side=signal.side,
                order_type=getattr(signal, 'order_type', 'LIMIT'),
                time_in_force="GTC"
            )
        except Exception as e:
            logger.error(f"Paper execute failed: {e}")

    async def _handle_exchange_update(self, data: dict):
        """
        Callback from MockExchangeClient when an order fills.
        Mimics receiving a WebSocket message from Binance.
        """
        # We only care about fills here
        if data['x'] == 'TRADE' and data['X'] == 'FILLED':
            client_order_id = data['c']
            filled_qty = float(data['l'])
            fill_price = float(data['L'])
            commission = float(data['n'])
            realized_pnl = float(data.get('q', 0.0)) # q for quote/pnl
            
            # Process via OrderManager
            await self.order_manager.process_fill(
                client_order_id=client_order_id,
                filled_qty=filled_qty,
                fill_price=fill_price,
                commission=commission,
                pnl=realized_pnl
            )
            
            # Create FillEvent for System
            fill_event = FillEvent(
                symbol=data['s'],
                side=data['S'],
                quantity=filled_qty,
                price=fill_price,
                commission=commission,
                pnl=realized_pnl # Pass calculated PnL
            )
            await self.queue.put(fill_event)
            logger.info(f"🧻 PAPER FILL: {data['S']} {filled_qty} {data['s']} @ {fill_price} (PnL: {realized_pnl:.4f})")

    async def cancel_all_orders(self):
        """Cancel all orders."""
        # For paper trading, we just clear local state if needed
        # But properly we should ask OrderManager to cancel
        await self.order_manager.cancel_all_orders()

    async def close(self):
        pass