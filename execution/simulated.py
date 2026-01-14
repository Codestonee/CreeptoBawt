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
    Now includes a REALISTIC matching engine.
    """
    def __init__(self):
        self.orders = {}  # exchange_order_id -> order_dict
        self.active_orders = defaultdict(list) # symbol -> list of order_ids
        self.positions = defaultdict(lambda: {'amount': 0.0, 'entryPrice': 0.0, 'unrealizedPnl': 0.0})
        self.current_prices = {} # symbol -> price
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
        
        # Check current market price
        current_market_price = self.current_prices.get(symbol)
        
        # Market order logic
        if order_type == 'MARKET':
            if not current_market_price:
                current_market_price = price if price > 0 else 0.0
                if current_market_price == 0:
                    logger.warning(f"Market order for {symbol} rejected - no price data")
                    raise Exception("No market price available")
            
            # Slippage simulation (TAKER)
            # Add 0.05% slippage against us
            slippage = current_market_price * 0.0005
            fill_price = current_market_price + slippage if side == 'BUY' else current_market_price - slippage
            
        else:
            # LIMIT order logic initialization
            fill_price = price # Only fills at this price or better
            
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
        
        # LOGIC BRANCHING
        if order_type == 'MARKET':
            # Fill immediately
            asyncio.create_task(self._simulate_fill(exchange_id, fill_price, quantity))
        else:
            # LIMIT ORDER: Check if marketable immediately
            filled = False
            if current_market_price:
                # Buy Limit >= Market Price -> Fill (Taker)
                if side == 'BUY' and price >= current_market_price:
                    asyncio.create_task(self._simulate_fill(exchange_id, price, quantity))
                    filled = True
                # Sell Limit <= Market Price -> Fill (Taker)
                elif side == 'SELL' and price <= current_market_price:
                    asyncio.create_task(self._simulate_fill(exchange_id, price, quantity))
                    filled = True
            
            if not filled:
                logger.info(f"?? Order {client_order_id} QUEUED @ ${price}")
                self.active_orders[symbol].append(exchange_id)
        
        return order


    async def futures_account(self, **kwargs):
        """Simulate fetching account info."""
        balance = self.balances.get('USDT', 0.0)
        return {
            'totalWalletBalance': str(balance),
            'totalMarginBalance': str(balance),
            'availableBalance': str(balance),
            'positions': []
        }

    async def futures_cancel_order(self, **kwargs):
        """Simulate cancelling an order."""
        symbol = kwargs.get('symbol', '').lower()
        order_id = kwargs.get('orderId')
        orig_client_order_id = kwargs.get('origClientOrderId')
        
        target_id = None
        if order_id:
            target_id = str(order_id)
        elif orig_client_order_id:
            # Find by client ID
            for oid, o in self.orders.items():
                if o['clientOrderId'] == orig_client_order_id:
                    target_id = oid
                    break
        
        if target_id and target_id in self.orders:
            self.orders[target_id]['status'] = 'CANCELED'
            # Remove from active queue
            if symbol and symbol in self.active_orders:
                if target_id in self.active_orders[symbol]:
                    self.active_orders[symbol].remove(target_id)
            return self.orders[target_id]
            
        return {'status': 'CANCELED'} # Fallback

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
        return {'symbols': []} 

    async def on_tick(self, symbol: str, price: float):
        """Process price updates and trigger fills for pending orders."""
        symbol = symbol.lower()
        self.current_prices[symbol] = price
        
        # Check pending orders for this symbol
        pending_ids = list(self.active_orders[symbol]) # Copy list
        
        for oid in pending_ids:
            order = self.orders.get(oid)
            if not order or order['status'] != 'NEW':
                if oid in self.active_orders[symbol]:
                    self.active_orders[symbol].remove(oid)
                continue
                
            limit_price = order['price']
            side = order['side']
            
            should_fill = False
            
            # Simple Matching Engine Logic
            if side == 'BUY':
                # Price dropped below limit
                if price <= limit_price:
                    should_fill = True
            elif side == 'SELL':
                # Price rose above limit
                if price >= limit_price:
                    should_fill = True
                    
            if should_fill:
                # Remove from queue confirmed
                self.active_orders[symbol].remove(oid)
                # Fill at LIMIT price (Maker) - technically could be better but stick to limit
                asyncio.create_task(self._simulate_fill(oid, limit_price, order['origQty']))

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
        # Feed tick to matching engine
        await self.client.on_tick(event.symbol, event.price)

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