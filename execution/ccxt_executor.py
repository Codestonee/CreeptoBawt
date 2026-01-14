import asyncio
import ccxt.async_support as ccxt
import logging
import time
import os
from typing import Optional, Dict
from core.events import MarketEvent, FillEvent
from database.db_manager import DatabaseManager

class CCXTExecutor:
    """
    Unified Execution Handler using CCXT (Async).
    Supports: Binance, OKX, and others.
    Handles: Connection, Order Placement, Balance, Ticker.
    """
    
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None, password: str = None, testnet: bool = True, event_queue = None, use_db: bool = True):
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.queue = event_queue
        self.logger = logging.getLogger(f"Execution.{exchange_id.upper()}")
        self.db = DatabaseManager() if use_db else None # Persistent DB Connection (Optional)
        
        # dynamic exchange class loading
        exchange_class = getattr(ccxt, exchange_id)
        
        target_env = 'default' # CCXT often uses 'default' or 'test'
        
        self.config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} # Default to futures (perps)
        }
        
        if password: # Required for OKX
            self.config['password'] = password
            
        self.exchange = exchange_class(self.config)
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        self.markets = None
        self.initialized = False
        self.running = False
        self.positions = {} 
        self._last_fill_check = 0

    async def start_polling(self, symbols: list):
        """Start background polling for Tickers and Fills."""
        self.running = True
        self.logger.info(f"🔄 Starting polling for {len(symbols)} symbols on {self.exchange_id}")
        asyncio.create_task(self._poll_tickers(symbols))
        asyncio.create_task(self._poll_fills(symbols))

    async def _poll_tickers(self, symbols):
        while self.running:
            for s in symbols:
                try:
                    # Map symbol format if needed (e.g. solusdt -> SOL/USDT:USDT)
                    # For now assume ccxt works with standard formatting or mapped
                    # ccxt usually needs 'SOL/USDT' or 'SOL/USDT:USDT' for futures
                    # We might need a mapper. For now try raw.
                    tick = await self.exchange.fetch_ticker(s.upper()) # Tries standard
                    if self.queue:
                        evt = MarketEvent(
                            exchange=self.exchange_id,
                            symbol=s,
                            price=tick['last'],
                            volume=tick['quoteVolume'] or 0,
                            side='SELL' if tick['last'] < tick.get('open', 0) else 'BUY' # Hacky estimation
                        )
                        await self.queue.put(evt)
                except Exception as e:
                    pass # Silence polling errors to avoid spam
            await asyncio.sleep(1.0)

    async def _poll_fills(self, symbols):
        while self.running:
            try:
                # Poll closed orders / trades
                # This is heavy. Optimized: use fetch_my_trades if supported
                now = self.exchange.milliseconds()
                if self._last_fill_check == 0:
                    start_time = now - 60000 # Look back 1 min on startup
                else:
                    start_time = self._last_fill_check
                
                # We iterate symbols or use generic fetch_my_trades?
                # CCXT often requires symbol
                for s in symbols:
                    try:
                        trades = await self.exchange.fetch_my_trades(s.upper(), since=start_time)
                        for t in trades:
                            if t['timestamp'] > self._last_fill_check:
                                # New Fill
                                if self.queue:
                                    # Canonical Event for Strategy
                                    evt = FillEvent(
                                        symbol=s,
                                        side=t['side'].upper(),
                                        quantity=float(t['amount']),
                                        price=float(t['price']),
                                        commission=float(t['fee']['cost']) if t.get('fee') else 0.0,
                                        commission_asset=t['fee']['currency'] if t.get('fee') else 'USDT',
                                        is_maker=t['info'].get('maker', False),
                                        exchange=self.exchange_id
                                    )
                                    
                                    # Log to DB with Prefix (e.g. okx_btcusdt) for Dashboard Visibility
                                    try:
                                        db_symbol = f"{self.exchange_id}_{s}" if self.exchange_id != 'binance' else s
                                        db_evt = FillEvent(
                                            symbol=db_symbol,
                                            side=evt.side,
                                            quantity=evt.quantity,
                                            price=evt.price,
                                            commission=evt.commission,
                                            commission_asset=evt.commission_asset,
                                            is_maker=evt.is_maker,
                                            exchange=evt.exchange,
                                            pnl=0.0 # No position tracking yet
                                        )
                                        await self.db.log_trade(db_evt, strategy_id="multi")
                                    except Exception as db_e:
                                        self.logger.error(f"DB Log Failed: {db_e}")
                                        
                                    await self.queue.put(evt)
                                    self.logger.info(f"⚡ FILL DETECTED ({self.exchange_id}): {evt.side} {evt.quantity} @ {evt.price}")
                        
                    except Exception:
                        pass
                
                self._last_fill_check = now
            except Exception as e:
                self.logger.error(f"Fill poll error: {e}")
            
            await asyncio.sleep(2.0)


    async def initialize(self):
        """Load markets and verify connection."""
        try:
            self.logger.info(f"Connecting to {self.exchange_id} (Testnet: {self.testnet})...")
            await self.exchange.load_markets()
            self.markets = self.exchange.markets
            
            # Check balance to verify auth
            balance = await self.exchange.fetch_balance()
            self.logger.info(f"✅ Connected to {self.exchange_id}. Balance available.")
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            return False

    async def close(self):
        """Close connection."""
        await self.exchange.close()

    async def execute(self, signal):
        """Execute a SignalEvent."""
        if not self.initialized:
            self.logger.error("Executor not initialized.")
            return None
            
        try:
            # Basic validation
            if not signal.price and signal.order_type == 'LIMIT':
                self.logger.error("Limit order missing price")
                return None
                
            response = await self.create_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.quantity,
                price=signal.price,
                order_type=signal.order_type.lower()
            )
            
            if response:
                return response.get('id')
            return None
        except Exception as e:
            self.logger.error(f"Execution Error: {e}")
            return None

    async def fetch_ticker(self, symbol: str):
        """Fetch current ticker price."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last']
            }
        except Exception as e:
            self.logger.warning(f"Ticker fail {symbol}: {e}")
            return None

    async def create_order(self, symbol: str, side: str, amount: float, price: float = None, order_type: str = 'limit', params: dict = {}):
        """
        Unified order placement.
        """
        try:
            if not self.initialized:
                self.logger.error("Executor not initialized.")
                return None
                
            response = await self.exchange.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=amount,
                price=price,
                params=params
            )
            
            self.logger.info(f"✅ Order Placed ({self.exchange_id}): {side} {amount} {symbol} @ {price}")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Order Failed ({self.exchange_id}): {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str):
        """Cancel order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"🚫 Canceled {order_id} on {self.exchange_id}")
            return True
        except Exception as e:
            self.logger.error(f"Cancel failed: {e}")
            return False

    async def fetch_position(self, symbol: str):
        """
        Fetch positions. CCXT normalization for positions takes some work.
        """
        try:
            # fetchPositions is not always standard, often requires params
            positions = await self.exchange.fetch_positions([symbol])
            
            # Find the target symbol
            for p in positions:
                if p['symbol'] == symbol:
                    qty = float(p['contracts']) if 'contracts' in p else float(p['info'].get('size', 0)) # Logic varies by exchange
                    # Normalize:
                    # CCXT structure: 'side': 'long'/'short', 'contracts': 1.0 (qty)
                    # We want signed quantity.
                    
                    raw_qty = float(p.get('contracts', 0) or p.get('amount', 0))
                    side = p.get('side')
                    
                    if side == 'short':
                        raw_qty = -abs(raw_qty)
                    else:
                        raw_qty = abs(raw_qty)
                        
                    return {
                        'symbol': symbol,
                        'quantity': raw_qty,
                        'entry_price': p.get('entryPrice', 0),
                        'unrealized_pnl': p.get('unrealizedPnl', 0)
                    }
            return None
        except Exception as e:
            self.logger.error(f"Position fetch error: {e}")
            return None
