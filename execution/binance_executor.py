import logging
import os
import asyncio
import json
from collections import defaultdict

# Binance Imports
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

# System Imports
from core.events import SignalEvent, FillEvent
from database.db_manager import DatabaseManager

logger = logging.getLogger("Execution.Binance")

class BinanceExecutionHandler:
    def __init__(self, event_queue, risk_manager, testnet=True):
        self.queue = event_queue
        self.risk_manager = risk_manager
        self.testnet = testnet
        self.client = None
        self.bsm = None # Socket Manager
        
        self.db = DatabaseManager()
        
        # Position Tracking (Symbol -> Quantity)
        self.positions = defaultdict(float)
        self.avg_entry_price = defaultdict(float)

        self.api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        self.api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")

        if not self.api_key or not self.api_secret:
            logger.critical("❌ MISSING API KEYS in .env file!")

    async def connect(self):
        """Startar API-klienten OCH WebSocket-lyssnaren."""
        try:
            self.client = await AsyncClient.create(
                self.api_key, 
                self.api_secret, 
                testnet=self.testnet
            )
            logger.info("✅ Connected to Binance REST API")
            
            # Starta User Data Stream (Lyssna på fills)
            self.bsm = BinanceSocketManager(self.client)
            # Starta lyssnar-loopen i bakgrunden
            asyncio.create_task(self._user_data_stream())
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")

    async def _user_data_stream(self):
        """Bakgrundsprocess som lyssnar på uppdateringar från Binance."""
        logger.info("🎧 Starting User Data Stream (Waiting for fills)...")
        
        # Välj rätt socket beroende på om det är Futures eller Spot (här Futures)
        ts = self.bsm.futures_user_socket()
        
        async with ts as tscm:
            while True:
                try:
                    res = await tscm.recv()
                    
                    if res is None:
                        continue
                        
                    # Event Type 'e': ORDER_TRADE_UPDATE är det vi vill ha
                    if res.get('e') == 'ORDER_TRADE_UPDATE':
                        order_data = res['o']
                        await self._handle_order_update(order_data)
                        
                except asyncio.CancelledError:
                    logger.info("User Stream cancelled.")
                    break
                except Exception as e:
                    logger.error(f"User Stream Error: {e}")
                    # Vänta lite innan vi försöker igen så vi inte spammar vid fel
                    await asyncio.sleep(5)

    async def _handle_order_update(self, data):
        """Bearbetar en orderuppdatering."""
        status = data['X']      # Order Status (NEW, FILLED, CANCELED...)
        exec_type = data['x']   # Execution Type (TRADE = Fill)
        
        # Vi agerar bara om det faktiskt skedde en handel (TRADE)
        if exec_type == 'TRADE' and status in ['FILLED', 'PARTIALLY_FILLED']:
            symbol = data['s'].lower()
            side = data['S'].upper()
            last_filled_qty = float(data['l'])     # Antal i just denna fill
            last_filled_price = float(data['L'])   # Priset i just denna fill
            commission = float(data.get('n', 0))   # Avgift
            
            logger.info(f"⚡ REAL FILL DETECTED: {side} {last_filled_qty} {symbol} @ {last_filled_price}")
            
            # 1. Beräkna PnL & Uppdatera innehav
            realized_pnl = self._calculate_pnl(symbol, side, last_filled_qty, last_filled_price)
            self._update_position(symbol, side, last_filled_qty, last_filled_price)
            
            # 2. Skapa FillEvent
            fill_event = FillEvent(
                symbol=symbol,
                side=side,
                quantity=last_filled_qty,
                price=last_filled_price,
                commission=commission,
                pnl=realized_pnl
            )
            
            # 3. Logga till databas
            await self.db.log_trade(fill_event, strategy_id="binance_live")
            logger.info(f"💾 Trade logged to DB. PnL: {realized_pnl:.4f}")
            
            # 4. Skicka till motorn (så strategin vet)
            await self.queue.put(fill_event)

    async def execute(self, signal: SignalEvent):
        """Skickar order till Binance (men skapar INTE Fake Fills längre)."""
        if not self.client:
            logger.error("Cannot execute: Client not connected.")
            return

        symbol = signal.symbol.upper()
        side = signal.side.upper()
        
        price_prec, qty_prec = self._get_precision(symbol)
        quantity = round(signal.quantity, qty_prec)
        price_str = str(round(signal.price, price_prec))

        logger.info(f"📤 SENDING ORDER: {side} {quantity} {symbol} @ {price_str}")

        try:
            # Notera: Inget 'await queue.put(fill)' här längre!
            # Vi litar på att _user_data_stream fångar upp det när det händer.
            order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price_str,
                recvWindow=10000 # Fix för timestamp-felet
            )
            logger.info(f"✅ ORDER PLACED: ID {order['orderId']} (Waiting for fill...)")

        except BinanceAPIException as e:
            logger.error(f"❌ BINANCE API ERROR: {e.message} (Code: {e.code})")
        except Exception as e:
            logger.error(f"❌ EXECUTION ERROR: {e}")

    def _get_precision(self, symbol):
        if symbol.upper().startswith("BTC"): return 1, 3
        elif symbol.upper().startswith("ETH"): return 2, 3
        return 2, 3

    def _calculate_pnl(self, symbol, side, qty, exit_price):
        current_pos = self.positions[symbol]
        entry_price = self.avg_entry_price[symbol]
        pnl = 0.0
        
        if side == 'SELL' and current_pos > 0:
            pnl = (exit_price - entry_price) * qty
        elif side == 'BUY' and current_pos < 0:
            pnl = (entry_price - exit_price) * qty
        return pnl

    def _update_position(self, symbol, side, qty, price):
        prev_pos = self.positions[symbol]
        if side == 'BUY':
            if prev_pos >= 0:
                total_cost = (prev_pos * self.avg_entry_price[symbol]) + (qty * price)
                new_qty = prev_pos + qty
                if new_qty > 0: self.avg_entry_price[symbol] = total_cost / new_qty
            self.positions[symbol] += qty
        elif side == 'SELL':
            if prev_pos <= 0:
                total_cost = (abs(prev_pos) * self.avg_entry_price[symbol]) + (qty * price)
                new_qty = abs(prev_pos) + qty
                if new_qty > 0: self.avg_entry_price[symbol] = total_cost / new_qty
            self.positions[symbol] -= qty

    async def close(self):
        if self.client:
            await self.client.close_connection()
    
    async def on_tick(self, event): pass 
    def get_total_equity(self, current_balance): return current_balance