# execution/okx_executor.py
import logging
import os
import asyncio
import json
import time
import hmac
import base64
import hashlib
import aiohttp
import datetime  # Added for time sync
from typing import Optional, Dict


# System Imports
from core.events import SignalEvent, FillEvent
from core.event_store import get_event_store, EventType
from execution.order_manager import get_order_manager, OrderState
from utils.nonce_service import get_nonce_service
from database.db_manager import DatabaseManager

logger = logging.getLogger("Execution.OKX")

class OkxExecutionHandler:
    """
    OKX Futures Execution Handler (V5 API).
    Uses aiohttp for REST and WebSocket interactions.
    """
    
    
    # Defaults (will be overridden by settings if available)
    DEFAULT_BASE_URL = "https://eea.okx.com" 
    WS_URL = "wss://ws.okx.com:8443/ws/v5/private" # WS is usually same
    
    def __init__(self, event_queue, risk_manager, testnet: bool = False):
        self.queue = event_queue
        self.risk_manager = risk_manager
        self.testnet = testnet
        
        # Load URL from Settings (via env or config import if we had it, but here we can check env/settings)
        from config.settings import settings
        self.BASE_URL = getattr(settings, 'OKX_API_URL', self.DEFAULT_BASE_URL)
        
        self.time_offset = 0 # ms offset
        
        # Dynamic WS URL based on REST URL (critical for EU users)
        if "eea.okx.com" in self.BASE_URL:
             self.WS_URL = "wss://wseea.okx.com:8443/ws/v5/private"
             logger.info(f"🌍 Detected OKX EU/EEA endpoint. Using WS: {self.WS_URL}")
        else:
             self.WS_URL = "wss://ws.okx.com:8443/ws/v5/private"
        
        self.api_key = os.getenv("OKX_API_KEY")
        self.api_secret = os.getenv("OKX_SECRET_KEY")
        self.passphrase = os.getenv("OKX_PASSPHRASE")
        
        if not self.api_key or not self.api_secret or not self.passphrase:
            logger.critical("❌ MISSING OKX API KEYS in .env file!")
            
        self.session = None
        self.ws = None
        
        # Services
        self.order_manager = get_order_manager()
        self.event_store = get_event_store()
        self.db = DatabaseManager()
        
        # Cache
        self.instrument_info: Dict[str, dict] = {} # symbol -> {tickSz, lotSz, instId}
        self._order_trace_map: Dict[str, str] = {}
        
    async def connect(self):
        """Connect to OKX API."""
        try:
            # Sync time first
            await self._sync_time()
            
            # OKX V5 uses headers for auth, no persistent session setup needed for REST
            # But we need WebSocket for data
            self.session = aiohttp.ClientSession()
            logger.info("✅ OKX Execution Handler Connected")
            
            # 1. Fetch Instruments (Dynamic Precision)
            await self._fetch_instrument_info()
            
            # 2. Connect WebSocket
            asyncio.create_task(self._connect_ws())
            
            await self.event_store.append(
                EventType.SYSTEM_START,
                payload={"exchange": "okx"},
                trace_id=self.event_store.generate_trace_id()
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to OKX: {e}")

    async def _fetch_instrument_info(self):
        """Fetch instrument info for active symbols."""
        # We fetch ALL SWAP instruments to cache precision
        path = "/api/v5/public/instruments?instType=SWAP"
        data = await self._rest_request("GET", path)
        
        if data and data.get('code') == '0':
            for item in data['data']:
                # item['instId'] e.g. "BTC-USDT-SWAP"
                # Map back to "btcusdt" if needed, or just store by instId
                inst_id = item['instId']
                
                # Simple mapping: BTC-USDT-SWAP -> btcusdt
                if "USDT-SWAP" in inst_id:
                    base = inst_id.split("-")[0] # BTC
                    symbol = f"{base}usdt".lower()
                    
                    self.instrument_info[symbol] = {
                        'tickSz': float(item['tickSz']) if item.get('tickSz') else 0.0,
                        'lotSz': float(item['lotSz']) if item.get('lotSz') else 0.0,
                        'minSz': float(item['minSz']) if item.get('minSz') else 0.0,
                        'ctVal': float(item['ctVal']) if item.get('ctVal') else 0.0,  # Contract Value (e.g. 0.01 BTC)
                        'instId': inst_id
                    }
                    self.instrument_info[inst_id] = self.instrument_info[symbol] # Store both keys
            
            logger.info(f"Loaded precision info for {len(self.instrument_info)} symbols")

    async def _connect_ws(self):
        """Maintain WebSocket connection."""
        while True:
            try:
                async with self.session.ws_connect(self.WS_URL) as ws:
                    self.ws = ws
                    
                    # Login
                    await self._ws_login()
                    
                    # Subscribe to orders and positions
                    await self._ws_subscribe()
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_ws_message(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
            except Exception as e:
                logger.error(f"WS Error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _ws_login(self):
        """Perform WS Login."""
        timestamp = str(int(time.time()))
        message = f"{timestamp}GET/users/self/verify"
        signature = self._generate_signature(message)
        
        login_cmd = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        }
        await self.ws.send_json(login_cmd)
        logger.debug("OKX WS: Sending login...")

    async def _ws_subscribe(self):
        """Subscribe to private channels."""
        # Wait a bit for login to process (naive)
        await asyncio.sleep(1)
        
        sub_cmd = {
            "op": "subscribe",
            "args": [
                {"channel": "orders", "instType": "SWAP"},
                {"channel": "positions", "instType": "SWAP"}
            ]
        }
        await self.ws.send_json(sub_cmd)

    async def _process_ws_message(self, data: dict):
        """Handle WS messages."""
        if  data.get('event') == 'login':
            logger.debug("✅ OKX WS: Logged in")
            return
            
        if 'arg' not in data or 'data' not in data:
            return
            
        channel = data['arg']['channel']
        
        for item in data['data']:
            if channel == 'orders':
                await self._handle_order_update(item)

    async def _handle_order_update(self, item: dict):
        """Handle order status update."""
        client_oid = item.get('clOrdId')
        state = item.get('state') # live, filled, canceled
        
        # Map OKX state to internal
        if state == 'live':
            # Could be new or partially filled
            pass
        elif state == 'filled':
            # FULL FILL
            filled_sz = float(item.get('fillSz', 0)) # Last fill size
            price = float(item.get('fillPx', 0))
            fee = float(item.get('fee', 0))
            
            if filled_sz > 0:
                await self.order_manager.process_fill(
                    client_order_id=client_oid,
                    filled_qty=filled_sz,
                    fill_price=price,
                    commission=abs(fee)
                )
                
                # Dispatch FillEvent
                # Need to map instId (BTC-USDT-SWAP) back to symbol (btcusdt)
                inst_id = item['instId']
                symbol = inst_id.split("-")[0].lower() + "usdt"
                
                fill_event = FillEvent(
                    symbol=symbol,
                    side=item['side'].upper(), # buy/sell
                    quantity=filled_sz,
                    price=price,
                    commission=abs(fee),
                    pnl=float(item.get('pnl', 0))
                )
                await self.queue.put(fill_event)

    async def execute(self, signal: SignalEvent) -> Optional[str]:
        """Execute order via REST."""
        symbol = signal.symbol.lower() # btcusdt
        
        info = self.instrument_info.get(symbol)
        if not info:
            logger.error(f"No instrument info for {symbol}")
            return None
            
        inst_id = info['instId']
        
        # ------------------------------------------------------------------
        # FIX: OKX Contract Sizing (The "Unit Trap")
        # ------------------------------------------------------------------
        # Strategy sends quantity in COINS (e.g. 0.05 BTC).
        # OKX Swaps require 'sz' in CONTRACTS (e.g. 5 contracts if ctVal=0.01).
        
        ct_val = info.get('ctVal', 0.0)
        
        if ct_val > 0:
            # We are likely trading a SWAP/FUTURES pair where sz = contracts
            # Calculate number of contracts
            num_contracts = int(signal.quantity / ct_val)
            
            if num_contracts < 1:
                logger.warning(f"Quantity {signal.quantity} too small for {symbol} (ctVal={ct_val})")
                return None
                
            qty = num_contracts * ct_val  # Actual coin quantity we are trading
            sz_param = str(num_contracts) # Params for OKX (Contracts)
            
            logger.info(f"OKX Sizing: {signal.quantity} {symbol} -> {num_contracts} contracts (sz={sz_param})")
            
        else:
            # Fallback for SPOT or if ctVal missing (treat as coins)
            # Round quantity (lot size)
            qty = self._round_step(signal.quantity, info['lotSz'])
            sz_param = str(qty)
            
            if qty == 0:
                logger.warning(f"Quantity 0 after rounding for {symbol}")
                return None
        
        # Round price (tick size)
        price = self._round_step(signal.price, info['tickSz'])
        
        # Create Order Record with ACTUAL coin quantity
        order = await self.order_manager.create_order(
            symbol=symbol,
            side=signal.side,
            quantity=qty,
            price=price,
            order_type="LIMIT",
            time_in_force="GTC"
        )
        
        # Submit to OKX
        path = "/api/v5/trade/order"
        body = {
            "instId": inst_id,
            "tdMode": "cross", 
            "side": signal.side.lower(),
            "ordType": "limit",
            "sz": sz_param,  # NOW CORRECTLY SET TO CONTRACTS OR COINS
            "px": str(price),
            "clOrdId": order.client_order_id
        }
        
        response = await self._rest_request("POST", path, body)
        
        if response and response.get('code') == '0':
             okx_id = response['data'][0]['ordId']
             await self.event_store.log_order_submitted(
                trace_id=order.trace_id,
                client_order_id=order.client_order_id,
                exchange_order_id=okx_id,
                symbol=symbol,
                side=signal.side,
                quantity=qty,
                price=price
             )
             logger.info(f"✅ OKX Order Placed: {okx_id}")
             return order.client_order_id
        else:
            logger.error(f"OKX Order Failed: {response}")
            await self.order_manager.mark_rejected(order.client_order_id, str(response))
            return None
            
    async def _rest_request(self, method, path, body=None):
        """Send Signed REST Request."""
        if not self.session: return None
        
        url = self.BASE_URL + path
        if False:
            # TODO: Add time sync
            pass
        timestamp = self._get_timestamp()
        
        body_str = json.dumps(body) if body else ""
        message = f"{timestamp}{method}{path}{body_str}"
        signature = self._generate_signature(message)
        
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        
        if self.testnet:
             headers["x-simulated-trading"] = "1"
             
        try:
            async with self.session.request(method, url, headers=headers, json=body) as resp:
                 return await resp.json()
        except Exception as e:
            logger.error(f"REST Request Error: {e}")
            return None

    def _generate_signature(self, message):
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    async def _sync_time(self):
        """Sync local time with OKX server time."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/api/v5/public/time"
                async with session.get(url) as response:
                    data = await response.json()
                    if data['code'] == '0':
                        server_time = int(data['data'][0]['ts'])
                        local_time = int(time.time() * 1000)
                        self.time_offset = server_time - local_time
                        logger.info(f"⏱️ OKX Time Offset: {self.time_offset}ms")
                    else:
                        logger.warning(f"Failed to sync time: {data}")
        except Exception as e:
            logger.error(f"Time sync error: {e}")

    def _get_timestamp(self):
        # Use sync offset
        offset_ms = getattr(self, 'time_offset', 0)
        ts = int(time.time() * 1000) + offset_ms
        return datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc).isoformat()[:-3]+'Z'

    def _round_step(self, value, step):
        if step == 0: return value
        # Simple rounding
        return round(value / step) * step
        
    async def close(self):
        if self.session: await self.session.close()     
