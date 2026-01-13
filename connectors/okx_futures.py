# connectors/okx_futures.py
import asyncio
import json
import logging
import aiohttp
from core.events import MarketEvent

logger = logging.getLogger("Connector.OKX")

class OkxFuturesConnector:
    """
    OKX Futures WebSocket Connector (Public Data).
    Connects to OKX V5 Public WebSocket.
    """
    def __init__(self, event_queue: asyncio.Queue, symbols: list[str], ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"):
        self.queue = event_queue
        # OKX symbols typically "BTC-USDT-SWAP" for perpetuals
        # Map generic "btcusdt" -> "BTC-USDT-SWAP"
        self.symbol_map = self._create_symbol_map(symbols)
        self.url = ws_url
        self.session = None
        self.ws = None
        self.running = False
        
    def _create_symbol_map(self, symbols: list[str]) -> dict:
        """Map 'btcusdt' -> 'BTC-USDT-SWAP'."""
        mapping = {}
        for s in symbols:
            s_clean = s.upper().replace("USDT", "") # BTC
            # OKX Perpetual format: BTC-USDT-SWAP
            okx_sym = f"{s_clean}-USDT-SWAP"
            mapping[okx_sym] = s.lower()
        return mapping

    async def connect(self):
        """Connect to OKX WebSocket."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        while self.running:
            try:
                logger.info(f"Connecting to {self.url}...")
                async with self.session.ws_connect(self.url) as ws:
                    self.ws = ws
                    await self._subscribe()
                    logger.info("✅ Connected and subscribed to OKX.")
                    
                    await self._listen()
                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _subscribe(self):
        """Subscribe to ticker and trades."""
        if not self.ws:
            return
            
        # Channels: 'tickers' gives best bid/ask, 'trades' gives fills
        args = []
        for okx_sym in self.symbol_map.keys():
            args.append({"channel": "tickers", "instId": okx_sym})
            args.append({"channel": "trades", "instId": okx_sym})
            
        payload = {
            "op": "subscribe",
            "args": args
        }
        await self.ws.send_json(payload)

    async def _listen(self):
        """Listen for messages."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                await self._process_message(data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket connection closed with error")
                break

    async def _process_message(self, data: dict):
        """Process OKX V5 message."""
        # Check for event responses (subscribe success/error)
        if 'event' in data:
            if data['event'] == 'error':
                logger.error(f"OKX API Error: {data.get('msg')}")
            return

        if 'arg' not in data or 'data' not in data:
            return

        channel = data['arg']['channel']
        inst_id = data['arg']['instId']
        
        # Map OKX symbol back to internal symbol
        internal_symbol = self.symbol_map.get(inst_id)
        if not internal_symbol:
            return

        for item in data['data']:
            # Ticker Data (Best Bid/Ask) - treat as TICK
            if channel == 'tickers':
                try:
                    price = float(item['last']) # Last traded price
                    # OKX also gives best bid/ask: bidPx, askPx
                    # For simplicity, sending last price as TICK
                    event = MarketEvent(
                        exchange='okx',
                        symbol=internal_symbol,
                        price=price,
                        volume=0.0, # Ticker doesn't give volume of last trade directly here
                        timestamp=float(item['ts']) / 1000,
                        event_type='TICK'
                    )
                    await self.queue.put(event)
                except Exception as e:
                    logger.warning(f"Error parsing ticker: {e}")

            # Trade Data
            elif channel == 'trades':
                try:
                    price = float(item['px'])
                    qty = float(item['sz'])
                    event = MarketEvent(
                        exchange='okx',
                        symbol=internal_symbol,
                        price=price,
                        volume=qty,
                        timestamp=float(item['ts']) / 1000,
                        event_type='TRADE'
                    )
                    await self.queue.put(event)
                except Exception as e:
                    logger.warning(f"Error parsing trade: {e}")

    async def close(self):
        """Close connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("OKX connector closed.")
