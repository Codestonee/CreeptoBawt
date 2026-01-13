# connectors/binance_futures.py
import asyncio
import logging
import aiohttp

# PERFORMANCE: Use orjson for 3-5x faster JSON parsing
try:
    import orjson
    def json_loads(data):
        return orjson.loads(data)
except ImportError:
    import json
    def json_loads(data):
        return json.loads(data)

from core.events import MarketEvent

logger = logging.getLogger("Connector.Binance")

class BinanceFuturesConnector:
    def __init__(self, event_queue: asyncio.Queue, symbols: list[str], ws_url: str):
        self.queue = event_queue
        self.symbols = [s.lower() for s in symbols]
        self.url = ws_url
        self.session = None
        self.ws = None
        self.running = False

    async def connect(self):
        """Hanterar anslutning och automatisk återanslutning."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        while self.running:
            try:
                logger.info(f"Connecting to {self.url}...")
                async with self.session.ws_connect(self.url) as ws:
                    self.ws = ws
                    await self._subscribe()
                    logger.info("Connected and subscribed.")
                    
                    await self._listen()
                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _subscribe(self):
        """Prenumererar på aggregerade trades för valda symboler."""
        if not self.ws:
            return
            
        params = [f"{s}@aggTrade" for s in self.symbols]
        payload = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": 1
        }
        await self.ws.send_json(payload)

    async def _listen(self):
        """Lyssnar på inkommande meddelanden."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json_loads(msg.data)
                await self._process_message(data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket connection closed with error")
                break

    async def _process_message(self, data: dict):
        """Omvandlar rå JSON till MarketEvent."""
        # Ignorera heartbeat/response meddelanden
        if 'e' not in data:
            return

        if data['e'] == 'aggTrade':
            event = MarketEvent(
                exchange='binance_futures',
                symbol=data['s'],      # Symbol
                price=float(data['p']), # Price
                volume=float(data['q']), # Quantity
                timestamp=float(data['T']) / 1000,
                event_type='TICK'
            )
            # Lägg eventet i kön för motorn att bearbeta
            await self.queue.put(event)

    async def close(self):
        """Stänger ner anslutningen snyggt."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("Binance connector closed.")