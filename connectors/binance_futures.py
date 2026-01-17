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

from core.events import MarketEvent, FundingRateEvent

logger = logging.getLogger("Connector.Binance")

class BinanceFuturesConnector:
    """
    WebSocket connector for Binance market data.
    Supports both Futures and Spot modes.
    """
    def __init__(self, event_queue: asyncio.Queue, symbols: list[str], ws_url: str, spot_mode: bool = False):
        self.queue = event_queue
        self.symbols = [s.lower() for s in symbols]
        self.url = ws_url
        self.spot_mode = spot_mode  # NEW: Spot vs Futures mode
        self.session = None
        self.ws = None
        self.running = False

    async def connect(self):
        """Handles connection and automatic reconnection."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        mode = "SPOT" if self.spot_mode else "FUTURES"
        logger.info(f"🔌 Connector starting in {mode} mode")
        
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
        """Subscribe to market data streams (spot or futures)."""
        if not self.ws:
            return
        
        # aggTrade works for both spot and futures
        agg_params = [f"{s}@aggTrade" for s in self.symbols]
        
        if self.spot_mode:
            # SPOT: Use bookTicker for best bid/ask (no markPrice in spot!)
            # bookTicker gives real-time best bid/ask which is what we need for market making
            ticker_params = [f"{s}@bookTicker" for s in self.symbols]
            all_params = agg_params + ticker_params
            logger.info(f"📊 Subscribing to SPOT streams: aggTrade + bookTicker for {len(self.symbols)} symbols")
        else:
            # FUTURES: Use markPrice for funding rate info
            mark_params = [f"{s}@markPrice@1s" for s in self.symbols]
            all_params = agg_params + mark_params
            logger.info(f"📊 Subscribing to FUTURES streams: aggTrade + markPrice for {len(self.symbols)} symbols")
        
        payload = {
            "method": "SUBSCRIBE",
            "params": all_params,
            "id": 1
        }
        await self.ws.send_json(payload)

    async def _listen(self):
        """Listen for incoming messages."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json_loads(msg.data)
                await self._process_message(data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket connection closed with error")
                break

    async def _process_message(self, data: dict):
        """Process raw JSON into MarketEvent or FundingRateEvent."""
        # Ignore heartbeat/response messages
        if 'e' not in data:
            return

        # 1. Trade Update (works for both spot and futures)
        if data['e'] == 'aggTrade':
            exchange_name = 'binance_spot' if self.spot_mode else 'binance_futures'
            event = MarketEvent(
                exchange=exchange_name,
                symbol=data['s'],       # Symbol
                price=float(data['p']), # Price
                volume=float(data['q']), # Quantity
                timestamp=float(data['T']) / 1000,
                event_type='TICK'
            )
            await self.queue.put(event)
        
        # 2. Book Ticker (SPOT ONLY) - Best bid/ask updates
        elif data['e'] == 'bookTicker' and self.spot_mode:
            # bookTicker format: {s: symbol, b: bestBid, B: bestBidQty, a: bestAsk, A: bestAskQty}
            # We convert this to a MarketEvent for the strategy to consume
            event = MarketEvent(
                exchange='binance_spot',
                symbol=data['s'],
                price=float(data['a']),  # Use ask price as reference
                volume=0,  # bookTicker doesn't have volume
                timestamp=float(data.get('E', 0)) / 1000 if 'E' in data else 0,
                event_type='BOOK_TICKER',
                bid=float(data['b']),
                ask=float(data['a'])
            )
            await self.queue.put(event)
            
        # 3. Mark Price Update (FUTURES ONLY - Funding Rate)
        elif data['e'] == 'markPriceUpdate' and not self.spot_mode:
            # "r": Funding Rate
            # "p": Mark Price
            # "T": Next Funding Time
            event = FundingRateEvent(
                symbol=data['s'],
                rate=float(data['r']),
                mark_price=float(data['p']),
                next_funding_time=float(data['T']) / 1000
            )
            await self.queue.put(event)

    async def close(self):
        """Gracefully close the connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info("Binance connector closed.")
