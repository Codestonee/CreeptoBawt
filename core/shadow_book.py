import asyncio
import aiohttp
import time
import logging
from typing import Dict, Optional, Tuple, List

# Try using orjson for faster parsing
try:
    import orjson
    def json_loads(data):
        return orjson.loads(data)
except ImportError:
    import json
    def json_loads(data):
        return json.loads(data)

logger = logging.getLogger("Data.ShadowBook")

class ShadowOrderBook:
    """
    Maintains a local view of the order book for low-latency access.
    Optimized for Partial Depth (Snapshot) streams from Binance.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.last_update_id = 0
        self.last_update_time = 0.0
        
    def update_from_snapshot(self, data: dict):
        """
        Update book from a partial depth snapshot (e.g. @depth20).
        Replaces the entire book state.
        """
        try:
            u_id = data.get('u') or data.get('lastUpdateId') # Futures vs Spot format
            if u_id and u_id <= self.last_update_id:
                return
            
            if u_id:
                self.last_update_id = u_id
            
            self.last_update_time = time.time()
            
            # Rebuild Bids
            # Data format: [['price', 'qty'], ...]
            self.bids = {float(p): float(q) for p, q in data.get('b', [])}
            
            # Rebuild Asks
            self.asks = {float(p): float(q) for p, q in data.get('a', [])}
            
            # Recalculate bests
            if self.bids:
                self.best_bid = max(self.bids.keys())
            else:
                self.best_bid = 0.0
                
            if self.asks:
                self.best_ask = min(self.asks.keys())
            else:
                self.best_ask = 0.0
                
        except Exception as e:
            logger.error(f"ShadowBook update error: {e}")
            
    def get_best_bid_ask(self) -> Tuple[float, float]:
        return (self.best_bid, self.best_ask)
        
    def get_mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    def get_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance: (BidVol - AskVol) / (BidVol + AskVol)
        Range: [-1, 1]. Positive = Buy Pressure, Negative = Sell Pressure.
        """
        if not self.bids or not self.asks:
            return 0.0
            
        # Get top N levels (sorted keys)
        top_bids = sorted(self.bids.keys(), reverse=True)[:levels]
        top_asks = sorted(self.asks.keys())[:levels]
        
        bid_vol = sum(self.bids[p] for p in top_bids)
        ask_vol = sum(self.asks[p] for p in top_asks)
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / total
        
    def is_stale(self, max_age_seconds: float = 2.0) -> bool:
        """Check if book data is older than max_age_seconds."""
        if self.last_update_time == 0:
            return True # Stale if never updated
        return (time.time() - self.last_update_time) > max_age_seconds


class ShadowBookService:
    """
    Manages ShadowOrderBooks for multiple symbols and handles WebSocket connections.
    """
    def __init__(self, symbols: List[str], testnet: bool = True):
        self.symbols = [s.lower() for s in symbols]
        self.testnet = testnet
        
        # FIX: Use correct WebSocket URL based on SPOT_MODE
        from config.settings import settings
        if settings.SPOT_MODE:
            # Spot WebSocket URL (Combined Streams)
            if testnet:
                self.ws_url = "wss://testnet.binance.vision/stream"
            else:
                self.ws_url = "wss://stream.binance.com:9443/stream"
            logger.info(f"ShadowBook configured for SPOT mode (Combined): {self.ws_url}")
        else:
            # Futures WebSocket URL
            if testnet:
                self.ws_url = "wss://fstream.binancefuture.com/ws"
            else:
                self.ws_url = settings.BINANCE_WS_URL  # wss://fstream.binance.com/ws
            logger.info(f"ShadowBook configured for FUTURES mode: {self.ws_url}")
            
        self.books: Dict[str, ShadowOrderBook] = {
            s: ShadowOrderBook(s) for s in self.symbols
        }
        self.running = False
        self.session = None
        self.ws = None
        
        # Reconnection with exponential backoff
        self._reconnect_delay = 5
        self._max_reconnect_delay = 60
        
        # Latency tracking
        self._last_update_latencies: List[float] = []
        self._max_latency_samples = 100
        
        # Ping task
        self._ping_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start WebSocket streams."""
        self.running = True
        logger.info(f"ShadowBookService starting for {len(self.symbols)} symbols...")
        
        while self.running:
            try:
                self.session = aiohttp.ClientSession()
                logger.info(f"ShadowBook connecting to {self.ws_url}...")
                
                async with self.session.ws_connect(self.ws_url, heartbeat=30) as ws:
                    self.ws = ws
                    self._reconnect_delay = 5  # Reset on successful connect
                    await self._subscribe()
                    logger.info("ShadowBook connected and subscribed.")
                    
                    # Start ping task
                    self._ping_task = asyncio.create_task(self._ping_loop())
                    
                    await self._listen()
                    
            except aiohttp.ClientError as e:
                logger.error(f"ShadowBook connection error: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            except Exception as e:
                logger.error(f"ShadowBook unexpected error: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            finally:
                if self._ping_task:
                    self._ping_task.cancel()
                    try:
                        await self._ping_task
                    except asyncio.CancelledError:
                        pass
                if self.session:
                    await self.session.close()

    async def _ping_loop(self):
        """Send periodic pings to keep connection alive."""
        while self.running and self.ws:
            try:
                await asyncio.sleep(30)
                if self.ws and not self.ws.closed:
                    await self.ws.ping()
                    logger.debug("ShadowBook ping sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Ping failed: {e}")

    async def _subscribe(self):
        """Subscribe to @depth20@100ms streams for all symbols."""
        if not self.ws:
            return
            
        # Batch subscriptions
        # Params: <symbol>@depth20@100ms
        params = [f"{s}@depth20@100ms" for s in self.symbols]
        
        payload = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": 100
        }
        await self.ws.send_json(payload)
        logger.info(f"Subscribed to {len(params)} depth streams")

    async def _listen(self):
        """Process incoming messages."""
        async for msg in self.ws:
            if not self.running:
                break
                
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    receive_time = time.time()
                    raw_data = json_loads(msg.data)
                    
                    # Handle Subscription Response
                    if 'result' in raw_data and 'id' in raw_data:
                         logger.info(f"ShadowBook subscription confirmed: {raw_data}")
                         continue
                         
                    # Normalize Data (Handle Combined Streams vs Raw)
                    stream_name = raw_data.get('stream', '')
                    data = raw_data.get('data', raw_data)
                    
                    symbol = ""
                    is_depth = False
                    
                    # Case 1: Combined Stream (Spot)
                    if stream_name:
                        # stream_name format: 'ltcusdc@depth20@100ms'
                        symbol = stream_name.split('@')[0].lower()
                        is_depth = True # Combined streams are what we subscribed to
                        
                    # Case 2: Futures Raw Stream (has 'e' and 's')
                    elif data.get('e') == 'depthUpdate':
                        symbol = data.get('s', '').lower()
                        is_depth = True
                        
                    if is_depth and symbol:
                        book = self.books.get(symbol)
                        if book:
                            book.update_from_snapshot(data)
                            
                            # Track latency
                            event_time = data.get('E', 0)
                            if event_time == 0 and 'lastUpdateId' in data:
                                # Spot partial depth doesn't have E (event time), use receive time
                                event_time = receive_time * 1000
                                
                            if event_time:
                                latency_ms = (receive_time * 1000) - event_time
                                self._last_update_latencies.append(latency_ms)
                                if len(self._last_update_latencies) > self._max_latency_samples:
                                    self._last_update_latencies.pop(0)
                            
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("ShadowBook WebSocket connection closed with error")
                break

    async def stop(self):
        """Stop WebSocket streams."""
        self.running = False
        logger.info("ShadowBookService stopping...")
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        
    def get_order_book(self, symbol: str) -> Optional[ShadowOrderBook]:
        return self.books.get(symbol.lower())
        
    def get_mid_price(self, symbol: str) -> float:
        book = self.get_order_book(symbol)
        return book.get_mid_price() if book else 0.0

    def get_imbalance(self, symbol: str, levels: int = 5) -> float:
        book = self.get_order_book(symbol)
        return book.get_imbalance(levels) if book else 0.0

    def is_stale(self, symbol: str, max_age_seconds: float = 2.0) -> bool:
        """Check if book data is older than max_age_seconds."""
        book = self.get_order_book(symbol)
        if not book:
            return True
        return book.is_stale(max_age_seconds)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics for order book updates."""
        if not self._last_update_latencies:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "samples": 0}
        
        return {
            "avg_ms": sum(self._last_update_latencies) / len(self._last_update_latencies),
            "min_ms": min(self._last_update_latencies),
            "max_ms": max(self._last_update_latencies),
            "samples": len(self._last_update_latencies)
        }

def get_shadow_book(symbols: List[str], testnet: bool = True) -> ShadowBookService:
    return ShadowBookService(symbols, testnet)
