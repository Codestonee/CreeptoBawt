"""
Shadow Order Book - Local L2 orderbook maintained from WebSocket depth updates.

Provides:
- Real-time bid/ask at any depth level
- Expected fill price calculation for market orders
- Order book imbalance signals
- Implementation shortfall measurement
"""

import asyncio
import aiohttp
import logging
import time

# PERFORMANCE: Use orjson for 3-5x faster JSON parsing (Rust-based)
try:
    import orjson
    def json_loads(data):
        return orjson.loads(data)
except ImportError:
    import json
    def json_loads(data):
        return json.loads(data)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sortedcontainers import SortedDict

logger = logging.getLogger("Data.ShadowBook")


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    quantity: float
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class OrderBookSnapshot:
    """Point-in-time snapshot of the order book."""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]  # Sorted descending (best bid first)
    asks: List[OrderBookLevel]  # Sorted ascending (best ask first)
    last_update_id: int = 0
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None


@dataclass
class FillEstimate:
    """Estimated fill for a market order."""
    side: str
    quantity: float
    avg_price: float
    total_cost: float
    slippage_bps: float  # Slippage from mid price in basis points
    levels_consumed: int


class ShadowOrderBook:
    """
    Local order book maintained from WebSocket depth updates.
    
    Features:
    - Connects to Binance @depth@100ms stream
    - Maintains sorted bid/ask levels
    - Calculates expected fill prices
    - Measures order book imbalance
    """
    
    # WebSocket URLs
    WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol}@depth@100ms"
    WS_URL_TESTNET = "wss://stream.binancefuture.com/ws/{symbol}@depth@100ms"
    REST_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"
    REST_DEPTH_TESTNET = "https://testnet.binancefuture.com/fapi/v1/depth"
    
    # Configuration
    MAX_LEVELS = 20  # Keep top N levels
    RECONNECT_DELAY = 5
    
    def __init__(self, symbols: List[str], testnet: bool = True):
        self.symbols = [s.lower() for s in symbols]
        self.testnet = testnet
        
        # Order books: symbol -> {bids: SortedDict, asks: SortedDict}
        self._books: Dict[str, Dict] = {}
        for symbol in self.symbols:
            self._books[symbol] = {
                'bids': SortedDict(),  # price -> qty (descending)
                'asks': SortedDict(),  # price -> qty (ascending)
                'last_update_id': 0,
                'last_update_time': 0
            }
        
        # WebSocket tasks
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start(self):
        """Start depth streams for all symbols."""
        if self._running:
            return
        
        self._running = True
        
        # Fetch initial snapshots
        for symbol in self.symbols:
            try:
                await self._fetch_snapshot(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch initial depth for {symbol}: {e}")
        
        # Start WebSocket streams
        for symbol in self.symbols:
            self._ws_tasks[symbol] = asyncio.create_task(
                self._ws_stream(symbol)
            )
        
        logger.info(f"ShadowOrderBook started for {len(self.symbols)} symbols")
    
    async def stop(self):
        """Stop all streams."""
        self._running = False
        
        for task in self._ws_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._ws_tasks.clear()
        logger.info("ShadowOrderBook stopped")
    
    async def _fetch_snapshot(self, symbol: str, limit: int = 100):
        """Fetch initial order book snapshot via REST."""
        url = self.REST_DEPTH_TESTNET if self.testnet else self.REST_DEPTH_URL
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={
                "symbol": symbol.upper(),
                "limit": limit
            }) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                data = await response.json()
        
        book = self._books[symbol]
        book['bids'].clear()
        book['asks'].clear()
        
        for bid in data.get('bids', []):
            price, qty = float(bid[0]), float(bid[1])
            if qty > 0:
                book['bids'][-price] = qty  # Negative for descending order
        
        for ask in data.get('asks', []):
            price, qty = float(ask[0]), float(ask[1])
            if qty > 0:
                book['asks'][price] = qty
        
        book['last_update_id'] = data.get('lastUpdateId', 0)
        book['last_update_time'] = time.time()
        
        logger.info(f"Loaded depth snapshot for {symbol}: {len(book['bids'])} bids, {len(book['asks'])} asks")
    
    async def _ws_stream(self, symbol: str):
        """WebSocket stream for depth updates."""
        url = (
            self.WS_URL_TESTNET.format(symbol=symbol)
            if self.testnet else
            self.WS_URL_TEMPLATE.format(symbol=symbol)
        )
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info(f"Depth WebSocket connected for {symbol}")
                        
                        async for msg in ws:
                            if not self._running:
                                break
                            
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._handle_depth_update(symbol, msg.data)
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                break
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Depth WebSocket error for {symbol}: {e}")
            
            if self._running:
                await asyncio.sleep(self.RECONNECT_DELAY)
    
    def _handle_depth_update(self, symbol: str, data: str):
        """Process depth update from WebSocket."""
        try:
            msg = json_loads(data)
            
            # Binance depth update format
            book = self._books[symbol]
            
            # Check sequence
            update_id = msg.get('u', 0)
            if update_id <= book['last_update_id']:
                return  # Old update, skip
            
            # Update bids
            for bid in msg.get('b', []):
                price, qty = float(bid[0]), float(bid[1])
                if qty == 0:
                    book['bids'].pop(-price, None)  # Remove level
                else:
                    book['bids'][-price] = qty
            
            # Update asks
            for ask in msg.get('a', []):
                price, qty = float(ask[0]), float(ask[1])
                if qty == 0:
                    book['asks'].pop(price, None)  # Remove level
                else:
                    book['asks'][price] = qty
            
            # Trim to max levels
            while len(book['bids']) > self.MAX_LEVELS:
                book['bids'].popitem()  # Remove worst bid
            while len(book['asks']) > self.MAX_LEVELS:
                book['asks'].popitem()  # Remove worst ask
            
            book['last_update_id'] = update_id
            book['last_update_time'] = time.time()
            
        except Exception as e:
            logger.error(f"Error processing depth update: {e}")
    
    def get_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot."""
        symbol = symbol.lower()
        book = self._books.get(symbol)
        
        if not book:
            return None
        
        bids = [
            OrderBookLevel(price=-k, quantity=v)
            for k, v in list(book['bids'].items())[:self.MAX_LEVELS]
        ]
        asks = [
            OrderBookLevel(price=k, quantity=v)
            for k, v in list(book['asks'].items())[:self.MAX_LEVELS]
        ]
        
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=book['last_update_time'],
            bids=bids,
            asks=asks,
            last_update_id=book['last_update_id']
        )
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices."""
        symbol = symbol.lower()
        book = self._books.get(symbol)
        
        if not book:
            return None, None
        
        best_bid = -next(iter(book['bids']), None) if book['bids'] else None
        best_ask = next(iter(book['asks']), None) if book['asks'] else None
        
        return best_bid, best_ask
    
    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price."""
        bid, ask = self.get_best_bid_ask(symbol)
        if bid and ask:
            return (bid + ask) / 2
        return None
    
    def estimate_fill(self, symbol: str, side: str, quantity: float) -> Optional[FillEstimate]:
        """
        Estimate fill price for a market order.
        
        Walks the book to calculate average fill price and slippage.
        """
        symbol = symbol.lower()
        book = self._books.get(symbol)
        
        if not book:
            return None
        
        mid = self.get_mid_price(symbol)
        if not mid:
            return None
        
        remaining = quantity
        total_cost = 0.0
        levels_consumed = 0
        
        # Walk the appropriate side
        if side.upper() == "BUY":
            # Buy = take from asks (ascending)
            for price, qty in book['asks'].items():
                if remaining <= 0:
                    break
                fill_qty = min(remaining, qty)
                total_cost += price * fill_qty
                remaining -= fill_qty
                levels_consumed += 1
        else:
            # Sell = take from bids (descending by negative key)
            for neg_price, qty in book['bids'].items():
                if remaining <= 0:
                    break
                price = -neg_price
                fill_qty = min(remaining, qty)
                total_cost += price * fill_qty
                remaining -= fill_qty
                levels_consumed += 1
        
        if remaining > 0:
            logger.warning(f"Insufficient liquidity for {quantity} {symbol}")
            return None
        
        avg_price = total_cost / quantity
        slippage = abs(avg_price - mid) / mid * 10000  # In basis points
        
        return FillEstimate(
            side=side.upper(),
            quantity=quantity,
            avg_price=avg_price,
            total_cost=total_cost,
            slippage_bps=slippage,
            levels_consumed=levels_consumed
        )
    
    def get_imbalance(self, symbol: str, levels: int = 5) -> Optional[float]:
        """
        Calculate order book imbalance.
        
        Returns value between -1 (all asks) and +1 (all bids).
        Positive = buy pressure, negative = sell pressure.
        
        Optimized: Uses islice for O(k) instead of O(N) list conversion.
        """
        from itertools import islice
        
        symbol = symbol.lower()
        book = self._books.get(symbol)
        
        if not book:
            return None
        
        # O(k) instead of O(N) - only iterate first 'levels' items
        bid_volume = sum(islice(book['bids'].values(), levels)) if book['bids'] else 0
        ask_volume = sum(islice(book['asks'].values(), levels)) if book['asks'] else 0
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def is_stale(self, symbol: str, max_age_seconds: float = 5.0) -> bool:
        """Check if order book data is stale."""
        symbol = symbol.lower()
        book = self._books.get(symbol)
        
        if not book:
            return True
        
        return (time.time() - book['last_update_time']) > max_age_seconds


# Factory function
def get_shadow_book(symbols: List[str], testnet: bool = True) -> ShadowOrderBook:
    """Create a ShadowOrderBook instance."""
    return ShadowOrderBook(symbols=symbols, testnet=testnet)
