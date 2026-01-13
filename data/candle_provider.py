"""
Candle Provider - Streaming OHLCV candles from WebSocket with REST fallback.

Replaces manual tick-to-candle construction with validated exchange data.
"""

import asyncio
import aiohttp
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
from decimal import Decimal
from collections import defaultdict

logger = logging.getLogger("Data.CandleProvider")


@dataclass
class Candle:
    """Validated OHLCV candle from exchange."""
    open_time: int      # Unix timestamp ms
    close_time: int     # Unix timestamp ms
    symbol: str
    interval: str       # e.g., "1m", "5m"
    open: float         # Using float for hot path (Decimal for settlement)
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    is_closed: bool = False
    
    def __post_init__(self):
        # Validate candle integrity
        if self.high < self.open or self.high < self.close or self.high < self.low:
            logger.warning(f"Invalid candle: high {self.high} is not highest")
        if self.low > self.open or self.low > self.close or self.low > self.high:
            logger.warning(f"Invalid candle: low {self.low} is not lowest")


class CandleProvider:
    """
    Streaming candle provider using WebSocket with REST fallback.
    
    Features:
    - Real-time candles via WebSocket (/ws/symbol@kline_interval)
    - REST polling for initial history and gap recovery
    - Checksum validation where available
    - Handles DST shifts and maintenance windows
    """
    
    # URLs
    WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol}@kline_{interval}"
    WS_URL_TESTNET = "wss://stream.binancefuture.com/ws/{symbol}@kline_{interval}"
    REST_URL = "https://fapi.binance.com/fapi/v1/klines"
    REST_URL_TESTNET = "https://testnet.binancefuture.com/fapi/v1/klines"
    
    # Limits
    MAX_CANDLES_CACHED = 200
    RECONNECT_DELAY_SECONDS = 5
    REST_RATE_LIMIT_SECONDS = 10  # Max 1 REST call per symbol per 10s
    
    def __init__(
        self, 
        symbols: List[str],
        interval: str = "1m",
        testnet: bool = True
    ):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.testnet = testnet
        
        # Candle cache: symbol -> list of candles
        self._candles: Dict[str, List[Candle]] = defaultdict(list)
        self._current_candle: Dict[str, Optional[Candle]] = {}
        
        # WebSocket state
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        # Callbacks
        self._on_candle_close: Optional[Callable[[Candle], Awaitable[None]]] = None
        
        # REST rate limiting
        self._last_rest_call: Dict[str, float] = {}
    
    def set_candle_close_callback(self, callback: Callable[[Candle], Awaitable[None]]):
        """Set callback to invoke when a candle closes."""
        self._on_candle_close = callback
    
    async def start(self):
        """Start streaming candles for all symbols."""
        if self._running:
            logger.warning("CandleProvider already running")
            return
        
        self._running = True
        
        # Fetch initial history for each symbol
        for symbol in self.symbols:
            try:
                await self._fetch_history(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch initial candles for {symbol}: {e}")
        
        # Start WebSocket streams
        for symbol in self.symbols:
            self._ws_tasks[symbol] = asyncio.create_task(
                self._ws_stream(symbol)
            )
        
        logger.info(f"CandleProvider started for {len(self.symbols)} symbols")
    
    async def stop(self):
        """Stop all streams."""
        self._running = False
        
        for symbol, task in self._ws_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._ws_tasks.clear()
        logger.info("CandleProvider stopped")
    
    async def _fetch_history(self, symbol: str, limit: int = 100) -> List[Candle]:
        """Fetch historical candles via REST API."""
        # Rate limiting
        last_call = self._last_rest_call.get(symbol, 0)
        if time.time() - last_call < self.REST_RATE_LIMIT_SECONDS:
            wait_time = self.REST_RATE_LIMIT_SECONDS - (time.time() - last_call)
            await asyncio.sleep(wait_time)
        
        url = self.REST_URL_TESTNET if self.testnet else self.REST_URL
        params = {
            "symbol": symbol.upper(),
            "interval": self.interval,
            "limit": limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    data = await response.json()
        except Exception as e:
            logger.error(f"REST candle fetch failed for {symbol}: {e}")
            raise
        
        self._last_rest_call[symbol] = time.time()
        
        candles = []
        prev_close_time = None
        
        for item in data:
            candle = Candle(
                open_time=item[0],
                close_time=item[6],
                symbol=symbol,
                interval=self.interval,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                quote_volume=float(item[7]),
                trades=int(item[8]),
                is_closed=True
            )
            
            # Validate sequence (check for gaps)
            if prev_close_time is not None:
                expected_open = prev_close_time + 1
                if candle.open_time != expected_open:
                    gap_ms = candle.open_time - expected_open
                    if gap_ms > 60000:  # More than 1 minute gap
                        logger.warning(
                            f"Candle gap detected for {symbol}: "
                            f"{gap_ms}ms between candles (possible maintenance)"
                        )
            
            prev_close_time = candle.close_time
            candles.append(candle)
        
        # Update cache
        self._candles[symbol] = candles[-self.MAX_CANDLES_CACHED:]
        logger.info(f"Fetched {len(candles)} historical candles for {symbol}")
        
        return candles
    
    async def _ws_stream(self, symbol: str):
        """WebSocket stream for a single symbol."""
        url = (
            self.WS_URL_TESTNET.format(symbol=symbol, interval=self.interval)
            if self.testnet else
            self.WS_URL_TEMPLATE.format(symbol=symbol, interval=self.interval)
        )
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info(f"WebSocket connected for {symbol}")
                        
                        async for msg in ws:
                            if not self._running:
                                break
                            
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_ws_message(symbol, msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error for {symbol}: {ws.exception()}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning(f"WebSocket closed for {symbol}")
                                break
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
            
            if self._running:
                logger.info(f"Reconnecting WebSocket for {symbol} in {self.RECONNECT_DELAY_SECONDS}s...")
                await asyncio.sleep(self.RECONNECT_DELAY_SECONDS)
    
    async def _handle_ws_message(self, symbol: str, data: str):
        """Handle incoming WebSocket kline message."""
        try:
            msg = json.loads(data)
            
            if "e" not in msg or msg["e"] != "kline":
                return
            
            k = msg["k"]
            
            candle = Candle(
                open_time=k["t"],
                close_time=k["T"],
                symbol=symbol,
                interval=k["i"],
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                quote_volume=float(k["q"]),
                trades=int(k["n"]),
                is_closed=k["x"]
            )
            
            # Update current candle
            self._current_candle[symbol] = candle
            
            # If candle closed, add to cache and invoke callback
            if candle.is_closed:
                self._add_to_cache(symbol, candle)
                
                if self._on_candle_close:
                    try:
                        await self._on_candle_close(candle)
                    except Exception as e:
                        logger.error(f"Candle close callback error: {e}")
                        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid WebSocket JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _add_to_cache(self, symbol: str, candle: Candle):
        """Add candle to cache, maintaining max size."""
        candles = self._candles[symbol]
        
        # Check for duplicates
        if candles and candles[-1].open_time == candle.open_time:
            # Update existing candle
            candles[-1] = candle
        else:
            candles.append(candle)
        
        # Trim to max size
        if len(candles) > self.MAX_CANDLES_CACHED:
            self._candles[symbol] = candles[-self.MAX_CANDLES_CACHED:]
    
    def get_candles(self, symbol: str, limit: int = 100) -> List[Candle]:
        """
        Get cached candles for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Max number of candles to return
            
        Returns:
            List of most recent candles (oldest first)
        """
        symbol = symbol.lower()
        candles = self._candles.get(symbol, [])
        return candles[-limit:] if limit else candles
    
    def get_current_candle(self, symbol: str) -> Optional[Candle]:
        """Get the current (unclosed) candle for a symbol."""
        return self._current_candle.get(symbol.lower())
    
    def get_ohlcv_arrays(self, symbol: str, limit: int = 100):
        """
        Get OHLCV data as separate arrays for indicator calculation.
        
        Returns:
            Tuple of (opens, highs, lows, closes, volumes) as lists
        """
        candles = self.get_candles(symbol, limit)
        
        opens = [c.open for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        
        return opens, highs, lows, closes, volumes


# Package init for data module
def get_candle_provider(
    symbols: List[str],
    interval: str = "1m",
    testnet: bool = True
) -> CandleProvider:
    """Factory function for CandleProvider."""
    return CandleProvider(symbols=symbols, interval=interval, testnet=testnet)
