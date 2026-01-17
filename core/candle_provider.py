from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Dict
import numpy as np

@dataclass
class Candle:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True

class CandleProvider:
    """
    Provides OHLCV data for strategies.
    Can be linked to an exchange client or stream.
    """
    def __init__(self, symbols: List[str] = None, interval: str = "1m", testnet: bool = True, exchange_client=None):
        self.symbols = symbols or []
        self.interval = interval
        self.testnet = testnet
        self.client = exchange_client
        self.candles: Dict[str, List[Candle]] = {} # symbol -> list[Candle]
        self._callback: Optional[Callable] = None

    async def start(self):
        """Start candle polling or streaming (placeholder)."""
        pass

    async def stop(self):
        """Stop candle provider."""
        pass

    def set_candle_close_callback(self, callback: Callable):
        """Set callback for when a candle closes."""
        self._callback = callback

    def get_ohlcv_arrays(self, symbol: str, limit: int = 100) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Get arrays of OHLCV data for analysis."""
        symbol = symbol.lower()
        if symbol not in self.candles:
            return ([], [], [], [], [])
            
        candles = self.candles[symbol][-limit:]
        
        opens = [c.open for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        
        return (opens, highs, lows, closes, volumes)
        
    async def process_candle(self, candle: Candle):
        """Process a new candle update."""
        symbol = candle.symbol.lower()
        if symbol not in self.candles:
            self.candles[symbol] = []
            
        self.candles[symbol].append(candle)
        
        # Trim history
        if len(self.candles[symbol]) > 1000:
            self.candles[symbol] = self.candles[symbol][-1000:]
            
        if self._callback:
            try:
                if asyncio.iscoroutinefunction(self._callback):
                    await self._callback(candle)
                else:
                    self._callback(candle)
            except Exception:
                pass

import asyncio
