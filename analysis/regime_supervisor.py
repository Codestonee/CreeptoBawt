"""
Regime Supervisor - Market regime detection using streaming candles.

Refactored to use CandleProvider instead of manual tick-to-candle construction.
"""

import logging
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List

from core.events import MarketEvent, RegimeEvent
from analysis.indicators import calculate_adx, calculate_atr
from data.candle_provider import CandleProvider, Candle

logger = logging.getLogger("Analysis.Regime")


class RegimeSupervisor:
    """
    Market regime detection using validated exchange candles.
    
    Classifies market into:
    - TRENDING: ADX > 25
    - RANGING: ADX < 20
    - UNCERTAIN: 20 <= ADX <= 25
    """
    
    def __init__(self, event_queue, candle_provider: Optional[CandleProvider] = None):
        self.queue = event_queue
        self.candle_provider = candle_provider
        self.last_regime: Dict[str, Optional[str]] = defaultdict(lambda: None)
        self._last_analysis_time: Dict[str, int] = defaultdict(int)
        
        # Register for candle close events if provider exists
        if self.candle_provider:
            self.candle_provider.set_candle_close_callback(self._on_candle_close)
    
    def set_candle_provider(self, provider: CandleProvider):
        """Set or update the candle provider."""
        self.candle_provider = provider
        provider.set_candle_close_callback(self._on_candle_close)
    
    async def _on_candle_close(self, candle: Candle):
        """Called when a candle closes - trigger regime analysis."""
        await self._analyze_regime(candle.symbol)
    
    async def update(self, event: MarketEvent):
        """
        Handle tick events.
        
        With CandleProvider, we don't build candles from ticks anymore.
        This method is kept for backward compatibility and triggers
        periodic analysis based on time.
        """
        symbol = event.symbol.lower()
        current_minute = int(event.timestamp // 60)
        
        # Only analyze once per minute to avoid redundant computation
        if current_minute > self._last_analysis_time.get(symbol, 0):
            self._last_analysis_time[symbol] = current_minute
            await self._analyze_regime(symbol)
    
    async def _analyze_regime(self, symbol: str):
        """Calculate ADX/ATR and publish RegimeEvent."""
        symbol = symbol.lower()
        
        # Get candles from provider
        if self.candle_provider:
            opens, highs, lows, closes, volumes = self.candle_provider.get_ohlcv_arrays(symbol, limit=100)
        else:
            # Fallback: no candles available
            logger.debug(f"No candle provider for {symbol}")
            return
        
        # Need at least 30 candles for meaningful ADX
        if len(closes) < 30:
            logger.debug(f"Not enough candles for {symbol}: {len(closes)}/30")
            return
        
        try:
            highs_arr = np.array(highs)
            lows_arr = np.array(lows)
            closes_arr = np.array(closes)
            
            adx = calculate_adx(highs_arr, lows_arr, closes_arr)
            volatility = calculate_atr(highs_arr, lows_arr, closes_arr)[-1]
            
            # Regime classification
            if adx > 25:
                regime = "TRENDING"
            elif adx < 20:
                regime = "RANGING"
            else:
                regime = "UNCERTAIN"
            
            # Only publish on regime change
            if regime != self.last_regime.get(symbol):
                self.last_regime[symbol] = regime
                
                logger.info(f"REGIME [{symbol}]: {regime} (ADX={adx:.2f}, Vol={volatility:.4f})")
                
                event = RegimeEvent(
                    symbol=symbol,
                    regime=regime,
                    adx=adx,
                    volatility=volatility
                )
                await self.queue.put(event)
                
        except Exception as e:
            logger.error(f"Error analyzing regime for {symbol}: {e}")
    
    def get_current_regime(self, symbol: str) -> Optional[str]:
        """Get the current regime for a symbol."""
        return self.last_regime.get(symbol.lower())
    
    def get_volatility(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get current ATR volatility for a symbol."""
        if not self.candle_provider:
            return None
        
        _, highs, lows, closes, _ = self.candle_provider.get_ohlcv_arrays(symbol, limit=period + 10)
        
        if len(closes) < period:
            return None
        
        try:
            atr = calculate_atr(
                np.array(highs),
                np.array(lows),
                np.array(closes),
                period=period
            )
            return atr[-1] if len(atr) > 0 else None
        except Exception:
            return None