"""
Intensity Calibrator for GLT Market Making.

Implements rolling calibration of A (intensity scale) and k (decay rate) parameters
from market trade data. Based on methodology from:
- HFTBacktest GLFT tutorial
- Avellaneda-Stoikov parameter learning

The key insight: λ(δ) = A * exp(-k * δ) describes how fill probability
decreases with quote distance from mid-price.

Usage:
    calibrator = IntensityCalibrator("btcusdt")
    
    # On each trade:
    calibrator.on_trade(trade_price, mid_price)
    
    # Periodically get calibrated params:
    A, k = calibrator.get_params()
"""

import logging
import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import time

logger = logging.getLogger("Analysis.Intensity")


@dataclass
class CalibrationConfig:
    """Configuration for intensity calibration."""
    window_seconds: float = 600.0      # 10-minute rolling window
    min_samples: int = 100             # Minimum trades for valid calibration
    shallow_depth_ticks: int = 70      # Only fit shallow depths (per HFTBacktest)
    tick_size_bps: float = 0.5         # Tick size in basis points (5 bps)
    recalibrate_interval: float = 60.0  # Recalibrate every 60 seconds
    
    # Default fallback values (from HFTBacktest ETH/USDT calibration)
    default_A: float = 2.98            # Fills per second at mid
    default_k: float = 0.042           # Decay rate


class IntensityCalibrator:
    """
    Rolling calibrator for order arrival intensity parameters.
    
    Implements the methodology from HFTBacktest:
    1. Record trade arrival depths (distance from mid-price)
    2. Bucket by tick distance
    3. Fit exponential: λ(δ) = A * exp(-k * δ)
    4. Refit specifically for shallow depths where quotes are placed
    
    This provides dynamic A and k that adapt to current market conditions.
    """
    
    def __init__(self, symbol: str, config: Optional[CalibrationConfig] = None):
        self.symbol = symbol.lower()
        self.config = config or CalibrationConfig()
        
        # Trade arrival records: (timestamp, depth_ticks)
        self._arrivals: deque = deque(maxlen=50000)
        
        # Current calibrated parameters
        self._A: float = self.config.default_A
        self._k: float = self.config.default_k
        self._last_calibration: float = 0.0
        self._calibration_count: int = 0
        
        # Stats
        self._total_trades: int = 0
        
        logger.info(
            f"[{symbol.upper()}] IntensityCalibrator initialized: "
            f"window={self.config.window_seconds}s, "
            f"shallow_depth={self.config.shallow_depth_ticks} ticks"
        )
    
    def on_trade(self, trade_price: float, mid_price: float, timestamp: Optional[float] = None):
        """
        Record a trade for intensity calibration.
        
        Args:
            trade_price: The execution price of the trade
            mid_price: Current mid-price at time of trade
            timestamp: Unix timestamp (uses current time if None)
        """
        if mid_price <= 0:
            return
        
        ts = timestamp or time.time()
        
        # Calculate depth in basis points, then convert to ticks
        depth_bps = abs(trade_price - mid_price) / mid_price * 10000
        depth_ticks = depth_bps / self.config.tick_size_bps
        
        # Only record trades within reasonable range
        if depth_ticks >= 0 and depth_ticks < 500:
            self._arrivals.append((ts, depth_ticks))
            self._total_trades += 1
        
        # Check if recalibration needed
        if ts - self._last_calibration >= self.config.recalibrate_interval:
            self._recalibrate(ts)
    
    def _recalibrate(self, current_time: float):
        """Perform rolling recalibration of A and k."""
        self._last_calibration = current_time
        
        # Get arrivals within window
        cutoff = current_time - self.config.window_seconds
        recent = [(ts, depth) for ts, depth in self._arrivals if ts >= cutoff]
        
        if len(recent) < self.config.min_samples:
            logger.debug(
                f"[{self.symbol.upper()}] Insufficient samples for calibration: "
                f"{len(recent)}/{self.config.min_samples}"
            )
            return
        
        # Extract depths
        depths = np.array([depth for _, depth in recent])
        
        # Count arrivals per tick bucket
        max_tick = min(int(depths.max()) + 1, 500)
        counts = np.zeros(max_tick, dtype=np.float64)
        
        for depth in depths:
            tick_idx = int(depth)
            if 0 <= tick_idx < max_tick:
                # All quotes within this depth would have been filled
                counts[:tick_idx + 1] += 1
        
        # Convert to rate (per second)
        counts /= self.config.window_seconds
        
        # Filter to shallow depths for fitting (per HFTBacktest recommendation)
        shallow_limit = min(self.config.shallow_depth_ticks, max_tick)
        if shallow_limit < 10:
            return  # Not enough range to fit
        
        ticks = np.arange(shallow_limit) + 0.5  # Mid-tick values
        lambda_shallow = counts[:shallow_limit]
        
        # Remove zeros (can't take log)
        valid_mask = lambda_shallow > 0
        if valid_mask.sum() < 5:
            return
        
        ticks_valid = ticks[valid_mask]
        lambda_valid = lambda_shallow[valid_mask]
        
        # Linear regression on log(λ) = log(A) - k * δ
        try:
            log_lambda = np.log(lambda_valid)
            slope, intercept = self._linear_regression(ticks_valid, log_lambda)
            
            new_A = np.exp(intercept)
            new_k = -slope
            
            # Sanity checks
            if new_A <= 0 or new_k <= 0 or new_A > 100 or new_k > 1.0:
                logger.warning(
                    f"[{self.symbol.upper()}] Calibration produced invalid params: "
                    f"A={new_A:.3f}, k={new_k:.4f}. Using defaults."
                )
                return
            
            self._A = new_A
            self._k = new_k
            self._calibration_count += 1
            
            logger.info(
                f"[{self.symbol.upper()}] Intensity recalibrated: "
                f"A={self._A:.3f} fills/s, k={self._k:.4f}, "
                f"samples={len(recent)}"
            )
            
        except Exception as e:
            logger.warning(f"[{self.symbol.upper()}] Calibration failed: {e}")
    
    @staticmethod
    def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Simple linear regression returning (slope, intercept)."""
        n = len(x)
        sx = np.sum(x)
        sy = np.sum(y)
        sx2 = np.sum(x ** 2)
        sxy = np.sum(x * y)
        
        slope = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
        intercept = (sy - slope * sx) / n
        
        return slope, intercept
    
    def get_params(self) -> Tuple[float, float]:
        """
        Get current calibrated A and k parameters.
        
        Returns:
            (A, k) tuple - intensity scale and decay rate
        """
        return self._A, self._k
    
    def get_fill_probability(self, depth_bps: float) -> float:
        """
        Estimate fill probability at a given depth from mid.
        
        Args:
            depth_bps: Distance from mid-price in basis points
            
        Returns:
            Probability of fill (0 to 1)
        """
        depth_ticks = depth_bps / self.config.tick_size_bps
        # λ(δ) / A gives relative probability
        return math.exp(-self._k * depth_ticks)
    
    def get_stats(self) -> dict:
        """Get calibration statistics."""
        return {
            'symbol': self.symbol,
            'A': self._A,
            'k': self._k,
            'total_trades': self._total_trades,
            'calibration_count': self._calibration_count,
            'last_calibration': self._last_calibration,
            'arrivals_in_window': len(self._arrivals)
        }


# Symbol-specific calibrators
_calibrators: dict = {}


def get_intensity_calibrator(symbol: str) -> IntensityCalibrator:
    """Get or create calibrator for a symbol."""
    symbol = symbol.lower()
    if symbol not in _calibrators:
        _calibrators[symbol] = IntensityCalibrator(symbol)
    return _calibrators[symbol]
