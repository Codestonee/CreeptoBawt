"""
VPIN (Volume-Synchronized Probability of Informed Trading) Calculator.

Detects toxic order flow by measuring imbalance between buy and sell
volume within equal-volume buckets. High VPIN indicates informed traders
are active and passive quoting has negative expected value.

Thresholds (crypto-calibrated):
- VPIN < 0.30: NORMAL - full quoting
- 0.30 <= VPIN < 0.50: CAUTION - widen spreads
- VPIN >= 0.50: TOXIC - pause quotes

References:
- Easley, López de Prado, O'Hara (2012) "Flow Toxicity and Liquidity"
"""

import math
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger("Analysis.VPIN")


class VPINState(str, Enum):
    """VPIN toxicity states (tiered per Gemini research)."""
    WARMUP = "WARMUP"    # Not enough buckets yet
    NORMAL = "NORMAL"    # Safe to quote normally
    CAUTION = "CAUTION"  # Slightly widen spreads
    WARNING = "WARNING"  # NEW: Widen spreads 50%, reduce size 50%
    TOXIC = "TOXIC"      # Pause quoting


@dataclass
class VPINConfig:
    """Configuration for VPIN calculator."""
    bucket_volume: float = 0.01      # Volume per bucket (BTC units)
    num_buckets: int = 50            # Rolling window size
    warmup_buckets: int = 30         # Min buckets before trusting VPIN
    # Tiered thresholds (Gemini research recommended)
    caution_threshold: float = 0.50  # VPIN >= this = CAUTION
    warning_threshold: float = 0.65  # VPIN >= this = WARNING (widen spreads 50%)
    toxic_threshold: float = 0.80    # VPIN >= this = TOXIC (halt quoting)
    
    @classmethod
    def for_symbol(cls, symbol: str) -> "VPINConfig":
        """Get symbol-specific config with appropriate bucket sizes."""
        symbol = symbol.lower()
        if "btc" in symbol:
            return cls(bucket_volume=0.01)  # ~$400 at 40k
        elif "eth" in symbol:
            return cls(bucket_volume=0.2)   # ~$500 at 2.5k
        elif "sol" in symbol:
            return cls(bucket_volume=5.0)   # ~$500 at 100
        else:
            return cls(bucket_volume=0.01)  # Default


class VPINCalculator:
    """
    Volume-synchronized VPIN detection for a single symbol.
    
    O(1) per trade update, safe to call on main asyncio loop at 100+ trades/sec.
    
    Usage:
        vpin = VPINCalculator("btcusdt")
        
        # On each trade event:
        vpin_value = vpin.on_trade(price=42000, quantity=0.005, side="BUY")
        
        # Check state:
        if vpin.get_state() == VPINState.TOXIC:
            pause_quoting()
    """
    
    def __init__(self, symbol: str, config: Optional[VPINConfig] = None):
        self.symbol = symbol.lower()
        self.config = config or VPINConfig.for_symbol(symbol)
        
        # Bucket state
        self._current_bucket_volume: float = 0.0
        self._current_buy_volume: float = 0.0
        self._bucket_imbalances: deque = deque(
            maxlen=max(self.config.num_buckets, self.config.warmup_buckets)
        )
        
        # BVC state (for side classification fallback)
        self._last_price: Optional[float] = None
        self._volatility: float = 0.001  # Default estimate
        
        logger.info(
            f"[{symbol.upper()}] VPINCalculator initialized: "
            f"bucket={self.config.bucket_volume}, "
            f"thresholds={self.config.caution_threshold}/{self.config.toxic_threshold}"
        )
    
    def on_trade(
        self, 
        price: float, 
        quantity: float, 
        side: Optional[str] = None
    ) -> Optional[float]:
        """
        Process a trade and update VPIN state.
        
        Args:
            price: Trade price (used for BVC fallback if side unknown)
            quantity: Trade quantity in base units (e.g., BTC)
            side: 'BUY' or 'SELL' from aggressor perspective.
                  If None, uses Bulk Volume Classification (BVC).
        
        Returns:
            VPIN value if a bucket is completed, None otherwise.
        """
        # Classify side if not provided (BVC fallback)
        if side is None:
            buy_prob = self._classify_bvc(price)
            buy_volume = quantity * buy_prob
        else:
            buy_volume = quantity if side.upper() == "BUY" else 0.0
        
        # Accumulate into current bucket
        self._current_buy_volume += buy_volume
        self._current_bucket_volume += quantity
        
        # Check if bucket is full
        if self._current_bucket_volume >= self.config.bucket_volume:
            return self._close_bucket()
        
        return None
    
    def _classify_bvc(self, price: float) -> float:
        """
        Bulk Volume Classification (BVC) using price drift.
        
        Returns probability that trade was buy-initiated (0-1).
        Uses error function for Gaussian approximation.
        
        Note: This is a fallback when exchange doesn't provide aggressor side.
        Binance aggTrade does provide side, so this is mainly for robustness.
        """
        if self._last_price is None:
            self._last_price = price
            return 0.5  # Unknown direction
        
        # Price drift
        delta = price - self._last_price
        self._last_price = price
        
        if self._volatility <= 0:
            return 0.5
        
        # Normalize by volatility
        z = delta / (self._volatility * math.sqrt(2))
        
        # CDF approximation using erf (10x faster than scipy.stats.norm.cdf)
        buy_prob = 0.5 * (1 + math.erf(z))
        
        return buy_prob
    
    def _close_bucket(self) -> float:
        """Close current bucket and compute imbalance."""
        v0 = self.config.bucket_volume
        
        # Order Imbalance: |V_buy - V_sell| / (V_buy + V_sell)
        # Simplified: |V_buy - V_0/2| / (V_0/2) when bucket is exactly V_0
        # This gives imbalance in [0, 1] where 1 = completely one-sided
        
        # Handle case where bucket might overflow slightly
        total_vol = max(self._current_bucket_volume, v0)
        sell_volume = total_vol - self._current_buy_volume
        
        if total_vol > 0:
            imbalance = abs(self._current_buy_volume - sell_volume) / total_vol
        else:
            imbalance = 0.0
        
        self._bucket_imbalances.append(imbalance)
        
        # Reset bucket
        self._current_bucket_volume = 0.0
        self._current_buy_volume = 0.0
        
        vpin = self.get_vpin()
        
        if vpin is not None:
            logger.debug(
                f"[{self.symbol.upper()}] Bucket closed: imbalance={imbalance:.3f}, "
                f"VPIN={vpin:.3f}, state={self.get_state().value}"
            )
        
        return vpin
    
    def get_vpin(self) -> Optional[float]:
        """
        Get current VPIN value.
        
        Returns:
            VPIN value in [0, 1] if enough buckets, None during warmup.
        """
        if len(self._bucket_imbalances) < self.config.warmup_buckets:
            return None
        
        # Use last N buckets for rolling VPIN
        recent = list(self._bucket_imbalances)[-self.config.num_buckets:]
        return sum(recent) / len(recent)
    
    def get_state(self) -> VPINState:
        """
        Get qualitative VPIN state for trading decisions.
        
        Returns:
            VPINState enum value.
        """
        vpin = self.get_vpin()
        
        if vpin is None:
            return VPINState.WARMUP
        if vpin >= self.config.toxic_threshold:
            return VPINState.TOXIC
        if vpin >= self.config.warning_threshold:
            return VPINState.WARNING
        if vpin >= self.config.caution_threshold:
            return VPINState.CAUTION
        return VPINState.NORMAL
    
    def is_safe_to_quote(self) -> bool:
        """Quick check if quoting is safe (not TOXIC)."""
        return self.get_state() != VPINState.TOXIC
    
    def set_volatility(self, volatility: float):
        """
        Update volatility estimate for BVC classification.
        
        Should be called periodically with EWMA volatility from main strategy.
        """
        self._volatility = max(volatility, 1e-8)
    
    def get_stats(self) -> dict:
        """Get current VPIN statistics."""
        return {
            "symbol": self.symbol,
            "vpin": self.get_vpin(),
            "state": self.get_state().value,
            "buckets_filled": len(self._bucket_imbalances),
            "current_bucket_pct": self._current_bucket_volume / self.config.bucket_volume,
            "config": {
                "bucket_volume": self.config.bucket_volume,
                "num_buckets": self.config.num_buckets,
                "thresholds": (self.config.caution_threshold, self.config.toxic_threshold)
            }
        }
    
    def reset(self):
        """Reset VPIN state (e.g., on reconnect)."""
        self._current_bucket_volume = 0.0
        self._current_buy_volume = 0.0
        self._bucket_imbalances.clear()
        self._last_price = None
        logger.info(f"[{self.symbol.upper()}] VPIN state reset")
