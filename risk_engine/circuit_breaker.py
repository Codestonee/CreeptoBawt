"""
Multi-Level Circuit Breaker - Four-level protection system.

Levels:
L1: Position limit (per symbol) → pause entry
L2: Portfolio heat (Σ|position * volatility|) → reduce size
L3: Peak equity drawdown → hard stop
L4: Margin call threshold → EMERGENCY market close
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("Risk.CircuitBreaker")


class BreakerLevel(str, Enum):
    """Circuit breaker severity levels with ordinal for proper comparison."""
    NORMAL = "NORMAL"           # All systems go
    L1_POSITION = "L1_POSITION"  # Position limit hit
    L2_HEAT = "L2_HEAT"         # Portfolio heat high
    L3_DRAWDOWN = "L3_DRAWDOWN"  # Drawdown limit hit
    L4_EMERGENCY = "L4_EMERGENCY"  # Margin call imminent
    
    @property
    def severity(self) -> int:
        """Return numeric severity for proper comparison."""
        order = {
            "NORMAL": 0,
            "L1_POSITION": 1,
            "L2_HEAT": 2,
            "L3_DRAWDOWN": 3,
            "L4_EMERGENCY": 4,
        }
        return order.get(self.value, 0)


class BreakerAction(str, Enum):
    """Actions to take at each level."""
    ALLOW = "ALLOW"             # Trade normally
    REDUCE_ONLY = "REDUCE_ONLY"  # Only allow position reduction
    PAUSE_NEW = "PAUSE_NEW"     # Pause new entries
    HALT = "HALT"               # Stop all trading
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"  # Close all positions


@dataclass
class BreakerStatus:
    """Current circuit breaker status."""
    level: BreakerLevel
    action: BreakerAction
    reason: str
    triggered_at: float
    # Details
    current_value: Optional[float] = None
    threshold: Optional[float] = None


class CircuitBreaker:
    """
    Multi-level circuit breaker for capital protection.
    
    Levels:
    L1: Per-symbol position limit - pause entry to that symbol
    L2: Portfolio heat limit - reduce all position sizes
    L3: Peak equity drawdown - hard stop all trading
    L4: Margin utilization - emergency close everything
    """
    
    def __init__(
        self,
        # L1: Position limits
        max_position_usd: float = 10000.0,  # Max position value per symbol
        # L2: Portfolio heat
        max_portfolio_heat: float = 50000.0,  # Max Σ|position * volatility|
        # L3: Drawdown
        max_drawdown_pct: float = 0.10,  # 10% max drawdown from peak
        # L4: Margin
        margin_call_threshold: float = 0.80,  # 80% margin utilization
        # Consecutive losses
        max_consecutive_losses: int = 5
    ):
        # Thresholds
        self.max_position_usd = max_position_usd
        self.max_portfolio_heat = max_portfolio_heat
        self.max_drawdown_pct = max_drawdown_pct
        self.margin_call_threshold = margin_call_threshold
        self.max_consecutive_losses = max_consecutive_losses
        
        # State tracking
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.last_reset_date: str = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Current status per symbol
        self._symbol_limits: Dict[str, bool] = {}  # symbol -> is_limited
        
        # Global status
        self._current_level: BreakerLevel = BreakerLevel.NORMAL
        self._triggered_at: float = 0.0
        self._trigger_reason: str = ""
    
    def update_equity(self, equity: float):
        """Update current equity and check drawdown."""
        self.current_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Check L3: Drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown >= self.max_drawdown_pct:
                self._trigger(
                    BreakerLevel.L3_DRAWDOWN,
                    f"Drawdown {drawdown*100:.1f}% exceeds {self.max_drawdown_pct*100}% limit",
                    drawdown,
                    self.max_drawdown_pct
                )
    
    def update_margin_utilization(self, utilization: float):
        """Update margin utilization and check L4."""
        if utilization >= self.margin_call_threshold:
            self._trigger(
                BreakerLevel.L4_EMERGENCY,
                f"Margin utilization {utilization*100:.1f}% exceeds {self.margin_call_threshold*100}%",
                utilization,
                self.margin_call_threshold
            )
    
    def check_position_limit(
        self,
        symbol: str,
        current_position_value: float,
        proposed_addition: float
    ) -> bool:
        """Check L1: Position limit for a symbol."""
        total = abs(current_position_value) + abs(proposed_addition)
        
        if total > self.max_position_usd:
            self._symbol_limits[symbol] = True
            logger.warning(
                f"L1: Position limit for {symbol}: ${total:.0f} > ${self.max_position_usd:.0f}"
            )
            return False
        
        return True
    
    def check_portfolio_heat(
        self,
        positions: Dict[str, float],
        volatilities: Dict[str, float]
    ) -> bool:
        """Check L2: Portfolio heat (position * volatility sum)."""
        heat = sum(
            abs(positions.get(sym, 0)) * volatilities.get(sym, 0.01)
            for sym in set(positions) | set(volatilities)
        )
        
        if heat > self.max_portfolio_heat:
            self._trigger(
                BreakerLevel.L2_HEAT,
                f"Portfolio heat ${heat:.0f} exceeds ${self.max_portfolio_heat:.0f}",
                heat,
                self.max_portfolio_heat
            )
            return False
        
        return True
    
    def record_trade_result(self, pnl: float):
        """Record trade PnL for consecutive loss tracking."""
        self._check_daily_reset()
        
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning(
                    f"L1: {self.consecutive_losses} consecutive losses - consider pausing"
                )
        else:
            self.consecutive_losses = 0
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        action = self.get_action()
        return action in (BreakerAction.ALLOW, BreakerAction.REDUCE_ONLY)
    
    def can_open_new(self) -> bool:
        """Check if new positions are allowed."""
        action = self.get_action()
        return action == BreakerAction.ALLOW
    
    def get_status(self) -> BreakerStatus:
        """Get current circuit breaker status."""
        return BreakerStatus(
            level=self._current_level,
            action=self.get_action(),
            reason=self._trigger_reason or "Normal operation",
            triggered_at=self._triggered_at,
            current_value=self._get_current_value(),
            threshold=self._get_current_threshold()
        )
    
    def get_action(self) -> BreakerAction:
        """Get the action for current level."""
        actions = {
            BreakerLevel.NORMAL: BreakerAction.ALLOW,
            BreakerLevel.L1_POSITION: BreakerAction.PAUSE_NEW,
            BreakerLevel.L2_HEAT: BreakerAction.REDUCE_ONLY,
            BreakerLevel.L3_DRAWDOWN: BreakerAction.HALT,
            BreakerLevel.L4_EMERGENCY: BreakerAction.EMERGENCY_CLOSE,
        }
        return actions.get(self._current_level, BreakerAction.HALT)
    
    def reset(self, level: Optional[BreakerLevel] = None):
        """Reset circuit breaker (manual intervention)."""
        if level:
            if self._current_level == level:
                self._current_level = BreakerLevel.NORMAL
                self._trigger_reason = ""
                logger.info(f"Circuit breaker level {level} reset")
        else:
            self._current_level = BreakerLevel.NORMAL
            self._trigger_reason = ""
            self._symbol_limits.clear()
            logger.info("Circuit breaker fully reset")
    
    def _trigger(
        self,
        level: BreakerLevel,
        reason: str,
        current: float,
        threshold: float
    ):
        """Trigger a circuit breaker level."""
        # Only escalate, never de-escalate automatically (use severity for proper ordering)
        if level.severity > self._current_level.severity:
            self._current_level = level
            self._trigger_reason = reason
            self._triggered_at = time.time()
            
            logger.critical(f"CIRCUIT BREAKER {level.value}: {reason}")
    
    def _check_daily_reset(self):
        """Reset daily PnL at midnight UTC."""
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            logger.info("New day - resetting daily PnL")
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            # Note: We do NOT auto-reset L3/L4 breakers
    
    def _get_current_value(self) -> Optional[float]:
        """Get current metric for active level."""
        if self._current_level == BreakerLevel.L3_DRAWDOWN:
            if self.peak_equity > 0:
                return (self.peak_equity - self.current_equity) / self.peak_equity
        return None
    
    def _get_current_threshold(self) -> Optional[float]:
        """Get threshold for active level."""
        thresholds = {
            BreakerLevel.L1_POSITION: self.max_position_usd,
            BreakerLevel.L2_HEAT: self.max_portfolio_heat,
            BreakerLevel.L3_DRAWDOWN: self.max_drawdown_pct,
            BreakerLevel.L4_EMERGENCY: self.margin_call_threshold,
        }
        return thresholds.get(self._current_level)