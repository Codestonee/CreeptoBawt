"""
Shadow Mode Controller for Phased Deployment.

Enables gradual rollout from paper trading to full production:
1. SHADOW (0%): Generate orders, don't execute - validate logic
2. MICRO (1%): Tiny positions for real fee/fill data
3. RAMP phases (10% → 25% → 50% → 100%): Gradual size increase

This prevents catastrophic losses during validation of new components.

Reference: CreeptBaws optimization research documents
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("Execution.ShadowMode")


class DeploymentPhase(Enum):
    """Deployment phases with execution multipliers."""
    SHADOW = ("SHADOW", 0.00)      # Paper trade only
    MICRO = ("MICRO", 0.01)        # 1% size
    RAMP_10 = ("RAMP_10", 0.10)    # 10% size
    RAMP_25 = ("RAMP_25", 0.25)    # 25% size
    RAMP_50 = ("RAMP_50", 0.50)    # 50% size
    RAMP_75 = ("RAMP_75", 0.75)    # 75% size
    RAMP_100 = ("RAMP_100", 1.00)  # Full production
    
    @property
    def multiplier(self) -> float:
        return self.value[1]
    
    @property
    def phase_name(self) -> str:
        return self.value[0]


@dataclass
class ShadowTradeResult:
    """Result of a shadow trade for comparison."""
    symbol: str
    side: str
    quantity: float
    target_price: float
    would_fill: bool
    simulated_fill_price: float
    market_mid_at_fill: float
    simulated_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)


class ShadowModeController:
    """
    Controller for phased deployment from shadow to production.
    
    Usage:
        controller = ShadowModeController(phase=DeploymentPhase.MICRO)
        
        # When placing an order:
        if controller.should_execute():
            adjusted_qty = controller.adjust_quantity(original_qty)
            result = await place_order(qty=adjusted_qty)
        else:
            controller.record_shadow_trade(...)
    """
    
    def __init__(
        self,
        phase: DeploymentPhase = DeploymentPhase.SHADOW,
        min_notional_usd: float = 10.0
    ):
        """
        Initialize shadow mode controller.
        
        Args:
            phase: Current deployment phase
            min_notional_usd: Minimum order value (exchange limit)
        """
        self.phase = phase
        self.min_notional_usd = min_notional_usd
        
        # Shadow trade records for comparison
        self.shadow_trades: list = []
        self.max_shadow_history = 1000
        
        # Statistics
        self.stats = {
            'total_shadow_trades': 0,
            'would_have_filled': 0,
            'simulated_pnl': 0.0,
            'actual_trades': 0,
            'actual_pnl': 0.0
        }
        
        logger.info(f"ShadowModeController initialized: {phase.phase_name} ({phase.multiplier*100:.0f}%)")
    
    def should_execute(self) -> bool:
        """Check if orders should be executed in current phase."""
        return self.phase.multiplier > 0
    
    def adjust_quantity(self, quantity: float, price: float = 0.0) -> float:
        """
        Adjust order quantity based on current phase.
        
        Args:
            quantity: Original intended quantity
            price: Current price (for min notional check)
            
        Returns:
            Adjusted quantity (may be 0 if below min notional)
        """
        adjusted = quantity * self.phase.multiplier
        
        # Check minimum notional
        if price > 0:
            notional = adjusted * price
            if notional < self.min_notional_usd:
                logger.debug(
                    f"Adjusted qty ${notional:.2f} below min ${self.min_notional_usd}, "
                    f"skipping order"
                )
                return 0.0
        
        return adjusted
    
    def record_shadow_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        target_price: float,
        market_mid: float,
        would_fill: bool = True
    ) -> ShadowTradeResult:
        """
        Record a shadow trade for later comparison.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Intended quantity
            target_price: Intended entry price
            market_mid: Market mid price at fill time
            would_fill: Whether order would have filled
            
        Returns:
            ShadowTradeResult with simulated outcome
        """
        # Simulate fill price (assume 1 bps slippage)
        slippage = 0.0001
        if side == "BUY":
            fill_price = target_price * (1 + slippage)
        else:
            fill_price = target_price * (1 - slippage)
        
        # Calculate simulated P&L
        if would_fill:
            if side == "BUY":
                pnl = (market_mid - fill_price) * quantity
            else:
                pnl = (fill_price - market_mid) * quantity
        else:
            pnl = 0.0
        
        result = ShadowTradeResult(
            symbol=symbol,
            side=side,
            quantity=quantity,
            target_price=target_price,
            would_fill=would_fill,
            simulated_fill_price=fill_price,
            market_mid_at_fill=market_mid,
            simulated_pnl=pnl
        )
        
        # Store and maintain history limit
        self.shadow_trades.append(result)
        if len(self.shadow_trades) > self.max_shadow_history:
            self.shadow_trades.pop(0)
        
        # Update stats
        self.stats['total_shadow_trades'] += 1
        if would_fill:
            self.stats['would_have_filled'] += 1
            self.stats['simulated_pnl'] += pnl
        
        logger.debug(
            f"Shadow trade: {side} {quantity:.6f} {symbol} @ {target_price:.2f}, "
            f"would_fill={would_fill}, sim_pnl=${pnl:.2f}"
        )
        
        return result
    
    def record_actual_trade(self, pnl: float) -> None:
        """Record an actual executed trade result."""
        self.stats['actual_trades'] += 1
        self.stats['actual_pnl'] += pnl
    
    def get_comparison_stats(self) -> Dict[str, Any]:
        """
        Get comparison between shadow and actual performance.
        
        Returns:
            Dict with comparison metrics
        """
        shadow_fill_rate = (
            self.stats['would_have_filled'] / self.stats['total_shadow_trades'] * 100
            if self.stats['total_shadow_trades'] > 0 else 0
        )
        
        return {
            'phase': self.phase.phase_name,
            'multiplier': self.phase.multiplier,
            'shadow_trades': self.stats['total_shadow_trades'],
            'shadow_fill_rate': shadow_fill_rate,
            'simulated_pnl': self.stats['simulated_pnl'],
            'actual_trades': self.stats['actual_trades'],
            'actual_pnl': self.stats['actual_pnl'],
            'pnl_difference': self.stats['actual_pnl'] - self.stats['simulated_pnl']
        }
    
    def advance_phase(self) -> DeploymentPhase:
        """
        Advance to next deployment phase.
        
        Returns:
            New phase
        """
        phase_order = list(DeploymentPhase)
        current_idx = phase_order.index(self.phase)
        
        if current_idx < len(phase_order) - 1:
            self.phase = phase_order[current_idx + 1]
            logger.info(f"Advanced to phase: {self.phase.phase_name} ({self.phase.multiplier*100:.0f}%)")
        else:
            logger.info("Already at maximum phase (RAMP_100)")
        
        return self.phase
    
    def set_phase(self, phase: DeploymentPhase) -> None:
        """Manually set deployment phase."""
        old_phase = self.phase
        self.phase = phase
        logger.info(f"Phase changed: {old_phase.phase_name} → {phase.phase_name}")
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            'total_shadow_trades': 0,
            'would_have_filled': 0,
            'simulated_pnl': 0.0,
            'actual_trades': 0,
            'actual_pnl': 0.0
        }
        self.shadow_trades.clear()
        logger.info("Shadow mode stats reset")


# Singleton instance for global access
_shadow_controller: Optional[ShadowModeController] = None


def get_shadow_controller(
    phase: DeploymentPhase = DeploymentPhase.SHADOW
) -> ShadowModeController:
    """
    Get or create the global shadow mode controller.
    
    Args:
        phase: Initial phase if creating new controller
        
    Returns:
        ShadowModeController instance
    """
    global _shadow_controller
    
    if _shadow_controller is None:
        _shadow_controller = ShadowModeController(phase=phase)
    
    return _shadow_controller
