"""
Trading Mode Controller - Manages system operational modes.

Modes:
- LIVE: Real money, real orders
- PAPER: Simulated execution, live data
- SHADOW: Dual execution (log both real and simulated)
- REPLAY: Historical data at real speed
- BACKTEST: Historical data at max speed
- EMERGENCY_STOP: All trading halted
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Callable, List, Optional

import structlog

log = structlog.get_logger()


class TradingMode(str, Enum):
    """System operational modes."""
    LIVE = "live"
    PAPER = "paper"
    SHADOW = "shadow"
    REPLAY = "replay"
    BACKTEST = "backtest"
    EMERGENCY_STOP = "emergency_stop"
    
    def allows_real_orders(self) -> bool:
        """Check if mode allows real order execution."""
        return self in {TradingMode.LIVE, TradingMode.SHADOW}
    
    def uses_live_data(self) -> bool:
        """Check if mode uses live market data."""
        return self in {TradingMode.LIVE, TradingMode.PAPER, TradingMode.SHADOW}
    
    def is_simulation(self) -> bool:
        """Check if mode is simulated."""
        return self in {TradingMode.PAPER, TradingMode.REPLAY, TradingMode.BACKTEST}


class ModeController:
    """
    Thread-safe trading mode controller.
    
    Manages mode transitions and notifies listeners of changes.
    """
    
    # Valid mode transitions
    VALID_TRANSITIONS = {
        TradingMode.LIVE: {
            TradingMode.PAPER,
            TradingMode.EMERGENCY_STOP,
        },
        TradingMode.PAPER: {
            TradingMode.LIVE,
            TradingMode.SHADOW,
            TradingMode.EMERGENCY_STOP,
        },
        TradingMode.SHADOW: {
            TradingMode.LIVE,
            TradingMode.PAPER,
            TradingMode.EMERGENCY_STOP,
        },
        TradingMode.REPLAY: {
            TradingMode.PAPER,
            TradingMode.EMERGENCY_STOP,
        },
        TradingMode.BACKTEST: {
            TradingMode.PAPER,
            TradingMode.EMERGENCY_STOP,
        },
        TradingMode.EMERGENCY_STOP: {
            TradingMode.PAPER,  # Only to paper mode for safety
        },
    }
    
    def __init__(self, initial_mode: TradingMode = TradingMode.PAPER) -> None:
        self._mode = initial_mode
        self._lock = asyncio.Lock()
        self._mode_history: List[tuple[float, TradingMode, str]] = [
            (time.time(), initial_mode, "initialization")
        ]
        self._listeners: List[Callable[[TradingMode, TradingMode], None]] = []
        
        log.info(
            "mode_controller_initialized",
            initial_mode=initial_mode,
        )
    
    @property
    def current_mode(self) -> TradingMode:
        """Get current trading mode."""
        return self._mode
    
    @property
    def mode_history(self) -> List[tuple[float, TradingMode, str]]:
        """Get mode transition history."""
        return list(self._mode_history)
    
    def add_listener(self, callback: Callable[[TradingMode, TradingMode], None]) -> None:
        """Add a mode change listener."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[TradingMode, TradingMode], None]) -> None:
        """Remove a mode change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def can_transition_to(self, new_mode: TradingMode) -> bool:
        """Check if transition to new mode is valid."""
        if new_mode == self._mode:
            return True  # No-op transition is always valid
        return new_mode in self.VALID_TRANSITIONS.get(self._mode, set())
    
    async def switch_mode(
        self,
        new_mode: TradingMode,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """
        Switch to a new trading mode.
        
        Args:
            new_mode: Target mode
            reason: Reason for the switch
            force: Skip validity check (use with caution)
            
        Returns:
            True if switch was successful
        """
        async with self._lock:
            old_mode = self._mode
            
            # No-op if same mode
            if new_mode == old_mode:
                log.debug("mode_switch_noop", mode=new_mode)
                return True
            
            # Validate transition
            if not force and not self.can_transition_to(new_mode):
                log.warning(
                    "invalid_mode_transition",
                    old_mode=old_mode,
                    new_mode=new_mode,
                    reason=reason,
                )
                return False
            
            # Perform transition
            self._mode = new_mode
            self._mode_history.append((time.time(), new_mode, reason))
            
            log.info(
                "mode_switched",
                old_mode=old_mode,
                new_mode=new_mode,
                reason=reason,
                forced=force,
            )
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(old_mode, new_mode)
                except Exception as e:
                    log.error(
                        "mode_listener_error",
                        listener=str(listener),
                        error=str(e),
                    )
            
            return True
    
    async def emergency_stop(self, reason: str = "manual") -> bool:
        """
        Activate emergency stop mode.
        
        This is always allowed regardless of current mode.
        """
        return await self.switch_mode(
            TradingMode.EMERGENCY_STOP,
            reason=f"emergency_stop: {reason}",
            force=True,
        )
    
    def should_execute_real_orders(self) -> bool:
        """Check if real orders should be executed."""
        return self._mode.allows_real_orders()
    
    def should_execute_paper_orders(self) -> bool:
        """Check if paper orders should be executed."""
        return self._mode in {
            TradingMode.PAPER,
            TradingMode.SHADOW,
            TradingMode.REPLAY,
            TradingMode.BACKTEST,
        }
    
    def is_trading_allowed(self) -> bool:
        """Check if any trading is allowed."""
        return self._mode != TradingMode.EMERGENCY_STOP
    
    def get_status(self) -> dict:
        """Get mode controller status."""
        return {
            "current_mode": self._mode.value,
            "allows_real_orders": self._mode.allows_real_orders(),
            "uses_live_data": self._mode.uses_live_data(),
            "is_simulation": self._mode.is_simulation(),
            "is_trading_allowed": self.is_trading_allowed(),
            "transitions_count": len(self._mode_history),
            "last_transition": self._mode_history[-1] if self._mode_history else None,
        }
