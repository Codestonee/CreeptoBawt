"""
Base Strategy - Abstract interface for all trading strategies.

All strategies must implement:
- on_market_event: Process market data
- on_fill_event: Handle order fills
- get_health_status: Report strategy health
- emergency_shutdown: Graceful shutdown
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

from core.events import FillEvent, MarketEvent, SignalEvent, HealthStatus

log = structlog.get_logger()


@dataclass
class StrategyConfig:
    """Base strategy configuration."""
    strategy_id: str
    symbols: List[str]
    enabled: bool = True
    max_position_usd: Decimal = Decimal("10000")
    max_order_size_usd: Decimal = Decimal("1000")
    
    # Additional params can be added in subclass configs
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies inherit from this and implement the core methods.
    The strategy receives market events and emits trading signals.
    """
    
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self.symbols = config.symbols
        self.enabled = config.enabled
        
        # State
        self._positions: Dict[str, Decimal] = {s: Decimal("0") for s in config.symbols}
        self._pending_signals: List[SignalEvent] = []
        self._last_prices: Dict[str, Decimal] = {}
        
        log.info(
            "strategy_initialized",
            strategy_id=self.strategy_id,
            symbols=self.symbols,
        )
    
    @property
    def positions(self) -> Dict[str, Decimal]:
        """Get current positions."""
        return self._positions.copy()
    
    def get_position(self, symbol: str) -> Decimal:
        """Get position for a symbol."""
        return self._positions.get(symbol, Decimal("0"))
    
    # =========================================================================
    # Abstract Methods (Implement in Subclass)
    # =========================================================================
    
    @abstractmethod
    async def on_market_event(self, event: MarketEvent) -> Optional[List[SignalEvent]]:
        """
        Process market update and potentially emit trading signals.
        
        Args:
            event: Market data event (trade, book update, etc.)
            
        Returns:
            List of trading signals, or None if no action
        """
        pass
    
    @abstractmethod
    async def on_fill_event(self, fill: FillEvent) -> None:
        """
        Handle order fill notification.
        
        Update internal state (positions, inventory, PnL).
        
        Args:
            fill: Fill event from execution
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """
        Return strategy health for monitoring.
        
        Returns:
            Dict with health status, positions, PnL, etc.
        """
        pass
    
    @abstractmethod
    async def emergency_shutdown(self) -> List[SignalEvent]:
        """
        Gracefully wind down positions.
        
        Called when kill switch is activated.
        Should emit signals to close all positions.
        
        Returns:
            List of exit signals
        """
        pass
    
    # =========================================================================
    # Common Methods
    # =========================================================================
    
    def update_position(self, symbol: str, delta: Decimal) -> None:
        """Update position by delta."""
        current = self._positions.get(symbol, Decimal("0"))
        self._positions[symbol] = current + delta
        
        log.debug(
            "position_updated",
            strategy=self.strategy_id,
            symbol=symbol,
            old=str(current),
            new=str(self._positions[symbol]),
        )
    
    def update_last_price(self, symbol: str, price: Decimal) -> None:
        """Update last known price."""
        self._last_prices[symbol] = price
    
    def get_last_price(self, symbol: str) -> Optional[Decimal]:
        """Get last known price."""
        return self._last_prices.get(symbol)
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        log.info("strategy_enabled", strategy_id=self.strategy_id)
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        log.info("strategy_disabled", strategy_id=self.strategy_id)
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled."""
        return self.enabled
    
    def get_pnl(self) -> Dict[str, Decimal]:
        """
        Calculate unrealized PnL for all positions.
        
        Override in subclass for realized PnL tracking.
        """
        pnl = {}
        
        for symbol, position in self._positions.items():
            if position == 0:
                pnl[symbol] = Decimal("0")
                continue
            
            last_price = self._last_prices.get(symbol)
            if not last_price:
                pnl[symbol] = Decimal("0")
                continue
            
            # Simple unrealized PnL (would need entry price for accuracy)
            pnl[symbol] = position * last_price
        
        return pnl
