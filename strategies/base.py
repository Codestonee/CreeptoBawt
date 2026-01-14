from abc import ABC, abstractmethod
from typing import List, Optional, Any
from core.events import MarketEvent, FillEvent, RegimeEvent, SignalEvent, FundingRateEvent

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    
    Enforces a standard interface for:
    - Event handling (tick, fill, regime, funding)
    - Lifecycle management (startup/shutdown)
    - State persistence
    """
    
    def __init__(self, event_queue: Any, symbols: List[str]):
        """
        Initialize the strategy.
        
        Args:
            event_queue: Async queue to put SignalEvents into.
            symbols: List of trading symbols this strategy manages.
        """
        self.queue = event_queue
        # Store symbols in lower case for consistency
        self.symbols = [s.lower() for s in symbols]
        
    @abstractmethod
    async def on_tick(self, event: MarketEvent):
        """
        Handle incoming market ticks (price updates).
        
        This is the main loop for most strategies.
        Should include logic for generating signals based on price action.
        """
        pass
        
    @abstractmethod
    async def on_fill(self, event: FillEvent):
        """
        Handle order fill events.
        
        Use this to update internal inventory/position state
        and trigger any post-trade logic (e.g. hedging).
        """
        pass
    
    async def on_regime_change(self, event: RegimeEvent):
        """
        Handle market regime changes (Optional).
        
        Override if strategy should adapt behavior based on market state
        (e.g., pause in high volatility).
        """
        pass
        
    async def on_funding_rate(self, event: FundingRateEvent):
        """
        Handle funding rate updates (Optional).
        
        Override for funding arbitrage or holding cost calculations.
        """
        pass
    
    async def start(self):
        """
        Perform any necessary startup tasks (loading state, etc).
        """
        pass
        
    async def stop(self):
        """
        Perform any cleanup on shutdown.
        """
        pass
