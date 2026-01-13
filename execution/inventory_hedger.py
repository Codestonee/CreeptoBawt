"""
Inventory Hedger - Delta-Neutral Hedging for Market Making.

When market making inventory exceeds the configured threshold,
automatically hedges using perpetual futures to neutralize delta risk.

Per Gemini research:
- Trigger at 80% of MAX_POSITION_USD
- Use perpetual futures to hedge spot exposure
- Collect funding rate as compensation for holding hedge

Usage:
    hedger = InventoryHedger(event_queue)
    await hedger.start()
    
    # Check and hedge if needed:
    await hedger.check_and_hedge(symbol, inventory, price)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from config.settings import settings
from core.events import SignalEvent

logger = logging.getLogger("Execution.InventoryHedger")


class HedgeAction(str, Enum):
    """Hedge action types."""
    NONE = "NONE"
    OPEN_HEDGE = "OPEN_HEDGE"
    CLOSE_HEDGE = "CLOSE_HEDGE"
    ADJUST_HEDGE = "ADJUST_HEDGE"


@dataclass
class HedgePosition:
    """Tracks a hedge position for a symbol."""
    symbol: str
    mm_inventory: float       # Current MM inventory (spot/perp position)
    hedge_qty: float          # Current hedge quantity (opposite direction)
    net_delta: float          # Net exposure (mm_inventory + hedge_qty)
    last_hedge_time: float    # When hedge was last adjusted
    hedge_count: int = 0      # Number of hedge adjustments


class InventoryHedger:
    """
    Automatic delta-neutral hedging for market making inventory.
    
    When MM inventory exceeds threshold:
    1. Calculate required hedge size
    2. Submit hedge order (opposite direction)
    3. Track net delta exposure
    
    Benefits:
    - Eliminates directional risk from stuck positions
    - Collects funding rate on perp hedge
    - Converts losing position to "cash and carry" arbitrage
    """
    
    # Configuration
    MIN_HEDGE_INTERVAL_SECONDS = 30.0  # Don't hedge too frequently
    MIN_HEDGE_SIZE_USD = 50.0          # Minimum hedge worth doing
    
    def __init__(self, event_queue):
        self.queue = event_queue
        self._positions: Dict[str, HedgePosition] = {}
        self._enabled = settings.ENABLE_AUTO_HEDGE
        self._threshold_pct = settings.HEDGE_THRESHOLD_PCT
        self._max_position_usd = settings.MAX_POSITION_USD
        
        self._stats = {
            'total_hedges': 0,
            'hedged_usd': 0.0,
            'symbols_hedged': set()
        }
        
        logger.info(
            f"InventoryHedger initialized: enabled={self._enabled}, "
            f"threshold={self._threshold_pct*100:.0f}%"
        )
    
    async def start(self):
        """Start the hedger (for future background tasks)."""
        if self._enabled:
            logger.info("InventoryHedger started")
    
    async def stop(self):
        """Stop the hedger."""
        logger.info("InventoryHedger stopped")
    
    def is_enabled(self) -> bool:
        """Check if auto-hedging is enabled."""
        return self._enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable auto-hedging."""
        self._enabled = enabled
        logger.info(f"InventoryHedger {'enabled' if enabled else 'disabled'}")
    
    async def check_and_hedge(
        self,
        symbol: str,
        inventory: float,
        price: float
    ) -> HedgeAction:
        """
        Check if inventory needs hedging and execute if necessary.
        
        Args:
            symbol: Trading symbol
            inventory: Current MM inventory (positive=long, negative=short)
            price: Current price
            
        Returns:
            HedgeAction taken
        """
        if not self._enabled:
            return HedgeAction.NONE
        
        symbol = symbol.lower()
        
        # Calculate inventory value
        inventory_usd = abs(inventory * price)
        threshold_usd = self._max_position_usd * self._threshold_pct
        
        # Get or create position tracker
        position = self._positions.get(symbol)
        if position is None:
            position = HedgePosition(
                symbol=symbol,
                mm_inventory=inventory,
                hedge_qty=0.0,
                net_delta=inventory,
                last_hedge_time=0.0
            )
            self._positions[symbol] = position
        
        # Update MM inventory
        position.mm_inventory = inventory
        position.net_delta = inventory + position.hedge_qty
        
        # Check if hedge needed
        if inventory_usd < threshold_usd:
            # Below threshold - no hedge needed
            # But if we have an existing hedge and inventory recovered, close it
            if abs(position.hedge_qty) > 0 and inventory_usd < threshold_usd * 0.5:
                return await self._close_hedge(position, price)
            return HedgeAction.NONE
        
        # Check cooldown
        now = time.time()
        if now - position.last_hedge_time < self.MIN_HEDGE_INTERVAL_SECONDS:
            return HedgeAction.NONE
        
        # Calculate required hedge
        # Target: net_delta = 0
        required_hedge = -inventory  # Opposite of inventory
        current_hedge = position.hedge_qty
        adjustment = required_hedge - current_hedge
        
        # Skip if adjustment too small
        adjustment_usd = abs(adjustment * price)
        if adjustment_usd < self.MIN_HEDGE_SIZE_USD:
            return HedgeAction.NONE
        
        # Execute hedge
        return await self._execute_hedge(position, adjustment, price)
    
    async def _execute_hedge(
        self,
        position: HedgePosition,
        adjustment: float,
        price: float
    ) -> HedgeAction:
        """Execute a hedge adjustment."""
        symbol = position.symbol
        
        # Determine side
        if adjustment > 0:
            side = 'BUY'  # Need to buy to hedge short inventory
        else:
            side = 'SELL'  # Need to sell to hedge long inventory
        
        quantity = abs(adjustment)
        
        # Create hedge signal - use aggressive LIMIT order at mid price
        # This avoids market order slippage while still getting quick fills
        hedge_signal = SignalEvent(
            strategy_id='inventory_hedger',
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            price=price,  # Use current price for aggressive limit
            order_type='LIMIT',  # Changed from MARKET to reduce slippage
            exchange='binance'
        )
        
        logger.warning(
            f"🔒 [{symbol.upper()}] HEDGING: {side} {quantity:.4f} @ MARKET "
            f"(inventory={position.mm_inventory:.4f}, "
            f"hedge_adjustment={adjustment:.4f})"
        )
        
        await self.queue.put(hedge_signal)
        
        # Update tracking
        position.hedge_qty += adjustment
        position.net_delta = position.mm_inventory + position.hedge_qty
        position.last_hedge_time = time.time()
        position.hedge_count += 1
        
        # Update stats
        self._stats['total_hedges'] += 1
        self._stats['hedged_usd'] += abs(adjustment * price)
        self._stats['symbols_hedged'].add(symbol)
        
        action = HedgeAction.OPEN_HEDGE if abs(position.hedge_qty) > abs(position.hedge_qty - adjustment) else HedgeAction.ADJUST_HEDGE
        return action
    
    async def _close_hedge(self, position: HedgePosition, price: float) -> HedgeAction:
        """Close an existing hedge when inventory recovers."""
        if abs(position.hedge_qty) < 0.0001:
            return HedgeAction.NONE
        
        symbol = position.symbol
        
        # Close hedge = trade in opposite direction
        if position.hedge_qty > 0:
            side = 'SELL'
        else:
            side = 'BUY'
        
        quantity = abs(position.hedge_qty)
        
        close_signal = SignalEvent(
            strategy_id='inventory_hedger_close',
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            price=price,  # Use current price for aggressive limit
            order_type='LIMIT',  # Changed from MARKET to reduce slippage
            exchange='binance'
        )
        
        logger.info(
            f"🔓 [{symbol.upper()}] CLOSING HEDGE: {side} {quantity:.4f} @ MARKET "
            f"(inventory recovered to {position.mm_inventory:.4f})"
        )
        
        await self.queue.put(close_signal)
        
        position.hedge_qty = 0.0
        position.net_delta = position.mm_inventory
        position.last_hedge_time = time.time()
        
        return HedgeAction.CLOSE_HEDGE
    
    def get_position(self, symbol: str) -> Optional[HedgePosition]:
        """Get hedge position for a symbol."""
        return self._positions.get(symbol.lower())
    
    def get_net_delta(self, symbol: str) -> float:
        """Get net delta exposure for a symbol."""
        position = self._positions.get(symbol.lower())
        return position.net_delta if position else 0.0
    
    def get_stats(self) -> dict:
        """Get hedger statistics."""
        return {
            **self._stats,
            'symbols_hedged': list(self._stats['symbols_hedged']),
            'active_positions': len([p for p in self._positions.values() if abs(p.hedge_qty) > 0])
        }


# Global instance
_inventory_hedger: Optional[InventoryHedger] = None


def get_inventory_hedger(event_queue=None) -> InventoryHedger:
    """Get or create the global inventory hedger."""
    global _inventory_hedger
    if _inventory_hedger is None:
        if event_queue is None:
            raise ValueError("event_queue required for first initialization")
        _inventory_hedger = InventoryHedger(event_queue)
    return _inventory_hedger
