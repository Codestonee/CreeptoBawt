"""
Avellaneda-Stoikov Market Maker Strategy.

Mathematical framework for optimal market making with inventory risk.

Reference:
- Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from math import exp, log, sqrt
from typing import Any, Deque, Dict, List, Optional

import structlog

from core.events import (
    FillEvent,
    MarketEvent,
    OrderType,
    SignalEvent,
    SignalType,
    TimeInForce,
)
from strategies.base import BaseStrategy, StrategyConfig

log = structlog.get_logger()


@dataclass
class MarketMakerConfig(StrategyConfig):
    """Market maker specific configuration."""
    # Core parameters
    risk_aversion: Decimal = Decimal("0.5")  # γ (gamma)
    target_inventory: Decimal = Decimal("0")  # Q target
    max_inventory: Decimal = Decimal("1.0")  # Hard limit
    
    # Order sizing
    order_size: Decimal = Decimal("0.01")  # BTC per side
    
    # Volatility estimation
    volatility_window: int = 300  # 5 minutes in seconds
    volatility_ewma_span: int = 60  # EWMA span
    
    # Spread constraints
    min_spread_bps: Decimal = Decimal("5")  # 0.05%
    max_spread_bps: Decimal = Decimal("100")  # 1%
    
    # Time horizon
    session_length_seconds: int = 3600  # 1 hour
    
    # Inventory management
    inventory_skew_factor: Decimal = Decimal("1.0")  # How much to skew quotes
    panic_threshold: Decimal = Decimal("0.8")  # Panic flatten at 80% max inventory


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility estimator."""
    
    def __init__(self, span: int = 60) -> None:
        self.span = span
        self.alpha = 2 / (span + 1)
        self._variance = Decimal("0")
        self._last_price: Optional[Decimal] = None
        self._initialized = False
    
    @property
    def value(self) -> Decimal:
        """Get current volatility estimate."""
        if self._variance <= 0:
            return Decimal("0.01")  # Default 1%
        return Decimal(str(sqrt(float(self._variance))))
    
    def update(self, price: Decimal) -> None:
        """Update with new price."""
        if self._last_price is None:
            self._last_price = price
            return
        
        # Calculate return
        ret = (price - self._last_price) / self._last_price
        ret_squared = ret * ret
        
        # Update EWMA variance
        if not self._initialized:
            self._variance = ret_squared
            self._initialized = True
        else:
            self._variance = (
                Decimal(str(self.alpha)) * ret_squared +
                (1 - Decimal(str(self.alpha))) * self._variance
            )
        
        self._last_price = price


class OrderFlowIntensity:
    """Estimate order arrival intensity (κ/k)."""
    
    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._trade_times: Deque[float] = deque(maxlen=1000)
    
    def record_trade(self) -> None:
        """Record a trade arrival."""
        self._trade_times.append(time.time())
    
    def estimate(self) -> Decimal:
        """Estimate order arrival rate per second."""
        now = time.time()
        recent = sum(1 for t in self._trade_times if now - t < self.window)
        
        if recent < 2:
            return Decimal("1.0")  # Default
        
        return Decimal(str(recent / self.window))


class AvellanedaStoikovMM(BaseStrategy):
    """
    Avellaneda-Stoikov optimal market making strategy.
    
    Key concepts:
    - Reservation price (r): Fair value adjusted for inventory risk
    - Optimal spread (δ): Based on risk aversion and order flow
    - Inventory skew: Quote asymmetrically based on inventory
    """
    
    def __init__(self, config: MarketMakerConfig) -> None:
        super().__init__(config)
        self.mm_config = config
        
        # State
        self._inventory: Dict[str, Decimal] = {s: Decimal("0") for s in config.symbols}
        self._mid_prices: Dict[str, Decimal] = {}
        
        # Volatility estimators per symbol
        self._volatility: Dict[str, EWMAVolatility] = {
            s: EWMAVolatility(config.volatility_ewma_span)
            for s in config.symbols
        }
        
        # Order flow intensity
        self._order_flow: Dict[str, OrderFlowIntensity] = {
            s: OrderFlowIntensity()
            for s in config.symbols
        }
        
        # Session tracking
        self._session_start = time.time()
        
        # Active quotes (for cancellation)
        self._active_quotes: Dict[str, List[str]] = {s: [] for s in config.symbols}
        
        log.info(
            "market_maker_initialized",
            strategy_id=self.strategy_id,
            risk_aversion=str(config.risk_aversion),
            order_size=str(config.order_size),
        )
    
    async def on_market_event(self, event: MarketEvent) -> Optional[List[SignalEvent]]:
        """Process market update and emit quotes."""
        if not self.enabled:
            return None
        
        symbol = event.symbol
        if symbol not in self.symbols:
            return None
        
        # Update volatility
        self._volatility[symbol].update(event.price)
        
        # Record trade for order flow
        if event.event_type == "trade":
            self._order_flow[symbol].record_trade()
        
        # Update mid price
        if event.book_snapshot and event.book_snapshot.mid_price:
            self._mid_prices[symbol] = event.book_snapshot.mid_price
        else:
            self._mid_prices[symbol] = event.price
        
        self.update_last_price(symbol, event.price)
        
        # Calculate quotes
        return await self._calculate_quotes(symbol)
    
    async def _calculate_quotes(self, symbol: str) -> Optional[List[SignalEvent]]:
        """Calculate optimal bid/ask quotes using Avellaneda-Stoikov."""
        mid = self._mid_prices.get(symbol)
        if not mid:
            return None
        
        # Get current state
        q = self._inventory.get(symbol, Decimal("0"))  # Current inventory
        sigma = self._volatility[symbol].value  # Volatility
        gamma = self.mm_config.risk_aversion  # Risk aversion
        k = self._order_flow[symbol].estimate()  # Order intensity
        
        # Time to session end (in seconds)
        elapsed = time.time() - self._session_start
        T_minus_t = max(
            1.0,
            self.mm_config.session_length_seconds - elapsed,
        )
        
        # =====================================================================
        # Avellaneda-Stoikov Formulas
        # =====================================================================
        
        # Reservation price (adjusted for inventory risk)
        # r = s - q * γ * σ² * (T - t)
        inventory_adjustment = q * gamma * (sigma ** 2) * Decimal(str(T_minus_t))
        reservation_price = mid - inventory_adjustment
        
        # Optimal spread (simplified formula)
        # δ = (1/γ) * ln(1 + γ/k)
        gamma_float = float(gamma)
        k_float = float(k) if k > 0 else 1.0
        
        try:
            spread = Decimal(str((1 / gamma_float) * log(1 + gamma_float / k_float)))
        except (ValueError, ZeroDivisionError):
            spread = Decimal("0.001")  # Default 0.1%
        
        # Convert to absolute spread
        half_spread = spread / 2 * mid
        
        # Apply min/max constraints
        min_spread = mid * self.mm_config.min_spread_bps / Decimal("10000")
        max_spread = mid * self.mm_config.max_spread_bps / Decimal("10000")
        half_spread = max(min_spread / 2, min(max_spread / 2, half_spread))
        
        # =====================================================================
        # Inventory Skew
        # =====================================================================
        
        # Skew quotes based on inventory
        # Positive inventory: widen ask, tighten bid (encourage selling)
        # Negative inventory: widen bid, tighten ask (encourage buying)
        inventory_ratio = q / self.mm_config.max_inventory if self.mm_config.max_inventory != 0 else Decimal("0")
        skew = inventory_ratio * self.mm_config.inventory_skew_factor * half_spread
        
        bid_price = reservation_price - half_spread - skew
        ask_price = reservation_price + half_spread - skew
        
        # =====================================================================
        # Panic Check
        # =====================================================================
        
        # If inventory exceeds panic threshold, aggressive flatten
        if abs(inventory_ratio) > self.mm_config.panic_threshold:
            log.warning(
                "inventory_panic",
                symbol=symbol,
                inventory=str(q),
                ratio=str(inventory_ratio),
            )
            
            if q > 0:
                # Need to sell urgently
                return [
                    SignalEvent(
                        signal_type=SignalType.EXIT_LONG,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="sell",
                        order_type=OrderType.MARKET,
                        quantity=abs(q) * Decimal("0.5"),  # Sell half
                        urgency="critical",
                        reason="inventory_panic",
                    )
                ]
            else:
                # Need to buy urgently
                return [
                    SignalEvent(
                        signal_type=SignalType.EXIT_SHORT,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="buy",
                        order_type=OrderType.MARKET,
                        quantity=abs(q) * Decimal("0.5"),
                        urgency="critical",
                        reason="inventory_panic",
                    )
                ]
        
        # =====================================================================
        # Emit Quote Signals
        # =====================================================================
        
        signals = [
            # Bid
            SignalEvent(
                signal_type=SignalType.ENTRY_LONG,
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="buy",
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=self.mm_config.order_size,
                time_in_force=TimeInForce.GTC,
                urgency="low",
                reason="market_making",
                metadata={
                    "reservation_price": str(reservation_price),
                    "spread": str(half_spread * 2),
                    "inventory": str(q),
                },
            ),
            # Ask
            SignalEvent(
                signal_type=SignalType.ENTRY_SHORT,
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="sell",
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=self.mm_config.order_size,
                time_in_force=TimeInForce.GTC,
                urgency="low",
                reason="market_making",
                metadata={
                    "reservation_price": str(reservation_price),
                    "spread": str(half_spread * 2),
                    "inventory": str(q),
                },
            ),
        ]
        
        return signals
    
    async def on_fill_event(self, fill: FillEvent) -> None:
        """Update inventory on fills."""
        symbol = fill.symbol
        if symbol not in self.symbols:
            return
        
        # Update inventory
        if fill.side == "buy":
            self._inventory[symbol] = self._inventory.get(symbol, Decimal("0")) + fill.quantity
        else:
            self._inventory[symbol] = self._inventory.get(symbol, Decimal("0")) - fill.quantity
        
        log.info(
            "mm_fill",
            strategy=self.strategy_id,
            symbol=symbol,
            side=fill.side,
            quantity=str(fill.quantity),
            price=str(fill.price),
            new_inventory=str(self._inventory[symbol]),
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get market maker health status."""
        return {
            "strategy_id": self.strategy_id,
            "enabled": self.enabled,
            "inventory": {k: str(v) for k, v in self._inventory.items()},
            "volatility": {k: str(v.value) for k, v in self._volatility.items()},
            "mid_prices": {k: str(v) for k, v in self._mid_prices.items()},
            "session_elapsed": time.time() - self._session_start,
        }
    
    async def emergency_shutdown(self) -> List[SignalEvent]:
        """Close all positions immediately."""
        signals = []
        
        for symbol, inventory in self._inventory.items():
            if inventory == 0:
                continue
            
            if inventory > 0:
                signals.append(
                    SignalEvent(
                        signal_type=SignalType.EXIT_LONG,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="sell",
                        order_type=OrderType.MARKET,
                        quantity=abs(inventory),
                        urgency="critical",
                        reason="emergency_shutdown",
                    )
                )
            else:
                signals.append(
                    SignalEvent(
                        signal_type=SignalType.EXIT_SHORT,
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side="buy",
                        order_type=OrderType.MARKET,
                        quantity=abs(inventory),
                        urgency="critical",
                        reason="emergency_shutdown",
                    )
                )
        
        self.enabled = False
        log.warning("mm_emergency_shutdown", signals=len(signals))
        
        return signals
