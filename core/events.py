from dataclasses import dataclass, field
import time
from typing import Any, Optional

# Vi lägger till kw_only=True för att lösa arvs-problemet med default-värden
@dataclass(kw_only=True)
class Event:
    """Grundklass för alla händelser i systemet."""
    timestamp: float = field(default_factory=time.time)

@dataclass(kw_only=True)
class MarketEvent(Event):
    """Marknadsdata (Prisuppdateringar)."""
    exchange: str
    symbol: str
    price: float
    volume: float = 0.0
    side: str = ""  # 'BUY' or 'SELL' for aggTrade (for VPIN)
    event_type: str = "TICK"  # TICK, BOOK, TRADE

@dataclass(kw_only=True)
class RegimeEvent(Event):
    """Marknadsregim (Trend/Range) - Kritiskt för strategi-val."""
    symbol: str
    regime: str  # 'TRENDING', 'RANGING', 'VOLATILE'
    adx: float
    volatility: float

@dataclass(kw_only=True)
class SignalEvent(Event):
    """Köp/Sälj-signal från en strategi."""
    strategy_id: str
    symbol: str
    side: str  # 'BUY', 'SELL'
    quantity: float
    price: Optional[float] = None  # None = Market Order
    exchange: str = "binance"    # Target Exchange
    order_type: str = "LIMIT"    # 'LIMIT' or 'MARKET'
    arb_id: Optional[str] = None # Arbitrage tracking ID (for hedge manager)

@dataclass(kw_only=True)
class OrderEvent(Event):
    """En order som skickats till börsen (eller simulatorn)."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    price: Optional[float] = None

@dataclass(kw_only=True)
class FillEvent(Event):
    """Ett bekräftat avslut (Trade)."""
    timestamp: float = field(default_factory=time.time)
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float = 0.0
    commission_asset: str = "USDT"
    is_maker: bool = False # Taker by default
    pnl: float = 0.0  # Realiserad vinst/förlust (om denna trade stänger en position)

@dataclass(kw_only=True)
class FundingRateEvent(Event):
    """Funding Rate uppdatering från Futures."""
    symbol: str
    rate: float            # Aktuell funding rate (e.g. 0.0001 = 0.01%)
    mark_price: float      # Mark price vid tillfället
    next_funding_time: float # Timestamp för nästa funding betalning