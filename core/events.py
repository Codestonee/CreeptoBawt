"""
Event Dataclasses - Type-safe events for the trading system.

All monetary values use Decimal for precision. Timestamps are in microseconds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class EventType(str, Enum):
    """All event types in the system."""
    # Market events
    TRADE = "trade"
    BOOK_UPDATE = "book_update"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"
    
    # Order events
    ORDER_CREATED = "order_created"
    ORDER_PENDING = "order_pending"
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"
    ORDER_FAILED = "order_failed"
    
    # Execution events
    FILL = "fill"
    
    # Strategy events
    SIGNAL = "signal"
    
    # System events
    HEALTH = "health"
    MODE_CHANGE = "mode_change"
    KILL_SWITCH = "kill_switch"


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Single price level in the order book."""
    price: Decimal
    quantity: Decimal
    
    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")


@dataclass(slots=True)
class OrderBook:
    """Order book snapshot with bids and asks."""
    bids: List[OrderBookLevel]  # Descending by price
    asks: List[OrderBookLevel]  # Ascending by price
    timestamp: int  # Microseconds
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * Decimal("10000")
        return None


@dataclass(slots=True)
class MarketEvent:
    """
    Normalized market event from any exchange.
    
    All field values are normalized across exchanges:
    - symbol: "BTC-USDT" format (not BTCUSDT or BTC/USDT)
    - timestamps: microseconds since epoch
    - prices/quantities: Decimal for precision
    """
    event_type: Literal["trade", "book_update", "funding", "liquidation"]
    exchange: str
    symbol: str  # Normalized format: "BTC-USDT"
    timestamp_exchange: int  # Microseconds from exchange
    timestamp_received: int  # Local timestamp in microseconds
    price: Decimal
    quantity: Decimal
    side: Literal["buy", "sell"]
    book_snapshot: Optional[OrderBook] = None
    trade_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def latency_us(self) -> int:
        """Latency from exchange to local in microseconds."""
        return self.timestamp_received - self.timestamp_exchange
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transmission."""
        return {
            "event_type": self.event_type,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp_exchange": self.timestamp_exchange,
            "timestamp_received": self.timestamp_received,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "side": self.side,
            "trade_id": self.trade_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketEvent:
        """Deserialize from transmission."""
        return cls(
            event_type=data["event_type"],
            exchange=data["exchange"],
            symbol=data["symbol"],
            timestamp_exchange=data["timestamp_exchange"],
            timestamp_received=data["timestamp_received"],
            price=Decimal(data["price"]),
            quantity=Decimal(data["quantity"]),
            side=data["side"],
            trade_id=data.get("trade_id"),
            metadata=data.get("metadata", {}),
        )


class OrderState(str, Enum):
    """Order lifecycle states."""
    CREATED = "created"
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELING = "canceling"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


class TimeInForce(str, Enum):
    """Time in force options."""
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date


@dataclass(slots=True)
class OrderEvent:
    """Order state change event."""
    event_type: EventType
    client_order_id: str
    exchange_order_id: Optional[str]
    exchange: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: OrderType
    state: OrderState
    price: Optional[Decimal]  # None for market orders
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    reason: str = ""
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }


@dataclass(slots=True)
class FillEvent:
    """Individual fill/execution event."""
    event_type: EventType = EventType.FILL
    fill_id: str = ""
    client_order_id: str = ""
    exchange_order_id: str = ""
    exchange: str = ""
    symbol: str = ""
    side: Literal["buy", "sell"] = "buy"
    price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    fee_currency: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    is_maker: bool = False
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def value(self) -> Decimal:
        """Total value of the fill (price * quantity)."""
        return self.price * self.quantity
    
    @property
    def net_value(self) -> Decimal:
        """Value after fees."""
        if self.side == "buy":
            return self.value + self.fee
        return self.value - self.fee


class SignalType(str, Enum):
    """Trading signal types."""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    CANCEL = "cancel"
    CANCEL_ALL = "cancel_all"


@dataclass(slots=True)
class SignalEvent:
    """Strategy trading signal."""
    event_type: EventType = EventType.SIGNAL
    signal_type: SignalType = SignalType.ENTRY_LONG
    strategy_id: str = ""
    symbol: str = ""
    side: Literal["buy", "sell"] = "buy"
    order_type: OrderType = OrderType.LIMIT
    price: Optional[Decimal] = None
    quantity: Decimal = Decimal("0")
    urgency: Literal["low", "normal", "high", "critical"] = "normal"
    time_in_force: TimeInForce = TimeInForce.GTC
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class HealthEvent:
    """System health update."""
    event_type: EventType = EventType.HEALTH
    component: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModeChangeEvent:
    """Trading mode change event."""
    event_type: EventType = EventType.MODE_CHANGE
    old_mode: str = ""
    new_mode: str = ""
    reason: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))


@dataclass(slots=True)
class KillSwitchEvent:
    """Kill switch activation event."""
    event_type: EventType = EventType.KILL_SWITCH
    activated: bool = True
    reason: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    triggered_by: str = ""  # manual, risk_limit, error, etc.
