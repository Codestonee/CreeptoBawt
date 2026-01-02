"""
Prometheus Metrics - Real-time system observability.

Metrics for:
- Order execution
- Market data latency
- System health
- Strategy performance
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

log = structlog.get_logger()

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    log.warning("prometheus_not_installed", message="Metrics will not be collected")


class TradingMetrics:
    """
    Prometheus metrics for the trading system.
    
    Exposes metrics for orders, fills, latency, inventory, PnL, etc.
    """
    
    def __init__(self, registry: Optional[Any] = None) -> None:
        if not HAS_PROMETHEUS:
            self._enabled = False
            return
        
        self._enabled = True
        self._registry = registry or CollectorRegistry()
        
        # =====================================================================
        # Order Metrics
        # =====================================================================
        self.orders_placed = Counter(
            "trading_orders_placed_total",
            "Total number of orders placed",
            ["exchange", "symbol", "side", "order_type"],
            registry=self._registry,
        )
        
        self.orders_filled = Counter(
            "trading_orders_filled_total",
            "Total number of orders filled",
            ["exchange", "symbol", "side"],
            registry=self._registry,
        )
        
        self.orders_canceled = Counter(
            "trading_orders_canceled_total",
            "Total number of orders canceled",
            ["exchange", "symbol", "reason"],
            registry=self._registry,
        )
        
        self.orders_rejected = Counter(
            "trading_orders_rejected_total",
            "Total number of orders rejected",
            ["exchange", "symbol", "reason"],
            registry=self._registry,
        )
        
        self.order_latency = Histogram(
            "trading_order_latency_seconds",
            "Order submission latency in seconds",
            ["exchange"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self._registry,
        )
        
        # =====================================================================
        # Market Data Metrics
        # =====================================================================
        self.messages_received = Counter(
            "trading_messages_received_total",
            "Total WebSocket messages received",
            ["exchange", "message_type"],
            registry=self._registry,
        )
        
        self.message_latency = Histogram(
            "trading_message_latency_milliseconds",
            "Market data latency in milliseconds (exchange to local)",
            ["exchange"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self._registry,
        )
        
        self.websocket_reconnects = Counter(
            "trading_websocket_reconnects_total",
            "WebSocket reconnection count",
            ["exchange"],
            registry=self._registry,
        )
        
        self.websocket_connected = Gauge(
            "trading_websocket_connected",
            "WebSocket connection status (1=connected, 0=disconnected)",
            ["exchange"],
            registry=self._registry,
        )
        
        # =====================================================================
        # Inventory & Position Metrics
        # =====================================================================
        self.inventory = Gauge(
            "trading_inventory",
            "Current inventory position",
            ["symbol"],
            registry=self._registry,
        )
        
        self.inventory_value_usd = Gauge(
            "trading_inventory_value_usd",
            "Current inventory value in USD",
            ["symbol"],
            registry=self._registry,
        )
        
        # =====================================================================
        # PnL Metrics
        # =====================================================================
        self.realized_pnl = Gauge(
            "trading_realized_pnl_usd",
            "Realized PnL in USD",
            ["strategy"],
            registry=self._registry,
        )
        
        self.unrealized_pnl = Gauge(
            "trading_unrealized_pnl_usd",
            "Unrealized PnL in USD",
            ["strategy"],
            registry=self._registry,
        )
        
        self.fees_paid = Counter(
            "trading_fees_paid_usd_total",
            "Total fees paid in USD",
            ["exchange"],
            registry=self._registry,
        )
        
        # =====================================================================
        # Risk Metrics
        # =====================================================================
        self.kill_switch_active = Gauge(
            "trading_kill_switch_active",
            "Kill switch status (1=active, 0=inactive)",
            registry=self._registry,
        )
        
        self.daily_loss = Gauge(
            "trading_daily_loss_usd",
            "Current daily loss in USD",
            registry=self._registry,
        )
        
        self.max_drawdown = Gauge(
            "trading_max_drawdown_percent",
            "Maximum drawdown percentage",
            registry=self._registry,
        )
        
        # =====================================================================
        # System Metrics
        # =====================================================================
        self.strategy_active = Gauge(
            "trading_strategy_active",
            "Strategy active status (1=active, 0=inactive)",
            ["strategy"],
            registry=self._registry,
        )
        
        self.open_orders = Gauge(
            "trading_open_orders",
            "Number of open orders",
            ["exchange"],
            registry=self._registry,
        )
        
        self.system_info = Info(
            "trading_system",
            "Trading system information",
            registry=self._registry,
        )
    
    @property
    def enabled(self) -> bool:
        """Check if metrics are enabled."""
        return self._enabled
    
    def set_system_info(self, info: Dict[str, str]) -> None:
        """Set system info labels."""
        if self._enabled:
            self.system_info.info(info)
    
    def record_order_placed(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
    ) -> None:
        """Record an order placement."""
        if self._enabled:
            self.orders_placed.labels(
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
            ).inc()
    
    def record_order_filled(self, exchange: str, symbol: str, side: str) -> None:
        """Record an order fill."""
        if self._enabled:
            self.orders_filled.labels(
                exchange=exchange,
                symbol=symbol,
                side=side,
            ).inc()
    
    def record_order_latency(self, exchange: str, latency_seconds: float) -> None:
        """Record order submission latency."""
        if self._enabled:
            self.order_latency.labels(exchange=exchange).observe(latency_seconds)
    
    def record_message(self, exchange: str, message_type: str) -> None:
        """Record a received message."""
        if self._enabled:
            self.messages_received.labels(
                exchange=exchange,
                message_type=message_type,
            ).inc()
    
    def record_message_latency(self, exchange: str, latency_ms: float) -> None:
        """Record message latency in milliseconds."""
        if self._enabled:
            self.message_latency.labels(exchange=exchange).observe(latency_ms)
    
    def set_websocket_status(self, exchange: str, connected: bool) -> None:
        """Set WebSocket connection status."""
        if self._enabled:
            self.websocket_connected.labels(exchange=exchange).set(1 if connected else 0)
    
    def record_reconnect(self, exchange: str) -> None:
        """Record a WebSocket reconnection."""
        if self._enabled:
            self.websocket_reconnects.labels(exchange=exchange).inc()
    
    def set_inventory(self, symbol: str, quantity: float) -> None:
        """Set current inventory."""
        if self._enabled:
            self.inventory.labels(symbol=symbol).set(quantity)
    
    def set_pnl(self, strategy: str, realized: float, unrealized: float) -> None:
        """Set PnL values."""
        if self._enabled:
            self.realized_pnl.labels(strategy=strategy).set(realized)
            self.unrealized_pnl.labels(strategy=strategy).set(unrealized)
    
    def set_kill_switch(self, active: bool) -> None:
        """Set kill switch status."""
        if self._enabled:
            self.kill_switch_active.set(1 if active else 0)
    
    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        if self._enabled:
            return generate_latest(self._registry)
        return b""
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        if HAS_PROMETHEUS:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# Global metrics instance
_metrics: Optional[TradingMetrics] = None


def get_metrics() -> TradingMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics
