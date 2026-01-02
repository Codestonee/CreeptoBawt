"""
Base Connector - Abstract interface for all exchange connectors.

All exchange connectors must implement this interface for:
- WebSocket connection management
- Market data normalization
- Resilient reconnection
- Health status reporting
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from core.events import MarketEvent, OrderBook

log = structlog.get_logger()


class ConnectorState(str, Enum):
    """Connector lifecycle states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ConnectorStats:
    """Connector statistics."""
    messages_received: int = 0
    messages_processed: int = 0
    errors: int = 0
    reconnects: int = 0
    last_message_time: float = 0.0
    connected_since: Optional[float] = None
    
    @property
    def uptime(self) -> float:
        """Return uptime in seconds."""
        if self.connected_since:
            return time.time() - self.connected_since
        return 0.0
    
    @property
    def time_since_last_message(self) -> float:
        """Time since last message in seconds."""
        if self.last_message_time:
            return time.time() - self.last_message_time
        return float("inf")


@dataclass
class ConnectionConfig:
    """Connection configuration."""
    # Reconnection settings
    max_reconnect_attempts: int = 10
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 30000
    backoff_multiplier: float = 2.0
    
    # Stale data detection
    stale_threshold_seconds: float = 5.0
    
    # Rate limiting
    rate_limit_backoff_seconds: float = 60.0
    
    # Heartbeat
    ping_interval_seconds: float = 30.0
    pong_timeout_seconds: float = 10.0


# Type alias for market event handlers
MarketEventHandler = Callable[[MarketEvent], None]


class BaseConnector(ABC):
    """
    Abstract base class for exchange connectors.
    
    Provides:
    - Connection lifecycle management
    - Exponential backoff reconnection
    - Stale data detection
    - Event handler registration
    - Symbol normalization
    """
    
    # Exchange-specific symbol mapping (override in subclass)
    SYMBOL_MAP: Dict[str, str] = {}
    
    def __init__(
        self,
        exchange_name: str,
        config: Optional[ConnectionConfig] = None,
    ) -> None:
        self.exchange_name = exchange_name
        self.config = config or ConnectionConfig()
        
        self._state = ConnectorState.DISCONNECTED
        self._subscribed_symbols: Set[str] = set()
        self._handlers: List[MarketEventHandler] = []
        self._stats = ConnectorStats()
        self._reconnect_attempt = 0
        self._ws = None
        self._running = False
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stale_check_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> ConnectorState:
        """Get current connection state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._state == ConnectorState.CONNECTED
    
    @property
    def stats(self) -> ConnectorStats:
        """Get connector statistics."""
        return self._stats
    
    # =========================================================================
    # Abstract Methods (Implement in Subclass)
    # =========================================================================
    
    @abstractmethod
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        pass
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Close WebSocket connection."""
        pass
    
    @abstractmethod
    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols."""
        pass
    
    @abstractmethod
    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for symbols."""
        pass
    
    @abstractmethod
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive and parse a WebSocket message."""
        pass
    
    @abstractmethod
    def _parse_message(self, message: Dict[str, Any]) -> Optional[MarketEvent]:
        """Parse raw message into MarketEvent."""
        pass
    
    @abstractmethod
    async def _send_ping(self) -> None:
        """Send ping/heartbeat to keep connection alive."""
        pass
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def add_handler(self, handler: MarketEventHandler) -> None:
        """Register a market event handler."""
        if handler not in self._handlers:
            self._handlers.append(handler)
            log.info("handler_added", exchange=self.exchange_name)
    
    def remove_handler(self, handler: MarketEventHandler) -> None:
        """Remove a market event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    async def start(self) -> None:
        """Start the connector."""
        if self._running:
            return
        
        self._running = True
        log.info("connector_starting", exchange=self.exchange_name)
        
        await self._connect_with_retry()
        
        # Start background tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._stale_check_task = asyncio.create_task(self._stale_check_loop())
        
        log.info("connector_started", exchange=self.exchange_name)
    
    async def stop(self) -> None:
        """Stop the connector."""
        if not self._running:
            return
        
        self._running = False
        self._state = ConnectorState.STOPPED
        
        # Cancel tasks
        for task in [self._receive_task, self._heartbeat_task, self._stale_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        await self._disconnect()
        
        log.info("connector_stopped", exchange=self.exchange_name, stats=self._stats)
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        # Normalize symbols
        normalized = [self.normalize_symbol(s) for s in symbols]
        new_symbols = [s for s in normalized if s not in self._subscribed_symbols]
        
        if not new_symbols:
            return
        
        if self.is_connected:
            await self._subscribe_symbols(new_symbols)
        
        self._subscribed_symbols.update(new_symbols)
        log.info(
            "subscribed",
            exchange=self.exchange_name,
            symbols=new_symbols,
        )
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        normalized = [self.normalize_symbol(s) for s in symbols]
        to_remove = [s for s in normalized if s in self._subscribed_symbols]
        
        if not to_remove:
            return
        
        if self.is_connected:
            await self._unsubscribe_symbols(to_remove)
        
        self._subscribed_symbols.difference_update(to_remove)
        log.info(
            "unsubscribed",
            exchange=self.exchange_name,
            symbols=to_remove,
        )
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to standard format: BASE-QUOTE (e.g., BTC-USDT).
        
        Override in subclass for exchange-specific normalization.
        """
        # Check mapping first
        if symbol in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol]
        
        # Already normalized
        if "-" in symbol:
            return symbol.upper()
        
        # Try common patterns
        symbol = symbol.upper()
        
        # BTCUSDT -> BTC-USDT
        for quote in ["USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP"]:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}-{quote}"
        
        # BTC/USDT -> BTC-USDT
        if "/" in symbol:
            return symbol.replace("/", "-")
        
        return symbol
    
    def denormalize_symbol(self, symbol: str) -> str:
        """
        Convert normalized symbol back to exchange format.
        
        Override in subclass for exchange-specific format.
        """
        # Reverse lookup
        for exchange_sym, normalized_sym in self.SYMBOL_MAP.items():
            if normalized_sym == symbol:
                return exchange_sym
        
        # Default: remove dash
        return symbol.replace("-", "")
    
    def get_health(self) -> Dict[str, Any]:
        """Get connector health status."""
        is_stale = self._stats.time_since_last_message > self.config.stale_threshold_seconds
        
        return {
            "exchange": self.exchange_name,
            "state": self._state.value,
            "connected": self.is_connected,
            "stale": is_stale,
            "uptime": self._stats.uptime,
            "messages_received": self._stats.messages_received,
            "errors": self._stats.errors,
            "reconnects": self._stats.reconnects,
            "subscribed_symbols": list(self._subscribed_symbols),
            "time_since_last_message": self._stats.time_since_last_message,
        }
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    async def _connect_with_retry(self) -> None:
        """Connect with exponential backoff retry."""
        while self._running:
            try:
                self._state = ConnectorState.CONNECTING
                await self._connect()
                
                self._state = ConnectorState.CONNECTED
                self._stats.connected_since = time.time()
                self._reconnect_attempt = 0
                
                log.info("connected", exchange=self.exchange_name)
                
                # Resubscribe to symbols
                if self._subscribed_symbols:
                    await self._subscribe_symbols(list(self._subscribed_symbols))
                
                return
                
            except Exception as e:
                self._stats.errors += 1
                self._reconnect_attempt += 1
                
                if self._reconnect_attempt > self.config.max_reconnect_attempts:
                    self._state = ConnectorState.ERROR
                    log.error(
                        "max_reconnect_attempts_exceeded",
                        exchange=self.exchange_name,
                        error=str(e),
                    )
                    raise
                
                # Calculate backoff
                backoff_ms = min(
                    self.config.initial_backoff_ms * (
                        self.config.backoff_multiplier ** self._reconnect_attempt
                    ),
                    self.config.max_backoff_ms,
                )
                backoff_s = backoff_ms / 1000
                
                self._state = ConnectorState.RECONNECTING
                self._stats.reconnects += 1
                
                log.warning(
                    "reconnecting",
                    exchange=self.exchange_name,
                    attempt=self._reconnect_attempt,
                    backoff_seconds=backoff_s,
                    error=str(e),
                )
                
                await asyncio.sleep(backoff_s)
    
    async def _receive_loop(self) -> None:
        """Main message receive loop."""
        while self._running:
            try:
                if not self.is_connected:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self._receive_message()
                
                if message is None:
                    continue
                
                self._stats.messages_received += 1
                self._stats.last_message_time = time.time()
                
                # Parse message
                event = self._parse_message(message)
                
                if event:
                    self._stats.messages_processed += 1
                    
                    # Dispatch to handlers
                    for handler in self._handlers:
                        try:
                            handler(event)
                        except Exception as e:
                            log.error(
                                "handler_error",
                                exchange=self.exchange_name,
                                error=str(e),
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats.errors += 1
                log.error(
                    "receive_error",
                    exchange=self.exchange_name,
                    error=str(e),
                )
                
                # Trigger reconnection
                if self._running:
                    await self._connect_with_retry()
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self._running:
            try:
                await asyncio.sleep(self.config.ping_interval_seconds)
                
                if self.is_connected:
                    await self._send_ping()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "heartbeat_error",
                    exchange=self.exchange_name,
                    error=str(e),
                )
    
    async def _stale_check_loop(self) -> None:
        """Check for stale data (no updates)."""
        while self._running:
            try:
                await asyncio.sleep(self.config.stale_threshold_seconds)
                
                if self.is_connected and self._subscribed_symbols:
                    if self._stats.time_since_last_message > self.config.stale_threshold_seconds:
                        log.warning(
                            "stale_data_detected",
                            exchange=self.exchange_name,
                            seconds_since_update=self._stats.time_since_last_message,
                        )
                        
                        # Trigger reconnection
                        await self._disconnect()
                        await self._connect_with_retry()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(
                    "stale_check_error",
                    exchange=self.exchange_name,
                    error=str(e),
                )
