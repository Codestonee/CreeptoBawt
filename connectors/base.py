"""
Resilient WebSocket Connector Base Class.

Provides exponential backoff, connection timeouts, max retry limits,
and heartbeat monitoring for all exchange connectors.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp

logger = logging.getLogger("Connector.Base")


class ResilientConnector(ABC):
    """
    Base class for resilient WebSocket connections.
    
    Features:
    - Exponential backoff on reconnection
    - Connection timeout handling
    - Maximum retry limit with circuit breaker
    - Heartbeat monitoring for stale connections
    """
    
    def __init__(
        self,
        url: str,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_multiplier: float = 2.0,
        max_consecutive_failures: int = 10,
        connection_timeout: float = 30.0,
        heartbeat_interval: float = 30.0,
        stale_threshold: float = 60.0,
    ):
        """
        Initialize the ResilientConnector.

        Args:
            url: WebSocket URL to connect to.
            initial_backoff: Initial delay (seconds) before first retry.
            max_backoff: Maximum delay (seconds) for exponential backoff.
            backoff_multiplier: Multiplier for backoff duration on failure.
            max_consecutive_failures: Max failures before triggering circuit breaker.
            connection_timeout: Timeout (seconds) for establishing connection.
            heartbeat_interval: Interval (seconds) to send/expect pings.
            stale_threshold: Time (seconds) without messages before forcing reconnect.
        """
        self.url = url
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.max_consecutive_failures = max_consecutive_failures
        self.connection_timeout = connection_timeout
        self.heartbeat_interval = heartbeat_interval
        self.stale_threshold = stale_threshold
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.running = False
        
        self._current_backoff = initial_backoff
        self._consecutive_failures = 0
        self._last_message_time = 0.0
        self._circuit_breaker_open = False
        
    async def connect(self):
        """Main connection loop with resilient reconnection."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        while self.running:
            # Circuit breaker check
            if self._circuit_breaker_open:
                logger.critical(
                    f"Circuit breaker OPEN after {self._consecutive_failures} failures. "
                    f"Waiting {self.max_backoff}s before retry..."
                )
                await asyncio.sleep(self.max_backoff)
                self._circuit_breaker_open = False
                self._consecutive_failures = 0
            
            try:
                logger.debug(f"Connecting to {self.url}...")
                
                async with asyncio.timeout(self.connection_timeout):
                    async with self.session.ws_connect(
                        self.url,
                        heartbeat=self.heartbeat_interval
                    ) as ws:
                        self.ws = ws
                        self._last_message_time = time.time()
                        
                        await self._on_connected()
                        
                        # Reset backoff on successful connection
                        self._current_backoff = self.initial_backoff
                        self._consecutive_failures = 0
                        
                        logger.debug("Connected and subscribed.")
                        
                        # Start heartbeat monitor
                        heartbeat_task = asyncio.create_task(
                            self._heartbeat_monitor()
                        )
                        
                        try:
                            await self._listen()
                        finally:
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass
                        
            except asyncio.TimeoutError:
                self._handle_failure("Connection timeout")
            except aiohttp.ClientError as e:
                self._handle_failure(f"Network error: {e}")
            except Exception as e:
                self._handle_failure(f"Unexpected error: {e}")
            
            # Wait before reconnecting
            if self.running and not self._circuit_breaker_open:
                logger.info(f"Reconnecting in {self._current_backoff:.1f}s...")
                await asyncio.sleep(self._current_backoff)
                
                # Exponential backoff
                self._current_backoff = min(
                    self._current_backoff * self.backoff_multiplier,
                    self.max_backoff
                )
    
    def _handle_failure(self, reason: str):
        """Handle connection failure with circuit breaker logic."""
        self._consecutive_failures += 1
        logger.error(
            f"{reason}. Failure {self._consecutive_failures}/{self.max_consecutive_failures}"
        )
        
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._circuit_breaker_open = True
            logger.critical(
                f"Too many consecutive failures ({self._consecutive_failures}). "
                f"Triggering circuit breaker."
            )
    
    async def _heartbeat_monitor(self):
        """Monitor for stale connections and trigger reconnect."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)
            
            time_since_last = time.time() - self._last_message_time
            if time_since_last > self.stale_threshold:
                logger.warning(
                    f"Connection stale ({time_since_last:.0f}s since last message). "
                    f"Forcing reconnect..."
                )
                if self.ws:
                    await self.ws.close()
                break
    
    async def _listen(self):
        """Listen for incoming messages."""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._last_message_time = time.time()
                await self._on_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket error received")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.info("WebSocket closed by server")
                break
    
    @abstractmethod
    async def _on_connected(self):
        """Called after successful connection. Override to subscribe to streams."""
        pass
    
    @abstractmethod
    async def _on_message(self, data: str):
        """Called for each message received. Override to process messages."""
        pass
    
    async def close(self):
        """Gracefully close the connection."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        logger.info(f"{self.__class__.__name__} closed.")
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.ws is not None and not self.ws.closed
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy (connected and receiving data)."""
        if not self.is_connected:
            return False
        return (time.time() - self._last_message_time) < self.stale_threshold

