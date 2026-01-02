"""
Event Bus - Redis-backed pub/sub for decoupled components.

Provides async message publishing and topic-based subscriptions.
Uses msgpack for fast binary serialization.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import structlog

log = structlog.get_logger()


@dataclass
class Message:
    """Event bus message wrapper."""
    topic: str
    payload: Dict[str, Any]
    timestamp: int  # Microseconds
    message_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


# Type alias for async message handlers
MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


class EventBus(ABC):
    """Abstract event bus interface."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the message broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message broker."""
        pass
    
    @abstractmethod
    async def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a topic with a message handler."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str, handler: Optional[MessageHandler] = None) -> None:
        """Unsubscribe from a topic."""
        pass


class InMemoryEventBus(EventBus):
    """
    In-memory event bus for development and testing.
    
    Provides the same interface as Redis-backed bus but runs entirely in memory.
    Useful for single-process deployments and testing.
    """
    
    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[MessageHandler]] = defaultdict(set)
        self._message_counter = 0
        self._connected = False
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Start the message processor."""
        if self._connected:
            return
        
        self._connected = True
        self._processor_task = asyncio.create_task(self._process_messages())
        log.info("event_bus_connected", bus_type="in_memory")
    
    async def disconnect(self) -> None:
        """Stop the message processor."""
        if not self._connected:
            return
        
        self._connected = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        log.info("event_bus_disconnected", bus_type="in_memory")
    
    async def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish a message to a topic."""
        if not self._connected:
            log.warning("publish_while_disconnected", topic=topic)
            return
        
        async with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self._message_counter}"
        
        message = Message(
            topic=topic,
            payload=payload,
            timestamp=int(time.time() * 1_000_000),
            message_id=message_id,
        )
        
        await self._queue.put(message)
        
        log.debug(
            "message_published",
            topic=topic,
            message_id=message_id,
        )
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a topic."""
        async with self._lock:
            self._subscribers[topic].add(handler)
        
        log.info(
            "subscription_added",
            topic=topic,
            handler=handler.__name__ if hasattr(handler, '__name__') else str(handler),
        )
    
    async def unsubscribe(self, topic: str, handler: Optional[MessageHandler] = None) -> None:
        """Unsubscribe from a topic."""
        async with self._lock:
            if handler:
                self._subscribers[topic].discard(handler)
            else:
                self._subscribers[topic].clear()
        
        log.info("subscription_removed", topic=topic)
    
    async def _process_messages(self) -> None:
        """Background task to process messages."""
        while self._connected:
            try:
                message = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                
                # Get handlers for this topic
                async with self._lock:
                    handlers = list(self._subscribers.get(message.topic, set()))
                    # Also notify wildcard subscribers
                    handlers.extend(self._subscribers.get("*", set()))
                
                # Dispatch to all handlers
                if handlers:
                    await asyncio.gather(
                        *[self._safe_call(h, message) for h in handlers],
                        return_exceptions=True,
                    )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("message_processing_error", error=str(e))
    
    async def _safe_call(self, handler: MessageHandler, message: Message) -> None:
        """Safely call a handler with error logging."""
        try:
            await handler(message)
        except Exception as e:
            log.error(
                "handler_error",
                handler=str(handler),
                topic=message.topic,
                error=str(e),
            )


class RedisEventBus(EventBus):
    """
    Redis-backed event bus for production.
    
    Uses Redis pub/sub for distributed message passing.
    Supports horizontal scaling of components.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "trading:",
    ) -> None:
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._redis = None
        self._pubsub = None
        self._subscribers: Dict[str, Set[MessageHandler]] = defaultdict(set)
        self._listener_task: Optional[asyncio.Task] = None
        self._connected = False
        self._message_counter = 0
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            import redis.asyncio as redis
            import msgpack
            
            self._redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=False,  # We use msgpack
            )
            self._pubsub = self._redis.pubsub()
            self._connected = True
            self._msgpack = msgpack
            
            # Start listener task
            self._listener_task = asyncio.create_task(self._listen())
            
            log.info("event_bus_connected", bus_type="redis", url=self._redis_url)
            
        except ImportError:
            log.warning("redis_not_installed", falling_back="in_memory")
            raise
        except Exception as e:
            log.error("redis_connection_failed", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if not self._connected:
            return
        
        self._connected = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
        
        log.info("event_bus_disconnected", bus_type="redis")
    
    async def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish a message to Redis."""
        if not self._connected or not self._redis:
            log.warning("publish_while_disconnected", topic=topic)
            return
        
        self._message_counter += 1
        message = Message(
            topic=topic,
            payload=payload,
            timestamp=int(time.time() * 1_000_000),
            message_id=f"msg_{self._message_counter}",
        )
        
        channel = f"{self._channel_prefix}{topic}"
        data = self._msgpack.packb(message.to_dict())
        
        await self._redis.publish(channel, data)
        
        log.debug("message_published", topic=topic, channel=channel)
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe to a Redis channel."""
        self._subscribers[topic].add(handler)
        
        channel = f"{self._channel_prefix}{topic}"
        if self._pubsub:
            await self._pubsub.subscribe(channel)
        
        log.info("subscription_added", topic=topic, channel=channel)
    
    async def unsubscribe(self, topic: str, handler: Optional[MessageHandler] = None) -> None:
        """Unsubscribe from a Redis channel."""
        if handler:
            self._subscribers[topic].discard(handler)
        else:
            self._subscribers[topic].clear()
        
        # Only unsubscribe from Redis if no handlers left
        if not self._subscribers[topic]:
            channel = f"{self._channel_prefix}{topic}"
            if self._pubsub:
                await self._pubsub.unsubscribe(channel)
        
        log.info("subscription_removed", topic=topic)
    
    async def _listen(self) -> None:
        """Background task to listen for Redis messages."""
        while self._connected and self._pubsub:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")
                    
                    # Extract topic from channel
                    topic = channel.replace(self._channel_prefix, "")
                    
                    # Deserialize payload
                    data = self._msgpack.unpackb(message["data"])
                    msg = Message(
                        topic=data["topic"],
                        payload=data["payload"],
                        timestamp=data["timestamp"],
                        message_id=data["message_id"],
                    )
                    
                    # Dispatch to handlers
                    handlers = list(self._subscribers.get(topic, set()))
                    handlers.extend(self._subscribers.get("*", set()))
                    
                    if handlers:
                        await asyncio.gather(
                            *[self._safe_call(h, msg) for h in handlers],
                            return_exceptions=True,
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("redis_listener_error", error=str(e))
                await asyncio.sleep(1.0)  # Brief pause on error
    
    async def _safe_call(self, handler: MessageHandler, message: Message) -> None:
        """Safely call a handler."""
        try:
            await handler(message)
        except Exception as e:
            log.error(
                "handler_error",
                handler=str(handler),
                topic=message.topic,
                error=str(e),
            )


def create_event_bus(use_redis: bool = False, redis_url: str = "redis://localhost:6379") -> EventBus:
    """
    Factory function to create an event bus.
    
    Args:
        use_redis: If True, use Redis-backed bus. Otherwise, use in-memory.
        redis_url: Redis connection URL.
    
    Returns:
        EventBus instance
    """
    if use_redis:
        return RedisEventBus(redis_url=redis_url)
    return InMemoryEventBus()
