"""
Redis State Manager for Crash Recovery.

Implements persistent state management using Redis as recommended in
Gemini's architectural plan. Ensures the bot survives crashes without
losing order tracking, position state, or PnL history.

Data Structures:
- Order Tracking: Redis Hash (order:{id}) for O(1) status lookup
- Active Queue: Sorted Set (orders:active) scored by timestamp
- PnL History: Redis Streams for equity snapshots
- Position State: Hash (positions) for current inventory

Requirements:
    pip install redis aioredis

Starting Redis (Docker):
    docker run -d -p 6379:6379 --name creeptbaws-redis redis:alpine
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger("State.Redis")

# Try to import redis, gracefully degrade if not available
try:
    import redis.asyncio as aioredis
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis not installed. State persistence disabled. Install with: pip install redis")


@dataclass
class OrderRecord:
    """Order record for Redis persistence."""
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str = "PENDING"
    exchange_order_id: Optional[str] = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = time.time()


@dataclass
class PositionRecord:
    """Position record for Redis persistence."""
    symbol: str
    quantity: float  # Signed: + long, - short
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    updated_at: float = 0.0


class RedisStateManager:
    """
    Persistent state management using Redis.
    
    Features:
    - Survives process crashes
    - O(1) order status lookup
    - Automatic timeout handling for stale orders
    - Equity curve persistence for risk analysis
    
    Usage:
        state = RedisStateManager()
        await state.connect()
        
        # Save order
        await state.save_order(order_record)
        
        # Get position
        pos = await state.get_position("btcusdt")
        
        # Log equity snapshot
        await state.log_equity(10500.0)
        
        # On restart: recover state
        orders = await state.get_active_orders()
    """
    
    # Key prefixes
    ORDER_PREFIX = "creeptbaws:order:"
    ACTIVE_ORDERS_KEY = "creeptbaws:orders:active"
    POSITIONS_KEY = "creeptbaws:positions"
    EQUITY_STREAM = "creeptbaws:equity"
    STATE_KEY = "creeptbaws:state"
    METRICS_KEY = "creeptbaws:metrics"
    
    # Timeouts
    ORDER_TIMEOUT_SECONDS = 300  # 5 minutes
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Initialize Redis state manager.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional password
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        
        # Fallback in-memory storage if Redis unavailable
        self._fallback_orders: Dict[str, OrderRecord] = {}
        self._fallback_positions: Dict[str, PositionRecord] = {}
        self._fallback_equity: List[tuple] = []
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._redis is not None
    
    async def connect(self) -> bool:
        """
        Connect to Redis server.
        
        Returns:
            True if connected, False if falling back to in-memory
        """
        if not HAS_REDIS:
            logger.warning("Redis library not installed, using in-memory fallback")
            return False
        
        try:
            self._redis = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, using in-memory fallback")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("Redis connection closed")
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def save_order(self, order: OrderRecord) -> None:
        """
        Save order to Redis Hash with automatic active queue tracking.
        
        Uses pipelining to batch all operations in a single round-trip.
        
        Args:
            order: OrderRecord to persist
        """
        order.updated_at = time.time()
        order_key = f"{self.ORDER_PREFIX}{order.client_order_id}"
        order_data = asdict(order)
        
        if self.is_connected:
            try:
                # Use pipeline for all operations in single round-trip
                async with self._redis.pipeline() as pipe:
                    # 1. Save to hash
                    pipe.hset(order_key, mapping={
                        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        for k, v in order_data.items()
                    })
                    
                    # 2. Update active set
                    if order.status not in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        pipe.zadd(
                            self.ACTIVE_ORDERS_KEY,
                            {order.client_order_id: order.created_at}
                        )
                    else:
                        # Remove from active if terminal
                        pipe.zrem(self.ACTIVE_ORDERS_KEY, order.client_order_id)
                    
                    # 3. Set TTL for old filled orders (cleanup after 24h)
                    if order.status in ['FILLED', 'CANCELED']:
                        pipe.expire(order_key, 86400)
                    
                    # Execute all commands atomically
                    await pipe.execute()
                
            except Exception as e:
                logger.error(f"Failed to save order to Redis: {e}")
                self._fallback_orders[order.client_order_id] = order
        else:
            self._fallback_orders[order.client_order_id] = order
    
    async def get_order(self, client_order_id: str) -> Optional[OrderRecord]:
        """
        Get order by client_order_id.
        
        Args:
            client_order_id: Order ID
            
        Returns:
            OrderRecord or None
        """
        if self.is_connected:
            try:
                order_key = f"{self.ORDER_PREFIX}{client_order_id}"
                data = await self._redis.hgetall(order_key)
                
                if not data:
                    return None
                
                return OrderRecord(
                    client_order_id=data.get('client_order_id', ''),
                    symbol=data.get('symbol', ''),
                    side=data.get('side', ''),
                    quantity=float(data.get('quantity', 0)),
                    price=float(data.get('price', 0)),
                    status=data.get('status', 'UNKNOWN'),
                    exchange_order_id=data.get('exchange_order_id'),
                    filled_qty=float(data.get('filled_qty', 0)),
                    avg_fill_price=float(data.get('avg_fill_price', 0)),
                    created_at=float(data.get('created_at', 0)),
                    updated_at=float(data.get('updated_at', 0))
                )
            except Exception as e:
                logger.error(f"Failed to get order from Redis: {e}")
        
        return self._fallback_orders.get(client_order_id)
    
    async def get_active_orders(self) -> List[OrderRecord]:
        """
        Get all active (non-terminal) orders using pipeline for performance.
        
        Returns:
            List of active OrderRecords
        """
        orders = []
        
        if self.is_connected:
            try:
                # Get all active order IDs from sorted set
                order_ids = await self._redis.zrange(self.ACTIVE_ORDERS_KEY, 0, -1)
                
                if not order_ids:
                    return []
                
                # Use pipeline to fetch all orders in one round-trip
                async with self._redis.pipeline() as pipe:
                    for order_id in order_ids:
                        order_key = f"{self.ORDER_PREFIX}{order_id}"
                        pipe.hgetall(order_key)
                    
                    results = await pipe.execute()
                
                # Parse results
                for data in results:
                    if not data:
                        continue
                        
                    try:
                        order = OrderRecord(
                            client_order_id=data.get('client_order_id', ''),
                            symbol=data.get('symbol', ''),
                            side=data.get('side', ''),
                            quantity=float(data.get('quantity', 0)),
                            price=float(data.get('price', 0)),
                            status=data.get('status', 'UNKNOWN'),
                            exchange_order_id=data.get('exchange_order_id'),
                            filled_qty=float(data.get('filled_qty', 0)),
                            avg_fill_price=float(data.get('avg_fill_price', 0)),
                            created_at=float(data.get('created_at', 0)),
                            updated_at=float(data.get('updated_at', 0))
                        )
                        orders.append(order)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse order data: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to get active orders from Redis: {e}")
        else:
            orders = [
                o for o in self._fallback_orders.values()
                if o.status not in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']
            ]
        
        return orders
    
    async def cleanup_stale_orders(self) -> int:
        """
        Remove orders that have been active too long (likely orphaned).
        
        Returns:
            Number of orders cleaned up
        """
        cutoff = time.time() - self.ORDER_TIMEOUT_SECONDS
        removed = 0
        
        if self.is_connected:
            try:
                # Get orders older than cutoff
                stale_ids = await self._redis.zrangebyscore(
                    self.ACTIVE_ORDERS_KEY, 
                    0, 
                    cutoff
                )
                
                for order_id in stale_ids:
                    order = await self.get_order(order_id)
                    if order:
                        order.status = "EXPIRED"
                        await self.save_order(order)
                        removed += 1
                        logger.warning(f"Cleaned up stale order: {order_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup stale orders: {e}")
        
        return removed
    
    # ==================== POSITION MANAGEMENT ====================
    
    async def save_position(self, position: PositionRecord) -> None:
        """
        Save position to Redis.
        
        Args:
            position: PositionRecord to persist
        """
        position.updated_at = time.time()
        
        if self.is_connected:
            try:
                await self._redis.hset(
                    self.POSITIONS_KEY,
                    position.symbol.lower(),
                    json.dumps(asdict(position))
                )
            except Exception as e:
                logger.error(f"Failed to save position to Redis: {e}")
                self._fallback_positions[position.symbol.lower()] = position
        else:
            self._fallback_positions[position.symbol.lower()] = position
    
    async def get_position(self, symbol: str) -> Optional[PositionRecord]:
        """
        Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            PositionRecord or None
        """
        symbol = symbol.lower()
        
        if self.is_connected:
            try:
                data = await self._redis.hget(self.POSITIONS_KEY, symbol)
                if data:
                    parsed = json.loads(data)
                    return PositionRecord(**parsed)
            except Exception as e:
                logger.error(f"Failed to get position from Redis: {e}")
        
        return self._fallback_positions.get(symbol)
    
    async def get_all_positions(self) -> Dict[str, PositionRecord]:
        """Get all positions."""
        positions = {}
        
        if self.is_connected:
            try:
                data = await self._redis.hgetall(self.POSITIONS_KEY)
                for symbol, pos_json in data.items():
                    parsed = json.loads(pos_json)
                    positions[symbol] = PositionRecord(**parsed)
            except Exception as e:
                logger.error(f"Failed to get positions from Redis: {e}")
        else:
            positions = dict(self._fallback_positions)
        
        return positions
    
    # ==================== EQUITY / PNL TRACKING ====================
    
    async def log_equity(self, equity: float, pnl: float = 0.0) -> None:
        """
        Log equity snapshot to Redis Stream.
        
        Args:
            equity: Current total equity
            pnl: Realized PnL since last snapshot
        """
        entry = {
            'timestamp': str(time.time()),
            'equity': str(equity),
            'pnl': str(pnl),
            'datetime': datetime.now().isoformat()
        }
        
        if self.is_connected:
            try:
                await self._redis.xadd(
                    self.EQUITY_STREAM,
                    entry,
                    maxlen=10000  # Keep last 10k entries
                )
            except Exception as e:
                logger.error(f"Failed to log equity: {e}")
                self._fallback_equity.append((time.time(), equity, pnl))
        else:
            self._fallback_equity.append((time.time(), equity, pnl))
    
    async def get_equity_history(self, count: int = 100) -> List[Dict]:
        """
        Get recent equity snapshots.
        
        Args:
            count: Number of entries to retrieve
            
        Returns:
            List of equity entries
        """
        entries = []
        
        if self.is_connected:
            try:
                # Get last N entries
                raw = await self._redis.xrevrange(
                    self.EQUITY_STREAM,
                    count=count
                )
                
                for entry_id, data in raw:
                    entries.append({
                        'id': entry_id,
                        'timestamp': float(data.get('timestamp', 0)),
                        'equity': float(data.get('equity', 0)),
                        'pnl': float(data.get('pnl', 0)),
                        'datetime': data.get('datetime', '')
                    })
                
            except Exception as e:
                logger.error(f"Failed to get equity history: {e}")
        else:
            entries = [
                {'timestamp': t, 'equity': e, 'pnl': p}
                for t, e, p in self._fallback_equity[-count:]
            ]
        
        return list(reversed(entries))
    
    # ==================== FULL STATE SAVE/LOAD ====================
    
    async def save_full_state(self, state: Dict[str, Any]) -> None:
        """
        Save complete bot state for crash recovery.
        
        Args:
            state: Dict with all state to persist
        """
        state['saved_at'] = time.time()
        
        if self.is_connected:
            try:
                await self._redis.set(
                    self.STATE_KEY,
                    json.dumps(state)
                )
                logger.debug("Full state saved to Redis")
            except Exception as e:
                logger.error(f"Failed to save full state: {e}")
    
    async def load_full_state(self) -> Optional[Dict[str, Any]]:
        """
        Load complete bot state for crash recovery.
        
        Returns:
            State dict or None
        """
        if self.is_connected:
            try:
                data = await self._redis.get(self.STATE_KEY)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Failed to load full state: {e}")
        
        return None
    
    # ==================== METRICS ====================
    
    async def increment_metric(self, name: str, amount: float = 1.0) -> None:
        """
        Increment a metric counter.
        
        Args:
            name: Metric name
            amount: Amount to increment
        """
        if self.is_connected:
            try:
                await self._redis.hincrbyfloat(self.METRICS_KEY, name, amount)
            except Exception as e:
                logger.debug(f"Failed to increment metric: {e}")
    
    async def get_metrics(self) -> Dict[str, float]:
        """Get all metrics."""
        if self.is_connected:
            try:
                data = await self._redis.hgetall(self.METRICS_KEY)
                return {k: float(v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
        
        return {}


# Singleton instance
_redis_state: Optional[RedisStateManager] = None


async def get_redis_state(
    host: str = "localhost",
    port: int = 6379
) -> RedisStateManager:
    """
    Get or create the global Redis state manager.
    
    Args:
        host: Redis host
        port: Redis port
        
    Returns:
        RedisStateManager instance (connected or in fallback mode)
    """
    global _redis_state
    
    if _redis_state is None:
        _redis_state = RedisStateManager(host=host, port=port)
        await _redis_state.connect()
    
    return _redis_state
