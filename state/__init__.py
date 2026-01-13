"""
State management module for CreeptBaws.

Provides persistent state storage using Redis with in-memory fallback.
"""

from state.redis_state import (
    RedisStateManager,
    OrderRecord,
    PositionRecord,
    get_redis_state,
    HAS_REDIS
)

__all__ = [
    'RedisStateManager',
    'OrderRecord',
    'PositionRecord',
    'get_redis_state',
    'HAS_REDIS'
]
