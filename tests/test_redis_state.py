"""
Unit tests for Redis State Manager.

Tests both Redis-connected and fallback in-memory modes.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import time

from state.redis_state import (
    RedisStateManager,
    OrderRecord,
    PositionRecord,
    HAS_REDIS
)


class TestOrderRecord:
    """Test OrderRecord dataclass."""
    
    def test_order_record_defaults(self):
        """OrderRecord should have sensible defaults."""
        order = OrderRecord(
            client_order_id="test123",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0
        )
        
        assert order.status == "PENDING"
        assert order.filled_qty == 0.0
        assert order.created_at > 0
    
    def test_order_record_with_all_fields(self):
        """OrderRecord should accept all fields."""
        order = OrderRecord(
            client_order_id="test123",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            status="FILLED",
            exchange_order_id="12345",
            filled_qty=0.1,
            avg_fill_price=50100.0
        )
        
        assert order.status == "FILLED"
        assert order.exchange_order_id == "12345"


class TestPositionRecord:
    """Test PositionRecord dataclass."""
    
    def test_position_record_creation(self):
        """PositionRecord should be creatable."""
        pos = PositionRecord(
            symbol="btcusdt",
            quantity=0.5,
            avg_entry_price=50000.0
        )
        
        assert pos.symbol == "btcusdt"
        assert pos.quantity == 0.5


class TestRedisStateManagerFallback:
    """Test RedisStateManager in fallback (no Redis) mode."""
    
    @pytest.fixture
    def manager(self):
        """Create manager that won't connect to Redis."""
        return RedisStateManager(host="nonexistent", port=6379)
    
    @pytest.mark.asyncio
    async def test_connect_fails_gracefully(self, manager):
        """Should handle connection failure gracefully."""
        result = await manager.connect()
        
        # May return True if Redis lib not installed, False if connection refused
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_save_order_fallback(self, manager):
        """Should save order to fallback storage."""
        order = OrderRecord(
            client_order_id="test123",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0
        )
        
        await manager.save_order(order)
        
        # Should be in fallback storage
        assert "test123" in manager._fallback_orders
    
    @pytest.mark.asyncio
    async def test_get_order_fallback(self, manager):
        """Should retrieve order from fallback storage."""
        order = OrderRecord(
            client_order_id="test123",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0
        )
        
        manager._fallback_orders["test123"] = order
        
        retrieved = await manager.get_order("test123")
        assert retrieved is not None
        assert retrieved.client_order_id == "test123"
    
    @pytest.mark.asyncio
    async def test_save_position_fallback(self, manager):
        """Should save position to fallback storage."""
        pos = PositionRecord(
            symbol="btcusdt",
            quantity=0.5,
            avg_entry_price=50000.0
        )
        
        await manager.save_position(pos)
        
        assert "btcusdt" in manager._fallback_positions
    
    @pytest.mark.asyncio
    async def test_log_equity_fallback(self, manager):
        """Should log equity to fallback storage."""
        await manager.log_equity(10500.0, pnl=100.0)
        
        assert len(manager._fallback_equity) == 1
        ts, equity, pnl = manager._fallback_equity[0]
        assert equity == 10500.0
        assert pnl == 100.0


class TestActiveOrderTracking:
    """Test active order queue functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create fallback manager."""
        return RedisStateManager(host="nonexistent", port=6379)
    
    @pytest.mark.asyncio
    async def test_get_active_orders_filters(self, manager):
        """Should only return non-terminal orders."""
        # Add active order
        active = OrderRecord(
            client_order_id="active1",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            status="SUBMITTED"
        )
        manager._fallback_orders["active1"] = active
        
        # Add filled order
        filled = OrderRecord(
            client_order_id="filled1",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0,
            status="FILLED"
        )
        manager._fallback_orders["filled1"] = filled
        
        active_orders = await manager.get_active_orders()
        
        assert len(active_orders) == 1
        assert active_orders[0].client_order_id == "active1"


class TestEquityHistory:
    """Test equity tracking functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create fallback manager."""
        return RedisStateManager(host="nonexistent", port=6379)
    
    @pytest.mark.asyncio
    async def test_equity_history_returns_list(self, manager):
        """Should return list of equity entries."""
        await manager.log_equity(10000.0)
        await manager.log_equity(10100.0, pnl=100.0)
        await manager.log_equity(10050.0, pnl=-50.0)
        
        history = await manager.get_equity_history(count=10)
        
        assert len(history) == 3
        assert all('equity' in entry for entry in history)


class TestStatePersistence:
    """Test full state save/load."""
    
    @pytest.fixture
    def manager(self):
        """Create fallback manager."""
        return RedisStateManager(host="nonexistent", port=6379)
    
    @pytest.mark.asyncio
    async def test_full_state_returns_none_when_not_connected(self, manager):
        """Should return None when not connected."""
        state = await manager.load_full_state()
        assert state is None


@pytest.mark.skipif(not HAS_REDIS, reason="Redis library not installed")
class TestWithRedis:
    """Integration tests that require actual Redis connection."""
    
    @pytest.fixture
    async def manager(self):
        """Create connected manager."""
        m = RedisStateManager()
        connected = await m.connect()
        if not connected:
            pytest.skip("Redis not available")
        yield m
        await m.disconnect()
    
    @pytest.mark.asyncio
    async def test_order_roundtrip(self, manager):
        """Order should survive save/load cycle."""
        order = OrderRecord(
            client_order_id=f"test_{time.time()}",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000.0
        )
        
        await manager.save_order(order)
        retrieved = await manager.get_order(order.client_order_id)
        
        assert retrieved is not None
        assert retrieved.symbol == order.symbol
