"""Tests for State Reconciliation Engine."""
import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from reconciliation.state_reconciler import (
    StateReconciler,
    BalanceCorrectionEvent,
    OrderDiscrepancy,
)


class MockExchangeConnector:
    """Mock exchange connector for testing."""
    
    def __init__(self):
        self.exchange_name = "test_exchange"
        self.balances = {}
        self.orders = []
        self.positions = {}
    
    async def fetch_balance(self):
        """Mock balance fetch."""
        return self.balances
    
    async def fetch_open_orders(self):
        """Mock open orders fetch."""
        return self.orders
    
    async def fetch_positions(self):
        """Mock positions fetch."""
        return [
            {"symbol": symbol, "contracts": float(contracts)}
            for symbol, contracts in self.positions.items()
        ]


class MockOrderManager:
    """Mock order manager for testing."""
    
    def __init__(self):
        self.open_orders = []
        self.closed_orders = []
        self.orphan_orders_added = []
        self.synced_orders = []
    
    def get_open_orders(self):
        """Mock get open orders."""
        return self.open_orders
    
    async def mark_order_closed(self, client_order_id, reason):
        """Mock mark order closed."""
        self.closed_orders.append({
            "client_order_id": client_order_id,
            "reason": reason,
        })
    
    async def add_orphan_order(self, order_data):
        """Mock add orphan order."""
        self.orphan_orders_added.append(order_data)
    
    async def sync_order_state(self, client_order_id, exchange_state):
        """Mock sync order state."""
        self.synced_orders.append({
            "client_order_id": client_order_id,
            "exchange_state": exchange_state,
        })


class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        self.published_events = []
    
    async def publish(self, topic, data):
        """Mock publish."""
        self.published_events.append({
            "topic": topic,
            "data": data,
        })


class TestStateReconciler:
    """Tests for State Reconciler."""
    
    @pytest.fixture
    def reconciler(self):
        """Create state reconciler."""
        return StateReconciler(
            sync_interval_seconds=30,
            order_sync_interval_seconds=10,
        )
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange connector."""
        return MockExchangeConnector()
    
    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        return MockOrderManager()
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        return MockEventBus()
    
    def test_reconciler_initialization(self, reconciler):
        """Test reconciler initializes correctly."""
        assert reconciler.sync_interval == 30
        assert reconciler.order_sync_interval == 10
        assert reconciler.discrepancy_count == 0
    
    def test_set_dependencies(self, reconciler, mock_exchange, mock_order_manager, mock_event_bus):
        """Test setting dependencies."""
        reconciler.set_dependencies(
            mock_exchange,
            mock_order_manager,
            mock_event_bus,
        )
        
        assert reconciler.exchange_connector is mock_exchange
        assert reconciler.order_manager is mock_order_manager
        assert reconciler.event_bus is mock_event_bus
    
    @pytest.mark.asyncio
    async def test_reconcile_balances_no_discrepancy(
        self,
        reconciler,
        mock_exchange,
        mock_event_bus,
    ):
        """Test balance reconciliation with matching balances."""
        reconciler.set_dependencies(mock_exchange, None, mock_event_bus)
        
        # Set up matching balances
        mock_exchange.balances = {
            "BTC": 1.5,
            "USDT": 10000.0,
        }
        
        # Mock local balances to match
        reconciler._get_local_balances = lambda: {
            "BTC": Decimal("1.5"),
            "USDT": Decimal("10000.0"),
        }
        reconciler._correct_local_balance = AsyncMock()
        
        corrections = await reconciler.reconcile_balances()
        
        assert len(corrections) == 0
        assert reconciler.discrepancy_count == 0
    
    @pytest.mark.asyncio
    async def test_reconcile_balances_with_discrepancy(
        self,
        reconciler,
        mock_exchange,
        mock_event_bus,
    ):
        """Test balance reconciliation detects discrepancy."""
        reconciler.set_dependencies(mock_exchange, None, mock_event_bus)
        
        # Exchange has more BTC than local
        mock_exchange.balances = {
            "BTC": 2.0,
            "USDT": 10000.0,
        }
        
        # Local has less
        reconciler._get_local_balances = lambda: {
            "BTC": Decimal("1.5"),
            "USDT": Decimal("10000.0"),
        }
        reconciler._correct_local_balance = AsyncMock()
        
        corrections = await reconciler.reconcile_balances()
        
        assert len(corrections) == 1
        assert corrections[0].currency == "BTC"
        assert corrections[0].local_balance == Decimal("1.5")
        assert corrections[0].exchange_balance == Decimal("2.0")
        assert corrections[0].discrepancy == Decimal("0.5")
        assert corrections[0].corrected is True
        
        # Should have published event
        assert len(mock_event_bus.published_events) == 1
        assert "balance_correction" in mock_event_bus.published_events[0]["topic"]
    
    @pytest.mark.asyncio
    async def test_reconcile_orders_ghost_order(
        self,
        reconciler,
        mock_exchange,
        mock_order_manager,
    ):
        """Test detection of ghost orders."""
        reconciler.set_dependencies(mock_exchange, mock_order_manager, None)
        
        # Local has order that exchange doesn't
        mock_order_manager.open_orders = [
            {
                "client_order_id": "order-123",
                "exchange_order_id": "ex-123",
                "state": "acknowledged",
            }
        ]
        
        # Exchange has no orders
        mock_exchange.orders = []
        
        discrepancies = await reconciler.reconcile_orders()
        
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == "ghost"
        assert discrepancies[0].client_order_id == "order-123"
        
        # Should have marked order as closed
        assert len(mock_order_manager.closed_orders) == 1
        assert mock_order_manager.closed_orders[0]["client_order_id"] == "order-123"
    
    @pytest.mark.asyncio
    async def test_reconcile_orders_orphan_order(
        self,
        reconciler,
        mock_exchange,
        mock_order_manager,
    ):
        """Test detection of orphan orders."""
        reconciler.set_dependencies(mock_exchange, mock_order_manager, None)
        
        # Exchange has order that local doesn't know about
        mock_exchange.orders = [
            {
                "client_order_id": "order-456",
                "exchange_order_id": "ex-456",
                "status": "open",
                "symbol": "BTC-USDT",
            }
        ]
        
        # Local has no orders
        mock_order_manager.open_orders = []
        
        discrepancies = await reconciler.reconcile_orders()
        
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == "orphan"
        assert discrepancies[0].client_order_id == "order-456"
        
        # Should have added orphan order
        assert len(mock_order_manager.orphan_orders_added) == 1
        assert mock_order_manager.orphan_orders_added[0]["client_order_id"] == "order-456"
    
    @pytest.mark.asyncio
    async def test_reconcile_orders_status_mismatch(
        self,
        reconciler,
        mock_exchange,
        mock_order_manager,
    ):
        """Test detection of order status mismatches."""
        reconciler.set_dependencies(mock_exchange, mock_order_manager, None)
        
        # Both have the order but different states
        mock_order_manager.open_orders = [
            {
                "client_order_id": "order-789",
                "exchange_order_id": "ex-789",
                "state": "acknowledged",
            }
        ]
        
        mock_exchange.orders = [
            {
                "client_order_id": "order-789",
                "exchange_order_id": "ex-789",
                "status": "closed",
            }
        ]
        
        discrepancies = await reconciler.reconcile_orders()
        
        assert len(discrepancies) == 1
        assert discrepancies[0].discrepancy_type == "status_mismatch"
        assert discrepancies[0].local_state == "acknowledged"
        assert discrepancies[0].exchange_state == "closed"
        
        # Should have synced the state
        assert len(mock_order_manager.synced_orders) == 1
        assert mock_order_manager.synced_orders[0]["exchange_state"] == "closed"
    
    @pytest.mark.asyncio
    async def test_reconcile_positions(
        self,
        reconciler,
        mock_exchange,
    ):
        """Test position reconciliation."""
        reconciler.set_dependencies(mock_exchange, None, None)
        
        # Set up position discrepancy
        mock_exchange.positions = {
            "BTC-PERP": 10.0,
            "ETH-PERP": 50.0,
        }
        
        reconciler._get_local_positions = lambda: {
            "BTC-PERP": Decimal("8.0"),  # Local has less
            "ETH-PERP": Decimal("50.0"),  # Matches
        }
        reconciler._correct_local_position = AsyncMock()
        
        corrections = await reconciler.reconcile_positions()
        
        assert len(corrections) == 1
        assert corrections[0]["symbol"] == "BTC-PERP"
        assert corrections[0]["local_position"] == "8.0"
        assert corrections[0]["exchange_position"] == "10.0"
    
    @pytest.mark.asyncio
    async def test_reconcile_state_full_cycle(
        self,
        reconciler,
        mock_exchange,
        mock_order_manager,
    ):
        """Test full three-phase reconciliation."""
        reconciler.set_dependencies(mock_exchange, mock_order_manager, None)
        
        # Set up mocks
        mock_exchange.balances = {"BTC": 1.0}
        mock_exchange.orders = []
        mock_exchange.positions = {}
        
        reconciler._get_local_balances = lambda: {"BTC": Decimal("1.0")}
        reconciler._get_local_positions = lambda: {}
        reconciler._correct_local_balance = AsyncMock()
        
        # Should not raise any errors
        await reconciler.reconcile_state()
    
    def test_states_match(self, reconciler):
        """Test state matching logic."""
        # Test matching states
        assert reconciler._states_match("acknowledged", "open") is True
        assert reconciler._states_match("filled", "closed") is True
        assert reconciler._states_match("canceled", "cancelled") is True
        
        # Test non-matching states
        assert reconciler._states_match("acknowledged", "closed") is False
        assert reconciler._states_match("filled", "open") is False
    
    @pytest.mark.asyncio
    async def test_reconciler_start_stop(self, reconciler):
        """Test starting and stopping reconciler."""
        await reconciler.start()
        assert reconciler._running is True
        assert len(reconciler._tasks) == 2  # Two background loops
        
        await reconciler.stop()
        assert reconciler._running is False
        # Tasks are cancelled and cleared
        for task in reconciler._tasks:
            assert task.cancelled()
    
    @pytest.mark.asyncio
    async def test_balance_correction_event(self):
        """Test BalanceCorrectionEvent creation."""
        event = BalanceCorrectionEvent(
            exchange="binance",
            currency="BTC",
            local_balance=Decimal("1.0"),
            exchange_balance=Decimal("1.5"),
            discrepancy=Decimal("0.5"),
        )
        
        assert event.exchange == "binance"
        assert event.currency == "BTC"
        assert event.discrepancy == Decimal("0.5")
        assert event.corrected is False
    
    @pytest.mark.asyncio
    async def test_order_discrepancy(self):
        """Test OrderDiscrepancy creation."""
        discrepancy = OrderDiscrepancy(
            client_order_id="order-123",
            exchange_order_id="ex-123",
            local_state="acknowledged",
            exchange_state="filled",
            discrepancy_type="status_mismatch",
        )
        
        assert discrepancy.client_order_id == "order-123"
        assert discrepancy.discrepancy_type == "status_mismatch"
        assert discrepancy.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_reconcile_balances_without_connector(self, reconciler):
        """Test balance reconciliation without exchange connector."""
        corrections = await reconciler.reconcile_balances()
        
        assert len(corrections) == 0  # Should skip gracefully
    
    @pytest.mark.asyncio
    async def test_reconcile_orders_without_dependencies(self, reconciler):
        """Test order reconciliation without dependencies."""
        discrepancies = await reconciler.reconcile_orders()
        
        assert len(discrepancies) == 0  # Should skip gracefully
