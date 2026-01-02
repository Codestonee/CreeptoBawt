"""Tests for Order Manager and FSM."""
import pytest
from decimal import Decimal
from execution.order_manager import (
    Order,
    OrderManager,
    InvalidStateTransition,
)
from core.events import OrderState, OrderType, TimeInForce


class TestOrderStateMachine:
    """Tests for order state transitions."""
    
    def test_valid_transition_created_to_pending(self):
        """Test valid transition from CREATED to PENDING."""
        order = Order(
            client_order_id="test-1",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        assert order.state == OrderState.CREATED
        order.transition(OrderState.PENDING, "submitting")
        assert order.state == OrderState.PENDING
    
    def test_valid_transition_pending_to_acknowledged(self):
        """Test valid transition from PENDING to ACKNOWLEDGED."""
        order = Order(
            client_order_id="test-2",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.PENDING,
        )
        
        order.transition(OrderState.ACKNOWLEDGED, "exchange_confirmed")
        assert order.state == OrderState.ACKNOWLEDGED
    
    def test_valid_transition_acknowledged_to_partially_filled(self):
        """Test valid transition from ACKNOWLEDGED to PARTIALLY_FILLED."""
        order = Order(
            client_order_id="test-3",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.ACKNOWLEDGED,
        )
        
        order.transition(OrderState.PARTIALLY_FILLED, "partial_fill")
        assert order.state == OrderState.PARTIALLY_FILLED
    
    def test_valid_transition_partially_filled_to_filled(self):
        """Test valid transition from PARTIALLY_FILLED to FILLED."""
        order = Order(
            client_order_id="test-4",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.5"),
        )
        
        order.transition(OrderState.FILLED, "fully_filled")
        assert order.state == OrderState.FILLED
    
    def test_valid_transition_acknowledged_to_canceling(self):
        """Test valid transition from ACKNOWLEDGED to CANCELING."""
        order = Order(
            client_order_id="test-5",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.ACKNOWLEDGED,
        )
        
        order.transition(OrderState.CANCELING, "user_requested")
        assert order.state == OrderState.CANCELING
    
    def test_valid_transition_canceling_to_canceled(self):
        """Test valid transition from CANCELING to CANCELED."""
        order = Order(
            client_order_id="test-6",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.CANCELING,
        )
        
        order.transition(OrderState.CANCELED, "canceled_by_exchange")
        assert order.state == OrderState.CANCELED
    
    def test_invalid_transition_created_to_filled(self):
        """Test invalid transition from CREATED to FILLED."""
        order = Order(
            client_order_id="test-7",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        with pytest.raises(InvalidStateTransition) as exc_info:
            order.transition(OrderState.FILLED, "invalid")
        
        # Check that error message contains key information
        error_msg = str(exc_info.value).lower()
        assert "cannot transition" in error_msg
        assert "created" in error_msg
        assert "filled" in error_msg
    
    def test_invalid_transition_filled_to_canceled(self):
        """Test invalid transition from FILLED (terminal) to CANCELED."""
        order = Order(
            client_order_id="test-8",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.FILLED,
        )
        
        with pytest.raises(InvalidStateTransition):
            order.transition(OrderState.CANCELED, "invalid")
    
    def test_state_history_tracking(self):
        """Test that state transitions are recorded in history."""
        order = Order(
            client_order_id="test-9",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        order.transition(OrderState.PENDING, "submitting")
        order.transition(OrderState.ACKNOWLEDGED, "confirmed")
        
        assert len(order.state_history) == 2
        assert order.state_history[0]["from_state"] == "created"
        assert order.state_history[0]["to_state"] == "pending"
        assert order.state_history[1]["from_state"] == "pending"
        assert order.state_history[1]["to_state"] == "acknowledged"


class TestOrderProperties:
    """Tests for order properties and helpers."""
    
    def test_is_complete_for_filled(self):
        """Test is_complete for FILLED state."""
        order = Order(
            client_order_id="test-10",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.FILLED,
        )
        
        assert order.is_complete is True
    
    def test_is_complete_for_canceled(self):
        """Test is_complete for CANCELED state."""
        order = Order(
            client_order_id="test-11",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.CANCELED,
        )
        
        assert order.is_complete is True
    
    def test_is_complete_for_open_order(self):
        """Test is_complete for open order."""
        order = Order(
            client_order_id="test-12",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.ACKNOWLEDGED,
        )
        
        assert order.is_complete is False
    
    def test_is_open(self):
        """Test is_open property."""
        order = Order(
            client_order_id="test-13",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            state=OrderState.ACKNOWLEDGED,
        )
        
        assert order.is_open is True
        
        order.state = OrderState.FILLED
        assert order.is_open is False
    
    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            client_order_id="test-14",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            filled_quantity=Decimal("0.3"),
        )
        
        assert order.remaining_quantity == Decimal("0.7")
    
    def test_is_timed_out(self):
        """Test timeout detection."""
        import time
        
        # Create order with very short timeout
        order = Order(
            client_order_id="test-15",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            timeout_seconds=0.1,
            created_at=time.time() - 0.2,  # Created 200ms ago
        )
        
        assert order.is_timed_out is True
        
        # Test non-timed-out order
        order2 = Order(
            client_order_id="test-16",
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            timeout_seconds=300,
        )
        
        assert order2.is_timed_out is False


class TestOrderManager:
    """Tests for OrderManager."""
    
    @pytest.fixture
    def manager(self):
        """Create order manager for tests."""
        return OrderManager(
            max_retries=3,
            base_retry_delay_ms=10,
            max_retry_delay_seconds=1,
        )
    
    def test_create_order(self, manager):
        """Test order creation."""
        order = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        assert order.state == OrderState.CREATED
        assert order.client_order_id is not None
        assert order.symbol == "BTC-USDT"
        assert order.quantity == Decimal("1.0")
    
    def test_create_order_with_custom_id(self, manager):
        """Test order creation with custom client order ID."""
        custom_id = "my-custom-order-123"
        order = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
            client_order_id=custom_id,
        )
        
        assert order.client_order_id == custom_id
    
    def test_get_order(self, manager):
        """Test retrieving order by ID."""
        order = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        retrieved = manager.get_order(order.client_order_id)
        assert retrieved is order
    
    def test_get_open_orders(self, manager):
        """Test getting open orders."""
        # Create multiple orders
        order1 = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        order2 = manager.create_order(
            exchange="binance",
            symbol="ETH-USDT",
            side="sell",
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("3000"),
        )
        
        # Mark one as filled
        order2.state = OrderState.FILLED
        
        open_orders = manager.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0]["client_order_id"] == order1.client_order_id
    
    def test_get_completed_orders(self, manager):
        """Test getting completed orders."""
        order1 = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        order2 = manager.create_order(
            exchange="binance",
            symbol="ETH-USDT",
            side="sell",
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("3000"),
        )
        
        # Mark as completed
        order1.state = OrderState.FILLED
        order2.state = OrderState.CANCELED
        
        completed = manager.get_completed_orders()
        assert len(completed) == 2
    
    @pytest.mark.asyncio
    async def test_update_order_fill_partial(self, manager):
        """Test partial fill update."""
        order = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        # Set to acknowledged first
        order.state = OrderState.ACKNOWLEDGED
        
        # Update with partial fill
        await manager.update_order_fill(
            order.client_order_id,
            filled_quantity=Decimal("0.5"),
            fill_price=Decimal("44999"),
        )
        
        assert order.state == OrderState.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("0.5")
        assert order.average_fill_price == Decimal("44999")
    
    @pytest.mark.asyncio
    async def test_update_order_fill_complete(self, manager):
        """Test complete fill update."""
        order = manager.create_order(
            exchange="binance",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("45000"),
        )
        
        order.state = OrderState.ACKNOWLEDGED
        
        # Update with full fill
        await manager.update_order_fill(
            order.client_order_id,
            filled_quantity=Decimal("1.0"),
            fill_price=Decimal("45000"),
        )
        
        assert order.state == OrderState.FILLED
        assert order.filled_quantity == Decimal("1.0")
