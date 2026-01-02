"""Tests for core events module."""
import pytest
from decimal import Decimal

from core.events import (
    MarketEvent,
    OrderBook,
    OrderBookLevel,
    OrderEvent,
    OrderState,
    OrderType,
    FillEvent,
)


class TestOrderBookLevel:
    """Tests for OrderBookLevel."""
    
    def test_valid_level(self):
        level = OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.5"))
        assert level.price == Decimal("50000")
        assert level.quantity == Decimal("1.5")
    
    def test_invalid_price(self):
        with pytest.raises(ValueError, match="Price must be positive"):
            OrderBookLevel(price=Decimal("-100"), quantity=Decimal("1"))
    
    def test_invalid_quantity(self):
        with pytest.raises(ValueError, match="Quantity cannot be negative"):
            OrderBookLevel(price=Decimal("100"), quantity=Decimal("-1"))


class TestOrderBook:
    """Tests for OrderBook."""
    
    def test_mid_price(self):
        book = OrderBook(
            bids=[OrderBookLevel(Decimal("49900"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50100"), Decimal("1"))],
            timestamp=1000000,
        )
        assert book.mid_price == Decimal("50000")
    
    def test_spread(self):
        book = OrderBook(
            bids=[OrderBookLevel(Decimal("49900"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50100"), Decimal("1"))],
            timestamp=1000000,
        )
        assert book.spread == Decimal("200")
    
    def test_spread_bps(self):
        book = OrderBook(
            bids=[OrderBookLevel(Decimal("49900"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50100"), Decimal("1"))],
            timestamp=1000000,
        )
        # spread = 200, mid = 50000, bps = 200/50000 * 10000 = 40
        assert book.spread_bps == Decimal("40")
    
    def test_empty_book(self):
        book = OrderBook(bids=[], asks=[], timestamp=1000000)
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None


class TestMarketEvent:
    """Tests for MarketEvent."""
    
    def test_create_trade_event(self):
        event = MarketEvent(
            event_type="trade",
            exchange="binance",
            symbol="BTC-USDT",
            timestamp_exchange=1700000000000000,
            timestamp_received=1700000000001000,
            price=Decimal("45000.50"),
            quantity=Decimal("0.5"),
            side="buy",
        )
        
        assert event.exchange == "binance"
        assert event.symbol == "BTC-USDT"
        assert event.price == Decimal("45000.50")
        assert event.latency_us == 1000
    
    def test_serialization(self):
        event = MarketEvent(
            event_type="trade",
            exchange="binance",
            symbol="BTC-USDT",
            timestamp_exchange=1700000000000000,
            timestamp_received=1700000000001000,
            price=Decimal("45000.50"),
            quantity=Decimal("0.5"),
            side="buy",
            trade_id="12345",
        )
        
        data = event.to_dict()
        assert data["price"] == "45000.50"
        assert data["trade_id"] == "12345"
        
        # Deserialize
        restored = MarketEvent.from_dict(data)
        assert restored.price == event.price
        assert restored.symbol == event.symbol


class TestOrderEvent:
    """Tests for OrderEvent."""
    
    def test_remaining_quantity(self):
        order = OrderEvent(
            event_type=None,
            client_order_id="test-123",
            exchange_order_id=None,
            exchange="paper",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            state=OrderState.PARTIALLY_FILLED,
            price=Decimal("45000"),
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.3"),
        )
        
        assert order.remaining_quantity == Decimal("0.7")
    
    def test_is_complete(self):
        order = OrderEvent(
            event_type=None,
            client_order_id="test-123",
            exchange_order_id=None,
            exchange="paper",
            symbol="BTC-USDT",
            side="buy",
            order_type=OrderType.LIMIT,
            state=OrderState.FILLED,
            price=Decimal("45000"),
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
        )
        
        assert order.is_complete is True
        
        order.state = OrderState.ACKNOWLEDGED
        assert order.is_complete is False


class TestFillEvent:
    """Tests for FillEvent."""
    
    def test_fill_value(self):
        fill = FillEvent(
            fill_id="fill-1",
            client_order_id="order-1",
            exchange_order_id="ex-1",
            exchange="paper",
            symbol="BTC-USDT",
            side="buy",
            price=Decimal("45000"),
            quantity=Decimal("0.5"),
            fee=Decimal("22.5"),
            fee_currency="USDT",
        )
        
        assert fill.value == Decimal("22500")
        assert fill.net_value == Decimal("22522.5")  # buy: value + fee
    
    def test_sell_net_value(self):
        fill = FillEvent(
            fill_id="fill-1",
            client_order_id="order-1",
            exchange_order_id="ex-1",
            exchange="paper",
            symbol="BTC-USDT",
            side="sell",
            price=Decimal("45000"),
            quantity=Decimal("0.5"),
            fee=Decimal("22.5"),
            fee_currency="USDT",
        )
        
        assert fill.net_value == Decimal("22477.5")  # sell: value - fee
