
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from strategies.funding_arb import FundingArbStrategy
from core.events import MarketEvent, FundingRateEvent

@pytest.fixture
def mock_queue():
    return asyncio.Queue()

@pytest.fixture
def strategy(mock_queue):
    with patch('strategies.funding_arb.get_order_manager') as mock_om:
        # Mock settings
        with patch('config.settings.settings.FUNDING_ARB_CONFIG', {
            "MIN_FUNDING_RATE_PCT": 0.01, # 0.01% basis
            "POSITION_SIZE_USD": 100.0,
            "ENTRY_SPREAD_THRESHOLD_PCT": 0.05,
            "EXIT_FUNDING_PCT": 0.001,
            "USE_SPOT_HEDGE": True
        }):
            strat = FundingArbStrategy(mock_queue, ["BTCUSDT"])
            return strat

@pytest.mark.asyncio
async def test_funding_arb_entry_on_high_rate(strategy, mock_queue):
    # 1. Setup Price (Mid = 50,000)
    # Mock event with order_book
    mock_event = MagicMock(spec=MarketEvent)
    mock_event.symbol = "BTCUSDT"
    mock_event.order_book = MagicMock()
    # Mock bids/asks as list of objects with price attr
    mock_bid = MagicMock(); mock_bid.price = 50000.0
    mock_ask = MagicMock(); mock_ask.price = 50000.0 # Zero spread
    mock_event.order_book.bids = [mock_bid]
    mock_event.order_book.asks = [mock_ask]
    
    await strategy.on_tick(mock_event)
    assert strategy.mid_prices["BTCUSDT"] == 50000.0
    
    # 2. Inject High Funding Rate (> 0.01%)
    # Rate = 0.0002 (0.02%)
    rate_event = FundingRateEvent(
        symbol="BTCUSDT",
        rate=0.0002,
        mark_price=50000.0,
        next_funding_time=1234567890,
        timestamp=1234567890
    )
    
    await strategy.on_funding_rate(rate_event)
    
    # 3. Verify Signals
    # Should produce 2 signals: BUY (Spot) and SELL (Perp)
    assert mock_queue.qsize() == 2
    
    sig1 = await mock_queue.get()
    sig2 = await mock_queue.get()
    
    # Check Leg 1 (Spot Buy)
    assert sig1.side == "BUY"
    assert sig1.symbol == "BTCUSDT"
    assert sig1.quantity == 100.0 / 50000.0 # 0.002
    
    # Check Leg 2 (Perp Sell)
    assert sig2.side == "SELL"
    assert sig2.symbol == "BTCUSDT"
    assert sig2.quantity == 0.002
    
    # Check internal state
    assert "BTCUSDT" in strategy.active_arbs
    assert strategy.active_arbs["BTCUSDT"].side == "LONG_SPOT_SHORT_PERP"

@pytest.mark.asyncio
async def test_funding_arb_exit_on_low_rate(strategy, mock_queue):
    # Setup Active Position
    from strategies.funding_arb import ArbPosition
    import time
    
    strategy.mid_prices["BTCUSDT"] = 50000.0
    strategy.active_arbs["BTCUSDT"] = ArbPosition(
        symbol="BTCUSDT",
        side="LONG_SPOT_SHORT_PERP",
        quantity=0.002,
        entry_funding_rate=0.0002,
        entry_time=time.time()
    )
    
    # Inject Low Rate (0.00001 < 0.001%)
    rate_event = FundingRateEvent(
        symbol="BTCUSDT",
        rate=0.000005,
        mark_price=50000.0,
        next_funding_time=1234567890,
        timestamp=1234567890
    )
    
    await strategy.on_funding_rate(rate_event)
    
    # Verify Exit Signals
    assert mock_queue.qsize() == 2
    
    sig1 = await mock_queue.get()
    sig2 = await mock_queue.get()
    
    # Should reverse: SELL Spot, BUY Perp
    assert sig1.side == "SELL" # Close Spot Long
    assert sig2.side == "BUY"  # Close Perp Short
    
    assert "BTCUSDT" not in strategy.active_arbs
