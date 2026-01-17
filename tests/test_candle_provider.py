import pytest
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.candle_provider import CandleProvider, Candle

@pytest.mark.asyncio
async def test_candle_provider_basic():
    """Test CandleProvider basics."""
    provider = CandleProvider()
    
    # 1. Process Candle
    candle = Candle(
        symbol="BTCUSDT",
        timestamp=1000,
        open=100, high=110, low=90, close=105, volume=10
    )
    
    await provider.process_candle(candle)
    
    # 2. Verify Storage
    # The provider normalizes keys to lowercase
    assert "btcusdt" in provider.candles
    assert len(provider.candles["btcusdt"]) == 1
    assert provider.candles["btcusdt"][0] == candle
    
    # 3. Get OHLCV
    opens, highs, lows, closes, vols = provider.get_ohlcv_arrays("BTCUSDT")
    assert opens == [100]
    assert closes == [105]
    
    # 4. Callback
    mock_callback_called = False
    async def cb(c):
        nonlocal mock_callback_called
        mock_callback_called = True
        assert c == candle
        
    provider.set_candle_close_callback(cb)
    await provider.process_candle(candle)
    assert mock_callback_called is True

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_candle_provider_basic())
    print("✅ Test Passed: CandleProvider verified.")
