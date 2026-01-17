import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os
sys.path.append(os.getcwd())
from execution.position_tracker import PositionTracker, Position
import config.settings

@pytest.mark.asyncio
async def test_spot_sync_preserves_entry_price():
    # Mock settings
    with patch('config.settings.settings') as mock_settings:
        mock_settings.SPOT_MODE = True
        
        # Mocks
        mock_exchange = AsyncMock()
        mock_db = AsyncMock()
        
        # Initialize Tracker
        tracker = PositionTracker(mock_exchange, mock_db)
        
        # 1. Pre-populate local state with a valid price
        symbol = 'btcusdt'
        tracker._positions[symbol] = Position(
            symbol=symbol,
            quantity=0.5,
            avg_entry_price=50000.0,  # Valid price
            unrealized_pnl=0.0,
            last_update=datetime.now()
        )
        
        # 2. Mock Exchange response (Spot Mode = 0 avgPrice)
        # Assuming the tracker adapts the response based on client type
        # In the code, it calls self.exchange.get_account() for spot
        mock_exchange.get_account.return_value = {
            'balances': [
                {'asset': 'BTC', 'free': '0.5', 'locked': '0.0'},
                {'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'}
            ]
        }
        
        # 3. Trigger Sync
        # We need to ensure logic hits the "SPOT_MODE" block
        # The logic checks: if hasattr(self.exchange, 'futures_account') or settings.SPOT_MODE:
        # So it enters the block. 
        # Then: if settings.SPOT_MODE: -> calls get_account
        
        success = await tracker.force_sync_with_exchange()
        
        # 4. Verify
        assert success is True
        
        pos = await tracker.get_position(symbol)
        assert pos is not None
        assert pos.quantity == 0.5
        # The CRITICAL check: Did it preserve 50000.0 or reset to 0.0?
        assert pos.avg_entry_price == 50000.0, f"Average price was reset to {pos.avg_entry_price}!"
        
        print("\n✅ Test Passed: Entry price preserved in Spot Mode")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(test_spot_sync_preserves_entry_price())
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
