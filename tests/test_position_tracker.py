import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.position_tracker import PositionTracker, Position

@pytest.mark.asyncio
async def test_position_versioning():
    """
    Test that PositionTracker correctly increments versions on updates.
    """
    # Mocks
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    
    # Init Tracker
    tracker = PositionTracker(mock_client, mock_db)
    tracker._is_synced = True # Hack: Assume synced for manual update testing
    
    # 1. New Position
    await tracker.update_position("BTCUSDT", 0.1, 50000.0)
    
    pos = await tracker.get_position("BTCUSDT")
    assert pos is not None
    assert pos.quantity == 0.1
    assert pos.version == 1
    assert pos.exchange_confirmed is False
    
    # 2. Update Position
    await tracker.update_position("BTCUSDT", 0.1, 51000.0) # Buy more
    
    pos = await tracker.get_position("BTCUSDT")
    assert pos.quantity == 0.2
    assert pos.version == 2
    
    # 3. Force Sync (Reset version? Or Increment?)
    # In my code, force_sync_with_exchange increments version if local exists.
    
    # Mock exchange return
    mock_client.futures_position_information.return_value = [{
        "symbol": "BTCUSDT",
        "positionAmt": "0.2001", # Slight diff
        "entryPrice": "50500.0",
        "unRealizedProfit": "10.0"
    }]
    # Set attribute for detection
    mock_client.futures_account = True 

    # We need to manually set synced=True because get_position checks it?
    # No, get_position checks is_synced.
    # We must run force_sync first to set is_synced = True?
    # Actually wait, my test skipped initialization which sets is_synced.
    # So get_position might return None or log error if I don't force sets.
    # Let's check get_position code:
    # if not self._is_synced: logger.error... return None.
    
    # So I must force sync or hack the flag.
    tracker._is_synced = True # Hack for unit test
    
    # Now retry the access assertions above (Retrospective)
    # The code above would have failed if I ran it.
    # I should restart the test sequence with the flag set.
    
    # Reset
    tracker = PositionTracker(mock_client, mock_db)
    tracker._is_synced = True
    
    await tracker.update_position("BTCUSDT", 0.1, 50000.0)
    pos = await tracker.get_position("BTCUSDT")
    assert pos.version == 1
    
    await tracker.update_position("BTCUSDT", 0.1, 51000.0)
    pos = await tracker.get_position("BTCUSDT")
    assert pos.version == 2

    # Test Force Sync
    # tracker.force_sync_with_exchange call
    # This matches the code I fixed with `try...except`
    
    result = await tracker.force_sync_with_exchange()
    assert result is True
    
    pos = await tracker.get_position("BTCUSDT")
    assert pos.quantity == 0.2001
    assert pos.version == 3 # Should increment
    assert pos.exchange_confirmed is True

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_position_versioning())
    print("✅ Test Passed: Position Versioning verified.")
