import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.shadow_book import get_shadow_book

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestShadowBook")

async def test_live_connection():
    symbols = ['BTCUSDT', 'ETHUSDT']
    logger.info(f"Testing ShadowBookService for {symbols}")
    
    # Initialize service
    service = get_shadow_book(symbols, testnet=False) # Use False to test real URL if settings default to mainnet
    
    # Start service in background task
    task = asyncio.create_task(service.start())
    
    # Monitor for 15 seconds
    try:
        for i in range(15):
            await asyncio.sleep(1)
            stale_count = 0
            for sym in symbols:
                is_stale = service.is_stale(sym)
                book = service.get_order_book(sym)
                bid, ask = book.get_best_bid_ask()
                logger.info(f"[{sym}] Stale: {is_stale}, Bid: {bid}, Ask: {ask}, UpdateID: {book.last_update_id}")
                
                if is_stale:
                    stale_count += 1
            
            if stale_count == 0 and i > 3:
                logger.info("✅ SUCCESS: All books are fresh!")
                break
                
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await service.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_live_connection())
