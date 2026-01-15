import asyncio
import logging
import time
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock
from database.db_manager import DatabaseManager
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.risk_gatekeeper import RiskGatekeeper
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

async def main():
    logger.info("🚀 Starting OrderManager Verification...")
    
    # Setup Temp DB
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    db = None
    try:
        db = DatabaseManager(db_path=db_path)
        logger.info(f"✅ Database Initialized at {db_path}")

        # 2. Patch Settings for Risk (BEFORE initializing Gatekeeper)
        settings.MAX_POSITION_USD = 1000000.0
        settings.RISK_MAX_ORDER_USD = 1000000.0
        settings.RISK_MIN_NOTIONAL_USD = 1.0
        settings.MIN_NOTIONAL_USD = 1.0
        settings.RISK_MAX_POSITION_PER_SYMBOL_USD = 1000000.0
        settings.RISK_MAX_POSITION_TOTAL_USD = 1000000.0
        settings.RISK_MAX_DAILY_LOSS_USD = 1000000.0
        logger.info("✅ Risk Settings Patched")

        # 3. Setup Mocks
        mock_exchange = AsyncMock()
        mock_exchange.create_order = AsyncMock(return_value={'orderId': '123456'})
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        # Mock positions to allow sync
        mock_exchange.get_positions = AsyncMock(return_value=[]) 
        
        # 4. Setup Dependencies
        position_tracker = PositionTracker(mock_exchange, db)
        risk_gatekeeper = RiskGatekeeper(position_tracker, mock_exchange)
        
        # 5. Create OrderManager
        om = OrderManager(mock_exchange, db, position_tracker, risk_gatekeeper)
        
        # Initialize implementation details
        await om.initialize()
        om.position_tracker._is_synced = True # Force sync for script
        logger.info("✅ OrderManager Initialized")
        
        # 6. Test: Submit Order
        logger.info("🧪 Testing: Submit Order...")
        try:
            order = await om.submit_order(
                symbol='btcusdt',
                side='BUY',
                quantity=0.1,
                price=50000.0
            )
            
            if order and order.state == OrderState.SUBMITTED.value:
                logger.info(f"✅ Order Submitted: {order.client_order_id}")
            else:
                logger.error(f"❌ Order Submit Failed: {order}")
                return
                
            # 7. Test: Process Fill
            logger.info("🧪 Testing: Process Fill...")
            await om.process_fill(
                client_order_id=order.client_order_id,
                filled_qty=0.1,
                fill_price=50000.0,
                commission=0.0
            )
            logger.info("✅ Fill Processed")
            
            # 8. Verify Position
            pos = await om.get_position('btcusdt')
            if pos and pos.quantity == 0.1:
                logger.info(f"✅ Position Verified: {pos.quantity} BTC")
            else:
                logger.error(f"❌ Position Mismatch: {pos}")
                
        except Exception as e:
            logger.error(f"❌ Test Failed: {e}", exc_info=True)

        logger.info("🏁 Verification Complete")
        
    finally:
        if db:
            db.close()
        try:
            os.unlink(db_path)
            logger.info("✅ DB Cleanup Complete")
        except Exception as e:
            logger.error(f"Failed to delete temp DB: {e}")

if __name__ == "__main__":
    asyncio.run(main())
