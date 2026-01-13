import asyncio
import os
import logging
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

from config.settings import settings
from execution.binance_executor import BinanceExecutionHandler
from asyncio import Queue as EventQueue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloseAllPositions")

async def close_all():
    print("🚨 FLATTENING ALL POSITIONS...")
    
    # Initialize basic components
    queue = EventQueue()
    # Mock RiskManager
    class DummyRiskManager:
        def check_order(self, *args, **kwargs): return True
    
    executor = BinanceExecutionHandler(queue, risk_manager=DummyRiskManager())
    await executor.connect()
    
    try:
        # Get all positions
        client = executor.client
        positions = await client.futures_position_information()
        
        tasks = []
        for pos in positions:
            amt = float(pos['positionAmt'])
            symbol = pos['symbol']
            
            if amt != 0:
                side = "SELL" if amt > 0 else "BUY"
                qty = abs(amt)
                print(f"📉 Closing {symbol}: {side} {qty}")
                
                # Place Market Order to Close
                task = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=qty,
                    reduceOnly=True
                )
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                print(f"Result: {res}")
        else:
            print("✅ No open positions found.")
            
    except Exception as e:
        print(f"❌ Error closing positions: {e}")
    finally:
        await executor.stop()
        await executor.close()
        print("🏁 Flattening complete.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(close_all())
