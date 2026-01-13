"""Quick check of current positions."""
import asyncio
import os
from dotenv import load_dotenv
from binance import AsyncClient

load_dotenv()

async def check():
    c = await AsyncClient.create(
        os.getenv('BINANCE_TESTNET_API_KEY'),
        os.getenv('BINANCE_TESTNET_SECRET_KEY'),
        testnet=True
    )
    positions = await c.futures_position_information()
    
    print("Current positions:")
    found = False
    for p in positions:
        qty = float(p['positionAmt'])
        if qty != 0:
            found = True
            print(f"  {p['symbol']}: {qty}")
    
    if not found:
        print("  (No open positions)")
    
    await c.close_connection()

asyncio.run(check())
