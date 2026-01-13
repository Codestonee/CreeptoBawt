"""Close positions one at a time with retries."""
import asyncio
import os
from dotenv import load_dotenv
from binance import AsyncClient
from binance.exceptions import BinanceAPIException

load_dotenv()

async def close_with_retry(client, symbol, qty, max_retries=5):
    """Close a single position with retry logic."""
    side = "SELL" if qty > 0 else "BUY"
    close_qty = abs(qty)
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            result = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=close_qty,
                reduceOnly=True
            )
            return True, float(result.get('avgPrice', 0))
        except BinanceAPIException as e:
            if e.code == -1007:
                delay = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"  Timeout, waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                return False, str(e)
    return False, "Max retries"

async def main():
    print("Connecting...")
    client = await AsyncClient.create(
        os.getenv('BINANCE_TESTNET_API_KEY'),
        os.getenv('BINANCE_TESTNET_SECRET_KEY'),
        testnet=True
    )
    
    # Get positions
    positions = await client.futures_position_information()
    open_positions = {p['symbol']: float(p['positionAmt']) for p in positions if float(p['positionAmt']) != 0}
    
    print(f"\nFound {len(open_positions)} positions to close\n")
    
    for symbol, qty in open_positions.items():
        print(f"Closing {symbol}: {qty}...")
        success, result = await close_with_retry(client, symbol, qty)
        if success:
            print(f"  ✅ Closed @ ${result:.4f}\n")
        else:
            print(f"  ❌ Failed: {result}\n")
    
    # Verify
    print("\n--- Verification ---")
    positions = await client.futures_position_information()
    remaining = [p for p in positions if float(p['positionAmt']) != 0]
    if remaining:
        print("Still open:")
        for p in remaining:
            print(f"  {p['symbol']}: {p['positionAmt']}")
    else:
        print("✅ All positions closed!")
    
    await client.close_connection()

asyncio.run(main())
