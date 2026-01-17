"""Quick API test to diagnose -2015 error."""
import asyncio
import os
import sys
sys.path.append(os.getcwd())

from binance import AsyncClient
from config.settings import settings

async def test_api():
    print("=== Binance API Connection Test ===\n")
    print(f"API Key (last 8): ...{settings.BINANCE_API_KEY[-8:]}")
    print(f"Testnet: {settings.TESTNET}")
    print(f"Spot Mode: {settings.SPOT_MODE}")
    print()
    
    try:
        # Create client
        if settings.TESTNET:
            client = await AsyncClient.create(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET,
                testnet=True
            )
        else:
            client = await AsyncClient.create(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET
            )
        
        print("✅ Client created successfully")
        
        # Test 1: Get server time (no auth needed)
        server_time = await client.get_server_time()
        print(f"✅ Server time: {server_time}")
        
        # Test 2: Get account info (requires auth)
        print("\nTesting authenticated endpoint (get_account)...")
        try:
            account = await client.get_account()
            print(f"✅ Account access successful!")
            print(f"   Account type: {account.get('accountType', 'N/A')}")
            print(f"   Can trade: {account.get('canTrade', 'N/A')}")
            print(f"   Balances: {len(account.get('balances', []))} assets")
        except Exception as e:
            print(f"❌ Account access failed: {e}")
        
        # Test 3: Try to create a test order (dry run)
        print("\nTesting order endpoint (test order)...")
        try:
            test_order = await client.create_test_order(
                symbol='BTCUSDC',
                side='BUY',
                type='LIMIT',
                timeInForce='GTC',
                quantity=0.001,
                price=50000.0
            )
            print(f"✅ Test order successful: {test_order}")
        except Exception as e:
            print(f"❌ Test order failed: {e}")
        
        await client.close_connection()
        
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api())
