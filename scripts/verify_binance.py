import asyncio
import os
import aiohttp
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load usage
load_dotenv()

async def verify():
    print("🔍 DIAGNOSTIC TOOL: Binance Connection Verify")
    print("---------------------------------------------")

    # 1. Check IP
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.ipify.org') as resp:
                ip = await resp.text()
                print(f"✅ DETECTED SERVER IP: {ip}")
                print(f"   (Please verify this IP is whitelisted in Binance API settings)")
    except Exception as e:
        print(f"⚠️ Could not detect IP: {e}")

    # 2. Check Keys
    key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_API_SECRET")
    
    if not key or not secret:
        print("❌ CRITICAL: API Keys missing in .env")
        return
        
    print(f"🔑 API Key loaded: {key[:6]}...{key[-6:]}")
    
    # 3. Connect to Binance
    print("\n📡 Connecting to Binance Mainnet...")
    client = await AsyncClient.create(key, secret)
    
    try:
        # 4. Check Read Permissions (Get Account)
        print("   Test 1: Fetching Account Balances (READ)...")
        account = await client.get_account()
        balances = [b for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        print("   ✅ READ SUCCESS! Balances found:")
        for b in balances:
            print(f"      - {b['asset']}: Free={b['free']}, Locked={b['locked']}")

        # 4b. Check Open Orders (Read)
        print("   Test 1b: Fetching Open Orders...")
        orders = await client.get_open_orders(symbol="BTCUSDC")
        print(f"   ✅ OPEN ORDERS SUCCESS! Count: {len(orders)}")
            
    except BinanceAPIException as e:
        print(f"   ❌ READ FAILED: {e}")
        print("   STOPPING: Read permissions are required.")
        await client.close_connection()
        return

    try:
        # 5. Check Trade Permissions (Test Order)
        # We use a TEST order - it validates permissions but does not execute
        symbol = "BTCUSDC"
        print(f"\n   Test 2: Placing TEST Limit Order on {symbol} (WRITE)...")
        
        # Verify symbol exists first
        info = await client.get_symbol_info(symbol)
        if not info:
             print(f"   ⚠️ Symbol {symbol} not found, trying BTCUSDT")
             symbol = "BTCUSDT"

        try:
            await client.create_test_order(
                symbol=symbol,
                side='BUY',
                type='LIMIT',
                timeInForce='GTC',
                quantity=0.001,
                price=10000.0 # Arbitrary price
            )
            print("   ✅ WRITE SUCCESS! (Permissions are correct)")
        except BinanceAPIException as e:
            print(f"   ❌ TEST ORDER FAILED (Expected if price is low): {e}")

    except BinanceAPIException as e:
        print(f"   ❌ SETUP FAILED: {e}")

    # 6. Check User Stream (Listen Key)
    try:
        print("\n   Test 3: Requesting Listen Key (Spot User Stream)...")
        try:
            listen_key = await client.stream_get_listen_key()
            print(f"   ✅ LISTEN KEY SUCCESS! Key: {listen_key[:6]}...")
        except BinanceAPIException as e:
            print(f"   ❌ FAILED via Lib: {e}")
            
            # Test 4: Raw Request
            print("\n   Test 4: Raw HTTP POST to userDataStream...")
            url = "https://api.binance.com/api/v3/userDataStream"
            headers = {"X-MBX-APIKEY": key}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    ptext = await resp.text()
                    print(f"   Raw Status: {resp.status}")
                    print(f"   Raw Response: {ptext}")
                    if resp.status == 200:
                        print("   ✅ RAW SUCCESS! (Library is the problem)")
                    else:
                        print("   ❌ RAW FAILED! (Binance is rejecting this endpoint)")

        print("\n🎉 CONCLUSION: Check results above.")
        print("   If Raw Success, we need to patch the bot to use raw requests.")

    except BinanceAPIException as e:
        print(f"   ❌ FAILED: {e}")
        
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(verify())
