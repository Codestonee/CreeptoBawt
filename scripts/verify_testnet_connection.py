
import asyncio
import os
import sys
import time
import hmac
import hashlib
import aiohttp
from urllib.parse import urlencode

# Add project root to path BEFORE imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from utils.logging_config import setup_logging

# Setup logging
setup_logging()

async def verify_connectivity():
    print("----------------------------------------------------------------")
    print("📡 Verifying Binance FUTURES Testnet Connectivity (Direct REST)...")
    print("----------------------------------------------------------------")
    
    api_key = settings.BINANCE_TESTNET_API_KEY
    api_secret = settings.BINANCE_TESTNET_SECRET_KEY
    base_url = "https://testnet.binancefuture.com"
    
    if not api_key or not api_secret:
        print("❌ FAILED: Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_SECRET_KEY in settings.")
        return

    print(f"🔑 API Key: {api_key[:4]}...{api_key[-4:]}")
    
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    async with aiohttp.ClientSession() as session:
        # 0. Sync Time
        time_offset = 0
        try:
            async with session.get(f"{base_url}/fapi/v1/time") as resp:
                if resp.status == 200:
                    server_data = await resp.json()
                    server_time = server_data['serverTime']
                    local_time = int(time.time() * 1000)
                    time_offset = server_time - local_time
                    print(f"⏱️ Time Sync: Local vs Server offset = {time_offset}ms")
                else:
                    print(f"⚠️ Failed to sync time: {resp.status}")
        except Exception as e:
            print(f"⚠️ Frequency Sync Error: {e}")

        # 1. Ping
        try:
            async with session.get(f"{base_url}/fapi/v1/ping") as resp:
                if resp.status == 200:
                    print("✅ Ping Successful (Public Endpoint)")
                else:
                    text = await resp.text()
                    print(f"❌ Ping Failed: {resp.status} - {text}")
                    return
        except Exception as e:
            print(f"❌ Network Error (Ping): {e}")
            return
            
        # 2. Account Info (Signed)
        try:
            endpoint = "/fapi/v2/account"
            # Apply offset
            timestamp = int(time.time() * 1000) + time_offset
            params = {
                'timestamp': timestamp,
                'recvWindow': 60000 # Increased recvWindow for resilience
            }
            
            # Sign
            query_string = urlencode(params)
            signature = hmac.new(
                api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
            full_url = f"{base_url}{endpoint}?{urlencode(params)}"
            
            async with session.get(full_url, headers=headers) as resp:
                data = await resp.json()
                
                if resp.status == 200:
                    print("✅ Authentication Successful (Futures Account Accessible)")
                    print(f"   Exchange: Binance Futures Testnet")
                    print(f"   Can Trade: {data.get('canTrade', False)}")
                    
                    print("\n💰 Futures Testnet Balances:")
                    has_funds = False
                    for asset in data.get('assets', []):
                        balance = float(asset.get('walletBalance', 0))
                        unrealized = float(asset.get('unrealizedProfit', 0))
                        if balance > 0:
                            print(f"   - {asset['asset']}: {balance:.4f} (Unrealized PnL: {unrealized:.4f})")
                            has_funds = True
                            
                    if not has_funds:
                        print("⚠️ WARNING: No funds found on Futures Testnet account.")
                    
                    print("\n✅ CONNECTIVITY VERIFIED")
                else:
                    print(f"❌ Account Check Failed: {data}")
                    
        except Exception as e:
            print(f"❌ Network/Signing Error: {e}")

if __name__ == "__main__":
    asyncio.run(verify_connectivity())
