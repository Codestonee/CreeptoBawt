import asyncio
import aiohttp
import hmac
import base64
import hashlib
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_KEY = os.getenv("OKX_API_KEY")
API_SECRET = os.getenv("OKX_SECRET_KEY")
PASSPHRASE = os.getenv("OKX_PASSPHRASE")

if not API_KEY or not API_SECRET:
    print("❌ Keys missing in .env")
    sys.exit(1)

print(f"🔑 Using Credentials (first 4): Key={API_KEY[:4]}... Secret={API_SECRET[:4]}... Pass={PASSPHRASE[:4 if PASSPHRASE else 0]}...")

def get_header(method, path, is_simulated="1"):
    dt = datetime.utcnow()
    # OKX ISO Format
    timestamp = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    message = f"{timestamp}{method}{path}"
    mac = hmac.new(bytes(API_SECRET, encoding='utf8'), bytes(message, encoding='utf8'), digestmod=hashlib.sha256)
    sign = base64.b64encode(mac.digest()).decode()
    
    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json",
        "x-simulated-trading": is_simulated
    }

async def check_connection():
    url = "https://www.okx.com"
    path = "/api/v5/account/balance?ccy=USDT"
    
    async with aiohttp.ClientSession() as session:
        # TEST 1: TESTNET (Simulated)
        print("\n📡 TEST 1: Connecting to TESTNET (x-simulated-trading: 1)...")
        headers = get_header("GET", path, is_simulated="1")
        async with session.get(url + path, headers=headers) as resp:
            data = await resp.json()
            if resp.status == 200 and data.get('code') == '0':
                print("✅ TESTNET SUCCESS!")
                print(f"   Balance Data: {data['data'][0]['details'][0]['cashBal']} USDT")
            else:
                print(f"❌ TESTNET FAILED: {resp.status}")
                print(f"   Response: {data}")

        # TEST 2: MAINNET (Real)
        print("\n📡 TEST 2: Connecting to MAINNET (x-simulated-trading: 0)...")
        headers = get_header("GET", path, is_simulated="0")
        async with session.get(url + path, headers=headers) as resp:
            data = await resp.json()
            if resp.status == 200 and data.get('code') == '0':
                print("✅ MAINNET SUCCESS!")
                print(f"   Balance Data: {data['data'][0]['details'][0]['cashBal']} USDT")
            else:
                print(f"❌ MAINNET FAILED: {resp.status}")
                print(f"   Response: {data}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_connection())
