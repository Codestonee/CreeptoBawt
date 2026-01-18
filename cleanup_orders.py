"""Quick script to check balance and cancel all open orders."""
from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Check USDC balance
account = c.get_account()
usdc = [a for a in account['balances'] if a['asset'] == 'USDC'][0]
print(f"USDC Balance: Free={usdc['free']}, Locked={usdc['locked']}")

# List open orders
orders = c.get_open_orders()
print(f"\nOpen Orders: {len(orders)}")
for o in orders:
    print(f"  {o['symbol']} {o['side']} {o['origQty']} @ {o['price']}")

# Cancel all open orders
if orders:
    print("\n🚮 Canceling all open orders...")
    for o in orders:
        try:
            c.cancel_order(symbol=o['symbol'], orderId=o['orderId'])
            print(f"  ✅ Canceled {o['symbol']} {o['orderId']}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

# Recheck balance
account = c.get_account()
usdc = [a for a in account['balances'] if a['asset'] == 'USDC'][0]
print(f"\n✅ USDC After Cancel: Free={usdc['free']}, Locked={usdc['locked']}")
