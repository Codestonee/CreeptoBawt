import sqlite3
import sys
import os

DB_PATH = "data/trading_data.db"

def check_db():
    print(f"🔍 Checking DB at {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print("❌ DB File not found!")
        return
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check Trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]
        print(f"✅ Trades Count: {trade_count}")
        
        # Check Positions
        cursor.execute("SELECT COUNT(*) FROM positions")
        pos_count = cursor.fetchone()[0]
        print(f"✅ Positions Count: {pos_count}")
        
        # Show Last 3 Trades
        cursor.execute("SELECT id, symbol, side, quantity, price, timestamp FROM trades ORDER BY id DESC LIMIT 3")
        rows = cursor.fetchall()
        print("\n📝 Last 3 Trades:")
        for r in rows:
            print(f"   ID: {r[0]} | {r[1]} {r[2]} {r[3]} @ {r[4]}")
            
        conn.close()
        print("\n✅ DB Integrity Check Passed.")
        
    except Exception as e:
        print(f"❌ DB Check Failed: {e}")

if __name__ == "__main__":
    check_db()
