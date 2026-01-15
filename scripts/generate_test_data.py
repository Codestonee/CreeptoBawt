"""Generate test trading data for dashboard development"""

import sqlite3
import time
import random
import os
from datetime import datetime, timezone

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

DB_PATH = "data/trading_data.db"

def generate_test_trades(n=50):
    """Generate realistic test trades"""
    
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing tables to ensure schema match
    cursor.execute("DROP TABLE IF EXISTS trades")
    cursor.execute("DROP TABLE IF EXISTS positions")
    
    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        symbol TEXT,
        side TEXT,
        quantity REAL,
        price REAL,
        commission REAL,
        realized_pnl REAL,
        strategy_id TEXT,
        trace_id TEXT,
        is_maker BOOLEAN,
        created_at REAL
    )
    """)
    
    # Create positions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        quantity REAL,
        avg_entry_price REAL,
        unrealized_pnl REAL,
        updated_at REAL
    )
    """)
    
    symbols = ['btcusdt', 'ethusdt', 'solusdt']
    current_time = int(time.time())
    
    print(f"Generating {n} test trades...")
    
    for i in range(n):
        symbol = random.choice(symbols)
        side = random.choice(['BUY', 'SELL'])
        quantity = round(random.uniform(0.001, 0.1), 4)
        
        # Realistic prices
        if 'btc' in symbol:
            price = round(random.uniform(95000, 100000), 2)
        elif 'eth' in symbol:
            price = round(random.uniform(3000, 3500), 2)
        else:
            price = round(random.uniform(180, 220), 2)
        
        # P&L with 60% win rate
        if random.random() < 0.6:
            pnl = round(random.uniform(0.5, 15.0), 2)  # Winner
        else:
            pnl = round(random.uniform(-12.0, -0.3), 2)  # Loser
        
        commission = round(quantity * price * 0.0004, 2)
        timestamp = current_time - (n - i) * 300  # 5 min apart
        
        cursor.execute("""
        INSERT INTO trades (timestamp, symbol, side, quantity, price, realized_pnl, commission)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, side, quantity, price, pnl, commission))
    
    # Add current positions
    cursor.execute("""
    INSERT OR REPLACE INTO positions 
    (symbol, quantity, avg_entry_price, unrealized_pnl, updated_at)
    VALUES 
    ('btcusdt', 0.05, 98500.0, 35.0, ?),
    ('ethusdt', -0.5, 3250.0, 25.0, ?)
    """, (current_time, current_time))
    
    conn.commit()
    conn.close()
    
    print("✅ Test data generated!")
    print(f"   Database: {DB_PATH}")
    print(f"   Trades: {n}")
    print(f"   Positions: 2")
    print("\nRefresh your dashboard to see data!")

if __name__ == "__main__":
    generate_test_trades(50)
