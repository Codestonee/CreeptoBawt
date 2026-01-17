import sqlite3
import pandas as pd
from tabulate import tabulate

def analyze_trades():
    db_path = 'data/trading_data.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get schema
        print("\n=== Trades Table Schema ===")
        cursor.execute("PRAGMA table_info(trades);")
        columns = cursor.fetchall()
        for col in columns:
            print(col)
            
        # Try to select all columns to see what's actually there
        print("\n=== Recent Trades (All Columns) ===")
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 5", conn)
        if not df.empty:
            print(tabulate(df, headers='keys', tablefmt='psql'))
        else:
            print("No trades found.")

    except Exception as e:
        print(f"Error accessing database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    analyze_trades()
