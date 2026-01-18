import sqlite3

conn = sqlite3.connect('data/trading_data.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in c.fetchall()])

# Check positions
try:
    c.execute("SELECT * FROM positions")
    rows = c.fetchall()
    print(f"\nPositions ({len(rows)} rows):")
    for row in rows:
        print(f"  {dict(row)}")
except Exception as e:
    print(f"Error reading positions: {e}")

conn.close()
