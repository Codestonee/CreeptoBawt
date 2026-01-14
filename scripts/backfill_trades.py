import sqlite3
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backfill")

DB_PATH = "data/trading_data.db"

def backfill_trades():
    logger.info("Starting trade backfill...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Get latest trade timestamp
        cursor.execute("SELECT MAX(timestamp) as last_ts FROM trades")
        res = cursor.fetchone()
        last_trade_ts = res['last_ts'] if res and res['last_ts'] else 0
        
        logger.info(f"Last trade timestamp in DB: {last_trade_ts}")

        # 2. Find FILLED orders strictly AFTER this timestamp
        # We look for orders that are FILLED and were active
        cursor.execute("""
            SELECT * FROM orders 
            WHERE state = 'FILLED' 
            AND updated_at > ?
            ORDER BY updated_at ASC
        """, (last_trade_ts,))
        
        missing_orders = cursor.fetchall()
        
        if not missing_orders:
            logger.info("✅ No missing trades found.")
            return

        logger.info(f"⚠️ Found {len(missing_orders)} filled orders missing from trades table.")

        count = 0
        for order in missing_orders:
            # Check if already exists (double check by client_order_id if possible, 
            # but trades table doesn't have client_order_id, so we rely on timestamp/symbol match 
            # or just trust the logical gap)
            
            # Map order to trade
            trade_data = {
                'timestamp': order['updated_at'], # Use update time as trade time
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['filled_quantity'],
                'price': order['avg_fill_price'],
                'commission': order['commission'],
                'commission_asset': 'USDT',
                'is_maker': 0, # Assume taker for now
                'pnl': order['pnl'],
                'strategy_id': 'backfill'
            }

            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, quantity, price, commission, commission_asset, is_maker, pnl, strategy_id)
                VALUES (:timestamp, :symbol, :side, :quantity, :price, :commission, :commission_asset, :is_maker, :pnl, :strategy_id)
            ''', trade_data)
            
            count += 1
            if count % 10 == 0:
                print(f"Restored {count} trades...")

        conn.commit()
        logger.info(f"✅ Successfully restored {count} trades to the dashboard.")

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    backfill_trades()
