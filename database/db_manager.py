import sqlite3
import aiosqlite
import logging
from datetime import datetime

logger = logging.getLogger("Database")

class DatabaseManager:
    def __init__(self, db_path="trading_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Skapar tabeller om de inte finns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabell för trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                commission REAL,
                pnl REAL,
                strategy_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    async def log_trade(self, fill_event, strategy_id="unknown"):
        """Sparar en trade till databasen (Asynkront)."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO trades (timestamp, symbol, side, quantity, price, commission, pnl, strategy_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fill_event.timestamp,
                    fill_event.symbol,
                    fill_event.side,
                    fill_event.quantity,
                    fill_event.price,
                    fill_event.commission,
                    fill_event.pnl,
                    strategy_id
                ))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to log trade to DB: {e}")

    def get_trades(self, limit=100):
        """Hämtar senaste trades för dashboarden."""
        conn = sqlite3.connect(self.db_path)
        import pandas as pd
        df = pd.read_sql_query(f"SELECT * FROM trades ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        return df