import sqlite3
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("Database")

class DatabaseManager:
    """
    High-Performance Non-Blocking Database Manager.
    
    Uses a Queue-Worker pattern to offload all DB writes to a background thread.
    The main trading loop never blocks on I/O.
    """
    
    def __init__(self, db_path="trading_data.db"):
        self.db_path = db_path
        self._write_queue = queue.Queue()
        self._running = True
        
        # Initialize DB (Tables) synchronously on startup
        self._init_db()
        
        # Start background worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="DB-Worker")
        self._worker_thread.start()
        logger.info("🚀 DB Background Worker started")

    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            # High timeout (30s) to wait for locks instead of crashing
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            
            # ENABLE WAL MODE (Crucial for concurrency)
            # Allows readers (Dashboard) to not block writers (Bot)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;") # Faster writes, safe for WAL
            
            cursor = conn.cursor()
            
            # Trades Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    commission REAL,
                    commission_asset TEXT,
                    is_maker BOOLEAN,
                    pnl REAL,
                    strategy_id TEXT
                )
            ''')
            
            # Trades Table (Historical Log)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    commission REAL,
                    commission_asset TEXT,
                    is_maker BOOLEAN,
                    pnl REAL,
                    strategy_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.critical(f"Failed to init DB: {e}")

    async def log_trade(self, fill_event, strategy_id="unknown"):
        """
        Non-blocking log trade.
        Just pushes to queue. Method is async for compatibility but returns instantly.
        """
        data = {
            'type': 'trade',
            'timestamp': fill_event.timestamp,
            'symbol': fill_event.symbol,
            'side': fill_event.side,
            'quantity': fill_event.quantity,
            'price': fill_event.price,
            'commission': fill_event.commission,
            'commission_asset': getattr(fill_event, 'commission_asset', 'USDT'),
            'is_maker': getattr(fill_event, 'is_maker', False),
            'pnl': fill_event.pnl,
            'strategy_id': strategy_id
        }
        self._write_queue.put(data)



    def _worker(self):
        """Background thread that consumes queue and writes to SQLite."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            while self._running or not self._write_queue.empty():
                try:
                    # Block for up to 1s waiting for items
                    task = self._write_queue.get(timeout=1.0)
                    
                    if task['type'] == 'trade':
                        cursor.execute('''
                            INSERT INTO trades (timestamp, symbol, side, quantity, price, commission, commission_asset, is_maker, pnl, strategy_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            task['timestamp'], task['symbol'], task['side'], task['quantity'],
                            task['price'], task['commission'], task['commission_asset'], task['is_maker'],
                            task['pnl'], task['strategy_id']
                        ))
                    
                    
                    # Batch commit optimization?
                    # For safety, we commit frequently in HFT context (crash resilience)
                    # But we could check queue size.
                    if self._write_queue.empty():
                        conn.commit()
                        
                    self._write_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"DB Worker Error: {e}")
                    
        finally:
            if conn:
                conn.close()
            logger.info("DB Worker stopped")

    def close(self):
        """Stop worker and flush queue."""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            
    def get_trades(self, limit=100):
        """Read latest trades (Synchronous - used by Dashboard)."""
        # Create NEW connection for read (SQLite allows multiple connections)
        conn = sqlite3.connect(self.db_path)
        import pandas as pd
        try:
            df = pd.read_sql_query(f"SELECT * FROM trades ORDER BY id DESC LIMIT {limit}", conn)
        except:
            df = pd.DataFrame()
        conn.close()
        return df