import sqlite3
import logging
import queue
import threading
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("Database")

class DatabaseManager:
    """
    High-Performance Non-Blocking Database Manager.
    
    Uses a Queue-Worker pattern to offload all DB writes to a background thread.
    The main trading loop never blocks on I/O.
    """
    
    def __init__(self, db_path="data/trading_data.db"):
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
            conn.execute("PRAGMA temp_store=MEMORY;")
            
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
            
            # Orders Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_order_id TEXT UNIQUE NOT NULL,
                    exchange_order_id TEXT,
                    trace_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    time_in_force TEXT DEFAULT 'GTC',
                    quantity REAL NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    remainder_quantity REAL DEFAULT 0,
                    price REAL,
                    avg_fill_price REAL DEFAULT 0,
                    state TEXT NOT NULL,
                    commission REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    error_message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Positions Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL DEFAULT 0,
                    avg_entry_price REAL NOT NULL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    last_reconciled_at REAL DEFAULT 0,
                    exchange_snapshot TEXT,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol_state ON orders(symbol, state)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_trace ON orders(trace_id)')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.critical(f"Failed to init DB: {e}")

    async def log_trade(self, fill_event, strategy_id="unknown"):
        """Non-blocking log trade."""
        data = {
            'type': 'trade',
            'timestamp': getattr(fill_event, 'timestamp', time.time()),
            'symbol': fill_event.symbol,
            'side': fill_event.side,
            'quantity': fill_event.quantity,
            'price': fill_event.price,
            'commission': getattr(fill_event, 'commission', 0.0),
            'commission_asset': getattr(fill_event, 'commission_asset', 'USDT'),
            'is_maker': getattr(fill_event, 'is_maker', False),
            'pnl': getattr(fill_event, 'pnl', 0.0),
            'strategy_id': strategy_id
        }
        self.submit_write_task(data)

    async def upsert_position(self, data: dict):
        """Queue a position upsert."""
        task = {'type': 'upsert_position', 'data': data}
        self.submit_write_task(task)

    async def insert_order(self, data: dict):
        """Queue an order insert."""
        task = {'type': 'insert_order', 'data': data}
        self.submit_write_task(task)
        
    async def update_order(self, data: dict):
        """Queue an order update."""
        task = {'type': 'update_order', 'data': data}
        self.submit_write_task(task)
        
    async def delete_position(self, symbol: str):
        """Queue a position deletion."""
        task = {'type': 'delete_position', 'symbol': symbol}
        self.submit_write_task(task)

    def submit_write_task(self, task: dict):
        """Submit a task to the write queue."""
        self._write_queue.put(task)

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
                    
                    try:
                        if task['type'] == 'trade':
                            # Handle different data structures safely
                            if 'data' in task:
                                # New format if wrapped
                                d = task['data']
                            else:
                                # Legacy/Direct format from log_trade
                                d = task
                                
                            cursor.execute('''
                                INSERT INTO trades (timestamp, symbol, side, quantity, price, commission, commission_asset, is_maker, pnl, strategy_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                d.get('timestamp'), d.get('symbol'), d.get('side'), d.get('quantity'),
                                d.get('price'), d.get('commission'), d.get('commission_asset'), d.get('is_maker'),
                                d.get('pnl'), d.get('strategy_id')
                            ))
                        
                        elif task['type'] == 'upsert_position':
                            d = task['data']
                            cursor.execute('''
                                INSERT INTO positions (symbol, quantity, avg_entry_price, unrealized_pnl, updated_at)
                                VALUES (?, ?, ?, ?, ?)
                                ON CONFLICT(symbol) DO UPDATE SET
                                    quantity = excluded.quantity,
                                    avg_entry_price = excluded.avg_entry_price,
                                    unrealized_pnl = excluded.unrealized_pnl,
                                    updated_at = excluded.updated_at
                            ''', (
                                d['symbol'], d['quantity'], d['avg_entry_price'], 
                                d.get('unrealized_pnl', 0), d['updated_at']
                            ))

                        elif task['type'] == 'delete_position':
                            cursor.execute('DELETE FROM positions WHERE symbol = ?', (task['symbol'],))

                        elif task['type'] == 'insert_order':
                            d = task['data']
                            cursor.execute('''
                                INSERT INTO orders 
                                (client_order_id, trace_id, symbol, side, order_type, time_in_force,
                                 quantity, filled_quantity, remainder_quantity, price, state, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                d['client_order_id'], d['trace_id'], d['symbol'], d['side'],
                                d['order_type'], d['time_in_force'], d['quantity'],
                                d.get('filled_quantity', 0), d.get('remainder_quantity', d['quantity']),
                                d['price'], d['state'], d['created_at'], d['updated_at']
                            ))
                            
                        elif task['type'] == 'update_order':
                            d = task['data']
                            # Dynamic update construction
                            fields = []
                            values = []
                            for k, v in d.items():
                                if k != 'client_order_id':
                                    fields.append(f"{k} = ?")
                                    values.append(v)
                            
                            if fields:
                                values.append(d['client_order_id'])
                                cursor.execute(f'''
                                    UPDATE orders SET {', '.join(fields)}
                                    WHERE client_order_id = ?
                                ''', values)
                                
                    except Exception as e:
                        logger.error(f"DB Write Error for task {task.get('type')}: {e}")
                    
                    # Batch commit optimization
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
        """Read latest trades (Synchronous - used by Dashboard) with Retry Logic."""
        import pandas as pd
        
        # Retry up to 3 times
        for i in range(3):
            conn = None
            try:
                # Use URI for Read-Only mode if possible to avoid locking
                # timeout=10s to wait for bot to finish writing
                db_uri = f"file:{self.db_path}?mode=ro"
                conn = sqlite3.connect(db_uri, uri=True, timeout=10.0)
                
                query = f"SELECT * FROM trades ORDER BY id DESC LIMIT {limit}"
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df
                
            except Exception as e:
                logger.warning(f"Dashboard Read Attempt {i+1} failed: {e}")
                if conn:
                    conn.close()
                time.sleep(0.5) # Wait before retry
                
        logger.error("Failed to read trades after 3 retries")
        return pd.DataFrame()

    # --- Read Methods (Can be direct or async, here using run_in_executor for async compat) ---
    
    async def get_all_positions(self):
        """Read all positions (direct read)."""
        # WAL mode allows concurrent reads
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._read_all_positions_sync)

    def _read_all_positions_sync(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM positions")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows