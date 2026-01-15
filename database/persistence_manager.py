"""
Centralized Data Persistence Manager

Handles all database operations with connection pooling,
retry logic, and async support.

Features:
- Connection pooling (avoid "database locked" errors)
- Automatic retry on conflicts
- Batch operations for performance
- Data validation
"""

import asyncio
import logging
import sqlite3
import aiosqlite
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

logger = logging.getLogger("Data.Persistence")


class PersistenceManager:
    """
    Async-first database manager with connection pooling.
    
    Fixes common issues:
    - "Database is locked" errors
    - Slow writes blocking reads
    - Transaction deadlocks
    """
    
    def __init__(
        self,
        db_path: str = "data/trading_data.db",
        pool_size: int = 5,
        timeout: float = 10.0
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.pool_size = pool_size
        self.timeout = timeout
        
        # Connection pool (simple implementation)
        self._pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection pool and create tables."""
        if self._initialized:
            return
        
        logger.info(f"Initializing database: {self.db_path}")
        
        # Create tables
        async with aiosqlite.connect(self.db_path, timeout=self.timeout) as db:
            await db.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
            await db.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            
            # Create tables
            await self._create_tables(db)
            await db.commit()
        
        # Initialize connection pool
        async with self._pool_lock:
            for _ in range(self.pool_size):
                conn = await aiosqlite.connect(self.db_path, timeout=self.timeout)
                await conn.execute("PRAGMA journal_mode=WAL")
                self._pool.append(conn)
        
        self._initialized = True
        logger.info(f"✅ Database initialized with {self.pool_size} connections")
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create all required tables."""
        
        # Trades table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                strategy_id TEXT,
                trace_id TEXT,
                is_maker BOOLEAN DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
            ON trades(timestamp DESC)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol 
            ON trades(symbol, timestamp DESC)
        """)
        
        # Positions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                unrealized_pnl REAL DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        
        # Orders table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                exchange_order_id TEXT,
                trace_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                time_in_force TEXT,
                quantity REAL NOT NULL,
                filled_quantity REAL DEFAULT 0,
                remainder_quantity REAL,
                price REAL,
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_orders_state 
            ON orders(state, updated_at DESC)
        """)
        
        # Events table (audit trail)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                trace_id TEXT,
                payload TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events(timestamp DESC)
        """)
        
        # Strategy state table (for persistence)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS strategy_state (
                symbol TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        
        logger.debug("Database tables created/verified")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a connection from the pool.
        
        Usage:
            async with persistence.get_connection() as db:
                await db.execute(...)
        """
        if not self._initialized:
            await self.initialize()
        
        conn = None
        async with self._pool_lock:
            if self._pool:
                conn = self._pool.pop()
        
        if conn is None:
            # Pool exhausted - create temporary connection
            conn = await aiosqlite.connect(self.db_path, timeout=self.timeout)
            await conn.execute("PRAGMA journal_mode=WAL")
        
        try:
            yield conn
        finally:
            # Return to pool
            async with self._pool_lock:
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                else:
                    await conn.close()
    
    async def insert_trade(self, trade_data: Dict[str, Any], max_retries: int = 3):
        """Insert trade with retry on lock."""
        
        for attempt in range(max_retries):
            try:
                async with self.get_connection() as db:
                    await db.execute("""
                        INSERT INTO trades (
                            timestamp, symbol, side, quantity, price,
                            commission, realized_pnl, strategy_id, trace_id, is_maker
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_data['timestamp'],
                        trade_data['symbol'],
                        trade_data['side'],
                        trade_data['quantity'],
                        trade_data['price'],
                        trade_data.get('commission', 0),
                        trade_data.get('realized_pnl', 0),
                        trade_data.get('strategy_id'),
                        trade_data.get('trace_id'),
                        trade_data.get('is_maker', False)
                    ))
                    await db.commit()
                    return
                    
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))
                    continue
                raise
    
    async def batch_insert_trades(self, trades: List[Dict[str, Any]]):
        """Insert multiple trades efficiently."""
        
        async with self.get_connection() as db:
            await db.executemany("""
                INSERT INTO trades (
                    timestamp, symbol, side, quantity, price,
                    commission, realized_pnl, strategy_id, trace_id, is_maker
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    t['timestamp'], t['symbol'], t['side'], t['quantity'],
                    t['price'], t.get('commission', 0), t.get('realized_pnl', 0),
                    t.get('strategy_id'), t.get('trace_id'), t.get('is_maker', False)
                )
                for t in trades
            ])
            await db.commit()
    
    async def upsert_position(self, position_data: Dict[str, Any]):
        """Insert or update position."""
        
        async with self.get_connection() as db:
            await db.execute("""
                INSERT INTO positions (symbol, quantity, avg_entry_price, unrealized_pnl, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_entry_price = excluded.avg_entry_price,
                    unrealized_pnl = excluded.unrealized_pnl,
                    updated_at = excluded.updated_at
            """, (
                position_data['symbol'],
                position_data['quantity'],
                position_data['avg_entry_price'],
                position_data.get('unrealized_pnl', 0),
                position_data['updated_at']
            ))
            await db.commit()
    
    async def close(self):
        """Close all connections."""
        async with self._pool_lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()
        
        logger.info("Database connections closed")


# Global instance
_persistence_manager: Optional[PersistenceManager] = None

def get_persistence_manager() -> PersistenceManager:
    """Get global persistence manager (singleton)."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager()
    return _persistence_manager
