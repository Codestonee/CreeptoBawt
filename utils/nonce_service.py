"""
Nonce Service - Centralized monotonic nonce generator with persistence.

Prevents race conditions on concurrent order submissions.
Binance requires monotonic timestamps per API key.
"""

import sqlite3
import aiosqlite
import asyncio
import logging
import time
import os
from typing import Optional
from threading import Lock

logger = logging.getLogger("Utils.NonceService")


class NonceService:
    """
    Centralized nonce generator for API request signing.
    
    Features:
    - Monotonic nonces per API key
    - Persistence across restarts
    - Thread-safe and async-safe
    - Prevents duplicate nonces on concurrent submits
    """
    
    def __init__(self, db_path: str = "nonce_store.db"):
        self.db_path = db_path
        self._lock = Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._last_nonce: int = 0
        self._init_db_sync()
        self._load_last_nonce()
    
    def _init_db_sync(self):
        """Initialize SQLite storage for nonce persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # WAL mode for performance
        cursor.execute("PRAGMA journal_mode=WAL")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nonces (
                api_key_hash TEXT PRIMARY KEY,
                last_nonce INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_last_nonce(self):
        """Load the last used nonce from persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT MAX(last_nonce) FROM nonces')
        result = cursor.fetchone()
        
        if result and result[0]:
            self._last_nonce = result[0]
        else:
            # Start from current time in milliseconds
            self._last_nonce = int(time.time() * 1000)
        
        conn.close()
        logger.info(f"NonceService initialized. Last nonce: {self._last_nonce}")
    
    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create async lock (must be in event loop context)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock
    
    def get_nonce_sync(self, api_key_hash: str = "default") -> int:
        """
        Get next monotonic nonce (synchronous, thread-safe).
        
        Args:
            api_key_hash: Hash of API key for multi-key support
            
        Returns:
            Monotonic nonce (millisecond timestamp, guaranteed increasing)
        """
        with self._lock:
            # Ensure monotonically increasing
            current_time_ms = int(time.time() * 1000)
            next_nonce = max(current_time_ms, self._last_nonce + 1)
            self._last_nonce = next_nonce
            
            # Persist (synchronous)
            self._persist_nonce_sync(api_key_hash, next_nonce)
            
            return next_nonce
    
    async def get_nonce(self, api_key_hash: str = "default") -> int:
        """
        Get next monotonic nonce (async, safe for concurrent use).
        
        Args:
            api_key_hash: Hash of API key for multi-key support
            
        Returns:
            Monotonic nonce (millisecond timestamp, guaranteed increasing)
        """
        async with self._get_async_lock():
            # Ensure monotonically increasing
            current_time_ms = int(time.time() * 1000)
            next_nonce = max(current_time_ms, self._last_nonce + 1)
            self._last_nonce = next_nonce
            
            # Persist asynchronously
            await self._persist_nonce_async(api_key_hash, next_nonce)
            
            return next_nonce
    
    def _persist_nonce_sync(self, api_key_hash: str, nonce: int):
        """Persist nonce to SQLite (synchronous)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO nonces (api_key_hash, last_nonce, updated_at)
                VALUES (?, ?, ?)
            ''', (api_key_hash, nonce, time.time()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist nonce: {e}")
    
    async def _persist_nonce_async(self, api_key_hash: str, nonce: int):
        """Persist nonce to SQLite (async)."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO nonces (api_key_hash, last_nonce, updated_at)
                    VALUES (?, ?, ?)
                ''', (api_key_hash, nonce, time.time()))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to persist nonce async: {e}")
    
    async def get_last_nonce(self, api_key_hash: str = "default") -> Optional[int]:
        """Get the last used nonce for an API key (for debugging)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                'SELECT last_nonce FROM nonces WHERE api_key_hash = ?',
                (api_key_hash,)
            )
            result = await cursor.fetchone()
            return result[0] if result else None
    
    def get_recv_window(self) -> int:
        """
        Get recommended recvWindow for Binance API.
        
        Increased to 10000ms per Gemini research to reduce -1007 errors.
        """
        return 10000  # 10 seconds - more tolerant of network latency


# Global nonce service instance
_nonce_service: Optional[NonceService] = None


def get_nonce_service() -> NonceService:
    """Get or create the global nonce service instance."""
    global _nonce_service
    if _nonce_service is None:
        _nonce_service = NonceService()
    return _nonce_service
