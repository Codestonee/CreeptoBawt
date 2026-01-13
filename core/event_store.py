"""
Event Store - Append-only event log for capital-critical decisions.

Uses SQLite with WAL mode for crash recovery via event replay.
All capital mutations are logged BEFORE and AFTER execution.
"""

import sqlite3
import aiosqlite
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from enum import Enum

logger = logging.getLogger("Core.EventStore")


class EventType(str, Enum):
    """Types of capital-critical events."""
    ORDER_SUBMIT_INTENT = "ORDER_SUBMIT_INTENT"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_CANCELED = "ORDER_CANCELED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_UPDATE_INTENT = "POSITION_UPDATE_INTENT"
    POSITION_UPDATED = "POSITION_UPDATED"
    RECONCILIATION_START = "RECONCILIATION_START"
    RECONCILIATION_COMPLETE = "RECONCILIATION_COMPLETE"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"


@dataclass
class EventRecord:
    """Immutable event record for the event store."""
    event_id: str
    trace_id: str
    timestamp: float
    event_type: str
    payload: Dict[str, Any]
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None


class EventStore:
    """
    Append-only event log for all capital-critical decisions.
    
    Features:
    - SQLite with WAL mode for durability
    - Log BEFORE execution (intent) and AFTER (result)
    - Trace ID propagation for correlation
    - Enables crash recovery via event replay
    """
    
    def __init__(self, db_path: str = "event_store.db"):
        self.db_path = db_path
        self._init_db_sync()
        logger.info(f"EventStore initialized with WAL mode at {db_path}")
    
    def _init_db_sync(self):
        """Initialize database with WAL mode (synchronous for startup)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency and crash recovery
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")  # Balance speed and safety
        
        # Create events table (append-only)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                trace_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                pre_state TEXT,
                post_state TEXT,
                created_at REAL DEFAULT (unixepoch('now', 'subsec'))
            )
        ''')
        
        # Index for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_trace_id 
            ON events(trace_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_type_timestamp 
            ON events(event_type, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_trace_id(self) -> str:
        """Generate a new trace ID for correlation across events."""
        return str(uuid.uuid4())
    
    def generate_event_id(self) -> str:
        """Generate unique event ID."""
        return str(uuid.uuid4())
    
    async def append(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        trace_id: str,
        pre_state: Optional[Dict[str, Any]] = None,
        post_state: Optional[Dict[str, Any]] = None
    ) -> EventRecord:
        """
        Append an event to the store (atomic, durable).
        
        Args:
            event_type: Type of event being logged
            payload: Event-specific data
            trace_id: Correlation ID for this operation chain
            pre_state: State snapshot before operation (optional)
            post_state: State snapshot after operation (optional)
            
        Returns:
            The created EventRecord
        """
        event = EventRecord(
            event_id=self.generate_event_id(),
            trace_id=trace_id,
            timestamp=time.time(),
            event_type=event_type.value,
            payload=payload,
            pre_state=pre_state,
            post_state=post_state
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO events 
                (event_id, trace_id, timestamp, event_type, payload, pre_state, post_state)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.trace_id,
                event.timestamp,
                event.event_type,
                json.dumps(event.payload),
                json.dumps(event.pre_state) if event.pre_state else None,
                json.dumps(event.post_state) if event.post_state else None
            ))
            await db.commit()
        
        logger.debug(f"Event appended: {event.event_type} trace={event.trace_id[:8]}...")
        return event
    
    async def get_events_by_trace(self, trace_id: str) -> List[EventRecord]:
        """Get all events for a given trace ID (for debugging/audit)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                'SELECT * FROM events WHERE trace_id = ? ORDER BY timestamp',
                (trace_id,)
            )
            rows = await cursor.fetchall()
            
        return [self._row_to_event(row) for row in rows]
    
    async def get_events_since(
        self, 
        since_timestamp: float, 
        event_types: Optional[List[EventType]] = None
    ) -> List[EventRecord]:
        """
        Get events since a timestamp (for replay/recovery).
        
        Args:
            since_timestamp: Unix timestamp to start from
            event_types: Optional filter for specific event types
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if event_types:
                placeholders = ','.join('?' * len(event_types))
                cursor = await db.execute(
                    f'''SELECT * FROM events 
                        WHERE timestamp > ? AND event_type IN ({placeholders})
                        ORDER BY timestamp''',
                    (since_timestamp, *[et.value for et in event_types])
                )
            else:
                cursor = await db.execute(
                    'SELECT * FROM events WHERE timestamp > ? ORDER BY timestamp',
                    (since_timestamp,)
                )
            rows = await cursor.fetchall()
            
        return [self._row_to_event(row) for row in rows]
    
    async def get_last_event(self, event_type: Optional[EventType] = None) -> Optional[EventRecord]:
        """Get the most recent event, optionally filtered by type."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if event_type:
                cursor = await db.execute(
                    'SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT 1',
                    (event_type.value,)
                )
            else:
                cursor = await db.execute(
                    'SELECT * FROM events ORDER BY timestamp DESC LIMIT 1'
                )
            row = await cursor.fetchone()
            
        return self._row_to_event(row) if row else None
    
    def _row_to_event(self, row) -> EventRecord:
        """Convert database row to EventRecord."""
        return EventRecord(
            event_id=row['event_id'],
            trace_id=row['trace_id'],
            timestamp=row['timestamp'],
            event_type=row['event_type'],
            payload=json.loads(row['payload']),
            pre_state=json.loads(row['pre_state']) if row['pre_state'] else None,
            post_state=json.loads(row['post_state']) if row['post_state'] else None
        )
    
    async def log_order_intent(
        self,
        trace_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        pre_state: Optional[Dict] = None
    ) -> EventRecord:
        """Convenience method: Log order submission intent BEFORE sending to exchange."""
        return await self.append(
            event_type=EventType.ORDER_SUBMIT_INTENT,
            payload={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type
            },
            trace_id=trace_id,
            pre_state=pre_state
        )
    
    async def log_order_submitted(
        self,
        trace_id: str,
        client_order_id: str,
        exchange_order_id: Optional[str],
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_state: Optional[Dict] = None
    ) -> EventRecord:
        """Convenience method: Log successful order submission."""
        return await self.append(
            event_type=EventType.ORDER_SUBMITTED,
            payload={
                "client_order_id": client_order_id,
                "exchange_order_id": exchange_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price
            },
            trace_id=trace_id,
            post_state=post_state
        )
    
    async def log_order_filled(
        self,
        trace_id: str,
        client_order_id: str,
        exchange_order_id: str,
        symbol: str,
        side: str,
        filled_qty: float,
        fill_price: float,
        commission: float,
        post_state: Optional[Dict] = None
    ) -> EventRecord:
        """Convenience method: Log order fill."""
        return await self.append(
            event_type=EventType.ORDER_FILLED,
            payload={
                "client_order_id": client_order_id,
                "exchange_order_id": exchange_order_id,
                "symbol": symbol,
                "side": side,
                "filled_qty": filled_qty,
                "fill_price": fill_price,
                "commission": commission
            },
            trace_id=trace_id,
            post_state=post_state
        )


# Global event store instance
_event_store: Optional[EventStore] = None


def get_event_store() -> EventStore:
    """Get or create the global event store instance."""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store
