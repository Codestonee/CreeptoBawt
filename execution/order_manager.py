"""
Order Manager - State machine for order lifecycle with atomic transactions.

Provides ACID guarantees for order + position updates.
"""

import sqlite3
import aiosqlite
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from decimal import Decimal

from core.event_store import get_event_store, EventType

logger = logging.getLogger("Execution.OrderManager")


class OrderState(str, Enum):
    """Order lifecycle states."""
    PENDING_SUBMIT = "PENDING_SUBMIT"   # About to submit to exchange
    SUBMITTED = "SUBMITTED"             # Confirmed on exchange
    PARTIAL_FILL = "PARTIAL_FILL"       # Partially filled
    FILLED = "FILLED"                   # Fully filled
    CANCELED = "CANCELED"               # Canceled by user or system
    REJECTED = "REJECTED"               # Rejected by exchange
    EXPIRED = "EXPIRED"                 # Time-in-force expired


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Post Only


@dataclass
class Order:
    """Order record with full lifecycle tracking."""
    id: int = 0                          # Database ID
    client_order_id: str = ""            # Our ID (for idempotency)
    exchange_order_id: Optional[str] = None  # Exchange ID
    trace_id: str = ""                   # Correlation ID
    symbol: str = ""
    side: str = ""
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"
    quantity: float = 0.0
    filled_quantity: float = 0.0
    remainder_quantity: float = 0.0       # For partial fills
    price: float = 0.0
    avg_fill_price: float = 0.0
    state: str = OrderState.PENDING_SUBMIT.value
    commission: float = 0.0
    pnl: float = 0.0
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"c_{uuid.uuid4().hex[:16]}"
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        self.remainder_quantity = self.quantity - self.filled_quantity


# Valid state transitions
VALID_TRANSITIONS = {
    OrderState.PENDING_SUBMIT: [OrderState.SUBMITTED, OrderState.REJECTED],
    OrderState.SUBMITTED: [OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED],
    OrderState.PARTIAL_FILL: [OrderState.PARTIAL_FILL, OrderState.FILLED, OrderState.CANCELED],
    OrderState.FILLED: [],  # Terminal
    OrderState.CANCELED: [],  # Terminal
    OrderState.REJECTED: [],  # Terminal
    OrderState.EXPIRED: [],  # Terminal
}


@dataclass
class Position:
    """Position record with persistence."""
    symbol: str
    quantity: float = 0.0               # Positive = long, negative = short
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    last_reconciled_at: float = 0.0
    exchange_snapshot: Optional[str] = None  # JSON blob of exchange state
    updated_at: float = field(default_factory=time.time)


class OrderManager:
    """
    Order lifecycle manager with ACID transactions.
    
    Features:
    - State machine for order lifecycle
    - Atomic transactions for order + position updates
    - Tracks pending_orders_value for pre-trade checks
    - Handles partial fills and remainders
    - Integration with EventStore for crash recovery
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self._init_db_sync()
        self._event_store = get_event_store()
        logger.info("OrderManager initialized")
    
    def _init_db_sync(self):
        """Initialize order and position tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enable WAL for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Orders table
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
        
        # Positions table
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
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_orders_symbol_state 
            ON orders(symbol, state)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_orders_trace 
            ON orders(trace_id)
        ''')
        
        conn.commit()
        conn.close()
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        trace_id: Optional[str] = None
    ) -> Order:
        """
        Create a new order in PENDING_SUBMIT state.
        
        This is step 1 of order submission - must call submit_order() after.
        """
        order = Order(
            symbol=symbol.lower(),
            side=side.upper(),
            order_type=order_type,
            time_in_force=time_in_force,
            quantity=quantity,
            remainder_quantity=quantity,
            price=price,
            state=OrderState.PENDING_SUBMIT.value,
            trace_id=trace_id or str(uuid.uuid4())
        )
        
        # Log intent to event store BEFORE any action
        await self._event_store.log_order_intent(
            trace_id=order.trace_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type,
            pre_state=await self._get_position_snapshot(order.symbol)
        )
        
        # Persist order
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                INSERT INTO orders 
                (client_order_id, trace_id, symbol, side, order_type, time_in_force,
                 quantity, filled_quantity, remainder_quantity, price, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
            ''', (
                order.client_order_id,
                order.trace_id,
                order.symbol,
                order.side,
                order.order_type,
                order.time_in_force,
                order.quantity,
                order.quantity,  # remainder = quantity initially
                order.price,
                order.state,
                order.created_at,
                order.updated_at
            ))
            order.id = cursor.lastrowid
            await db.commit()
        
        logger.debug(f"Order created: {order.client_order_id[:8]}... {order.side} {order.quantity} {order.symbol.upper()}")
        return order
    
    async def mark_submitted(
        self,
        client_order_id: str,
        exchange_order_id: str
    ) -> Order:
        """Mark order as successfully submitted to exchange."""
        return await self._update_state(
            client_order_id,
            OrderState.SUBMITTED,
            exchange_order_id=exchange_order_id
        )
    
    async def mark_rejected(
        self,
        client_order_id: str,
        error_message: str
    ) -> Order:
        """Mark order as rejected by exchange.
        
        If order is already in a terminal state (FILLED, CANCELED, etc.),
        log a warning and return the order without error. This handles
        race conditions where fills arrive before rejection messages.
        """
        # Check if order is already in a terminal state
        order = await self.get_order(client_order_id)
        if order and order.state in [OrderState.FILLED.value, OrderState.CANCELED.value, 
                                      OrderState.REJECTED.value, OrderState.EXPIRED.value]:
            logger.warning(
                f"Ignoring rejection for terminal order {client_order_id[:8]}... "
                f"(already {order.state}): {error_message}"
            )
            return order
        
        return await self._update_state(
            client_order_id,
            OrderState.REJECTED,
            error_message=error_message
        )
    
    async def process_fill(
        self,
        client_order_id: str,
        filled_qty: float,
        fill_price: float,
        commission: float = 0.0
    ) -> Order:
        """
        Process a fill event with ATOMIC position update.
        
        Uses BEGIN IMMEDIATE to ensure atomicity across order and position updates.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # BEGIN IMMEDIATE for write lock
            await db.execute("BEGIN IMMEDIATE")
            
            try:
                # 1. Get current order
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    'SELECT * FROM orders WHERE client_order_id = ?',
                    (client_order_id,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    await db.rollback()
                    raise ValueError(f"Order not found: {client_order_id}")
                
                order = self._row_to_order(row)
                
                # 2. Update order state
                new_filled = order.filled_quantity + filled_qty
                new_remainder = order.quantity - new_filled
                
                # Calculate weighted average fill price
                if order.filled_quantity > 0:
                    total_value = (order.avg_fill_price * order.filled_quantity) + (fill_price * filled_qty)
                    new_avg_price = total_value / new_filled
                else:
                    new_avg_price = fill_price
                
                # Determine new state
                if new_filled >= order.quantity:
                    new_state = OrderState.FILLED.value
                    new_remainder = 0
                else:
                    new_state = OrderState.PARTIAL_FILL.value
                
                # Update order
                await db.execute('''
                    UPDATE orders SET
                        filled_quantity = ?,
                        remainder_quantity = ?,
                        avg_fill_price = ?,
                        commission = commission + ?,
                        state = ?,
                        updated_at = ?
                    WHERE client_order_id = ?
                ''', (
                    new_filled,
                    new_remainder,
                    new_avg_price,
                    commission,
                    new_state,
                    time.time(),
                    client_order_id
                ))
                
                # 3. Update position ATOMICALLY
                await self._update_position_atomic(
                    db,
                    order.symbol,
                    order.side,
                    filled_qty,
                    fill_price
                )
                
                # 4. Commit transaction
                await db.commit()
                
                # 5. Log to event store
                await self._event_store.log_order_filled(
                    trace_id=order.trace_id,
                    client_order_id=client_order_id,
                    exchange_order_id=order.exchange_order_id or "",
                    symbol=order.symbol,
                    side=order.side,
                    filled_qty=filled_qty,
                    fill_price=fill_price,
                    commission=commission,
                    post_state=await self._get_position_snapshot(order.symbol)
                )
                
                order.filled_quantity = new_filled
                order.remainder_quantity = new_remainder
                order.avg_fill_price = new_avg_price
                order.state = new_state
                
                logger.info(
                    f"Fill processed: {client_order_id} "
                    f"+{filled_qty}@{fill_price} ({new_filled}/{order.quantity})"
                )
                
                return order
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Fill processing failed, rolled back: {e}")
                raise
    
    async def _update_position_atomic(
        self,
        db: aiosqlite.Connection,
        symbol: str,
        side: str,
        qty: float,
        price: float
    ):
        """Update position within an existing transaction."""
        # Get current position
        cursor = await db.execute(
            'SELECT * FROM positions WHERE symbol = ?',
            (symbol,)
        )
        row = await cursor.fetchone()
        
        if row:
            current_qty = row['quantity']
            current_avg = row['avg_entry_price']
        else:
            current_qty = 0.0
            current_avg = 0.0
        
        # Calculate new position
        if side == "BUY":
            # Adding to long or reducing short
            if current_qty >= 0:
                # Adding to long - weighted average
                total_cost = (current_qty * current_avg) + (qty * price)
                new_qty = current_qty + qty
                new_avg = total_cost / new_qty if new_qty > 0 else 0
            else:
                # Reducing short
                new_qty = current_qty + qty
                new_avg = current_avg if new_qty != 0 else 0
        else:
            # SELL - Adding to short or reducing long
            if current_qty <= 0:
                # Adding to short - weighted average
                total_cost = (abs(current_qty) * current_avg) + (qty * price)
                new_qty = current_qty - qty
                new_avg = total_cost / abs(new_qty) if new_qty != 0 else 0
            else:
                # Reducing long
                new_qty = current_qty - qty
                new_avg = current_avg if new_qty != 0 else 0
        
        # Upsert position
        await db.execute('''
            INSERT INTO positions (symbol, quantity, avg_entry_price, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = ?,
                avg_entry_price = ?,
                updated_at = ?
        ''', (
            symbol, new_qty, new_avg, time.time(),
            new_qty, new_avg, time.time()
        ))
    
    async def _update_state(
        self,
        client_order_id: str,
        new_state: OrderState,
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Order:
        """Update order state with validation."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                'SELECT * FROM orders WHERE client_order_id = ?',
                (client_order_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                raise ValueError(f"Order not found: {client_order_id}")
            
            order = self._row_to_order(row)
            current_state = OrderState(order.state)
            
            # Validate transition
            valid_next = VALID_TRANSITIONS.get(current_state, [])
            if new_state not in valid_next:
                raise ValueError(
                    f"Invalid state transition: {current_state} -> {new_state}"
                )
            
            # Update
            updates = ["state = ?", "updated_at = ?"]
            params = [new_state.value, time.time()]
            
            if exchange_order_id:
                updates.append("exchange_order_id = ?")
                params.append(exchange_order_id)
            
            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)
            
            params.append(client_order_id)
            
            await db.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE client_order_id = ?",
                params
            )
            await db.commit()
            
            order.state = new_state.value
            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
            
            logger.debug(f"Order {client_order_id[:8]}...: {current_state.value} → {new_state.value}")
            return order
    
    async def cancel_order(self, client_order_id: str) -> Order:
        """Cancel an open order."""
        return await self._update_state(client_order_id, OrderState.CANCELED)
    
    async def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client_order_id."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                'SELECT * FROM orders WHERE client_order_id = ?',
                (client_order_id,)
            )
            row = await cursor.fetchone()
            return self._row_to_order(row) if row else None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open (non-terminal) orders."""
        open_states = (
            OrderState.PENDING_SUBMIT.value,
            OrderState.SUBMITTED.value,
            OrderState.PARTIAL_FILL.value
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if symbol:
                cursor = await db.execute(
                    'SELECT * FROM orders WHERE symbol = ? AND state IN (?, ?, ?)',
                    (symbol.lower(), *open_states)
                )
            else:
                cursor = await db.execute(
                    'SELECT * FROM orders WHERE state IN (?, ?, ?)',
                    open_states
                )
            
            rows = await cursor.fetchall()
            return [self._row_to_order(row) for row in rows]
    
    async def get_pending_orders_value(self, symbol: Optional[str] = None) -> float:
        """
        Get total value of pending orders.
        
        Critical for pre-trade validation - must check pending + new order < margin.
        """
        open_orders = await self.get_open_orders(symbol)
        total = 0.0
        
        for order in open_orders:
            # Value of unfilled portion
            unfilled = order.quantity - order.filled_quantity
            total += unfilled * order.price
        
        return total
    
    async def get_position(self, symbol: str) -> Position:
        """Get current position for a symbol."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                'SELECT * FROM positions WHERE symbol = ?',
                (symbol.lower(),)
            )
            row = await cursor.fetchone()
            
            if row:
                return Position(
                    symbol=row['symbol'],
                    quantity=row['quantity'],
                    avg_entry_price=row['avg_entry_price'],
                    unrealized_pnl=row['unrealized_pnl'] or 0,
                    last_reconciled_at=row['last_reconciled_at'] or 0,
                    exchange_snapshot=row['exchange_snapshot'],
                    updated_at=row['updated_at']
                )
            else:
                return Position(symbol=symbol.lower())
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('SELECT * FROM positions')
            rows = await cursor.fetchall()
            
            return {
                row['symbol']: Position(
                    symbol=row['symbol'],
                    quantity=row['quantity'],
                    avg_entry_price=row['avg_entry_price'],
                    unrealized_pnl=row['unrealized_pnl'] or 0,
                    last_reconciled_at=row['last_reconciled_at'] or 0,
                    updated_at=row['updated_at']
                )
                for row in rows
            }
    
    async def set_position_from_exchange(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        exchange_snapshot: str = "",
        unrealized_pnl: float = 0.0
    ) -> Position:
        """
        Force-set position from exchange data (reconciliation sync).
        
        Overwrites local position with exchange truth. Use for:
        - Manual sync after position mismatch
        - Initial state bootstrap
        - Recovery from corrupted local state
        """
        symbol_lower = symbol.lower()
        now = time.time()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO positions (symbol, quantity, avg_entry_price, unrealized_pnl, last_reconciled_at, exchange_snapshot, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity = ?,
                    avg_entry_price = ?,
                    unrealized_pnl = ?,
                    last_reconciled_at = ?,
                    exchange_snapshot = ?,
                    updated_at = ?
            ''', (
                symbol_lower, quantity, entry_price, unrealized_pnl, now, exchange_snapshot, now,
                quantity, entry_price, unrealized_pnl, now, exchange_snapshot, now
            ))
            await db.commit()
        
        logger.info(f"Position force-synced from exchange: {symbol_lower} qty={quantity} price={entry_price} upnl={unrealized_pnl:.2f}")
        
        return Position(
            symbol=symbol_lower,
            quantity=quantity,
            avg_entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
            last_reconciled_at=now,
            exchange_snapshot=exchange_snapshot,
            updated_at=now
        )
    
    async def clear_all_positions(self) -> int:
        """
        Clear all local positions (reset to zero).
        
        Returns number of positions cleared.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('DELETE FROM positions')
            count = cursor.rowcount
            await db.commit()
        
        logger.warning(f"Cleared {count} positions from local storage")
        return count
    
    async def _get_position_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get position as dict for event store."""
        pos = await self.get_position(symbol)
        return {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_entry_price": pos.avg_entry_price
        }
    
    async def register_existing_order(
        self,
        client_order_id: str,
        exchange_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
        filled_quantity: float = 0.0
    ) -> Order:
        """
        Register an existing exchange order in the local database.
        
        Used during bootstrap to pre-register orders that already exist
        on the exchange, preventing them from being flagged as orphans.
        
        If order already exists in DB, this is a no-op.
        """
        symbol_lower = symbol.lower()
        
        # Check if order already exists
        existing = await self.get_order(client_order_id)
        if existing:
            logger.debug(f"Order {client_order_id[:8]}... already registered")
            return existing
        
        # Determine state based on fill
        if filled_quantity >= quantity:
            state = OrderState.FILLED.value
        elif filled_quantity > 0:
            state = OrderState.PARTIAL_FILL.value
        else:
            state = OrderState.SUBMITTED.value
        
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            symbol=symbol_lower,
            side=side.upper(),
            order_type=order_type,
            time_in_force="GTC",
            quantity=quantity,
            filled_quantity=filled_quantity,
            remainder_quantity=quantity - filled_quantity,
            price=price,
            state=state,
            trace_id=str(uuid.uuid4())
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR IGNORE INTO orders 
                (client_order_id, exchange_order_id, trace_id, symbol, side, order_type, 
                 time_in_force, quantity, filled_quantity, remainder_quantity, price, 
                 state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.client_order_id,
                order.exchange_order_id,
                order.trace_id,
                order.symbol,
                order.side,
                order.order_type,
                order.time_in_force,
                order.quantity,
                order.filled_quantity,
                order.remainder_quantity,
                order.price,
                order.state,
                order.created_at,
                order.updated_at
            ))
            await db.commit()
        
        logger.debug(f"📥 Registered existing order: {client_order_id[:8]}... {side} {quantity} {symbol_lower}")
        return order
    
    def _row_to_order(self, row) -> Order:
        """Convert database row to Order."""
        return Order(
            id=row['id'],
            client_order_id=row['client_order_id'],
            exchange_order_id=row['exchange_order_id'],
            trace_id=row['trace_id'],
            symbol=row['symbol'],
            side=row['side'],
            order_type=row['order_type'],
            time_in_force=row['time_in_force'],
            quantity=row['quantity'],
            filled_quantity=row['filled_quantity'],
            remainder_quantity=row['remainder_quantity'],
            price=row['price'] or 0,
            avg_fill_price=row['avg_fill_price'],
            state=row['state'],
            commission=row['commission'],
            pnl=row['pnl'],
            error_message=row['error_message'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )


# Global instance
_order_manager: Optional[OrderManager] = None


def get_order_manager() -> OrderManager:
    """Get or create global OrderManager instance."""
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager
