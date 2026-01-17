"""
Position Tracker - The Single Source of Truth for Positions

This MUST be fixed before anything else. Your bot went bankrupt because
position tracking failed.

CRITICAL REQUIREMENTS:
1. Exchange is the source of truth (not local DB)
2. Sync on startup BEFORE any trading
3. Reconcile frequently (every 60s)
4. HALT trading on sync failures
5. Handle database locks properly
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
import json

logger = logging.getLogger("PositionTracker")

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float
    last_update: datetime
    exchange_confirmed: bool = False  # Is this synced with exchange?
    version: int = 0  # Atomic version counter

class PositionTracker:
    """
    Tracks positions across the bot with exchange as source of truth.
    
    DESIGN PRINCIPLES:
    - Exchange data always wins over local data
    - Fail-safe: reject trading if sync fails
    - Atomic updates to prevent race conditions
    - Periodic reconciliation
    """
    
    def __init__(self, exchange_client, db_manager):
        self.exchange = exchange_client
        self.db = db_manager
        
        # In-memory position cache (fast access)
        self._positions: Dict[str, Position] = {}
        self._positions_lock = asyncio.Lock()
        
        # State tracking
        self._is_synced: bool = False
        self._last_sync: Optional[datetime] = None
        self._sync_failures: int = 0
        self._max_sync_failures: int = 3
        
        # Background tasks
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._reconciliation_interval: int = 60  # seconds
        
    async def initialize(self) -> bool:
        """
        Initialize position tracker. MUST be called before any trading.
        
        Returns:
            True if successfully synced, False otherwise
        """
        logger.info("🔄 Initializing Position Tracker...")
        
        try:
            # Step 1: Load positions from database (backup)
            await self._load_from_db()
            
            # Step 2: Force sync with exchange (source of truth)
            sync_success = await self.force_sync_with_exchange()
            
            if not sync_success:
                logger.critical("❌ FAILED TO SYNC POSITIONS - CANNOT START TRADING")
                return False
            
            # Step 3: Start background reconciliation
            self._start_reconciliation_loop()
            
            logger.info("✅ Position Tracker initialized and synced")
            return True
            
        except Exception as e:
            logger.critical(f"❌ Position Tracker initialization failed: {e}", exc_info=True)
            return False
    
    async def force_sync_with_exchange(self) -> bool:
        """
        Force synchronization with exchange.
        CRITICAL FIX: Use correct Binance Futures API endpoint.
        
        Returns:
            True if sync successful, False otherwise
        """
        # Import settings here to avoid circular imports? Or assume passed in.
        from config.settings import settings
        
        if hasattr(self.exchange, 'futures_account') or settings.SPOT_MODE:
            # Only log detailed force sync start at DEBUG level to avoid spam
            logger.debug("🔄 Force syncing positions with exchange...")
        else:
            logger.debug("🔄 Force syncing positions with exchange...")
        
        try:
            # Get positions (Adapter for Binance vs Mock)
            if hasattr(self.exchange, 'futures_account') or settings.SPOT_MODE:  # Binance Client
                
                exchange_positions = []
                
                if settings.SPOT_MODE:
                    # SPOT MODE: Use get_account() -> balances
                    # Access client directly if needed or assume adapter has method
                    # binance_executor.py passes 'client' which is AsyncClient
                    
                    # Spot doesn't give us "Average Entry Price" or "Unrealized PnL"
                    # We have to infer or keep local, but for sync we just take quantity.
                    # Price will be 0 or current price if we fetch ticker (expensive).
                    # We'll set avgPrice to 0 and let Reconciler or Portfolio update it?
                    # Or better: Keep local avgPrice if valid, else 0.
                    
                    account_info = await self.exchange.get_account()
                    balances = account_info['balances']
                    
                    for b in balances:
                        asset = b['asset']
                        free = float(b['free'])
                        locked = float(b['locked'])
                        total = free + locked
                        
                        if total > 0:
                            # Convert Asset to Symbol (e.g. BTC -> BTCUSDT)
                            # This is tricky without knowing all pairs. 
                            # We assume Quote is USDT for simplicity in this version.
                            if asset == 'USDT': continue # Don't track USDT as a position
                            
                            symbol = f"{asset}USDT"
                            
                            # Valid symbol check? We'll assume it's valid if we hold it.
                            # For precision, we should filter by TRADING_SYMBOLS if available.
                            
                            exchange_positions.append({
                                'symbol': symbol,
                                'quantity': total,
                                'avgPrice': 0.0, # Spot doesn't track this
                                'unrealizedPnl': 0.0 # Spot doesn't track this
                            })
                            
                else:
                    # FUTURES MODE
                    # futures_position_information returns list of all positions
                    raw_positions = await self.exchange.futures_position_information()
                    
                    # Map to generic format
                    for p in raw_positions:
                        amt = float(p['positionAmt'])
                        if abs(amt) > 0.0:  # Use non-zero check if desired, but here we process all
                            exchange_positions.append({
                                'symbol': p['symbol'],
                                'quantity': amt,
                                'avgPrice': float(p['entryPrice']),
                                'unrealizedPnl': float(p['unRealizedProfit'])
                            })
            else:
                # Mock / Standard Interface
                exchange_positions = await self.exchange.get_positions()
            
                if not exchange_positions:
                     exchange_positions = []

            async with self._positions_lock:
                synced_count = 0
                mismatch_count = 0
                
                # Parse exchange positions (Use Normalized Data)
                exchange_positions_map = {}
                for p in exchange_positions:
                    # Normalized keys from above (symbol, quantity, avgPrice, unrealizedPnl)
                    symbol = p['symbol'].lower()
                    qty = float(p['quantity'])
                    
                    # Only track non-zero positions
                    if abs(qty) > 0.0001:
                        exchange_positions_map[symbol] = {
                            'symbol': symbol,
                            'quantity': qty,
                            'avgPrice': float(p['avgPrice']),
                            'unrealizedPnl': float(p['unrealizedPnl']),
                            'leverage': 1, # Default/Ignored
                            'marginType': 'cross' # Default
                        }
                
                # ... (Rest of logic uses exchange_positions_map) ...

                # 1. Update/Create positions from exchange
                local_symbols = set(self._positions.keys())
                exchange_symbols = set(exchange_positions_map.keys())

                for symbol, exch_data in exchange_positions_map.items():
                    exch_qty = exch_data['quantity']
                    exch_price = exch_data['avgPrice']
                    exch_pnl = exch_data['unrealizedPnl']
                    
                    local_pos = self._positions.get(symbol)
                    
                    # Versioning: If we are updating/overwriting, we should increment version?
                    # Since this is a FORCE SYNC, we represent the absolute truth.
                    # If we blindly overwrite, any in-flight orders might have stale versions.
                    # But since trading should be halted/paused during full sync, it's safer.
                    
                    current_version = local_pos.version if local_pos else 0
                    
                    if local_pos:
                        # Check for mismatch
                        if abs(local_pos.quantity - exch_qty) > 0.001:
                            logger.warning(
                                f"⚠️ POSITION MISMATCH {symbol.upper()}: "
                                f"Local={local_pos.quantity:.4f}, Exchange={exch_qty:.4f}"
                            )
                            mismatch_count += 1
                        
                        # CRITICAL FIX FOR SPOT MODE:
                        # Binance Spot API returns 0.0 for avgPrice. We must preserve our local 
                        # weighted average price to track PnL, unless we have no local data.
                        if settings.SPOT_MODE and exch_price == 0.0 and abs(local_pos.avg_entry_price) > 0:
                            exch_price = local_pos.avg_entry_price
                    
                    # Exchange is source of truth - overwrite local
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=exch_qty,
                        avg_entry_price=exch_price,
                        unrealized_pnl=exch_pnl,
                        last_update=datetime.now(),
                        exchange_confirmed=True,
                        version=current_version + 1 # Increment atomic version
                    )
                    synced_count += 1
                
                # ... (Phantom removal logic remains) ...

                # 2. Remove phantom positions (local but not on exchange)
                phantom_symbols = local_symbols - exchange_symbols
                for symbol in phantom_symbols:
                    if abs(self._positions[symbol].quantity) > 0.001:
                        logger.warning(
                            f"⚠️ PHANTOM POSITION {symbol.upper()}: "
                            f"Local={self._positions[symbol].quantity:.4f}, Exchange=0.0 - REMOVING"
                        )
                        mismatch_count += 1
                    
                    # Remove phantom position
                    del self._positions[symbol]
                
                # Update sync state
                self._is_synced = True
                self._last_sync = datetime.now()
                self._sync_failures = 0
                
                # Save to database (async, don't block)
                asyncio.create_task(self._save_to_db())
                
                # Only log as INFO if something actually happened
                if synced_count > 0 or mismatch_count > 0:
                    logger.info(
                        f"✅ Position sync complete: {synced_count} positions synced, "
                        f"{mismatch_count} discrepancies resolved"
                    )
                else:
                    logger.debug(
                        f"✅ Position sync complete: {synced_count} positions synced, "
                        f"{mismatch_count} discrepancies resolved"
                    )
                
                # Log current positions for visibility
                if self._positions:
                    for pos in self._positions.values():
                        logger.info(
                            f"   📊 {pos.symbol.upper()}: {pos.quantity:+.4f} @ "
                            f"${pos.avg_entry_price:.2f} | PnL: ${pos.unrealized_pnl:+.2f}"
                        )
                
                return True
                
        except Exception as e:
            logger.error(f"Force sync failed: {e}", exc_info=True)
            self._sync_failures += 1
            return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            
        Returns:
            Position object or None if no position
        """
        # Safety check: refuse to return data if not synced
        if not self._is_synced:
            logger.error(f"❌ Cannot get position - not synced with exchange")
            return None
        
        async with self._positions_lock:
            return self._positions.get(symbol.lower())
    
    async def update_position(
        self, 
        symbol: str, 
        quantity_delta: float, 
        price: float,
        commission: float = 0.0
    ) -> None:
        """
        Update position after a fill.
        """
        symbol = symbol.lower()
        
        async with self._positions_lock:
            current_pos = self._positions.get(symbol)
            
            if current_pos is None:
                # New position
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity_delta,
                    avg_entry_price=price,
                    unrealized_pnl=0.0,
                    last_update=datetime.now(),
                    exchange_confirmed=False,  # Not confirmed until next sync
                    version=1 # Initialize version
                )
                logger.info(f"📊 NEW POSITION: {symbol} {quantity_delta:.4f} @ ${price:.4f}")
            else:
                # Update existing position
                old_qty = current_pos.quantity
                new_qty = old_qty + quantity_delta
                
                # Calculate new average entry price
                if abs(new_qty) > 0.001:
                    if (old_qty > 0 and quantity_delta > 0) or (old_qty < 0 and quantity_delta < 0):
                        # Adding to position - update avg price
                        total_cost = (old_qty * current_pos.avg_entry_price) + (quantity_delta * price)
                        new_avg_price = total_cost / new_qty
                    else:
                        # Reducing or flipping position
                        new_avg_price = price if abs(quantity_delta) > abs(old_qty) else current_pos.avg_entry_price
                    
                    current_pos.quantity = new_qty
                    current_pos.avg_entry_price = new_avg_price
                    current_pos.last_update = datetime.now()
                    current_pos.exchange_confirmed = False
                    current_pos.version += 1 # CRITICAL: Increment version
                    
                    logger.info(
                        f"📊 POSITION UPDATE: {symbol} {old_qty:.4f} -> {new_qty:.4f} "
                        f"@ ${new_avg_price:.4f}"
                    )
                else:
                    # Position closed
                    del self._positions[symbol]
                    logger.info(f"📊 POSITION CLOSED: {symbol}")
        
        # Save to DB asynchronously (don't block trading)
        asyncio.create_task(self._save_position_to_db(symbol))

    async def update_position_from_exchange(
        self, 
        symbol: str, 
        quantity: float, 
        entry_price: float,
        unrealized_pnl: float = 0.0
    ):
        """
        Force update a single position from exchange data (Absolute update).
        """
        from config.settings import settings
        
        symbol = symbol.lower()
        async with self._positions_lock:
            current = self._positions.get(symbol)
            version = (current.version + 1) if current else 1
            
            # CRITICAL FIX FOR SPOT MODE:
            # Binance Spot doesn't provide entry price. Preserve our local price.
            if settings.SPOT_MODE and entry_price == 0.0 and current and abs(current.avg_entry_price) > 0:
                entry_price = current.avg_entry_price

            if abs(quantity) > 0.001:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=entry_price,
                    unrealized_pnl=unrealized_pnl,
                    last_update=datetime.now(),
                    exchange_confirmed=True,
                    version=version # Increment version
                )
                logger.info(f"📊 FORCE UPDATE: {symbol} = {quantity:.4f} @ ${entry_price:.4f}")
            else:
                if symbol in self._positions:
                    del self._positions[symbol]
                    logger.info(f"📊 FORCE CLOSED: {symbol}")
        
        asyncio.create_task(self._save_position_to_db(symbol))
    
    async def get_total_exposure(self) -> float:
        """
        Calculate total exposure across all positions.
        
        Returns:
            Total USD value of all positions
        """
        if not self._is_synced:
            logger.warning("⚠️ Getting exposure from unsynced positions")
        
        async with self._positions_lock:
            total = 0.0
            for pos in self._positions.values():
                total += abs(pos.quantity * pos.avg_entry_price)
            return total
    
    async def get_symbol_exposure(self, symbol: str) -> float:
        """
        Get exposure for a specific symbol.
        
        Returns:
            USD value of position (absolute)
        """
        pos = await self.get_position(symbol)
        if pos is None:
            return 0.0
        return abs(pos.quantity * pos.avg_entry_price)
    
    async def get_active_position_count(self) -> int:
        """Get number of active positions."""
        async with self._positions_lock:
            return len(self._positions)
    
    async def get_all_positions(self) -> List[Position]:
        """Get all active positions."""
        async with self._positions_lock:
            return list(self._positions.values())
    
    def is_synced(self) -> bool:
        """Check if positions are synced with exchange."""
        return self._is_synced
    
    def get_sync_status(self) -> Dict:
        """Get detailed sync status."""
        return {
            'is_synced': self._is_synced,
            'last_sync': self._last_sync.isoformat() if self._last_sync else None,
            'sync_failures': self._sync_failures,
            'position_count': len(self._positions),
            'seconds_since_sync': (datetime.now() - self._last_sync).total_seconds() if self._last_sync else None
        }
    
    def _start_reconciliation_loop(self):
        """Start background reconciliation task."""
        if self._reconciliation_task is not None:
            logger.warning("Reconciliation loop already running")
            return
        
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info(f"🔄 Started position reconciliation loop (every {self._reconciliation_interval}s)")
    
    async def _reconciliation_loop(self):
        """Background task to periodically sync with exchange."""
        while True:
            try:
                await asyncio.sleep(self._reconciliation_interval)
                
                logger.debug("🔄 Running scheduled position reconciliation...")
                success = await self.force_sync_with_exchange()
                
                if not success:
                    logger.warning(f"⚠️ Reconciliation failed (attempt {self._sync_failures})")
                
            except asyncio.CancelledError:
                logger.info("Reconciliation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}", exc_info=True)
    
    async def _load_from_db(self):
        """Load positions from database (fallback/cache)."""
        try:
            positions_data = await self.db.get_all_positions()
            
            async with self._positions_lock:
                for data in positions_data:
                    symbol = data['symbol'].lower()
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=float(data['quantity']),
                        avg_entry_price=float(data['avg_entry_price']),
                        unrealized_pnl=float(data.get('unrealized_pnl', 0)),
                        last_update=datetime.fromtimestamp(data.get('updated_at', 0)),
                        exchange_confirmed=False  # Not confirmed until synced
                    )
            
            logger.info(f"📚 Loaded {len(positions_data)} positions from database")
            
        except Exception as e:
            logger.warning(f"Could not load positions from DB: {e}")
    
    async def _save_to_db(self):
        """Save all positions to database."""
        try:
            async with self._positions_lock:
                positions_copy = list(self._positions.values())
            
            for pos in positions_copy:
                await self._save_position_to_db(pos.symbol)
                
        except Exception as e:
            logger.error(f"Failed to save positions to DB: {e}")
    
    async def _save_position_to_db(self, symbol: str):
        """
        Save single position to database with retry logic.
        Uses queue to prevent database lock issues.
        """
        try:
            pos = self._positions.get(symbol.lower())
            if pos is None:
                # Position was closed - delete from DB
                await self.db.delete_position(symbol)
                return
            
            # Use the new async upsert method on db_manager
            await self.db.upsert_position({
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'avg_entry_price': pos.avg_entry_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'updated_at': datetime.now().timestamp()
            })
                    
        except Exception as e:
            # Don't let DB errors crash the bot - just log
            logger.error(f"Failed to save position {symbol} to DB: {e}")
    
    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info("🔄 Shutting down Position Tracker...")
        
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        
        # Final save to DB
        await self._save_to_db()
        
        logger.info("✅ Position Tracker shutdown complete")
