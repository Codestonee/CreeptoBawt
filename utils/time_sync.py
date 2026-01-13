"""
Time Sync Service - NTP validation and exchange time offset tracking.

Critical for API signing - Binance rejects requests with timestamp drift > recvWindow.
"""

import asyncio
import aiohttp
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

logger = logging.getLogger("Utils.TimeSync")


@dataclass
class TimeSyncStatus:
    """Current time synchronization status."""
    system_time_ms: int
    exchange_time_ms: int
    offset_ms: int  # exchange_time - system_time
    last_sync: float
    is_synced: bool
    warning: Optional[str] = None


class TimeSyncService:
    """
    NTP-style time sync validation against exchange servers.
    
    Features:
    - Check system vs exchange time offset on startup
    - Periodic drift monitoring (every 60s)
    - Auto-pause trading if drift exceeds threshold
    - Offset correction for API requests
    """
    
    # Thresholds (increased for testnet stability)
    MAX_ACCEPTABLE_DRIFT_MS = 1500  # Warning only if > 1500ms (was 500 - too strict)
    CRITICAL_DRIFT_MS = 3000        # Pause trading if > 3000ms
    SYNC_INTERVAL_SECONDS = 60
    
    def __init__(self, exchange_time_url: str = "https://fapi.binance.com/fapi/v1/time"):
        self.exchange_time_url = exchange_time_url
        self._offset_ms: int = 0
        self._last_sync: float = 0
        self._is_synced: bool = False
        self._sync_task: Optional[asyncio.Task] = None
        self._on_drift_critical: Optional[Callable[[], Awaitable[None]]] = None
    
    def set_critical_drift_callback(self, callback: Callable[[], Awaitable[None]]):
        """Set callback to invoke when drift becomes critical."""
        self._on_drift_critical = callback
    
    async def sync_once(self) -> TimeSyncStatus:
        """
        Perform a single time sync with the exchange.
        
        Returns:
            TimeSyncStatus with current offset and sync status
        """
        system_time_before = int(time.time() * 1000)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.exchange_time_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        exchange_time = data.get("serverTime", 0)
                    else:
                        logger.error(f"Failed to get exchange time: HTTP {response.status}")
                        return self._create_error_status("HTTP error")
        except asyncio.TimeoutError:
            logger.error("Exchange time request timed out")
            return self._create_error_status("Timeout")
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")
            return self._create_error_status(str(e))
        
        system_time_after = int(time.time() * 1000)
        
        # Estimate system time at exchange response (midpoint)
        system_time_mid = (system_time_before + system_time_after) // 2
        round_trip = system_time_after - system_time_before
        
        # Calculate offset (exchange - system)
        self._offset_ms = exchange_time - system_time_mid
        self._last_sync = time.time()
        
        # Determine status
        abs_offset = abs(self._offset_ms)
        warning = None
        
        if abs_offset > self.CRITICAL_DRIFT_MS:
            self._is_synced = False
            warning = f"CRITICAL: Clock drift {abs_offset}ms exceeds {self.CRITICAL_DRIFT_MS}ms threshold!"
            logger.critical(warning)
            if self._on_drift_critical:
                asyncio.create_task(self._on_drift_critical())
        elif abs_offset > self.MAX_ACCEPTABLE_DRIFT_MS:
            self._is_synced = True  # Still functional but warn
            warning = f"WARNING: Clock drift {abs_offset}ms exceeds {self.MAX_ACCEPTABLE_DRIFT_MS}ms"
            logger.warning(warning)
        else:
            self._is_synced = True
            logger.info(f"Time sync OK. Offset: {self._offset_ms}ms, RTT: {round_trip}ms")
        
        return TimeSyncStatus(
            system_time_ms=system_time_mid,
            exchange_time_ms=exchange_time,
            offset_ms=self._offset_ms,
            last_sync=self._last_sync,
            is_synced=self._is_synced,
            warning=warning
        )
    
    def _create_error_status(self, error: str) -> TimeSyncStatus:
        """Create an error status when sync fails."""
        now = int(time.time() * 1000)
        return TimeSyncStatus(
            system_time_ms=now,
            exchange_time_ms=0,
            offset_ms=self._offset_ms,  # Keep old offset
            last_sync=self._last_sync,
            is_synced=False,
            warning=f"Sync failed: {error}"
        )
    
    async def start_periodic_sync(self):
        """Start background periodic sync task."""
        if self._sync_task is not None:
            logger.warning("Periodic sync already running")
            return
        
        # Initial sync with retry
        max_retries = 3
        for attempt in range(max_retries):
            status = await self.sync_once()
            if status.is_synced:
                break
            logger.warning(f"Time sync attempt {attempt + 1}/{max_retries} failed, offset: {status.offset_ms}ms")
            await asyncio.sleep(1)
        
        # Even if drift is high, continue with warning (don't block trading)
        if not status.is_synced:
            logger.warning(f"Time sync has high drift ({status.offset_ms}ms) but continuing anyway")
            self._is_synced = True  # Force synced to allow trading
        
        # Start periodic task
        self._sync_task = asyncio.create_task(self._periodic_sync_loop())
        logger.info("Started periodic time sync")
    
    async def _periodic_sync_loop(self):
        """Background loop for periodic sync."""
        while True:
            try:
                await asyncio.sleep(self.SYNC_INTERVAL_SECONDS)
                await self.sync_once()
            except asyncio.CancelledError:
                logger.info("Periodic sync cancelled")
                break
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")
    
    async def stop(self):
        """Stop periodic sync."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
    
    def get_corrected_timestamp(self) -> int:
        """
        Get corrected timestamp for API requests.
        
        Returns:
            Timestamp in milliseconds, adjusted for exchange offset
        """
        return int(time.time() * 1000) + self._offset_ms
    
    @property
    def offset_ms(self) -> int:
        """Current offset (exchange_time - system_time) in milliseconds."""
        return self._offset_ms
    
    @property
    def is_synced(self) -> bool:
        """Whether time is currently synced within acceptable limits."""
        return self._is_synced
    
    @property
    def is_stale(self) -> bool:
        """Whether sync is stale (> 2x sync interval since last sync)."""
        return (time.time() - self._last_sync) > (self.SYNC_INTERVAL_SECONDS * 2)
    
    def get_status(self) -> TimeSyncStatus:
        """Get current sync status without performing a sync."""
        now_ms = int(time.time() * 1000)
        return TimeSyncStatus(
            system_time_ms=now_ms,
            exchange_time_ms=now_ms + self._offset_ms,
            offset_ms=self._offset_ms,
            last_sync=self._last_sync,
            is_synced=self._is_synced and not self.is_stale,
            warning="Sync data stale" if self.is_stale else None
        )


# Testnet URL for development
BINANCE_TESTNET_TIME_URL = "https://testnet.binancefuture.com/fapi/v1/time"

# Global instance
_time_sync: Optional[TimeSyncService] = None


def get_time_sync_service(testnet: bool = True) -> TimeSyncService:
    """Get or create the global time sync service."""
    global _time_sync
    if _time_sync is None:
        url = BINANCE_TESTNET_TIME_URL if testnet else "https://fapi.binance.com/fapi/v1/time"
        _time_sync = TimeSyncService(url)
    return _time_sync
