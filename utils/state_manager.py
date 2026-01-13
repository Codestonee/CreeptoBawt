
import json
import os
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger("Utils.StateManager")

class StateManager:
    """
    Persist volatile strategy state (Analytics, VPIN, HMM history) to disk.
    
    Why:
    - OrderManager handles Orders/Positions (Critical State)
    - StateManager handles Analytics (Volatile State) to prevent "amnesia" on restart.
    
    Storage:
    - Uses JSON file (simple, portable, human-readable).
    - Redis support can be added if performance demands it (currently <1MB state).
    """
    
    def __init__(self, file_path: str = "strategy_state.json", auto_save_interval: int = 60):
        self.file_path = file_path
        self.auto_save_interval = auto_save_interval
        self._state: Dict[str, Any] = {}
        self._running = False
        self._save_task: Optional[asyncio.Task] = None
        
        # Load immediately on init
        self.load()

    def load(self):
        """Load state from disk."""
        if not os.path.exists(self.file_path):
            logger.info("No existing state file found. Starting fresh.")
            return

        try:
            with open(self.file_path, 'r') as f:
                self._state = json.load(f)
            logger.info(f"Loaded strategy state from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load state file: {e}")
            # Backup corrupted file
            try:
                os.rename(self.file_path, f"{self.file_path}.corrupted.{int(time.time())}")
            except:
                pass
            self._state = {}

    def save(self):
        """Save state to disk atomically."""
        try:
            temp_path = f"{self.file_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(self._state, f, indent=2)
            
            # Atomic rename (windows compatible-ish)
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            os.rename(temp_path, self.file_path)
            
            logger.debug("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def start_auto_save(self):
        """Start background auto-save loop."""
        self._running = True
        self._save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("Auto-save loop started")

    async def stop(self):
        """Stop auto-save and force final save."""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        self.save()
        logger.info("StateManager stopped")

    async def _auto_save_loop(self):
        while self._running:
            await asyncio.sleep(self.auto_save_interval)
            self.save()

    def get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        """Get state dict for a specific symbol."""
        return self._state.get(symbol.lower(), {})

    def update_symbol_state(self, symbol: str, data: Dict[str, Any]):
        """Update state for a symbol (merges with existing)."""
        symbol = symbol.lower()
        if symbol not in self._state:
            self._state[symbol] = {}
        self._state[symbol].update(data)
