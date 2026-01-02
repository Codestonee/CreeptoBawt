"""
Write-Ahead Logging Manager - Crash recovery mechanism.

Logs operations to disk BEFORE executing them, enabling recovery
from crashes and ensuring no operations are lost.

Features:
- Sequential operation logging with sequence numbers
- Line-buffered writes with explicit flush
- WAL replay on startup to recover incomplete operations
- Automatic cleanup of completed operations
- Operation completion tracking
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

log = structlog.get_logger()


@dataclass
class Operation:
    """Represents an operation to be logged."""
    operation_type: str
    data: Dict[str, Any]
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "data": self.data,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "completed": self.completed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Operation:
        """Create from dictionary."""
        return cls(
            operation_type=data["operation_type"],
            data=data["data"],
            sequence=data.get("sequence", 0),
            timestamp=data.get("timestamp", time.time()),
            completed=data.get("completed", False),
        )


class WALManager:
    """
    Write-Ahead Logging manager for crash recovery.
    
    Logs critical operations before execution to enable recovery
    after system crashes or unexpected shutdowns.
    """
    
    def __init__(
        self,
        wal_path: str = "/var/lib/tradingbot/wal.log",
        enable_auto_cleanup: bool = True,
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ) -> None:
        """
        Initialize WAL manager.
        
        Args:
            wal_path: Path to WAL file
            enable_auto_cleanup: Whether to automatically cleanup completed operations
            cleanup_interval_seconds: How often to run cleanup
        """
        self.wal_path = Path(wal_path)
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_interval = cleanup_interval_seconds
        
        self._sequence = 0
        self._wal_file: Optional[Any] = None
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Ensure directory exists
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        
        log.info("wal_manager_initialized", wal_path=str(self.wal_path))
    
    async def start(self) -> None:
        """Start WAL manager and open log file."""
        if self._running:
            log.warning("wal_manager_already_running")
            return
        
        try:
            # Open WAL file in append mode with line buffering
            self._wal_file = open(self.wal_path, 'a', buffering=1)
            
            # Read existing WAL to determine next sequence number
            await self._initialize_sequence()
            
            self._running = True
            
            # Start auto-cleanup if enabled
            if self.enable_auto_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            log.info("wal_manager_started", next_sequence=self._sequence)
        
        except Exception as e:
            log.error("wal_manager_start_failed", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop WAL manager and close log file."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close WAL file
        if self._wal_file:
            self._wal_file.close()
            self._wal_file = None
        
        log.info("wal_manager_stopped")
    
    async def log_operation(self, operation: Operation) -> int:
        """
        Log an operation BEFORE executing it.
        
        Args:
            operation: Operation to log
            
        Returns:
            Sequence number assigned to the operation
        """
        if not self._running or not self._wal_file:
            raise RuntimeError("WAL manager not started")
        
        async with self._lock:
            # Assign sequence number
            operation.sequence = self.get_next_sequence()
            
            # Create WAL entry
            entry = {
                "timestamp": operation.timestamp,
                "operation": operation.operation_type,
                "data": operation.data,
                "sequence": operation.sequence,
                "completed": operation.completed,
            }
            
            # Write to WAL (line buffered, auto-flushes)
            self._wal_file.write(json.dumps(entry) + '\n')
            self._wal_file.flush()  # Explicit flush for safety
            
            log.debug(
                "operation_logged",
                operation=operation.operation_type,
                sequence=operation.sequence,
            )
            
            return operation.sequence
    
    async def mark_completed(self, sequence: int) -> None:
        """
        Mark an operation as completed.
        
        Args:
            sequence: Sequence number of the completed operation
        """
        if not self._running or not self._wal_file:
            raise RuntimeError("WAL manager not started")
        
        async with self._lock:
            entry = {
                "timestamp": time.time(),
                "operation": "mark_completed",
                "sequence": sequence,
                "completed": True,
            }
            
            self._wal_file.write(json.dumps(entry) + '\n')
            self._wal_file.flush()
            
            log.debug("operation_completed", sequence=sequence)
    
    def get_next_sequence(self) -> int:
        """Get the next sequence number."""
        self._sequence += 1
        return self._sequence
    
    async def replay_on_startup(self) -> List[Operation]:
        """
        Replay WAL on startup to recover incomplete operations.
        
        Returns:
            List of incomplete operations that need to be completed
        """
        incomplete_operations: List[Operation] = []
        
        if not self.wal_path.exists():
            log.info("wal_replay_skipped", reason="no_wal_file")
            return incomplete_operations
        
        try:
            # Read entire WAL
            with open(self.wal_path, 'r') as f:
                lines = f.readlines()
            
            # Parse operations
            operations: Dict[int, Operation] = {}
            completed_sequences: set[int] = set()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    if entry.get("operation") == "mark_completed":
                        # Mark sequence as completed
                        completed_sequences.add(entry["sequence"])
                    else:
                        # Store operation
                        op = Operation(
                            operation_type=entry["operation"],
                            data=entry.get("data", {}),
                            sequence=entry["sequence"],
                            timestamp=entry["timestamp"],
                            completed=entry.get("completed", False),
                        )
                        operations[op.sequence] = op
                
                except json.JSONDecodeError as e:
                    log.warning("wal_parse_error", line=line, error=str(e))
                    continue
            
            # Find incomplete operations
            for seq, op in operations.items():
                if seq not in completed_sequences and not op.completed:
                    incomplete_operations.append(op)
            
            if incomplete_operations:
                log.warning(
                    "incomplete_operations_found",
                    count=len(incomplete_operations),
                    sequences=[op.sequence for op in incomplete_operations],
                )
            else:
                log.info("wal_replay_complete", incomplete_count=0)
        
        except Exception as e:
            log.error("wal_replay_failed", error=str(e))
        
        return incomplete_operations
    
    async def is_operation_complete(self, sequence: int) -> bool:
        """
        Check if an operation is marked as complete.
        
        Args:
            sequence: Sequence number to check
            
        Returns:
            True if operation is complete
        """
        if not self.wal_path.exists():
            return False
        
        try:
            with open(self.wal_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        if (entry.get("operation") == "mark_completed" and 
                            entry.get("sequence") == sequence):
                            return True
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            log.error("operation_complete_check_failed", sequence=sequence, error=str(e))
        
        return False
    
    async def cleanup_completed(self) -> int:
        """
        Remove completed operations from WAL.
        
        Rewrites WAL file with only incomplete operations.
        
        Returns:
            Number of completed operations removed
        """
        if not self.wal_path.exists():
            return 0
        
        try:
            # Read current WAL
            with open(self.wal_path, 'r') as f:
                lines = f.readlines()
            
            # Parse and filter
            operations: Dict[int, str] = {}
            completed_sequences: set[int] = set()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    seq = entry.get("sequence")
                    
                    if entry.get("operation") == "mark_completed":
                        completed_sequences.add(seq)
                    else:
                        operations[seq] = line
                
                except json.JSONDecodeError:
                    continue
            
            # Count removals
            removed_count = len(completed_sequences)
            
            if removed_count == 0:
                return 0
            
            # Filter out completed operations
            incomplete_lines = [
                line for seq, line in operations.items()
                if seq not in completed_sequences
            ]
            
            # Write cleaned WAL to temporary file
            temp_path = self.wal_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                for line in incomplete_lines:
                    f.write(line + '\n')
            
            # Close current WAL file
            if self._wal_file:
                self._wal_file.close()
            
            # Replace with cleaned version
            temp_path.replace(self.wal_path)
            
            # Reopen WAL file
            if self._running:
                self._wal_file = open(self.wal_path, 'a', buffering=1)
            
            log.info("wal_cleanup_complete", removed=removed_count)
            return removed_count
        
        except Exception as e:
            log.error("wal_cleanup_failed", error=str(e))
            return 0
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_completed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("cleanup_loop_error", error=str(e))
    
    async def _initialize_sequence(self) -> None:
        """Initialize sequence number from existing WAL."""
        if not self.wal_path.exists():
            self._sequence = 0
            return
        
        try:
            max_sequence = 0
            with open(self.wal_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        seq = entry.get("sequence", 0)
                        max_sequence = max(max_sequence, seq)
                    except json.JSONDecodeError:
                        continue
            
            self._sequence = max_sequence
            log.info("sequence_initialized", max_sequence=max_sequence)
        
        except Exception as e:
            log.error("sequence_initialization_failed", error=str(e))
            self._sequence = 0
