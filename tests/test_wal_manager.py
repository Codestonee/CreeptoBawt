"""Tests for Write-Ahead Logging Manager."""
import pytest
import tempfile
import json
from pathlib import Path
from reconciliation.wal_manager import WALManager, Operation


class TestWALManager:
    """Tests for WAL Manager."""
    
    @pytest.fixture
    async def wal_manager(self):
        """Create a WAL manager with temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        manager = WALManager(
            wal_path=temp_path,
            enable_auto_cleanup=False,  # Disable for tests
        )
        
        await manager.start()
        
        yield manager
        
        await manager.stop()
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_log_operation(self, wal_manager):
        """Test logging an operation."""
        operation = Operation(
            operation_type="place_order",
            data={"symbol": "BTC-USDT", "quantity": "1.0"},
        )
        
        sequence = await wal_manager.log_operation(operation)
        
        assert sequence > 0
        assert operation.sequence == sequence
    
    @pytest.mark.asyncio
    async def test_sequence_increments(self, wal_manager):
        """Test that sequence numbers increment."""
        op1 = Operation(operation_type="order1", data={})
        op2 = Operation(operation_type="order2", data={})
        
        seq1 = await wal_manager.log_operation(op1)
        seq2 = await wal_manager.log_operation(op2)
        
        assert seq2 == seq1 + 1
    
    @pytest.mark.asyncio
    async def test_mark_completed(self, wal_manager):
        """Test marking operation as completed."""
        operation = Operation(
            operation_type="place_order",
            data={"symbol": "BTC-USDT"},
        )
        
        sequence = await wal_manager.log_operation(operation)
        await wal_manager.mark_completed(sequence)
        
        # Check that completion was logged
        is_complete = await wal_manager.is_operation_complete(sequence)
        assert is_complete is True
    
    @pytest.mark.asyncio
    async def test_replay_finds_incomplete_operations(self):
        """Test WAL replay finds incomplete operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        try:
            # Create manager and log operations
            manager1 = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            await manager1.start()
            
            op1 = Operation(operation_type="order1", data={"id": "1"})
            op2 = Operation(operation_type="order2", data={"id": "2"})
            op3 = Operation(operation_type="order3", data={"id": "3"})
            
            seq1 = await manager1.log_operation(op1)
            seq2 = await manager1.log_operation(op2)
            seq3 = await manager1.log_operation(op3)
            
            # Mark only op1 and op3 as completed
            await manager1.mark_completed(seq1)
            await manager1.mark_completed(seq3)
            
            await manager1.stop()
            
            # Create new manager and replay
            manager2 = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            incomplete = await manager2.replay_on_startup()
            
            # Should find op2 as incomplete
            assert len(incomplete) == 1
            assert incomplete[0].operation_type == "order2"
            assert incomplete[0].data["id"] == "2"
            assert incomplete[0].sequence == seq2
            
            await manager2.stop()
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_replay_empty_wal(self):
        """Test replay with no WAL file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.log') as f:
            temp_path = f.name
        
        # File doesn't exist
        manager = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
        incomplete = await manager.replay_on_startup()
        
        assert len(incomplete) == 0
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_cleanup_removes_completed(self):
        """Test cleanup removes completed operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        try:
            manager = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            await manager.start()
            
            # Log and complete operations
            op1 = Operation(operation_type="order1", data={})
            op2 = Operation(operation_type="order2", data={})
            op3 = Operation(operation_type="order3", data={})
            
            seq1 = await manager.log_operation(op1)
            seq2 = await manager.log_operation(op2)
            seq3 = await manager.log_operation(op3)
            
            await manager.mark_completed(seq1)
            await manager.mark_completed(seq2)
            # Leave op3 incomplete
            
            # Run cleanup
            removed = await manager.cleanup_completed()
            
            assert removed == 2
            
            # Stop and replay to verify
            await manager.stop()
            
            manager2 = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            incomplete = await manager2.replay_on_startup()
            
            # Should only find op3
            assert len(incomplete) == 1
            assert incomplete[0].operation_type == "order3"
            
            await manager2.stop()
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_wal_persists_across_restarts(self):
        """Test WAL data persists across manager restarts."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        try:
            # First manager session
            manager1 = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            await manager1.start()
            
            op = Operation(
                operation_type="place_order",
                data={"symbol": "BTC-USDT", "quantity": "1.0"},
            )
            sequence = await manager1.log_operation(op)
            
            await manager1.stop()
            
            # Second manager session
            manager2 = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            await manager2.start()
            
            # Check operation can be found
            is_complete = await manager2.is_operation_complete(sequence)
            assert is_complete is False  # Not marked complete
            
            incomplete = await manager2.replay_on_startup()
            assert len(incomplete) == 1
            assert incomplete[0].sequence == sequence
            
            await manager2.stop()
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_operation_to_from_dict(self):
        """Test Operation serialization."""
        op = Operation(
            operation_type="cancel_order",
            data={"order_id": "12345", "reason": "timeout"},
            sequence=42,
        )
        
        # Serialize
        data = op.to_dict()
        assert data["operation_type"] == "cancel_order"
        assert data["sequence"] == 42
        assert data["data"]["order_id"] == "12345"
        
        # Deserialize
        restored = Operation.from_dict(data)
        assert restored.operation_type == op.operation_type
        assert restored.sequence == op.sequence
        assert restored.data == op.data
    
    @pytest.mark.asyncio
    async def test_wal_file_format(self, wal_manager):
        """Test that WAL file is properly formatted JSON lines."""
        operation = Operation(
            operation_type="test_op",
            data={"key": "value"},
        )
        
        sequence = await wal_manager.log_operation(operation)
        await wal_manager.mark_completed(sequence)
        
        # Read WAL file directly
        with open(wal_manager.wal_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2  # One operation log + one completion log
        
        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert "timestamp" in data
            assert "operation" in data
            assert "sequence" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, wal_manager):
        """Test logging multiple operations concurrently."""
        import asyncio
        
        operations = [
            Operation(operation_type=f"order_{i}", data={"id": i})
            for i in range(10)
        ]
        
        # Log all operations concurrently
        sequences = await asyncio.gather(*[
            wal_manager.log_operation(op) for op in operations
        ])
        
        # All should have unique sequences
        assert len(set(sequences)) == 10
        
        # Sequences should be sequential
        sequences_sorted = sorted(sequences)
        for i in range(1, len(sequences_sorted)):
            assert sequences_sorted[i] == sequences_sorted[i-1] + 1
    
    @pytest.mark.asyncio
    async def test_is_operation_complete_for_incomplete(self, wal_manager):
        """Test checking completion status of incomplete operation."""
        operation = Operation(
            operation_type="test_op",
            data={},
        )
        
        sequence = await wal_manager.log_operation(operation)
        
        # Don't mark as completed
        is_complete = await wal_manager.is_operation_complete(sequence)
        assert is_complete is False
    
    @pytest.mark.asyncio
    async def test_get_next_sequence(self):
        """Test sequence number generation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        
        try:
            manager = WALManager(wal_path=temp_path, enable_auto_cleanup=False)
            await manager.start()
            
            # Get sequences
            seq1 = manager.get_next_sequence()
            seq2 = manager.get_next_sequence()
            seq3 = manager.get_next_sequence()
            
            assert seq1 == 1
            assert seq2 == 2
            assert seq3 == 3
            
            await manager.stop()
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
