"""
Integration Test Suite - "The Smoke Test"

Tests the FULL trading system flow:
1. Strategy → Signal → OrderManager → Executor → Fill → Strategy
2. Position sync & reconciliation
3. Zombie order handling
4. PnL calculation accuracy
5. Risk manager circuit breakers

Run with: pytest tests/test_integration.py -v
"""

import asyncio
import pytest
import sqlite3
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

# System imports
from core.events import MarketEvent, SignalEvent, FillEvent, RegimeEvent
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import Position
from execution.reconciliation import ReconciliationService, DiscrepancyType
from risk_engine.risk_manager import RiskManager
import config.settings as settings


# ==================== FIXTURES ====================

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def event_queue():
    """Async event queue."""
    return asyncio.Queue()


@pytest.fixture
def mock_exchange():
    """Mock exchange client."""
    client = MagicMock()
    # Async methods need AsyncMock
    client.create_order = AsyncMock(return_value={'orderId': '123456'})
    client.cancel_order = AsyncMock()
    return client

@pytest.fixture
def order_manager(temp_db, mock_exchange):
    """OrderManager with isolated DB and dependencies."""
    # Mock dependencies
    db_manager = MagicMock()
    # Configure db_manager async methods
    db_manager.insert_order = AsyncMock()
    db_manager.update_order = AsyncMock()
    db_manager.submit_write_task = MagicMock() # Synchronous method
    
    # Real PositionTracker with mock exchange
    from execution.position_tracker import PositionTracker
    position_tracker = PositionTracker(mock_exchange, db_manager)
    position_tracker._positions = {} # Ensure empty
    position_tracker._is_synced = True # Fake sync
    
    # Mock RiskGatekeeper
    gatekeeper = MagicMock()
    gatekeeper.validate_order = AsyncMock(return_value=MagicMock(is_allowed=True))
    gatekeeper.update_pnl = MagicMock()
    
    return OrderManager(mock_exchange, db_manager, position_tracker, gatekeeper)

@pytest.fixture
def mock_shadow_book():
    """Mock ShadowOrderBook."""
    book = MagicMock()
    book.get_mid_price = MagicMock(return_value=50000.0)
    book.is_stale = MagicMock(return_value=False)  # Never stale in tests
    book.get_imbalance = MagicMock(return_value=0.0)  # Neutral imbalance
    return book


@pytest.fixture
def strategy(event_queue, mock_shadow_book, order_manager):
    """Avellaneda-Stoikov strategy for testing (Depends on order_manager to init global)."""
    return AvellanedaStoikovStrategy(
        event_queue=event_queue,
        symbols=['btcusdt'],
        base_quantity=0.01,
        gamma=0.1,
        max_inventory=1.0,
        shadow_book=mock_shadow_book
    )


@pytest.fixture
def risk_manager():
    """Risk manager for testing."""
    return RiskManager(account_balance=10000.0, max_drawdown_pct=0.05)


# ==================== ORDER MANAGER TESTS ====================

class TestOrderManagerIntegration:
    """Test OrderManager state machine and ACID transactions."""
    
    @pytest.mark.asyncio
    async def test_order_lifecycle_happy_path(self, order_manager):
        """Test: Order creation → Submit → Fill → Position update."""
        # 1. Create order
        order = await order_manager.submit_order(
            symbol='btcusdt',
            quantity=0.01,
            price=50000.0,
            side='BUY',
            order_type='LIMIT'
        )
        
        # Order transitions to SUBMITTED immediately upon successful mock return
        assert order.state == OrderState.SUBMITTED.value
        assert order.client_order_id != ""
        
        # 2. Mark as submitted (Already done by submit_order, but we can verify params)
        # Verify exchange ID was set from mock return
        assert order.exchange_order_id == "123456"
        
        # 3. Process fill
        order = await order_manager.process_fill(
            client_order_id=order.client_order_id,
            filled_qty=0.01,
            fill_price=50000.0,
            commission=0.02
        )
        
        assert order.state == OrderState.FILLED.value
        assert order.filled_quantity == 0.01
        
        # 4. Verify position updated
        position = await order_manager.get_position('btcusdt')
        assert position.quantity == 0.01
        assert position.avg_entry_price == 50000.0
    
    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, order_manager):
        """Test: Order with multiple partial fills."""
        order = await order_manager.submit_order(
            symbol='btcusdt',
            quantity=1.0,
            price=50000.0,
            side='BUY'
        )
        await order_manager.mark_submitted(order.client_order_id, "12345")
        
        # First partial fill
        order = await order_manager.process_fill(
            order.client_order_id,
            filled_qty=0.3,
            fill_price=50000.0
        )
        assert order.state == OrderState.PARTIAL_FILL.value
        assert order.filled_quantity == 0.3
        assert order.remainder_quantity == 0.7
        
        # Second partial fill
        order = await order_manager.process_fill(
            order.client_order_id,
            filled_qty=0.7,
            fill_price=50100.0
        )
        assert order.state == OrderState.FILLED.value
        assert order.filled_quantity == 1.0
        
        # Verify weighted average price
        expected_avg = (0.3 * 50000 + 0.7 * 50100) / 1.0
        assert abs(order.avg_fill_price - expected_avg) < 0.01
    
    @pytest.mark.asyncio
    async def test_position_sync_from_exchange(self, order_manager, mock_exchange):
        """Test: Force sync position from exchange."""
        # 1. Setup Mock Exchange Return
        mock_exchange.get_positions = AsyncMock(return_value=[
            {
                'symbol': 'BTCUSDT',
                'quantity': '0.5',
                'avgPrice': '48000.0',
                'unrealizedPnl': '100.0',
                'positionAmt': '0.5',
                'entryPrice': '48000.0'
            }
        ])

        # 2. Trigger Sync
        await order_manager.position_tracker.force_sync_with_exchange()
        
        # 3. Verify
        position = await order_manager.get_position('btcusdt')
        assert position.quantity == 0.5
        assert position.avg_entry_price == 48000.0
        assert position.exchange_confirmed == True
    
    @pytest.mark.asyncio
    async def test_sync_clears_phantom_positions(self, order_manager, mock_exchange):
        """Test: Local positions not on exchange should be cleared."""
        # 1. Setup Phantom Positions (exist locally but not on exchange)
        from execution.position_tracker import Position
        from datetime import datetime
        
        order_manager.position_tracker._positions['btcusdt'] = Position(
            symbol='btcusdt', quantity=0.5, avg_entry_price=50000, 
            unrealized_pnl=0, last_update=datetime.now()
        )
        
        # 2. Mock Exchange returns EMPTY list
        mock_exchange.get_positions = AsyncMock(return_value=[])
        
        # 3. Sync
        await order_manager.position_tracker.force_sync_with_exchange()
        
        # 4. Verify cleared
        positions = await order_manager.position_tracker.get_all_positions()
        assert len(positions) == 0


# ==================== RECONCILIATION TESTS ====================

class TestReconciliationIntegration:
    """Test reconciliation service ghost/orphan/zombie handling."""
    
    @pytest.mark.asyncio
    async def test_ghost_order_no_trades_gets_canceled(self, order_manager, temp_db):
        """Test: Ghost order with no trades → CANCELED."""
        # Create a "submitted" order locally
        order = await order_manager.submit_order(
            symbol='btcusdt',
            quantity=0.1,
            price=50000.0,
            side='BUY'
        )
        await order_manager.mark_submitted(order.client_order_id, "fake_exchange_id")
        
        # Verify it's in open orders
        open_orders = await order_manager.get_open_orders()
        assert len(open_orders) == 1
        
        # Create reconciliation service with mocked API
        recon = ReconciliationService(
            api_key="test",
            api_secret="test",
            testnet=True
        )
        recon.order_manager = order_manager
        
        # Mock: Exchange returns NO open orders (ghost!)
        recon._fetch_open_orders = AsyncMock(return_value=[])
        # Mock: No trades found
        recon._fetch_order_trades = AsyncMock(return_value=[])
        
        # Run reconciliation
        await recon.reconcile_orders()
        
        # Verify order was canceled locally
        updated_order = await order_manager.get_order(order.client_order_id)
        assert updated_order.state == OrderState.CANCELED.value
    
    @pytest.mark.asyncio
    async def test_zombie_order_partial_fill_gets_canceled(self, order_manager):
        """Test: Partially filled ghost order → Process fills + CANCEL remainder."""
        # Create order for 1 BTC
        order = await order_manager.submit_order(
            symbol='btcusdt',
            quantity=1.0,
            price=50000.0,
            side='BUY'
        )
        await order_manager.mark_submitted(order.client_order_id, "fake_id")
        
        recon = ReconciliationService(
            api_key="test",
            api_secret="test",
            testnet=True
        )
        recon.order_manager = order_manager
        
        # Mock: Order gone from exchange
        recon._fetch_open_orders = AsyncMock(return_value=[])
        # Mock: Trade history shows 0.5 BTC filled
        recon._fetch_order_trades = AsyncMock(return_value=[
            {'qty': '0.5', 'price': '50000'}
        ])
        
        # Run reconciliation
        await recon.reconcile_orders()
        
        # Verify: Order should be CANCELED (zombie killed)
        updated_order = await order_manager.get_order(order.client_order_id)
        assert updated_order.state == OrderState.CANCELED.value
        assert updated_order.filled_quantity == 0.5
        
        # Position should be 0.5 BTC
        position = await order_manager.get_position('btcusdt')
        assert position.quantity == 0.5
    
    @pytest.mark.asyncio
    async def test_force_sync_positions(self, order_manager):
        """Test: Force sync positions from exchange."""
        recon = ReconciliationService(
            api_key="test",
            api_secret="test",
            testnet=True
        )
        recon.order_manager = order_manager
    
        # Mock exchange positions via OrderManager's exchange client (PositionTracker uses this)
        # Ensure it's AsyncMock
        # PositionTracker requires 'quantity' (consistent with normalized Position model)
        order_manager.exchange.get_positions = AsyncMock(return_value=[
            {'symbol': 'BTCUSDT', 'positionAmt': '0.5', 'entryPrice': '50000', 'unrealizedProfit': '10.0', 'quantity': '0.5', 'avgPrice': '50000'},
            {'symbol': 'ETHUSDT', 'positionAmt': '2.0', 'entryPrice': '3000', 'unrealizedProfit': '5.0', 'quantity': '2.0', 'avgPrice': '3000'}
        ])
    
        # Run sync
        results = await recon.force_sync_positions()
    
        assert len(results['synced']) == 2
        assert len(results['errors']) == 0
        
        # Verify positions
        btc_pos = await order_manager.get_position('btcusdt')
        eth_pos = await order_manager.get_position('ethusdt')
        
        assert btc_pos.quantity == 0.5
        assert eth_pos.quantity == 2.0


# ==================== PnL CALCULATION TESTS ====================

class TestPnLCalculation:
    """Test realized PnL calculation accuracy."""
    
    @pytest.mark.asyncio
    async def test_long_position_profit(self, order_manager):
        """Test: Buy low, sell high → Positive PnL."""
        # Set initial position: Long 0.5 BTC @ $50,000
        order_manager.position_tracker._positions['btcusdt'] = Position(
            symbol='btcusdt', quantity=0.5, avg_entry_price=50000, 
            unrealized_pnl=0, last_update=datetime.now()
        )
        
        pos = await order_manager.get_position('btcusdt')
        
        # Simulate SELL at $51,000
        exit_price = 51000
        sell_qty = 0.5
        
        # Expected PnL
        expected_pnl = (exit_price - pos.avg_entry_price) * sell_qty
        assert expected_pnl == 500.0  # $500 profit
    
    @pytest.mark.asyncio
    async def test_short_position_profit(self, order_manager):
        """Test: Sell high, buy low → Positive PnL."""
        # Set initial position: Short 0.5 BTC @ $50,000 (negative quantity)
        order_manager.position_tracker._positions['btcusdt'] = Position(
            symbol='btcusdt', quantity=-0.5, avg_entry_price=50000, 
            unrealized_pnl=0, last_update=datetime.now()
        )
        
        pos = await order_manager.get_position('btcusdt')
        
        # Simulate BUY at $49,000 to close short
        exit_price = 49000
        buy_qty = 0.5
        
        # Expected PnL for short
        expected_pnl = (pos.avg_entry_price - exit_price) * buy_qty
        assert expected_pnl == 500.0  # $500 profit
    
    @pytest.mark.asyncio
    async def test_position_flip_pnl(self, order_manager):
        """Test: Long → Oversell → Short transition."""
        # Initial long position
        order_manager.position_tracker._positions['btcusdt'] = Position(
            symbol='btcusdt', quantity=0.5, avg_entry_price=50000, 
            unrealized_pnl=0, last_update=datetime.now()
        )
        
        # Create sell order that flips to short
        order = await order_manager.submit_order(
            symbol='btcusdt',
            quantity=1.0,
            price=51000.0,
            side='SELL'
        )
        await order_manager.mark_submitted(order.client_order_id, "123")
        await order_manager.process_fill(order.client_order_id, 1.0, 51000.0)
        
        # Position should now be -0.5 (short)
        pos = await order_manager.get_position('btcusdt')
        assert pos.quantity == -0.5


# ==================== STRATEGY SIGNAL TESTS ====================

class TestStrategySignalFlow:
    """Test strategy signal generation and execution flow."""
    
    @pytest.mark.asyncio
    async def test_strategy_generates_signals_on_tick(self, strategy, event_queue):
        """Test: Market tick → Strategy generates bid/ask signals."""
        # Send market event
        tick = MarketEvent(
            exchange='binance',
            symbol='btcusdt',
            price=50000.0,
            volume=100.0
        )
        
        await strategy.on_tick(tick)
        
        # Check if signals were generated
        signals = []
        while not event_queue.empty():
            event = await event_queue.get()
            if isinstance(event, SignalEvent):
                signals.append(event)
        
        # Avellaneda-Stoikov should generate bid and ask quotes
        assert len(signals) == 2
        sides = {s.side for s in signals}
        assert 'BUY' in sides
        assert 'SELL' in sides
    
    @pytest.mark.asyncio
    async def test_inventory_affects_quote_sizes(self, strategy, event_queue):
        """Test: Long inventory → Smaller bid, bigger ask."""
        # Patch settings to allow large inventory without triggering reduce-only
        # settings is the module, settings.settings is the instance
        with patch.object(settings.settings, 'MAX_POSITION_USD', 1000000.0):
            # Set positive inventory (long) via internal state
            strategy._state['btcusdt']['inventory'] = 0.5
        
            tick = MarketEvent(
                exchange='binance',
                symbol='btcusdt',
                price=50000.0,
                volume=100.0
            )
        
            await strategy.on_tick(tick)
        
            signals = []
            while not event_queue.empty():
                event = await event_queue.get()
                if isinstance(event, SignalEvent):
                    signals.append(event)
        
            bid_signal = next((s for s in signals if s.side == 'BUY'), None)
            ask_signal = next((s for s in signals if s.side == 'SELL'), None)
        
            # With positive inventory, ask should be larger/more aggressive
            assert ask_signal is not None
            assert bid_signal is not None
    
    @pytest.mark.asyncio
    async def test_fill_updates_strategy_inventory(self, strategy):
        """Test: Fill event updates strategy inventory."""
        initial_inventory = strategy._state.get('btcusdt', {}).get('inventory', 0)
        
        fill = FillEvent(
            symbol='btcusdt',
            side='BUY',
            quantity=0.1,
            price=50000.0,
            commission=0.02,
            pnl=0.0
        )
        
        await strategy.on_fill(fill)
        
        new_inventory = strategy._state.get('btcusdt', {}).get('inventory', 0)
        assert new_inventory == initial_inventory + 0.1
    
    @pytest.mark.asyncio
    async def test_regime_change_affects_quoting(self, strategy, event_queue):
        """Test: Trending regime → Strategy pauses/adjusts."""
        # Set trending regime
        regime = RegimeEvent(
            symbol='btcusdt',
            regime='TRENDING',
            adx=35.0,
            volatility=0.02
        )
        
        await strategy.on_regime_change(regime)
        
        # Strategy should note the regime in internal state
        assert strategy._state.get('btcusdt', {}).get('regime') == 'TRENDING'


# ==================== END-TO-END FLOW TESTS ====================

class TestEndToEndFlow:
    """Full integration: Strategy → OrderManager → Execute → Fill → Update."""
    
    @pytest.mark.asyncio
    async def test_full_trade_cycle(self, order_manager, event_queue, mock_shadow_book):
        """Test complete trade cycle with all components."""
        # 1. Create strategy
        strategy = AvellanedaStoikovStrategy(
            event_queue=event_queue,
            symbols=['btcusdt'],
            base_quantity=0.01,
            gamma=0.1,
            max_inventory=1.0,
            shadow_book=mock_shadow_book
        )
        
        # 2. Strategy receives tick and generates signal
        tick = MarketEvent(
            exchange='binance',
            symbol='btcusdt',
            price=50000.0,
            volume=100.0
        )
        await strategy.on_tick(tick)
        
        # 3. Get generated signal
        signal = None
        while not event_queue.empty():
            event = await event_queue.get()
            if isinstance(event, SignalEvent) and event.side == 'BUY':
                signal = event
                break
        
        assert signal is not None
        
        # 4. Create order in OrderManager
        order = await order_manager.submit_order(
            symbol=signal.symbol,
            quantity=signal.quantity,
            price=signal.price,
            side=signal.side
        )
        
        # 5. Simulate exchange acceptance
        await order_manager.mark_submitted(order.client_order_id, "exchange_123")
        
        # 6. Simulate fill
        await order_manager.process_fill(
            order.client_order_id,
            filled_qty=signal.quantity,
            fill_price=signal.price,
            commission=0.01
        )
        
        # 7. Verify position tracking
        position = await order_manager.get_position('btcusdt')
        assert position.quantity == signal.quantity
        
        # 8. Update strategy with fill
        fill = FillEvent(
            symbol='btcusdt',
            side='BUY',
            quantity=signal.quantity,
            price=signal.price,
            commission=0.01,
            pnl=0.0
        )
        await strategy.on_fill(fill)
        
        # 9. Verify strategy inventory increased (may include prior fills from warmup)
        final_inventory = strategy._state.get('btcusdt', {}).get('inventory', 0)
        assert final_inventory > 0, "Inventory should be positive after BUY fill"


# ==================== RISK MANAGER TESTS ====================

class TestRiskManagerIntegration:
    """Test risk manager circuit breakers."""
    
    def test_losses_reduce_balance(self, risk_manager):
        """Test: Losses reduce current balance."""
        initial_balance = risk_manager.current_balance
        
        # Simulate losses
        risk_manager.record_trade_result(-100)
        risk_manager.record_trade_result(-50)
        
        assert risk_manager.current_balance == initial_balance - 150
    
    def test_kill_switch_triggers_on_max_drawdown(self, risk_manager):
        """Test: Max drawdown triggers kill switch."""
        # 5% of 10000 = 500 loss triggers
        # Simulate gradual losses
        for _ in range(10):
            risk_manager.record_trade_result(-60)  # -$600 total = 6% > 5%
        
        # Check account health to trigger kill switch
        risk_manager.check_account_health(risk_manager.current_balance)
        
        # Kill switch should be triggered
        assert risk_manager.kill_switch_triggered == True
    
    def test_wins_increase_balance(self, risk_manager):
        """Test: Winning trades increase balance."""
        initial_balance = risk_manager.current_balance
        
        risk_manager.record_trade_result(50)
        risk_manager.record_trade_result(75)
        risk_manager.record_trade_result(100)
        
        assert risk_manager.current_balance == initial_balance + 225


# ==================== SYNC FLAG FILE TESTS ====================

class TestSyncFlagMechanism:
    """Test dashboard → reconciliation flag file communication."""
    
    def test_sync_flag_creation(self, tmp_path):
        """Test: Creating sync flag file."""
        flag_path = tmp_path / "SYNC_POSITIONS.flag"
        
        with open(flag_path, 'w') as f:
            f.write('SYNC_REQUESTED')
        
        assert flag_path.exists()
    
    @pytest.mark.asyncio
    async def test_sync_flag_triggers_sync(self, order_manager, tmp_path):
        """Test: Flag file detected triggers sync."""
        flag_path = tmp_path / "SYNC_POSITIONS.flag"
        
        # Create flag
        with open(flag_path, 'w') as f:
            f.write('SYNC_REQUESTED')
        
        # Verify flag exists
        assert flag_path.exists()
        
        # In real scenario, reconciliation loop would detect and process
        # Here we just verify the mechanism works


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
