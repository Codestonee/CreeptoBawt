"""
Unit Tests for Analytics Module

Tests the risk metrics calculations (Sharpe, Sortino, MaxDD, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from utils.analytics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    get_performance_summary
)


class TestSharpeRatio:
    """Tests for Sharpe Ratio calculation."""
    
    def test_positive_returns(self):
        """Positive consistent returns should give positive Sharpe."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.02, 0.01, 0.012, 0.018])
        sharpe = calculate_sharpe_ratio(returns, annualization_factor=252)
        assert sharpe > 0, "Positive returns should yield positive Sharpe"
    
    def test_negative_returns(self):
        """Negative consistent returns should give negative Sharpe."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.02])
        sharpe = calculate_sharpe_ratio(returns, annualization_factor=252)
        assert sharpe < 0, "Negative returns should yield negative Sharpe"
    
    def test_empty_returns(self):
        """Empty returns should return 0."""
        returns = pd.Series([])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_single_return(self):
        """Single return should return 0 (not enough data)."""
        returns = pd.Series([0.05])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino Ratio calculation."""
    
    def test_all_positive_returns(self):
        """All positive returns should give high/infinite Sortino."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        sortino = calculate_sortino_ratio(returns, annualization_factor=252)
        # With no downside, Sortino should be very high or inf
        assert sortino > 0
    
    def test_mixed_returns(self):
        """Mixed returns should give finite Sortino."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.005])
        sortino = calculate_sortino_ratio(returns, annualization_factor=252)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)


class TestMaxDrawdown:
    """Tests for Maximum Drawdown calculation."""
    
    def test_no_drawdown(self):
        """Monotonically increasing equity should have 0 drawdown."""
        equity = pd.Series([100, 101, 102, 103, 104, 105])
        max_dd = calculate_max_drawdown(equity)
        assert max_dd == 0.0 or abs(max_dd) < 0.001
    
    def test_simple_drawdown(self):
        """Test a simple 10% drawdown."""
        equity = pd.Series([100, 110, 99, 105])  # Drop from 110 to 99 = -10%
        max_dd = calculate_max_drawdown(equity)
        assert max_dd < 0, "Max DD should be negative"
        assert abs(max_dd) > 0.05, "Should detect at least 5% drawdown"
    
    def test_empty_equity(self):
        """Empty equity should return 0."""
        equity = pd.Series([])
        max_dd = calculate_max_drawdown(equity)
        assert max_dd == 0.0


class TestWinRate:
    """Tests for win rate calculation."""
    
    def test_all_winners(self):
        """100% win rate."""
        trades = [{'pnl': 10}, {'pnl': 5}, {'pnl': 20}]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 1.0
    
    def test_all_losers(self):
        """0% win rate."""
        trades = [{'pnl': -10}, {'pnl': -5}, {'pnl': -20}]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.0
    
    def test_mixed_trades(self):
        """50% win rate."""
        trades = [{'pnl': 10}, {'pnl': -10}, {'pnl': 20}, {'pnl': -5}]
        win_rate = calculate_win_rate(trades)
        assert win_rate == 0.5
    
    def test_empty_trades(self):
        """Empty trades should return 0."""
        win_rate = calculate_win_rate([])
        assert win_rate == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""
    
    def test_profitable(self):
        """Profit factor > 1 when profitable."""
        trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 80}, {'pnl': -30}]
        pf = calculate_profit_factor(trades)
        assert pf > 1.0, "Profitable trades should have PF > 1"
    
    def test_unprofitable(self):
        """Profit factor < 1 when losing."""
        trades = [{'pnl': 50}, {'pnl': -100}]
        pf = calculate_profit_factor(trades)
        assert pf < 1.0
    
    def test_no_losses(self):
        """No losses should give infinite profit factor."""
        trades = [{'pnl': 100}, {'pnl': 50}]
        pf = calculate_profit_factor(trades)
        assert pf == float('inf')


class TestPerformanceSummary:
    """Tests for the full performance summary."""
    
    def test_basic_summary(self):
        """Test that summary returns all expected fields."""
        trades = [
            {'pnl': 10, 'timestamp': 1000, 'side': 'BUY', 'symbol': 'btcusdt'},
            {'pnl': -5, 'timestamp': 2000, 'side': 'SELL', 'symbol': 'btcusdt'},
            {'pnl': 15, 'timestamp': 3000, 'side': 'BUY', 'symbol': 'btcusdt'},
        ]
        summary = get_performance_summary(trades)
        
        expected_fields = [
            'total_trades', 'win_rate', 'profit_factor', 
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
            'total_pnl', 'avg_pnl_per_trade'
        ]
        
        for field in expected_fields:
            assert field in summary, f"Missing field: {field}"
        
        assert summary['total_trades'] == 3
        assert summary['total_pnl'] == 20.0  # 10 - 5 + 15
    
    def test_empty_trades(self):
        """Empty trades should return zeroed summary."""
        summary = get_performance_summary([])
        assert summary['total_trades'] == 0
        assert summary['win_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
