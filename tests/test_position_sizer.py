"""
Unit tests for VolatilityAwarePositionSizer.

Tests the Kelly criterion, Parkinson volatility, and multi-asset
correlation adjustments implemented in the optimization.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

# Import the module under test
from risk_engine.position_sizer import (
    VolatilityAwarePositionSizer,
    PositionSizer,
    MarketState
)


class TestParkinsonVolatility:
    """Test Parkinson volatility estimator (integrated into sizer)."""
    
    @pytest.fixture
    def sizer(self):
        """Create sizer for volatility tests."""
        return VolatilityAwarePositionSizer(
            target_volatility=0.02,
            lookback_window=20
        )
    
    def test_insufficient_data_returns_default(self, sizer):
        """With no data, should return target volatility."""
        vol = sizer.get_current_volatility('BTCUSDT')
        assert vol == sizer.target_volatility
    
    def test_stable_market_low_volatility(self, sizer):
        """Stable market (narrow ranges) should show low volatility."""
        # Feed 20 candles with 0.1% range
        for i in range(20):
            price = 100 + i * 0.01
            sizer.update_market_state_sync('BTCUSDT', price * 1.001, price * 0.999)
        
        vol = sizer.get_current_volatility('BTCUSDT')
        assert vol < 0.01, f"Expected low volatility, got {vol}"
    
    def test_volatile_market_high_volatility(self, sizer):
        """Volatile market (wide ranges) should show high volatility."""
        # Feed 20 candles with 5% range
        for i in range(20):
            sizer.update_market_state_sync('BTCUSDT', 105, 95)
        
        vol = sizer.get_current_volatility('BTCUSDT')
        assert vol > 0.02, f"Expected high volatility, got {vol}"


class TestVolatilityAwarePositionSizer:
    """Test the main position sizer."""
    
    @pytest.fixture
    def sizer(self):
        """Create a position sizer with default config."""
        s = VolatilityAwarePositionSizer(
            kelly_fraction=0.5,
            target_volatility=0.02,
            max_leverage=3.0,
            max_position_fraction=0.20,
            min_position_value=10.0
        )
        # Add some volatility data
        for i in range(20):
            s.update_market_state_sync('BTCUSDT', 51000, 49000)
        return s
    
    def test_basic_sizing_returns_positive(self, sizer):
        """Basic sizing should return a positive position."""
        position = sizer.calculate_size(
            symbol='BTCUSDT',
            account_balance=10000,
            entry_price=50000
        )
        
        assert position > 0
    
    def test_high_volatility_reduces_position(self, sizer):
        """Higher volatility should reduce position size."""
        # Low vol: 1% range
        sizer_low_vol = VolatilityAwarePositionSizer(target_volatility=0.02)
        for i in range(20):
            sizer_low_vol.update_market_state_sync('BTCUSDT', 50500, 49500)
        
        # High vol: 5% range
        sizer_high_vol = VolatilityAwarePositionSizer(target_volatility=0.02)
        for i in range(20):
            sizer_high_vol.update_market_state_sync('BTCUSDT', 52500, 47500)
        
        pos_low = sizer_low_vol.calculate_size('BTCUSDT', 10000, 50000)
        pos_high = sizer_high_vol.calculate_size('BTCUSDT', 10000, 50000)
        
        assert pos_high < pos_low, \
            f"High vol position ({pos_high}) should be < low vol ({pos_low})"
    
    def test_respects_max_position(self, sizer):
        """Should never exceed max position fraction."""
        # Very favorable conditions
        sizer.win_probability = 0.95
        sizer.profit_ratio = 10.0
        
        position = sizer.calculate_size('BTCUSDT', 10000, 50000)
        position_value = position * 50000
        max_value = 10000 * sizer.max_position_fraction
        
        assert position_value <= max_value * 1.01, \
            f"Position value ({position_value}) exceeds max ({max_value})"
    
    def test_zero_balance_returns_zero(self, sizer):
        """Zero account balance should return zero position."""
        position = sizer.calculate_size('BTCUSDT', 0, 50000)
        assert position == 0
    
    def test_zero_price_returns_zero(self, sizer):
        """Zero price should return zero position."""
        position = sizer.calculate_size('BTCUSDT', 10000, 0)
        assert position == 0


class TestPortfolioCorrelation:
    """Test multi-asset correlation adjustments."""
    
    @pytest.fixture
    def sizer(self):
        """Create sizer with correlation matrix."""
        s = VolatilityAwarePositionSizer()
        # Add data for both assets
        for i in range(20):
            s.update_market_state_sync('BTCUSDT', 51000, 49000)
            s.update_market_state_sync('ETHUSDT', 3100, 2900)
        return s
    
    def test_portfolio_positions_returns_dict(self, sizer):
        """Portfolio calculation should return position dict."""
        positions = sizer.calculate_portfolio_positions(
            assets=['BTCUSDT', 'ETHUSDT'],
            current_volatilities={'BTCUSDT': 0.03, 'ETHUSDT': 0.04},
            account_balance=10000,
            entry_prices={'BTCUSDT': 50000, 'ETHUSDT': 3000}
        )
        
        assert isinstance(positions, dict)
        assert 'BTCUSDT' in positions
        assert 'ETHUSDT' in positions


class TestLegacyCompatibility:
    """Test that legacy PositionSizer wrapper works."""
    
    def test_legacy_wrapper_exists(self):
        """Legacy PositionSizer should still be importable."""
        sizer = PositionSizer(risk_per_trade_pct=0.01)
        assert sizer is not None
    
    def test_legacy_inherits_methods(self):
        """Legacy class should have calculate_size method."""
        sizer = PositionSizer()
        assert hasattr(sizer, 'calculate_size')
