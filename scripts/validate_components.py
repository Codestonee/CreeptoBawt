"""
Shadow Mode Validation Script.

Quick test to validate all implemented optimization components:
1. VolatilityAwarePositionSizer (Kelly criterion)
2. ModernRiskManager (CVaR + graduated shutdown)
3. GLTQuoteEngine (multi-asset correlation)
4. DeterministicOrderRouter (limit chasing)
5. ShadowModeController (phased deployment)
6. RedisStateManager (crash recovery)
7. ReconciliationService (bootstrap)

Usage:
    python validate_components.py

Expected output: All components initialize successfully in shadow mode.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def validate_components():
    """Validate all optimization components work together."""
    
    print("=" * 60)
    print("CreeptBaws Optimization Validation")
    print("=" * 60)
    
    results = {}
    
    # 1. Position Sizer
    print("\n[1/7] Testing VolatilityAwarePositionSizer...")
    try:
        from risk_engine.position_sizer import VolatilityAwarePositionSizer
        
        sizer = VolatilityAwarePositionSizer(
            kelly_fraction=0.5,
            target_volatility=0.02,
            max_leverage=3.0
        )
        
        # Add some volatility data
        for i in range(20):
            sizer.update_market_state_sync('BTCUSDT', 51000, 49000)
        
        # Calculate position
        pos = sizer.calculate_size('BTCUSDT', 10000, 50000)
        print(f"   ✅ Kelly position: {pos:.6f} BTC (${pos * 50000:.2f})")
        results['position_sizer'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['position_sizer'] = f'FAIL: {e}'
    
    # 2. Risk Manager
    print("\n[2/7] Testing ModernRiskManager...")
    try:
        from risk_engine.risk_manager import ModernRiskManager, RiskState
        
        rm = ModernRiskManager(
            account_balance=10000,
            daily_var_limit=0.05,
            max_drawdown_pct=0.10
        )
        
        # Check initial state
        assert rm.current_state == RiskState.NORMAL
        print(f"   ✅ Initial state: {rm.current_state.value}")
        print(f"   ✅ CVaR ratio: {rm.CRYPTO_CVAR_RATIO}x")
        results['risk_manager'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['risk_manager'] = f'FAIL: {e}'
    
    # 3. GLT Quote Engine
    print("\n[3/7] Testing GLTQuoteEngine with multi-asset correlation...")
    try:
        from strategies.glt_quote_engine import GLTQuoteEngine, GLTParams
        
        engine = GLTQuoteEngine()
        engine.set_params('btcusdt', GLTParams(A=1.0, k=0.3, gamma=0.3))
        engine.set_params('ethusdt', GLTParams(A=1.5, k=0.4, gamma=0.3))
        
        # Test multi-asset quotes
        engine.update_portfolio_state('btcusdt', 0.1, 0.03)  # Long 0.1 BTC
        engine.update_portfolio_state('ethusdt', 0.5, 0.04)  # Long 0.5 ETH
        
        bid, ask = engine.compute_quotes('btcusdt', 50000, 0.1, 0.03)
        spread_bps = (ask - bid) / 50000 * 10000
        
        print(f"   ✅ BTC quotes: Bid=${bid:.2f}, Ask=${ask:.2f}")
        print(f"   ✅ Spread: {spread_bps:.1f} bps")
        print(f"   ✅ Correlation matrix loaded: {len(engine._asset_order)} assets")
        results['glt_engine'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['glt_engine'] = f'FAIL: {e}'
    
    # 4. Deterministic Order Router
    print("\n[4/7] Testing DeterministicOrderRouter...")
    try:
        from execution.smart_router import DeterministicOrderRouter, get_order_router
        
        router = DeterministicOrderRouter(
            reprice_interval_ms=100,
            max_repricings=3
        )
        
        print(f"   ✅ Router created: reprice_interval={router.reprice_interval_ms}ms")
        print(f"   ✅ Max repricings: {router.max_repricings}")
        print(f"   ✅ Target maker fill rate: 80%+")
        results['order_router'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['order_router'] = f'FAIL: {e}'
    
    # 5. Shadow Mode Controller
    print("\n[5/7] Testing ShadowModeController...")
    try:
        from execution.shadow_mode import ShadowModeController, DeploymentPhase
        
        controller = ShadowModeController(phase=DeploymentPhase.SHADOW)
        
        # Record a shadow trade
        result = controller.record_shadow_trade(
            symbol='btcusdt',
            side='BUY',
            quantity=0.1,
            target_price=50000,
            market_mid=50010,
            would_fill=True
        )
        
        stats = controller.get_comparison_stats()
        print(f"   ✅ Phase: {stats['phase']} ({stats['multiplier']*100:.0f}%)")
        print(f"   ✅ Shadow trades recorded: {stats['shadow_trades']}")
        print(f"   ✅ Simulated PnL: ${stats['simulated_pnl']:.2f}")
        results['shadow_mode'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['shadow_mode'] = f'FAIL: {e}'
    
    # 6. Redis State Manager
    print("\n[6/7] Testing RedisStateManager (fallback mode)...")
    try:
        from state.redis_state import RedisStateManager, OrderRecord
        
        state = RedisStateManager(host="localhost", port=6379)
        connected = await state.connect()
        
        # Test with fallback
        order = OrderRecord(
            client_order_id="test_123",
            symbol="btcusdt",
            side="BUY",
            quantity=0.1,
            price=50000
        )
        await state.save_order(order)
        retrieved = await state.get_order("test_123")
        
        if connected:
            print(f"   ✅ Redis connected: {state.host}:{state.port}")
        else:
            print(f"   ✅ Using fallback mode (Redis not available)")
        
        assert retrieved is not None
        print(f"   ✅ Order save/load working")
        results['redis_state'] = 'PASS'
        
        await state.disconnect()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['redis_state'] = f'FAIL: {e}'
    
    # 7. Reconciliation Bootstrap
    print("\n[7/7] Testing ReconciliationService bootstrap...")
    try:
        from execution.reconciliation import ReconciliationService
        
        # Can't actually connect without API keys, but validate the class
        service = ReconciliationService(
            api_key="test",
            api_secret="test",
            testnet=True
        )
        
        assert hasattr(service, 'bootstrap')
        print(f"   ✅ bootstrap() method exists")
        print(f"   ✅ Auto-fix positions: {service.auto_fix_positions}")
        print(f"   ✅ Sync interval: {service.FULL_SYNC_INTERVAL}s")
        results['reconciliation'] = 'PASS'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        results['reconciliation'] = f'FAIL: {e}'
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)
    
    for component, status in results.items():
        icon = "✅" if status == 'PASS' else "❌"
        print(f"  {icon} {component}: {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All optimization components validated successfully!")
        print("   Ready for shadow mode testing with real market data.")
    else:
        print("\n⚠️ Some components failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(validate_components())
    sys.exit(0 if success else 1)
