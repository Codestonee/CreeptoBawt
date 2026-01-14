"""
Test GLT theta computation methods.

Usage:
    python -m pytest tests/test_glt_theta.py -v
"""

import time
import numpy as np
import pytest
from strategies.glt_quote_engine import GLTQuoteEngine, GLTParams


def test_theta_methods_comparison():
    """Compare quadratic vs iterative theta calculation."""
    
    engine = GLTQuoteEngine(inventory_limit=10.0)
    symbol = "btcusdt"
    
    params = GLTParams(
        A=10.0,
        k=0.5,
        gamma=0.1,
        delta=0.001
    )
    
    # Test 1: Quadratic method (fast)
    start = time.perf_counter()
    engine._compute_theta_table(symbol, params)
    quad_time = time.perf_counter() - start
    theta_quad = engine.theta_tables[symbol].copy()
    
    print(f"\n✓ Quadratic theta: {len(theta_quad)} entries in {quad_time*1000:.2f}ms")
    
    # Test 2: Iterative method (accurate)
    engine.theta_tables[symbol] = {}  # Clear
    start = time.perf_counter()
    engine._compute_theta_table_iterative(symbol, params, volatility_estimate=0.02)
    iter_time = time.perf_counter() - start
    theta_iter = engine.theta_tables[symbol].copy()
    
    print(f"✓ Iterative theta: {len(theta_iter)} entries in {iter_time*1000:.2f}ms")
    
    # Compare values at key inventory levels
    print("\nTheta Value Comparison:")
    print("Inventory | Quadratic | Iterative | Diff %")
    print("-" * 50)
    
    for q_idx in [0, 500, 1000, 2000, 5000]:
        val_quad = theta_quad.get(q_idx, 0)
        val_iter = theta_iter.get(q_idx, 0)
        diff_pct = ((val_iter - val_quad) / val_quad * 100) if val_quad != 0 else 0
        
        print(f"{q_idx:8d}  | {val_quad:9.4f} | {val_iter:9.4f} | {diff_pct:+7.2f}%")
    
    # Test quote generation with both methods
    print("\n\nQuote Comparison (mid=$50000, inventory=1.0, vol=0.02):")
    
    # Quadratic
    engine.theta_tables[symbol] = theta_quad
    bid_q, ask_q = engine.compute_quotes(symbol, 50000, 1.0, 0.02)
    spread_q = ask_q - bid_q
    
    # Iterative
    engine.theta_tables[symbol] = theta_iter
    bid_i, ask_i = engine.compute_quotes(symbol, 50000, 1.0, 0.02)
    spread_i = ask_i - bid_i
    
    print(f"Quadratic:  Bid=${bid_q:.2f}, Ask=${ask_q:.2f}, Spread=${spread_q:.2f} ({spread_q/50000*10000:.1f}bps)")
    print(f"Iterative:  Bid=${bid_i:.2f}, Ask=${ask_i:.2f}, Spread=${spread_i:.2f} ({spread_i/50000*10000:.1f}bps)")
    print(f"Difference: Spread {(spread_i-spread_q)/spread_q*100:+.1f}%")
    
    assert len(theta_quad) > 0, "Quadratic theta table is empty"
    assert len(theta_iter) > 0, "Iterative theta table is empty"
    assert iter_time < 5.0, f"Iterative method too slow: {iter_time:.2f}s"
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_theta_methods_comparison()
