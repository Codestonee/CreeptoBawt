import asyncio
import logging
import math
import sys
import os
import time
from typing import List, Dict, Any

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.events import MarketEvent, SignalEvent, FillEvent
from execution.simulated import MockExecutionHandler
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from analysis.backtest_reporter import BacktestReporter
from config.settings import settings

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestSimulator")

# Constants
SYMBOL = 'BTCUSDT'
DURATION_MINUTES = 60 * 4  # 4 Hours
TICK_INTERVAL_SEC = 1.0    # 1 tick per sec
INITIAL_PRICE = 95000.0    # Current BTC price (Jan 2026)

# ============ PARAMETER SWEEP MODE ============
SWEEP_MODE = True  # Set to True to run parameter sweep

# Parameter grid for sweep mode
GAMMA_VALUES = [0.5, 1.0, 2.0, 3.0]
KAPPA_VALUES = [0.5, 1.0, 2.0, 4.0]


async def run_simulation(gamma: float = 0.5, kappa: float = 0.5) -> Dict[str, Any]:
    """
    Run market simulation with Avellaneda-Stoikov strategy.
    
    Args:
        gamma: Risk aversion parameter
        kappa: Inventory aversion parameter
        
    Returns:
        Dict with simulation results
    """
    logger.info(f"🚀 Starting Backtest (γ={gamma}, κ={kappa}) for {SYMBOL} ({DURATION_MINUTES} mins)...")
    
    # queues
    event_queue = asyncio.Queue()
    
    # Components
    handler = MockExecutionHandler(event_queue, risk_manager=None)
    await handler.connect()
    
    # Initialize Strategy
    class MockShadowBook:
        def get_mid_price(self, s):
            return handler.client.current_prices.get(s.lower(), 0.0)
            
        def is_stale(self, *args, **kwargs):
            return False
            
        def get_imbalance(self, *args, **kwargs):
            return 0.0
            
    shadow_book = MockShadowBook()
    
    strategy = AvellanedaStoikovStrategy(
        event_queue=event_queue,
        symbols=[SYMBOL],
        shadow_book=shadow_book,
        gamma=gamma,  # Use param
        base_quantity=0.005,  # Reduced to stay under $500 risk limit
        max_inventory=1.0
    )
    
    # Override Strategy state for test
    strategy._state[SYMBOL.lower()]['volatility'] = 0.002  # Fixed vol
    strategy._state[SYMBOL.lower()]['kappa'] = kappa  # Use param
    strategy.min_kappa = 0.5
    strategy.max_kappa = 5.0
    
    # Simulation Loop
    total_ticks = int((DURATION_MINUTES * 60) / TICK_INTERVAL_SEC)
    
    equity_history = []
    
    price = INITIAL_PRICE
    time_offset = 0
    
    try:
        for i in range(total_ticks):
            # 1. Generate Price Tick (Sine Wave + Noise)
            sine_component = math.sin((i / total_ticks) * 2 * math.pi) * 500
            noise = (hash(str(i)) % 100 - 50) / 10.0  # Deterministic noise
            price = INITIAL_PRICE + sine_component + noise
            
            timestamp = time.time() + time_offset
            time_offset += TICK_INTERVAL_SEC
            
            # 2. Feed to Handler (Mock Exchange)
            market_event = MarketEvent(
                exchange='binance',
                symbol=SYMBOL,
                timestamp=timestamp,
                price=price,
                volume=1.0
            )
            await handler.on_tick(market_event)
            
            # 3. Feed to Strategy
            await strategy.on_tick(market_event)
            
            # 4. Process Signal Queue
            while not event_queue.empty():
                evt = await event_queue.get()
                if isinstance(evt, SignalEvent):
                    await handler.execute(evt)
                elif isinstance(evt, FillEvent):
                    await strategy.on_fill(evt)
                    
            # 5. Capture Equity Snapshot (every 1 min)
            if i % 60 == 0:
                acct = await handler.client.futures_account()
                total_equity = float(acct['totalWalletBalance'])
                
                # Add Unrealized PnL from positions
                pos_list = await handler.client.get_positions()
                # MockExchangeClient uses 'unRealizedProfit' and 'positionAmt'
                upnl = sum([float(p.get('unRealizedProfit', 0)) for p in pos_list])
                inventory = sum([float(p.get('positionAmt', 0)) for p in pos_list])
                
                total_equity += upnl
                
                equity_history.append({
                    'timestamp': timestamp,
                    'total_equity': total_equity,
                    'mark_price': price,
                    'inventory': inventory
                })

    except Exception as e:
        logger.exception(f"CRASH AT TICK {i}: {e}")
        return {'error': str(e), 'gamma': gamma, 'kappa': kappa}
    
    # Calculate metrics
    fill_history = getattr(handler.client, 'fill_history', [])
    total_pnl = equity_history[-1]['total_equity'] - equity_history[0]['total_equity'] if equity_history else 0
    max_dd = min([eq['total_equity'] for eq in equity_history]) - equity_history[0]['total_equity'] if equity_history else 0
    
    return {
        'gamma': gamma,
        'kappa': kappa,
        'total_pnl': total_pnl,
        'max_drawdown': max_dd,
        'num_trades': len(fill_history),
        'equity_history': equity_history,
        'fill_history': fill_history
    }


async def run_parameter_sweep():
    """
    Run simulations across all gamma/kappa combinations and generate comparison report.
    """
    print(f"\n{'='*60}")
    print("🔬 PARAMETER SWEEP MODE")
    print(f"Testing {len(GAMMA_VALUES)} gamma × {len(KAPPA_VALUES)} kappa = {len(GAMMA_VALUES) * len(KAPPA_VALUES)} combinations")
    print(f"{'='*60}\n")
    
    results = []
    
    for gamma in GAMMA_VALUES:
        for kappa in KAPPA_VALUES:
            result = await run_simulation(gamma=gamma, kappa=kappa)
            results.append(result)
            print(f"✓ γ={gamma}, κ={kappa} → PnL: ${result.get('total_pnl', 0):.2f}, Trades: {result.get('num_trades', 0)}")
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("📊 PARAMETER SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Gamma':>8} {'Kappa':>8} {'PnL':>12} {'MaxDD':>12} {'Trades':>8}")
    print("-" * 60)
    
    best_result = max(results, key=lambda x: x.get('total_pnl', float('-inf')))
    
    for r in sorted(results, key=lambda x: x.get('total_pnl', 0), reverse=True):
        marker = "⭐" if r == best_result else "  "
        print(f"{marker}{r['gamma']:>6.1f} {r['kappa']:>8.1f} ${r.get('total_pnl', 0):>10.2f} ${r.get('max_drawdown', 0):>10.2f} {r.get('num_trades', 0):>8}")
    
    print(f"\n🏆 BEST PARAMS: γ={best_result['gamma']}, κ={best_result['kappa']} → PnL: ${best_result.get('total_pnl', 0):.2f}")
    
    # Generate individual report for best result
    if best_result.get('equity_history'):
        reporter = BacktestReporter()
        metrics = {'total_pnl': best_result['total_pnl']}
        report_path = reporter.generate_report(
            f"{SYMBOL}_BEST_g{best_result['gamma']}_k{best_result['kappa']}",
            best_result['equity_history'],
            best_result.get('fill_history', []),
            metrics
        )
        print(f"\nBest result report: {report_path}")


async def run_single():
    """Run single simulation with default params."""
    result = await run_simulation(gamma=settings.AS_GAMMA, kappa=settings.AS_KAPPA)
    
    print("\n✅ Simulation Complete.")
    
    # Generate Report
    if result.get('equity_history'):
        reporter = BacktestReporter()
        metrics = {'total_pnl': result['total_pnl']}
        report_path = reporter.generate_report(SYMBOL, result['equity_history'], result.get('fill_history', []), metrics)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    if SWEEP_MODE:
        asyncio.run(run_parameter_sweep())
    else:
        asyncio.run(run_single())
