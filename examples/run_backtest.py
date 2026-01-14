"""
Example: Run backtest on Avellaneda-Stoikov strategy.

Usage:
    python examples/run_backtest.py
"""

import asyncio
import logging
from datetime import datetime, timedelta

from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from backtesting.data_loader import DataLoader
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run backtest."""
    
    # Configuration
    config = BacktestConfig(
        symbols=['btcusdt', 'ethusdt'],
        start_date='2025-01-01',
        end_date='2025-01-14',
        initial_capital=10000.0,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        spread_cost_bps=5.0,
        impact_per_10k_usd=2.0,
        max_position_usd=5000.0
    )
    
    logger.info("="*60)
    logger.info("BACKTEST: Avellaneda-Stoikov Market Making")
    logger.info("="*60)
    
    # Load data
    loader = DataLoader()
    
    try:
        data = await loader.load_data(
            symbols=[s.upper() for s in config.symbols],
            start_date=config.start_date,
            end_date=config.end_date,
            interval='1m',
            source='binance'
        )
        
        # Convert symbol keys to lowercase for strategy
        data = {k.lower(): v for k, v in data.items()}
        
        # Create strategy instance
        # Use a mock event queue (backtest doesn't need real queue)
        from asyncio import Queue
        event_queue = Queue()
        
        strategy = AvellanedaStoikovStrategy(
            event_queue=event_queue,
            symbols=config.symbols,
            base_quantity=0.01,
            gamma=0.1,
            max_inventory=1.0
        )
        
        # Disable GLT for faster backtest (use linear A-S)
        strategy.use_glt = False
        
        # Run backtest
        engine = BacktestEngine(config)
        result = await engine.run(strategy, data)
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return:        {result.total_return_pct:+.2f}%")
        print(f"Annualized Return:   {result.annualized_return_pct:+.2f}%")
        print(f"Sharpe Ratio:        {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {result.sortino_ratio:.2f}")
        print(f"Max Drawdown:        {result.max_drawdown_pct:.2f}%")
        print(f"Max DD Duration:     {result.max_drawdown_duration_days} days")
        print(f"Volatility (annual): {result.volatility_annualized*100:.2f}%")
        print()
        print(f"Total Trades:        {result.total_trades}")
        print(f"Win Rate:            {result.win_rate_pct:.1f}%")
        print(f"Avg Win:             ${result.avg_win_usd:+.2f}")
        print(f"Avg Loss:            ${result.avg_loss_usd:+.2f}")
        print(f"Profit Factor:       {result.profit_factor:.2f}")
        print("="*60)
        
        # Save results
        # Ensure data directory exists
        import os
        os.makedirs('data', exist_ok=True)
        
        result.equity_curve.to_csv('data/backtest_equity.csv')
        result.trades.to_csv('data/backtest_trades.csv')
        
        logger.info("✅ Results saved to data/backtest_*.csv")
        
        # Warnings
        if result.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"   - {warning}")
        
    finally:
        await loader.close()


if __name__ == "__main__":
    asyncio.run(main())
