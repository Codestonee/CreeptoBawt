"""
Analytics Module - Risk Metrics using Empyrical

Provides professional-grade risk analytics for the trading bot.
Uses empyrical for calculations with safe wrappers.
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("Utils.Analytics")

# Try to import empyrical, fall back to manual calculations if unavailable
try:
    import empyrical
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    logger.warning("empyrical not installed - using fallback calculations")


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Series of period returns (not cumulative)
        risk_free_rate: Annual risk-free rate (default 0%)
        annualization_factor: Trading days per year (252 for daily, 365*24 for hourly)
        
    Returns:
        Annualized Sharpe Ratio
    """
    if returns is None or len(returns) < 2:
        return 0.0
    
    try:
        if EMPYRICAL_AVAILABLE:
            return float(empyrical.sharpe_ratio(
                returns, 
                risk_free=risk_free_rate,
                annualization=annualization_factor
            ))
        else:
            # Manual calculation
            excess_returns = returns - risk_free_rate / annualization_factor
            if excess_returns.std() == 0:
                return 0.0
            return float(
                (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor)
            )
    except Exception as e:
        logger.warning(f"Sharpe calculation failed: {e}")
        return 0.0


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sortino Ratio (downside deviation only).
    
    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Trading periods per year
        
    Returns:
        Annualized Sortino Ratio
    """
    if returns is None or len(returns) < 2:
        return 0.0
    
    try:
        if EMPYRICAL_AVAILABLE:
            return float(empyrical.sortino_ratio(
                returns,
                required_return=risk_free_rate,
                annualization=annualization_factor
            ))
        else:
            # Manual calculation using downside deviation
            excess_returns = returns - risk_free_rate / annualization_factor
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return float('inf') if excess_returns.mean() > 0 else 0.0
            downside_std = np.sqrt((downside_returns ** 2).mean())
            return float(
                (excess_returns.mean() / downside_std) * np.sqrt(annualization_factor)
            )
    except Exception as e:
        logger.warning(f"Sortino calculation failed: {e}")
        return 0.0


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Series of equity values (not returns)
        
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.15 = -15%)
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    
    try:
        if EMPYRICAL_AVAILABLE:
            # empyrical expects returns, so we calculate returns from equity
            returns = equity_curve.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            return float(empyrical.max_drawdown(returns))
        else:
            # Manual calculation
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            return float(drawdown.min())
    except Exception as e:
        logger.warning(f"Max drawdown calculation failed: {e}")
        return 0.0


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate from trade list.
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if not trades:
        return 0.0
    
    try:
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return winning_trades / len(trades)
    except Exception as e:
        logger.warning(f"Win rate calculation failed: {e}")
        return 0.0


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dicts with 'pnl' field
        
    Returns:
        Profit factor (>1 means profitable)
    """
    if not trades:
        return 0.0
    
    try:
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    except Exception as e:
        logger.warning(f"Profit factor calculation failed: {e}")
        return 0.0


def get_performance_summary(
    trades: List[Dict],
    equity_history: Optional[List[Dict]] = None,
    annualization_factor: int = 252 * 24  # Hourly by default for HFT
) -> Dict:
    """
    Get comprehensive performance summary.
    
    Args:
        trades: List of trade dicts with 'pnl', 'timestamp' fields
        equity_history: Optional list of equity snapshots with 'timestamp', 'total_equity'
        annualization_factor: Periods per year for ratio calculations
        
    Returns:
        Dict with all performance metrics
    """
    summary = {
        'total_trades': len(trades) if trades else 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown_pct': 0.0,
        'total_pnl': 0.0,
        'avg_pnl_per_trade': 0.0,
    }
    
    if not trades:
        return summary
    
    try:
        # Basic metrics from trades
        summary['win_rate'] = calculate_win_rate(trades)
        summary['profit_factor'] = calculate_profit_factor(trades)
        summary['total_pnl'] = sum(t.get('pnl', 0) for t in trades)
        summary['avg_pnl_per_trade'] = summary['total_pnl'] / len(trades) if trades else 0
        
        # Convert trades to returns series for Sharpe/Sortino
        if len(trades) >= 2:
            # Group by timestamp and sum PnL for each period
            df = pd.DataFrame(trades)
            if 'pnl' in df.columns and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                returns = df.groupby(df['timestamp'].dt.floor('H'))['pnl'].sum()
                
                if len(returns) >= 2:
                    # Normalize to returns (percentage)
                    initial_equity = 100  # Assume $100 starting for returns calc
                    cumulative = returns.cumsum() + initial_equity
                    pct_returns = cumulative.pct_change().dropna()
                    
                    if len(pct_returns) >= 2:
                        summary['sharpe_ratio'] = calculate_sharpe_ratio(
                            pct_returns, annualization_factor=annualization_factor
                        )
                        summary['sortino_ratio'] = calculate_sortino_ratio(
                            pct_returns, annualization_factor=annualization_factor
                        )
        
        # Max drawdown from equity history if available
        if equity_history and len(equity_history) >= 2:
            equity_series = pd.Series([e.get('total_equity', 0) for e in equity_history])
            dd = calculate_max_drawdown(equity_series)
            summary['max_drawdown_pct'] = dd * 100  # Convert to percentage
        
    except Exception as e:
        logger.error(f"Performance summary failed: {e}", exc_info=True)
    
    return summary


# Convenience function for dashboard
def get_risk_metrics_for_dashboard(db_manager) -> Dict:
    """
    Fetch trades from database and calculate metrics for dashboard.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        Performance summary dict
    """
    import asyncio
    
    async def _fetch_and_calculate():
        try:
            # Get all trades from DB
            trades = await db_manager.get_recent_trades(limit=1000)
            
            # Get equity history if available
            # This would need a method in db_manager - for now use trades
            
            return get_performance_summary(trades, annualization_factor=252 * 24)
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return get_performance_summary([])
    
    # Run async function
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _fetch_and_calculate())
                return future.result(timeout=5.0)
        else:
            return asyncio.run(_fetch_and_calculate())
    except Exception as e:
        logger.error(f"Risk metrics fetch error: {e}")
        return get_performance_summary([])
