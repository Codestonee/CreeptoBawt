"""
Parameter Optimization Script using Optuna

Bayesian hyperparameter optimization for Avellaneda-Stoikov strategy parameters.
Uses existing backtest simulation to evaluate different parameter combinations.
"""

import asyncio
import logging
import sys
import os
import json
from typing import Dict, Any

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.trial import Trial

# Import from existing backtest simulation
from scripts.run_backtest_simulation import run_simulation

# Analytics for objective function
from utils.analytics import calculate_sharpe_ratio
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptunaOptimizer")

# Suppress noisy loggers during optimization
logging.getLogger("BacktestSimulator").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Default search space
DEFAULT_SEARCH_SPACE = {
    "gamma": {"low": 0.1, "high": 5.0, "step": 0.1},
    "kappa": {"low": 0.1, "high": 5.0, "step": 0.1},
}

# Output file for best params
OUTPUT_FILE = "data/optimized_params.json"


def create_objective(search_space: Dict[str, Dict] = None):
    """
    Create Optuna objective function.
    
    The objective is to MAXIMIZE Sharpe Ratio.
    """
    space = search_space or DEFAULT_SEARCH_SPACE
    
    def objective(trial: Trial) -> float:
        # Sample parameters from search space
        gamma = trial.suggest_float(
            "gamma",
            space["gamma"]["low"],
            space["gamma"]["high"],
            step=space["gamma"].get("step")
        )
        kappa = trial.suggest_float(
            "kappa",
            space["kappa"]["low"],
            space["kappa"]["high"],
            step=space["kappa"].get("step")
        )
        
        # Run simulation with these parameters
        try:
            result = asyncio.run(run_simulation(gamma=gamma, kappa=kappa))
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return float('-inf')  # Penalize failed runs
        
        # Check for errors
        if 'error' in result:
            logger.warning(f"Trial {trial.number} errored: {result['error']}")
            return float('-inf')
        
        # Calculate Sharpe Ratio from equity curve
        equity_history = result.get('equity_history', [])
        if len(equity_history) < 3:
            return float('-inf')  # Not enough data
        
        # Convert to returns
        equity_series = pd.Series([e['total_equity'] for e in equity_history])
        returns = equity_series.pct_change().dropna()
        
        if len(returns) < 2:
            return float('-inf')
        
        # Calculate Sharpe (using our analytics module)
        sharpe = calculate_sharpe_ratio(returns, annualization_factor=252)
        
        # Log progress
        pnl = result.get('total_pnl', 0)
        trades = result.get('num_trades', 0)
        logger.info(
            f"Trial {trial.number}: γ={gamma:.2f}, κ={kappa:.2f} → "
            f"Sharpe={sharpe:.3f}, PnL=${pnl:.2f}, Trades={trades}"
        )
        
        return sharpe
    
    return objective


def run_optimization(
    n_trials: int = 50,
    timeout: int = None,
    search_space: Dict = None,
    study_name: str = "avellaneda_stoikov_optimization"
) -> Dict[str, Any]:
    """
    Run Optuna optimization study.
    
    Args:
        n_trials: Number of trials to run
        timeout: Maximum time in seconds (optional)
        search_space: Custom search space dict
        study_name: Name for the study (for persistence)
        
    Returns:
        Dict with best params and statistics
    """
    logger.info(f"🔬 Starting Optuna optimization with {n_trials} trials...")
    
    # Create study (maximize Sharpe Ratio)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Create objective function
    objective = create_objective(search_space)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)  # Continue even if some trials fail
    )
    
    # Extract results
    best_params = study.best_params
    best_value = study.best_value
    
    # Get top 5 trials for comparison
    top_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value > float('-inf')],
        key=lambda t: t.value,
        reverse=True
    )[:5]
    
    results = {
        "best_params": best_params,
        "best_sharpe": best_value,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "top_5_trials": [
            {
                "params": t.params,
                "sharpe": t.value
            }
            for t in top_trials
        ]
    }
    
    # Save to file
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Best params saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Failed to save params: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Sharpe Ratio: {best_value:.4f}")
    print(f"Best Parameters:")
    for param, value in best_params.items():
        print(f"  - {param}: {value:.4f}")
    print(f"\nTrials completed: {results['n_completed']}/{results['n_trials']}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Avellaneda-Stoikov parameters")
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=20,
        help="Number of optimization trials (default: 20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds (optional)"
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=0.1,
        help="Minimum gamma value (default: 0.1)"
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=5.0,
        help="Maximum gamma value (default: 5.0)"
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=0.1,
        help="Minimum kappa value (default: 0.1)"
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=5.0,
        help="Maximum kappa value (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Build custom search space from args
    search_space = {
        "gamma": {"low": args.gamma_min, "high": args.gamma_max, "step": 0.1},
        "kappa": {"low": args.kappa_min, "high": args.kappa_max, "step": 0.1},
    }
    
    # Run optimization
    run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout,
        search_space=search_space
    )


if __name__ == "__main__":
    main()
