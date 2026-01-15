"""
Dashboard Data Updater

Runs alongside the bot to write dashboard-compatible JSON files
from monitoring components.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

# Import monitoring components
import sys
sys.path.append('.')

from monitoring.health_monitor import get_health_monitor
from monitoring.performance_tracker import get_performance_tracker
from monitoring.alert_manager import get_alert_manager


async def update_dashboard_data():
    """Update dashboard JSON files every 2 seconds."""
    
    Path("data").mkdir(exist_ok=True)
    
    while True:
        try:
            # Health metrics
            health_monitor = get_health_monitor()
            metrics = health_monitor.get_current_metrics()
            
            if metrics:
                health_data = {
                    'ws_latency_ms': 15,  # TODO: Get from WebSocket
                    'position_sync_ok': metrics.position_sync_ok,
                    'fills_last_hour': metrics.fills_last_hour,
                    'memory_mb': metrics.memory_usage_mb,
                    'cpu_pct': metrics.cpu_usage_pct,
                    'uptime_seconds': time.time() - 0,  # TODO: Track start time
                    'is_healthy': metrics.is_healthy,
                    'issues': metrics.issues
                }
                
                with open('data/health_status.json', 'w') as f:
                    json.dump(health_data, f, indent=2)
            
            # Performance metrics
            perf_tracker = get_performance_tracker()
            perf_metrics = perf_tracker.get_metrics()
            
            perf_data = {
                'total_pnl': perf_metrics.total_pnl,
                'total_return_pct': perf_metrics.total_return_pct,
                'sharpe_ratio_30d': perf_metrics.sharpe_ratio_30d,
                'sortino_ratio_30d': perf_metrics.sortino_ratio_30d,
                'max_drawdown_pct': perf_metrics.max_drawdown_pct,
                'current_drawdown_pct': perf_metrics.current_drawdown_pct,
                'win_rate_pct': perf_metrics.win_rate_pct,
                'profit_factor': perf_metrics.profit_factor,
                'total_trades': perf_metrics.total_trades,
                'total_fees_paid': perf_metrics.total_fees_paid
            }
            
            with open('data/performance_metrics.json', 'w') as f:
                json.dump(perf_data, f, indent=2)
            
            # Alert stats
            alert_manager = get_alert_manager()
            alert_stats = alert_manager.get_stats()
            
            # Recent alerts (simplified - full implementation would track alert history)
            alerts_data = []  # TODO: Maintain alert history
            
            with open('data/alerts.json', 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            await asyncio.sleep(2)
        
        except Exception as e:
            print(f"Dashboard update error: {e}")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(update_dashboard_data())
