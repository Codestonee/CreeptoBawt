import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Analysis.BacktestReporter")

class BacktestReporter:
    """
    Generates interactive HTML reports for backtest results.
    """
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(
        self,
        symbol: str,
        equity_history: List[Dict[str, Any]],
        trade_history: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate HTML report and save to file.
        
        Args:
            symbol: Trading symbol
            equity_history: List of dicts (timestamp, total_equity, unrealized_pnl, etc)
            trade_history: List of dicts (timestamp, price, quantity, side, pnl)
            performance_metrics: Summary dict
            
        Returns:
            Path to generated report
        """
        if not equity_history:
            logger.warning("No equity history to report.")
            return ""
            
        # Convert to DataFrame
        df_equity = pd.DataFrame(equity_history)
        df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'], unit='s')
        
        df_trades = pd.DataFrame()
        if trade_history:
            df_trades = pd.DataFrame(trade_history)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='s')
            
        # Create Subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=(
                f"{symbol} Backtest Performance (Total PnL: ${performance_metrics.get('total_pnl', 0):.2f})",
                "Equity Curve & Drawdown",
                "Positions & Inventory"
            )
        )
        
        # === ROW 1: Price & Trades ===
        # Assuming we have price history? 
        # If simulated env doesn't pass tick history, we use equity timestamps marks?
        # Ideally we want the price chart.
        # Use 'mark_price' from equity history if available.
        if 'mark_price' in df_equity.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_equity['timestamp'], 
                    y=df_equity['mark_price'],
                    name="Price",
                    line=dict(color='gray', width=1)
                ),
                row=1, col=1
            )
            
        # Add Trades
        if not df_trades.empty:
            buys = df_trades[df_trades['side'] == 'BUY']
            sells = df_trades[df_trades['side'] == 'SELL']
            
            fig.add_trace(
                go.Scatter(
                    x=buys['timestamp'], y=buys['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sells['timestamp'], y=sells['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )

        # === ROW 2: Equity & Drawdown ===
        fig.add_trace(
            go.Scatter(
                x=df_equity['timestamp'], 
                y=df_equity['total_equity'],
                name="Total Equity",
                line=dict(color='cyan', width=2)
            ),
            row=2, col=1
        )
        
        # Drawdown (fill to zero)
        # Calculate local drawdown
        peak = df_equity['total_equity'].cummax()
        drawdown = df_equity['total_equity'] - peak
        
        fig.add_trace(
            go.Scatter(
                x=df_equity['timestamp'], 
                y=drawdown,
                name="Drawdown",
                fill='tozeroy',
                line=dict(color='red', width=0),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=1
        )
        
        # === ROW 3: Inventory ===
        # Assuming 'inventory' or 'position' in equity history
        if 'inventory' in df_equity.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_equity['timestamp'], 
                    y=df_equity['inventory'],
                    name="Inventory",
                    line=dict(color='yellow', width=1),
                    fill='tozeroy'
                ),
                row=3, col=1
            )
            
        # Layout Polish
        fig.update_layout(
            template="plotly_dark",
            height=1000,
            title_text=f"Backtest Report: {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            hovermode="x unified"
        )
        
        # Save output
        filename = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        
        logger.info(f"Report generated: {filepath}")
        return filepath
