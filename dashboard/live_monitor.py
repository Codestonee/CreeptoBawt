"""
Live Trading Dashboard

Real-time monitoring of:
- Current positions
- P&L (realized + unrealized)
- Recent trades
- Strategy state
- System health

Usage:
    streamlit run dashboard/live_monitor.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Trading Bot Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database path
DB_PATH = "data/trading_data.db"

def get_db_connection():
    """Get SQLite connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=2)  # Cache for 2 seconds
def load_recent_trades(limit=50):
    """Load recent trades from database."""
    conn = get_db_connection()
    query = f"""
        SELECT 
            timestamp,
            symbol,
            side,
            quantity,
            price,
            commission,
            realized_pnl,
            strategy_id
        FROM trades
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except Exception as e:
        conn.close()
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_current_positions():
    """Load current positions."""
    conn = get_db_connection()
    query = """
        SELECT 
            symbol,
            quantity,
            avg_entry_price,
            unrealized_pnl,
            updated_at
        FROM positions
        WHERE ABS(quantity) > 0.0001
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['updated_at'] = pd.to_datetime(df['updated_at'], unit='s')
        return df
    except Exception as e:
        conn.close()
        st.error(f"Error loading positions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_equity_curve(hours=24):
    """Load equity curve from trades."""
    conn = get_db_connection()
    
    # Get trades from last N hours
    cutoff = time.time() - (hours * 3600)
    query = f"""
        SELECT 
            timestamp,
            realized_pnl
        FROM trades
        WHERE timestamp > {cutoff}
        ORDER BY timestamp ASC
    """
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['cumulative_pnl'] = df['realized_pnl'].cumsum()
        return df
    except Exception as e:
        conn.close()
        st.error(f"Error loading equity: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_system_health():
    """Load system health metrics."""
    conn = get_db_connection()
    
    try:
        # Recent fill rate
        recent_cutoff = time.time() - 3600  # Last hour
        query = f"""
            SELECT COUNT(*) as fills_last_hour
            FROM trades
            WHERE timestamp > {recent_cutoff}
        """
        df = pd.read_sql_query(query, conn)
        fills_last_hour = df['fills_last_hour'].iloc[0] if not df.empty else 0
        
        # Total trades today
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        query = f"""
            SELECT 
                COUNT(*) as total_trades,
                SUM(realized_pnl) as total_pnl
            FROM trades
            WHERE timestamp > {today_start}
        """
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        return {
            'fills_last_hour': fills_last_hour,
            'total_trades_today': df['total_trades'].iloc[0] if not df.empty else 0,
            'total_pnl_today': df['total_pnl'].iloc[0] if not df.empty else 0.0
        }
    except Exception as e:
        conn.close()
        st.error(f"Error loading health: {e}")
        return {
            'fills_last_hour': 0,
            'total_trades_today': 0,
            'total_pnl_today': 0.0
        }

# Main dashboard
def main():
    st.title("🤖 Trading Bot Live Monitor")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
        show_debug = st.checkbox("Show debug info", value=False)
        
        st.markdown("---")
        st.markdown("**Database:** `data/trading_data.db`")
        
        if Path(DB_PATH).exists():
            db_size = Path(DB_PATH).stat().st_size / 1024 / 1024
            st.markdown(f"**Size:** {db_size:.2f} MB")
        else:
            st.error("❌ Database not found!")
            return
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    
    # System health metrics (top row)
    st.header("📊 System Health")
    
    health = load_system_health()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fills (Last Hour)",
            health['fills_last_hour'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Trades Today",
            health['total_trades_today']
        )
    
    with col3:
        pnl_today = health['total_pnl_today']
        st.metric(
            "P&L Today",
            f"${pnl_today:.2f}",
            delta=f"{pnl_today:+.2f}",
            delta_color="normal" if pnl_today >= 0 else "inverse"
        )
    
    with col4:
        # Last update time
        st.metric(
            "Last Update",
            datetime.now().strftime("%H:%M:%S")
        )
    
    st.markdown("---")
    
    # Current positions
    st.header("💼 Current Positions")
    
    positions = load_current_positions()
    
    if not positions.empty:
        # Add current P&L estimate
        positions['position_usd'] = positions['quantity'] * positions['avg_entry_price']
        positions['unrealized_pnl'] = positions['unrealized_pnl'].fillna(0)
        
        # Format for display
        display_df = positions[['symbol', 'quantity', 'avg_entry_price', 'position_usd', 'unrealized_pnl']].copy()
        display_df.columns = ['Symbol', 'Quantity', 'Avg Price', 'Position (USD)', 'Unrealized P&L']
        
        st.dataframe(
            display_df.style.format({
                'Quantity': '{:,.4f}',
                'Avg Price': '${:,.2f}',
                'Position (USD)': '${:,.2f}',
                'Unrealized P&L': '${:+,.2f}'
            }),
            use_container_width=True
        )
        
        # Position summary
        total_exposure = display_df['Position (USD)'].abs().sum()
        total_unrealized = display_df['Unrealized P&L'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Exposure", f"${total_exposure:,.2f}")
        with col2:
            st.metric("Total Unrealized P&L", f"${total_unrealized:+,.2f}")
    else:
        st.info("No open positions")
    
    st.markdown("---")
    
    # Equity curve
    st.header("📈 Equity Curve (24h)")
    
    equity_df = load_equity_curve(hours=24)
    
    if not equity_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#00CC96', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        
        fig.update_layout(
            title="Cumulative Realized P&L",
            xaxis_title="Time",
            yaxis_title="P&L (USD)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        total_pnl = equity_df['cumulative_pnl'].iloc[-1]
        max_pnl = equity_df['cumulative_pnl'].max()
        min_pnl = equity_df['cumulative_pnl'].min()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current P&L", f"${total_pnl:+,.2f}")
        with col2:
            st.metric("Peak P&L", f"${max_pnl:+,.2f}")
        with col3:
            st.metric("Drawdown", f"${total_pnl - max_pnl:+,.2f}")
    else:
        st.info("No trades in last 24 hours")
    
    st.markdown("---")
    
    # Recent trades
    st.header("📝 Recent Trades")
    
    trades = load_recent_trades(limit=50)
    
    if not trades.empty:
        # Format for display
        display_trades = trades.copy()
        display_trades['timestamp'] = display_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Color code by side
        def highlight_side(row):
            if row['side'] == 'BUY':
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            else:
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        
        st.dataframe(
            display_trades.style.apply(highlight_side, axis=1).format({
                'quantity': '{:,.4f}',
                'price': '${:,.2f}',
                'commission': '${:.4f}',
                'realized_pnl': '${:+,.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Trade statistics
        st.subheader("📊 Trade Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            win_rate = (trades['realized_pnl'] > 0).sum() / len(trades) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            avg_pnl = trades['realized_pnl'].mean()
            st.metric("Avg P&L", f"${avg_pnl:+,.2f}")
        
        with col3:
            total_commission = trades['commission'].sum()
            st.metric("Total Fees", f"${total_commission:,.2f}")
        
        with col4:
            total_realized = trades['realized_pnl'].sum()
            st.metric("Total Realized", f"${total_realized:+,.2f}")
    else:
        st.info("No recent trades")
    
    # Debug info
    if show_debug:
        st.markdown("---")
        st.header("🐛 Debug Info")
        
        with st.expander("Database Tables"):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            st.write([t[0] for t in tables])
            conn.close()
        
        with st.expander("Raw Positions Data"):
            st.dataframe(positions)
        
        with st.expander("Raw Trades Data"):
            st.dataframe(trades.head(10))

if __name__ == "__main__":
    main()
