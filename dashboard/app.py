import streamlit as st
import pandas as pd
import sqlite3
import time
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Titan HFT Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(r"""
<style>
    /* Global Styles */
    .stApp {
        background-color: #222831;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #393E46;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4B5563;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Text Colors */
    h1, h2, h3 {
        color: #00ADB5 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, span, div {
        color: #EEEEEE;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border: 1px solid #4B5563;
        border-radius: 5px;
    }
    
    /* Status Indicator */
    .status-online {
        color: #00FF00;
        font-weight: bold;
    }
    .status-offline {
        color: #FF0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
def get_data(db_path='trading_data.db'):
    """Fetch trade data from SQLite."""
    try:
        if not os.path.exists(db_path):
            return pd.DataFrame()
            
        conn = sqlite3.connect(db_path)
        # Read all data for calculations
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
        conn.close()
        
        # Convert timestamp to datetime if not already
        if not df.empty and 'timestamp' in df.columns:
            # Check if timestamp is float (unix) or string
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            except:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def get_logs(log_file="bot_execution.log", lines=50):
    """Read the last N lines of the log file."""
    if not os.path.exists(log_file):
        return ["Log file not found."]
    
    try:
        with open(log_file, "r", encoding='utf-8') as f:
            return f.readlines()[-lines:]
    except Exception as e:
        return [f"Error reading logs: {e}"]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_metrics(df):
    if df.empty:
        return None
        
    total_pnl = df['pnl'].sum()
    total_trades = len(df)
    
    closed_trades = df[df['pnl'] != 0]
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] < 0]
    
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if not closed_trades.empty else 0.0
    avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0.0
    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0.0
    
    # Max Drawdown
    df_sorted = df.sort_values('id')
    cumulative = df_sorted['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()
    
    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "cumulative_pnl": cumulative,
        "ids": df_sorted['id']
    }

def handle_emergency_stop():
    """Toggle emergency stop."""
    flag_file = 'EMERGENCY_STOP.flag'
    if os.path.exists(flag_file):
        os.remove(flag_file)
        return False # Not stopped
    else:
        with open(flag_file, 'w') as f:
            f.write(f'STOPPED at {datetime.now()}')
        return True # Stopped

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("⚙️ Control Panel")
    
    # Status
    is_stopped = os.path.exists('EMERGENCY_STOP.flag')
    status_color = "red" if is_stopped else "green"
    status_text = "STOPPED" if is_stopped else "RUNNING"
    st.sidebar.markdown(f"**Bot Status:** <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
    
    if st.sidebar.button("🚨 EMERGENCY STOP / RESUME", type="primary"):
        new_state = handle_emergency_stop()
        if new_state:
            st.sidebar.error("Bot STOPPED!")
        else:
            st.sidebar.success("Bot RESUMED!")
            
    st.sidebar.markdown("---")
    
    # Settings
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 2)
    show_logs = st.sidebar.checkbox("Show Live Logs", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Titan HFT v2.0\nCreated by Claude & Gemini")
    
    return refresh_rate, show_logs

def render_kpi_row(metrics):
    if not metrics:
        st.warning("No data available yet.")
        return

    cols = st.columns(4)
    
    # PnL Card
    with cols[0]:
        st.metric(
            label="💰 Net Profit",
            value=f"${metrics['total_pnl']:.2f}",
            delta=f"{metrics['total_pnl']:.2f}",
            delta_color="normal"
        )
        
    # Win Rate Card
    with cols[1]:
        st.metric(
            label="🎯 Win Rate",
            value=f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['total_trades']} trades"
        )
        
    # Drawdown Card
    with cols[2]:
        st.metric(
            label="📉 Max Drawdown",
            value=f"${metrics['max_dd']:.2f}",
            delta="risk metric",
            delta_color="inverse"
        )
        
    # Avg Trade Card
    with cols[3]:
        avg_trade = (metrics['avg_win'] * (metrics['win_rate']/100)) + (metrics['avg_loss'] * (1 - metrics['win_rate']/100))
        st.metric(
            label="⚖️ Expectancy",
            value=f"${avg_trade:.3f}",
            delta="per trade"
        )

def render_charts(df, metrics):
    if metrics is None or df.empty:
        return
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Equity Curve")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=metrics['ids'], 
            y=metrics['cumulative_pnl'],
            mode='lines',
            name='Equity',
            line=dict(color='#00ADB5', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 173, 181, 0.2)'
        ))
        fig_equity.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, title='Trade #'),
            yaxis=dict(showgrid=True, gridcolor='#393E46', title='USDT'),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode='x unified',
            font=dict(color='#EEEEEE')
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
    with col2:
        st.subheader("⚖️ Buy vs Sell")
        fig_pie = px.pie(
            df, 
            names='side', 
            color='side',
            color_discrete_map={'BUY':'#00ADB5', 'SELL':'#FF2E63'},
            hole=0.6
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(color='#EEEEEE'),
            annotations=[dict(text=f"{len(df)}", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def render_recent_trades(df):
    if df.empty:
        return
        
    st.subheader("📜 Recent Activity")
    
    # Filter columns
    display_df = df[['timestamp', 'symbol', 'side', 'quantity', 'price', 'pnl']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
    
    # Style the dataframe
    def style_pnl(val):
        color = '#00FF00' if val > 0 else '#FF0000' if val < 0 else '#AAAAAA'
        return f'color: {color}'

    st.dataframe(
        display_df.head(15).style.map(style_pnl, subset=['pnl'])
        .format({'price': '{:.2f}', 'quantity': '{:.4f}', 'pnl': '{:.4f}'}),
        use_container_width=True,
        height=400
    )

def render_logs(log_lines):
    st.subheader("💻 System Logs")
    log_text = "".join(log_lines)
    st.code(log_text, language="text")

# -----------------------------------------------------------------------------
# MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    refresh_rate, show_logs = render_sidebar()
    
    # Main Dashboard Container
    main_placeholder = st.empty()
    
    while True:
        with main_placeholder.container():
            # 1. Fetch Data
            df = get_data()
            metrics = calculate_metrics(df)
            
            # 2. Header
            st.title("🚀 Titan HFT Dashboard")
            last_update = datetime.now().strftime("%H:%M:%S")
            st.caption(f"Last updated: {last_update}")
            
            # 3. KPIs
            render_kpi_row(metrics)
            
            # 4. Divider
            st.markdown("---")
            
            # 5. Charts & Tables
            render_charts(df, metrics)
            render_recent_trades(df)
            
            # 6. Logs (Optional)
            if show_logs:
                st.markdown("---")
                logs = get_logs()
                render_logs(logs)
                
        time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
