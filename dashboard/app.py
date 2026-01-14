"""
CreeptBaws# Titan HFT Dashboard - Safety-First Edition (v3.0.1)al Trading Dashboard
3-Column Cockpit Layout: Controls | Market | Feed

Design Philosophy:
- Left: Control & Risk (Your hands)
- Center: Market & Strategy (Your eyes)  
- Right: Logs & Positions (Your memory)
- No scroll: Everything fits on 1080p/1440p
"""
import streamlit as st
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CreeptBaws | Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
def load_css():
    css_path = "dashboard/style.css"
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Session state
if 'panic_armed' not in st.session_state:
    st.session_state.panic_armed = False

# =============================================================================
# SIGNAL FILES (Emergency Controls)
# =============================================================================
STOP_SIGNAL_FILE = "data/STOP_SIGNAL"
PAUSE_SIGNAL_FILE = "data/PAUSE_SIGNAL"

def send_stop_signal():
    with open(STOP_SIGNAL_FILE, "w") as f:
        f.write(f"STOP:{datetime.now().isoformat()}")

def send_pause_signal():
    with open(PAUSE_SIGNAL_FILE, "w") as f:
        f.write(f"PAUSE:{datetime.now().isoformat()}")

def clear_signals():
    for f in [STOP_SIGNAL_FILE, PAUSE_SIGNAL_FILE]:
        if os.path.exists(f):
            os.remove(f)

def get_signal_status():
    if os.path.exists(STOP_SIGNAL_FILE):
        return "STOPPED", "error"
    elif os.path.exists(PAUSE_SIGNAL_FILE):
        return "PAUSED", "warn"
    return "ACTIVE", "ok"

# =============================================================================
# DATABASE
# =============================================================================
def get_db_connection():
    db_path = 'data/trading_data.db'
    if os.path.exists(db_path):
        return sqlite3.connect(db_path)
    return None

@st.cache_data(ttl=2)
def get_metrics():
    """Calculate all key metrics."""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Fetch ALL trades for accurate PnL/Win Rate (No LIMIT)
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
        positions = pd.read_sql_query("SELECT * FROM positions", conn)
        
        # Get all-time best/worst stats directly from DB
        best_trade_db = pd.read_sql_query("SELECT * FROM trades ORDER BY pnl DESC LIMIT 1", conn)
        worst_trade_db = pd.read_sql_query("SELECT * FROM trades ORDER BY pnl ASC LIMIT 1", conn)
        
        conn.close()
        
        if trades.empty:
            return {
                "balance": settings.INITIAL_CAPITAL,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "trades_count": 0,
                "win_rate": 0.0,
                "win_rate_total": 0.0,
                "win_rate_decision": 0.0,
                "winners": 0,
                "losers": 0,
                "breakevens": 0,
                "total_pnl_green": 0.0,
                "total_pnl_red": 0.0,
                "equity_curve": [settings.INITIAL_CAPITAL],
                "recent_trades": pd.DataFrame(),
                "positions": positions,
                "maker_rate": 0,
                "best_trade": 0,
                "worst_trade": 0,
                "best_info": "",
                "worst_info": "",
                "inventory_breakdown": {}
            }
        
        pnl_total = trades['pnl'].sum()
        trades_count = len(trades)
        winners = len(trades[trades['pnl'] > 0])
        losers = len(trades[trades['pnl'] < 0])
        breakevens = len(trades[trades['pnl'] == 0])
        
        # Calculate total PnL for winners and losers
        total_pnl_green = trades[trades['pnl'] > 0]['pnl'].sum()
        total_pnl_red = trades[trades['pnl'] < 0]['pnl'].sum()
        
        # Standard Win Rate: Winners / Total Trades
        win_rate_total = (winners / trades_count * 100) if trades_count > 0 else 0
        
        # Decision Win Rate: Winners / (Winners + Losers) - ignores breakeven
        decisive_trades = winners + losers
        win_rate_decision = (winners / decisive_trades * 100) if decisive_trades > 0 else 0
        
        # Equity curve
        # Equity curve
        trades_sorted = trades.sort_values('timestamp')
        # Create DataFrame with time and equity
        equity_df = pd.DataFrame()
        if not trades_sorted.empty:
            # Convert to UTC-aware timestamps immediately and floor to microseconds
            equity_df['timestamp'] = pd.to_datetime(trades_sorted['timestamp'], unit='s', utc=True).dt.floor('us')
            equity_df['equity'] = trades_sorted['pnl'].cumsum() + settings.INITIAL_CAPITAL
            # Add initial point? Maybe complex for now, let's stick to trade points.
        
        
        # Best/worst trades
        best_pnl = trades['pnl'].max()
        # Best/worst trades (All Time)
        if not best_trade_db.empty:
            best_row = best_trade_db.iloc[0]
            best_pnl = float(best_row['pnl'])
            best_info = f"{best_row.get('symbol', 'N/A').upper()}"
            if 'timestamp' in best_row:
                best_info += f" @ {pd.to_datetime(best_row['timestamp'], unit='s').strftime('%H:%M:%S')}"
        else:
            best_pnl = 0
            best_info = "N/A"

        if not worst_trade_db.empty:
            worst_row = worst_trade_db.iloc[0]
            worst_pnl = float(worst_row['pnl'])
            worst_info = f"{worst_row.get('symbol', 'N/A').upper()}"
            if 'timestamp' in worst_row:
                worst_info += f" @ {pd.to_datetime(worst_row['timestamp'], unit='s').strftime('%H:%M:%S')}"
        else:
            worst_pnl = 0
            worst_info = "N/A"
        
        # Unrealized PnL
        unrealized_pnl = positions['unrealized_pnl'].sum() if 'unrealized_pnl' in positions.columns else 0
        
        # Inventory breakdown
        inventory = {}
        if not positions.empty and 'symbol' in positions.columns:
            for _, row in positions.iterrows():
                if row['quantity'] != 0:
                    inventory[row['symbol'].upper()] = {
                        'qty': row['quantity'],
                        'entry': row.get('avg_entry_price', 0),
                        'pnl': row.get('unrealized_pnl', 0)
                    }
        
        # Maker rate
        maker_rate = 0
        if 'is_maker' in trades.columns:
            maker_rate = (trades['is_maker'].sum() / trades_count * 100) if trades_count > 0 else 0
        
        return {
            "balance": settings.INITIAL_CAPITAL + pnl_total + unrealized_pnl,
            "realized_pnl": pnl_total,
            "unrealized_pnl": unrealized_pnl,
            "trades_count": trades_count,
            "trades_count": trades_count,
            "win_rate_total": win_rate_total,
            "win_rate_decision": win_rate_decision,
            "winners": winners,
            "losers": losers,
            "breakevens": breakevens,
            "total_pnl_green": total_pnl_green,
            "total_pnl_red": total_pnl_red,
            "total_pnl_green": total_pnl_green,
            "total_pnl_red": total_pnl_red,
            "equity_curve": equity_df,
            "recent_trades": trades,
            "positions": positions,
            "maker_rate": maker_rate,
            "best_trade": best_pnl,
            "worst_trade": worst_pnl,
            "best_info": best_info,
            "worst_info": worst_info,
            "inventory_breakdown": inventory
        }
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

@st.cache_data(ttl=3)
def get_system_health():
    """Get system health metrics."""
    health = {
        "ws_status": "OK",
        "ws_latency": 12,
        "last_reconcile": "2s ago",
        "db_status": "OK",
        "router_mode": "LIMIT_CHASE",
        "gtx_rejections": 0
    }
    
    # Check for health file
    if os.path.exists("health_status.json"):
        try:
            import json
            with open("health_status.json") as f:
                health.update(json.load(f))
        except:
            pass
    
    return health

# =============================================================================
# HEADER
# =============================================================================
def render_header():
    """Top status bar."""
    status, status_type = get_signal_status()
    health = get_system_health()
    
    cols = st.columns([3, 2, 1, 1, 1, 1])
    
    with cols[0]:
        st.markdown("### 🛡️ CREEPTBAWS COMMAND")
    
    with cols[1]:
        if status == "ACTIVE":
            st.markdown(f"<span style='color:#3FB950'>● {status}</span>", unsafe_allow_html=True)
        elif status == "PAUSED":
            st.markdown(f"<span style='color:#D29922'>● {status}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#F85149'>● {status}</span>", unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"`WS: {health['ws_status']} ({health['ws_latency']}ms)`")
    
    with cols[3]:
        st.markdown(f"`RECON: {health['last_reconcile']}`")
    
    with cols[4]:
        st.markdown(f"`{health['router_mode']}`")
    
    with cols[5]:
        st.markdown(f"`{datetime.now().strftime('%H:%M:%S')}`")
    
    st.markdown("---")

# =============================================================================
# LEFT COLUMN: CONTROLS & RISK
# =============================================================================
def render_controls(metrics, health):
    """Control panel (Static to prevent button flicker)."""
    # EMERGENCY CONTROLS
    st.markdown("##### ⚠️ EMERGENCY")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏸️ PAUSE", help="Stop new orders", width="stretch"):
            send_pause_signal()
            st.toast("⏸️ Paused", icon="⏸️")
            st.rerun()
    
    with c2:
        if st.session_state.panic_armed:
            if st.button("🔥 CONFIRM", type="primary", width="stretch"):
                send_stop_signal()
                st.session_state.panic_armed = False
                st.toast("🛑 FLATTEN SENT", icon="🔥")
                st.rerun()
        else:
            if st.button("🛑 FLATTEN", help="Arm flatten", width="stretch"):
                st.session_state.panic_armed = True
                st.rerun()
    
    if st.session_state.panic_armed:
        st.warning("⚠️ Click CONFIRM to flatten all")
    
    status, _ = get_signal_status()
    if status != "ACTIVE":
        if st.button("▶️ RESUME", width="stretch"):
            clear_signals()
            st.session_state.panic_armed = False
            st.toast("▶️ Resumed", icon="✅")
            st.rerun()
    
    st.markdown("---")

def setup_account_stats(container):
    """Setup static layout for Left Column Stats to avoid flicker."""
    phs = {}
    with container:
        st.markdown("##### 💰 ACCOUNT")
        
        # Net Liq
        phs['net_liq'] = st.empty()
        
        col1, col2 = st.columns(2)
        phs['realized'] = col1.empty()
        phs['unrealized'] = col2.empty()
        
        st.markdown("---")
        st.markdown("##### 📊 PERFORMANCE")
        
        col1, col2 = st.columns(2)
        with col1:
             # Checkbox must be interactive, so it creates state.
             phs['wr_checkbox_val'] = st.checkbox("Excl. Break-even", value=False, key="wr_toggle", help="Show Win Rate excluding 0 PnL trades")
             phs['win_rate'] = st.empty()
             phs['win_count'] = st.empty()
        
        with col2:
             phs['maker_rate'] = st.empty()
             phs['gtx_rej'] = st.empty()
             
        col1, col2 = st.columns(2)
        # Fix: Separate placeholders to avoid overwrite
        with col1:
            phs['best_val'] = st.empty()
            phs['best_info'] = st.empty()
            
        with col2:
            phs['worst_val'] = st.empty()
            phs['worst_info'] = st.empty()
        
        return phs

def update_account_stats(phs, metrics, health):
    """Update values in existing placeholders."""
    if not metrics: return
    
    balance = metrics['balance']
    pnl_pct = ((balance - settings.INITIAL_CAPITAL) / settings.INITIAL_CAPITAL * 100)
    phs['net_liq'].metric("Net Liquidation", f"${balance:,.2f}", f"{pnl_pct:+.2f}%")
    
    phs['realized'].metric("Realized", f"${metrics['realized_pnl']:+.2f}")
    phs['unrealized'].metric("Unrealized", f"${metrics['unrealized_pnl']:+.2f}")
    
    # WR
    use_decision_wr = phs.get('wr_checkbox_val', False)
    wr_value = metrics['win_rate_decision'] if use_decision_wr else metrics['win_rate_total']
    wr_label = "Win Rate (Dec)" if use_decision_wr else "Win Rate (All)"
    phs['win_rate'].metric(wr_label, f"{wr_value:.1f}%")
    
    
    
    # Update caption with PnL totals (Exact User Format)
    # Format: ✅ x / ❌ x (green total / red total)
    counts_str = f"✅ {metrics['winners']} / ❌ {metrics['losers']}"
    totals_str = f"(:green[${metrics['total_pnl_green']:+.2f}] / :red[${metrics['total_pnl_red']:+.2f}])"
    
    final_md = f"{counts_str} {totals_str}"
    
    # Add Break-even count if Excl. Break-even is un-checked
    if not use_decision_wr:
        final_md += f" :grey[⚪ {metrics['breakevens']}]"
        
    phs['win_count'].markdown(final_md)
    
    # Maker
    phs['maker_rate'].metric("Maker Rate", f"{metrics['maker_rate']:.0f}%")
    phs['gtx_rej'].caption(f"GTX rej: {health.get('gtx_rejections', 0)}")
    
    # Best/Worst
    c1 = "normal" if metrics['best_trade'] > 0 else "off"
    phs['best_val'].metric("🏆 Best", f"${metrics['best_trade']:+.2f}", delta_color=c1)
    phs['best_info'].caption(metrics['best_info'])
    
    c2 = "normal" if metrics['worst_trade'] > 0 else "inverse"
    phs['worst_val'].metric("💀 Worst", f"${metrics['worst_trade']:+.2f}", delta_color=c2)
    phs['worst_info'].caption(metrics['worst_info']) # Added caption update
    phs['best_info'].caption(metrics['best_info'])   # Added caption update

# =============================================================================
# CENTER COLUMN: MARKET & STRATEGY
# =============================================================================
def setup_center_column(container):
    """Setup static layout for Center Column."""
    phs = {}
    with container:
        tab1, tab2 = st.tabs(["📈 EQUITY & POSITIONS", "🧠 STRATEGY STATE"])
        
        with tab1:
            # Header with Selector
            c1, c2 = st.columns([2, 2])
            with c1:
                st.markdown("##### 📈 EQUITY CURVE")
            with c2:
                # Time Window Selector
                phs['time_window'] = st.radio(
                    "Time Window", 
                    ["10M", "1H", "4H", "24H"], 
                    horizontal=True, 
                    label_visibility="collapsed",
                    key="equity_window_selector",
                    index=1 # Default 1H
                )
            
            # Equity Chart Placeholder
            phs['equity_chart'] = st.empty()
            st.markdown("---")
            st.markdown("##### 📍 POSITIONS")
            # Positions Placeholder
            phs['positions'] = st.empty()
            
        with tab2:
            st.markdown("##### 🤖 STRATEGY INTERNALS")
            # Strategy Stats Placeholder
            phs['strategy_stats'] = st.empty()
            
    return phs

def update_center_column(phs, metrics):
    """Update Center Column dynamic content."""
    
    # 1. EQUITY CHART
    if metrics is not None:
        eq_df = metrics['equity_curve']
        
        # Get Time Window
        window_map = {"10M": 10, "1H": 60, "4H": 240, "24H": 1440}
        selected_window = st.session_state.get("equity_window_selector", "1H")
        minutes = window_map.get(selected_window, 60)
        
        from datetime import timezone
        # Use UTC for all time calculations
        now = datetime.now(timezone.utc)
        start_time = now - pd.Timedelta(minutes=minutes)
        
        # Filter Data
        chart_data = pd.DataFrame()
        if not eq_df.empty:
            # Ensure timestamps are UTC-aware for comparison
            if eq_df['timestamp'].dt.tz is None:
                eq_df['timestamp'] = eq_df['timestamp'].dt.tz_localize('UTC')
            else:
                eq_df['timestamp'] = eq_df['timestamp'].dt.tz_convert('UTC')
                
            chart_data = eq_df[eq_df['timestamp'] >= start_time].copy()
            # Cast to microsecond to avoid Plotly warning (preserving previous fix)
            # chart_data['timestamp'] = chart_data['timestamp'].astype('datetime64[us]') 
            # Note: tz-aware datetime64[ns, UTC] might still trigger warnings if not handled, 
            # but usually the issue is specifically high-precision nanoseconds.
            # Let's trust pandas truncation or explicit cast if needed. 
            # Safe bet: Convert to standard separate ISO string or just let Plotly handle datetime objects if they are clean.
        
        start_capital = settings.INITIAL_CAPITAL
        
        # Determine Color based on latest equity vs start of WINDOW (or absolute start?)
        current_equity = chart_data['equity'].iloc[-1] if not chart_data.empty else start_capital
        line_color = '#3FB950' if current_equity >= start_capital else '#F85149'
        fill_color = 'rgba(63, 185, 80, 0.1)' if current_equity >= start_capital else 'rgba(248, 81, 73, 0.1)'

        fig = go.Figure()
        
        x_min_range = start_time
        x_max_range = now

        if not chart_data.empty:
            # Main Line
            fig.add_trace(go.Scatter(
                x=chart_data['timestamp'],
                y=chart_data['equity'],
                mode='lines',
                fill='tozeroy',
                line=dict(color=line_color, width=2),
                fillcolor=fill_color,
                hoverinfo='y+x'
            ))
            
            # Pulsing Dot (Current Value)
            last_pt = chart_data.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_pt['timestamp']],
                y=[last_pt['equity']],
                mode='markers',
                marker=dict(
                    color='#3FB950',
                    size=12,
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Dynamic X-Axis Start:
            # If data started AFTER the window start (e.g. bot just started), snap to data start.
            # If data started BEFORE window start (e.g. long running), snap to window start.
            data_start = chart_data['timestamp'].min()
            x_min_range = max(start_time, data_start)
            
            # Add small padding to right so the dot isn't cut off
            x_max_range = now + pd.Timedelta(seconds=30*minutes/60) # scaled padding

        # Dynamic Y-Axis Range logic
        y_min = start_capital
        y_max = start_capital
        if not chart_data.empty:
            y_min = chart_data['equity'].min()
            y_max = chart_data['equity'].max()
        
        # Add padding
        y_range = max(y_max - y_min, 10) # Min $10 range
        y_min -= y_range * 0.1
        y_max += y_range * 0.1

        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=False, 
                showticklabels=True,
                tickformat='%H:%M',
                range=[x_min_range, x_max_range], # Dynamic Range
                gridcolor='#21262D'
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#21262D', 
                tickformat='$,.0f',
                range=[y_min, y_max] # Dynamic Y based on data
            ),
            showlegend=False
        )
        phs['equity_chart'].plotly_chart(fig, width="stretch", key=f"eq_chart_{time.time()}")
    else:
        phs['equity_chart'].info("Loading...")

    # 2. POSITIONS
    CRYPTO_ICONS = {
        'BTC': '🟠', 'BTCUSDT': '🟠', 'ETH': '🔷', 'ETHUSDT': '🔷',
        'SOL': '🟣', 'SOLUSDT': '🟣', 'DOGE': '🐕', 'DOGEUSDT': '🐕',
        'XRP': '⚫', 'XRPUSDT': '⚫', 'BNB': '🟡', 'BNBUSDT': '🟡',
        'ADA': '🔵', 'ADAUSDT': '🔵', 'LTC': '⚪', 'LTCUSDT': '⚪'
    }
    
    pos_html = ""
    if metrics and metrics['inventory_breakdown']:
        for sym, data in metrics['inventory_breakdown'].items():
            icon = CRYPTO_ICONS.get(sym.upper(), '💎')
            qty = data['qty']
            pnl = data['pnl']
            
            if qty > 0:
                dir_emoji = "📈"; dir_text = "LONG"; border_color = "#3FB950"
            else:
                dir_emoji = "📉"; dir_text = "SHORT"; border_color = "#F85149"
            
            pnl_color = "#3FB950" if pnl > 0 else "#F85149" if pnl < 0 else "#8B949E"
            
            pos_html += f"""
<div style="display:flex; justify-content:space-between; padding:8px; margin:4px 0; background:#161B22; border-radius:6px; border-left:3px solid {border_color};">
<div><span style="font-size:1.1rem">{icon}</span> <b style="color:#E6EDF3">{sym.upper()}</b> <span style="color:{border_color}; font-size:0.75rem">{dir_emoji} {dir_text}</span></div>
<div style="text-align:right"><div style="color:#8B949E; font-size:0.75rem">Size: {qty:+.4f}</div><div style="color:{pnl_color}; font-weight:600">${pnl:+.2f}</div></div>
</div>"""
        phs['positions'].markdown(pos_html, unsafe_allow_html=True)
    else:
        phs['positions'].info("No open positions")

    # 3. STRATEGY INTERNALS
    # Since this is complex to update bit-by-bit, we'll redraw the container content
    # using the placeholder.
    if os.path.exists("data/strategy_state.json"):
        try:
            import json
            with open("data/strategy_state.json") as f:
                full_state = json.load(f)
            
            # Determine which symbol to show
            token_filter = st.session_state.get("trade_token_filter", "ALL")
            active_sym = None
            
            if token_filter != "ALL":
                # Filter format is "🔵 ADAUSDT" -> extract ADAUSDT
                selected_sym = token_filter.split(" ")[-1].lower()
                if selected_sym in full_state:
                    active_sym = selected_sym
            
            # If no specific match or ALL, pick first available
            if not active_sym and full_state:
                active_sym = next(iter(full_state))
                
            if active_sym and active_sym in full_state:
                state = full_state[active_sym]
                
                regime = state.get('regime', 'UNKNOWN')
                color = '#3FB950' if regime == 'MEAN_REVERTING' else '#D29922' if regime == 'TRENDING' else '#F85149'
                
                # Try to get VPIN/Spread if they exist, or calculate/default
                vpin = state.get('vpin', 0)
                v_color = '#3FB950' if vpin < 0.5 else '#D29922' if vpin < 0.7 else '#F85149'
                
                # Format helper
                def fmt_param(val):
                    return f"{val:.2e}" if val < 0.0001 and val > 0 else f"{val:.4f}"

                strategy_html = f"""
                <div style="margin-bottom:5px; text-align:center; color:#8B949E; font-size:0.8rem">
                    Displaying logic for: <b style="color:#E6EDF3">{active_sym.upper()}</b>
                </div>
                <div style="display:flex; justify-content:space-around; margin-bottom:10px;">
                    <div>Regime: <span style='color:{color}; font-weight:bold'>{regime}</span></div>
                    <div>VPIN: <span style='color:{v_color}; font-weight:bold'>{vpin:.2f}</span></div>
                    <div>Vol: <b>{state.get('volatility', 0)*10000:.1f} bps</b></div>
                </div>
                <hr style="margin:5px 0; border-color:#30363D">
                <div style="font-size:0.8rem; color:#8B949E; margin-bottom:4px"><b>Avellaneda-Stoikov Params:</b></div>
                <div style="display:flex; justify-content:space-around; font-family:'JetBrains Mono', monospace; font-size:0.9rem">
                   <div>γ: <span style="color:#E6EDF3">{fmt_param(state.get('gamma', 0))}</span></div>
                   <div>κ: <span style="color:#E6EDF3">{fmt_param(state.get('kappa', 0))}</span></div>
                   <div>σ: <span style="color:#E6EDF3">{fmt_param(state.get('volatility', 0))}</span></div>
                </div>
                """
                phs['strategy_stats'].markdown(strategy_html, unsafe_allow_html=True)
            else:
                 phs['strategy_stats'].info("No active strategy state found")
            
        except Exception as e:
            phs['strategy_stats'].error(f"Error reading state: {e}")
    else:
        phs['strategy_stats'].info("No strategy state file found")

# =============================================================================
# RIGHT COLUMN: LOGS & FEED
# =============================================================================
def setup_right_column(container, initial_metrics=None):
    """Setup static layout for Right Column."""
    phs = {}
    with container:
        st.markdown("##### ⚡ TRADES")
        
        # 1. Filters Row
        # Icons
        CRYPTO_ICONS = {
            'BTC': '🟠', 'BTCUSDT': '🟠', 'ETH': '🔷', 'ETHUSDT': '🔷',
            'SOL': '🟣', 'SOLUSDT': '🟣', 'DOGE': '🐕', 'DOGEUSDT': '🐕',
            'XRP': '⚫', 'XRPUSDT': '⚫', 'BNB': '🟡', 'BNBUSDT': '🟡',
            'ADA': '🔵', 'ADAUSDT': '🔵', 'LTC': '⚪', 'LTCUSDT': '⚪'
        }
        
        # 1. View Mode (Moved up as requested)
        options = ["PnL", "Winners", "Losers", "All"]
        st.radio("View", options, horizontal=True, label_visibility="collapsed", key="trade_view_filter")
        
        # 2. Filters Row
        # Get Symbols
        symbols = ["ALL"]
        if initial_metrics and 'recent_trades' in initial_metrics and not initial_metrics['recent_trades'].empty:
            raw_syms = sorted(initial_metrics['recent_trades']['symbol'].unique().tolist())
            # Format: 🔵 ADAUSDT
            for s in raw_syms:
                s_upper = s.upper()
                icon = CRYPTO_ICONS.get(s_upper, '💎')
                symbols.append(f"{icon} {s_upper}")
            
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Token", symbols, key="trade_token_filter", label_visibility="collapsed")
        with col2:
            st.selectbox("Sort", ["Newest", "Best PnL", "Worst PnL"], key="trade_sort_filter", label_visibility="collapsed")
            
        # Hide Dust Filter
        st.checkbox("Hide Dust (<$0.01)", value=True, key="trade_hide_dust", help="Hide trades with absolute PnL < $0.01")
            
        # Table Placeholder
        phs['trades_table'] = st.empty()
        
        st.markdown("---")
        st.markdown("##### 📜 EVENT LOG")
        
        # Log Placeholder
        phs['event_log'] = st.empty()
        
        st.markdown("---")
        st.markdown("##### 🔧 SYSTEM")
        
        # System Health Placeholder
        phs['system_health'] = st.empty()
        
        return phs

def update_right_column(phs, metrics):
    """Update Right Column Data."""
    
    # 1. TRADES LIST
    # Get filters
    token_selection = st.session_state.get("trade_token_filter", "ALL")
    sort_filter = st.session_state.get("trade_sort_filter", "Newest")
    view_filter = st.session_state.get("trade_view_filter", "PnL")
    hide_dust = st.session_state.get("trade_hide_dust", True)
    
    if metrics and not metrics['recent_trades'].empty:
        df = metrics['recent_trades'].copy()
        
        # Hide Dust
        if hide_dust:
            df = df[df['pnl'].abs() >= 0.01]
        
        # Token Filter
        if token_selection != "ALL":
            # Extract symbol from "🔵 ADAUSDT" -> "ADAUSDT"
            token_filter = token_selection.split(" ")[-1]
            df = df[df['symbol'].str.upper() == token_filter]
        
        # View Filter
        if view_filter == "PnL":
            df = df[df['pnl'] != 0]
        elif view_filter == "Winners":
            df = df[df['pnl'] > 0]
        elif view_filter == "Losers":
            df = df[df['pnl'] < 0]
            
        # Sort
        if sort_filter == "Best PnL":
            df = df.sort_values("pnl", ascending=False)
        elif sort_filter == "Worst PnL":
            df = df.sort_values("pnl", ascending=True)
        else: # Newest
            # Ensure sorting by time/id
            if 'timestamp' in df.columns:
                 df = df.sort_values("timestamp", ascending=False)
            else:
                 df = df.sort_values("id", ascending=False)
            
        # Limit Display
        df = df.head(10)
        
        # Format
        CRYPTO_ICONS = {
            'BTC': '🟠', 'BTCUSDT': '🟠', 'ETH': '🔷', 'ETHUSDT': '🔷',
            'SOL': '🟣', 'SOLUSDT': '🟣', 'DOGE': '🐕', 'DOGEUSDT': '🐕',
            'XRP': '⚫', 'XRPUSDT': '⚫', 'BNB': '🟡', 'BNBUSDT': '🟡',
            'ADA': '🔵', 'ADAUSDT': '🔵', 'LTC': '⚪', 'LTCUSDT': '⚪'
        }
        
        # Fixed Height to prevent jumping
        html_cards = '<div style="height: 500px; overflow-y: auto; padding-right: 5px;">'
        
        if df.empty:
             html_cards += '<div style="color:#8B949E; padding:10px; text-align:center">No trades found matching filters</div>'
        else:
            for _, row in df.iterrows():
                sym = str(row.get('symbol', 'N/A')).upper()
                icon = CRYPTO_ICONS.get(sym, '💎')
                side = str(row.get('side', 'N/A')).upper()
                qty = float(row.get('quantity', 0))
                pnl = float(row.get('pnl', 0))
                
                # Timestamp
                if 'timestamp' in row:
                    ts = pd.to_datetime(row['timestamp'], unit='s').strftime('%H:%M:%S')
                else:
                    ts = "--:--:--"
                
                # Styles & Precision
                is_win = pnl > 0
                is_loss = pnl < 0
                
                if is_win:
                    border_color = "#3FB950" # Green
                    pnl_color = "#3FB950"
                    pnl_str = f"${pnl:+.4f}" if pnl < 0.01 else f"${pnl:+.2f}"
                elif is_loss:
                    border_color = "#F85149" # Red
                    pnl_color = "#F85149"
                    pnl_str = f"${pnl:+.4f}" if pnl > -0.01 else f"${pnl:+.2f}"
                else:
                    border_color = "#30363D" # Grey
                    pnl_color = "#8B949E"
                    pnl_str = "$0.00"
                    
                side_badge = f'<span style="color:{border_color}; font-size:0.75rem; border:1px solid {border_color}; padding: 1px 4px; border-radius: 4px;">{side}</span>'
                
                # Card HTML
                html_cards += f"""<div style="display:flex; justify-content:space-between; align-items:center; padding:8px; margin:4px 0; background:#161B22; border-radius:6px; border-left:3px solid {border_color};"><div><div style="font-size:0.7rem; color:#8B949E; margin-bottom:2px;">{ts}</div><div style="display:flex; align-items:center; gap:6px;"><span style="font-size:1.1rem">{icon}</span><b style="color:#E6EDF3">{sym}</b>{side_badge}</div></div><div style="text-align:right"><div style="color:#8B949E; font-size:0.75rem">Size: {qty:.4f}</div><div style="color:{pnl_color}; font-weight:600; font-size:0.9rem">{pnl_str}</div></div></div>"""
            
        html_cards += "</div>"
        phs['trades_table'].markdown(html_cards, unsafe_allow_html=True)
    else:
        phs['trades_table'].info("No trades yet")

    # 2. EVENT LOG
    log_entries = []
    if os.path.exists("logs/dashboard_log.txt"): # Updated path
        try:
            with open("logs/dashboard_log.txt", "r") as f:
                log_entries = f.readlines()[-20:]
        except:
            pass
            
    log_html = """<div class="log-container" style="height: 200px; overflow-y: auto; background: #0D1117; border: 1px solid #30363D; border-radius: 6px; padding: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;">"""
    
    if log_entries:
        for entry in reversed(log_entries):
            entry = entry.strip()
            color = "#8B949E"
            if "ERROR" in entry or "CRITICAL" in entry: color = "#F85149"
            elif "WARNING" in entry or "risk" in entry.lower(): color = "#D29922"
            elif "BUY" in entry: color = "#3FB950"
            elif "SELL" in entry: color = "#F85149"
            elif "INFO" in entry: color = "#E6EDF3"
            
            log_html += f'<div style="color:{color}; padding:2px 0; border-bottom:1px solid #21262D; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{entry}</div>'
    else:
        log_html += '<div style="color:#8B949E">Waiting for events...</div>'
    
    log_html += "</div>"
    phs['event_log'].markdown(log_html, unsafe_allow_html=True)

    # 3. SYSTEM HEALTH
    health = get_system_health()
    phs['system_health'].markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #8B949E;">
        <div>🔌 WebSocket: <span style="color:#3FB950">{health['ws_status']}</span> ({health['ws_latency']}ms)</div>
        <div>📡 Reconcile: {health['last_reconcile']}</div>
        <div>💾 Database: <span style="color:#3FB950">{health['db_status']}</span></div>
        <div>🤖 Router: {health['router_mode']}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
def render_footer():
    """Bottom status bar."""
    st.markdown("---")
    
    cols = st.columns([2, 1, 1, 1])
    
    with cols[0]:
        is_testnet = getattr(settings, 'TESTNET', True)
        st.caption(f"🚀 CreeptBaws v3.0 | Titan Command | {'TESTNET' if is_testnet else 'MAINNET'}")
    
    with cols[1]:
        st.caption(f"Symbols: {', '.join(s.upper() for s in settings.TRADING_SYMBOLS[:3])}...")
    
    with cols[2]:
        st.caption(f"Max Position: ${settings.MAX_POSITION_USD:,}")
    
    with cols[3]:
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# Removed @st.fragment decorators to use manual loop
def render_header_fragment():
    render_header()

def render_center_fragment():
    metrics = get_metrics()
    render_center_column(metrics)

def render_right_fragment():
    metrics = get_metrics()
    render_right_column(metrics)

def main():
    # Create a placeholder for the header at the very top
    header_ph = st.empty()

    # 1. LAYOUT SETUP (Runs once per script execution)
    st.markdown("### 🛡️ CREEPTBAWS COMMAND")
    
    # Create main columns
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    # --- LEFT COLUMN (Controls & Account) ---
    with col_left:
        # Check cached metrics for initial render
        metrics_init = get_metrics() 
        health_init = get_system_health()
        
        # 1. Controls
        render_controls(metrics_init, health_init)
        
        # 2. Account Stats Container
        # Create a specific container for stats so we can update it without touching controls
        stats_container = st.container()
        account_phs = setup_account_stats(stats_container)

    # --- CENTER COLUMN (Charts & Strategy) ---
    with col_center:
        center_container = st.container()
        center_phs = setup_center_column(center_container)
        
    # --- RIGHT COLUMN (Logs & Feed) ---
    with col_right:
        right_container = st.container()
        right_phs = setup_right_column(right_container, metrics_init)

    # 2. DATA UPDATE LOOP
    # We use a placeholder for the "Health/Time" header to update it dynamically without shifting layout
    # NOTE: render_header uses st.columns, which expands to full width.
    # It should effectively replace the static header logic if we want dynamic time.
    # But wait, st.markdown("### ...") is already rendered. 
    # Let's put the header ABOVE title? Or below?
    # The user is used to header at top.
    
    # Best practice: Create a placeholder at the VERY top of `main` (before markdown title), 
    # but `main` is called where?
    
    # Let's use the `header_placeholder` passed from __main__ or defined here? No.
    # We can create a NEW placeholder at top of main.
    
    # Move it to top? Streamlit renders sequentially.
    # If we create `header_ph` here, it appears below `st.columns`.
    # Layout order matters.
    
    # FIX: Move `st.markdown("### ...")` and columns AFTER `header_ph`.
    
    try:
        # Streamlit Cloud/newer versions prefer st.rerun() or fragments. 
        # But 'while True' is acceptable for a local dashboard if we handle interruptions.
        while True:
            # Fetch fresh data
            metrics = get_metrics()
            health = get_system_health()
            
            # Update Header safely
            with header_ph.container():
                render_header()
            
            # --- UPDATE UI ELEMENTS ---
            update_account_stats(account_phs, metrics, health)
            update_center_column(center_phs, metrics)
            update_right_column(right_phs, metrics)
            
            time.sleep(1) # Faster update (1s) for clock
            
    except Exception:
        # If user interacts, Streamlit raises an RerunException (internal) or just restarts.
        # We catch generic to allow safe exit.
        pass

if __name__ == "__main__":
    main()