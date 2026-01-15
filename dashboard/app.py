"""
SERQET Trading Dashboard - Premium Terminal Edition
Professional HFT Command Center

Layout:
- Header: Logo, Latency, Risk Level, UTC Clock
- Left (HANDS): Emergency Controls, Risk Metrics
- Center (EYES): Equity Curve, Open Positions, Strategy State  
- Right (MEMORY): Trade History, Event Log, System Health
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import plotly.graph_objects as go
import textwrap

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

load_dotenv()

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SERQET | Command",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🦂</text></svg>",
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
if 'flatten_armed' not in st.session_state:
    st.session_state.flatten_armed = False

# =============================================================================
# CONSTANTS
# =============================================================================
STOP_SIGNAL_FILE = "data/STOP_SIGNAL"
PAUSE_SIGNAL_FILE = "data/PAUSE_SIGNAL"
FLATTEN_SIGNAL_FILE = "data/FLATTEN_SIGNAL"

SCORPION_SVG = '<svg viewBox="0 0 100 100" width="28" height="28" style="fill: #d4af37;"><path d="M50 10 C30 10 20 30 20 50 C20 70 35 85 50 90 C65 85 80 70 80 50 C80 30 70 10 50 10 Z M50 20 C60 20 70 35 70 50 C70 65 60 75 50 80 C40 75 30 65 30 50 C30 35 40 20 50 20 Z"/><path d="M25 45 L10 30 M75 45 L90 30 M30 60 L15 75 M70 60 L85 75"/></svg>'

# =============================================================================
# SIGNAL FUNCTIONS
# =============================================================================
def send_stop_signal():
    with open(STOP_SIGNAL_FILE, "w") as f:
        f.write(f"STOP:{datetime.now(timezone.utc).isoformat()}")

def send_pause_signal():
    with open(PAUSE_SIGNAL_FILE, "w") as f:
        f.write(f"PAUSE:{datetime.now(timezone.utc).isoformat()}")

def send_flatten_signal():
    with open(FLATTEN_SIGNAL_FILE, "w") as f:
        f.write(f"FLATTEN:{datetime.now(timezone.utc).isoformat()}")

def clear_signals():
    for f in [STOP_SIGNAL_FILE, PAUSE_SIGNAL_FILE, FLATTEN_SIGNAL_FILE]:
        if os.path.exists(f):
            os.remove(f)

def get_signal_status():
    if os.path.exists(STOP_SIGNAL_FILE):
        return "STOPPED", "error"
    elif os.path.exists(PAUSE_SIGNAL_FILE):
        return "PAUSED", "warning"
    return "LIVE", "active"

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
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", conn)
        positions = pd.read_sql_query("SELECT * FROM positions", conn)
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
                "winners": 0,
                "losers": 0,
                "equity_curve": pd.DataFrame(),
                "recent_trades": pd.DataFrame(),
                "positions": positions,
                "best_trade": 0,
                "worst_trade": 0,
                "best_info": "",
                "worst_info": "",
                "inventory": {}
            }
        
        pnl_total = trades['pnl'].sum()
        trades_count = len(trades)
        winners = len(trades[trades['pnl'] > 0])
        losers = len(trades[trades['pnl'] < 0])
        decisive = winners + losers
        win_rate = (winners / decisive * 100) if decisive > 0 else 0
        
        # Equity curve
        trades_sorted = trades.sort_values('timestamp')
        equity_df = pd.DataFrame()
        if not trades_sorted.empty:
            equity_df['timestamp'] = pd.to_datetime(trades_sorted['timestamp'], unit='s', utc=True)
            equity_df['equity'] = trades_sorted['pnl'].cumsum() + settings.INITIAL_CAPITAL
        
        # Best/worst
        best_pnl = float(best_trade_db.iloc[0]['pnl']) if not best_trade_db.empty else 0
        best_info = best_trade_db.iloc[0].get('symbol', 'N/A').upper() if not best_trade_db.empty else "N/A"
        worst_pnl = float(worst_trade_db.iloc[0]['pnl']) if not worst_trade_db.empty else 0
        worst_info = worst_trade_db.iloc[0].get('symbol', 'N/A').upper() if not worst_trade_db.empty else "N/A"
        
        unrealized = positions['unrealized_pnl'].sum() if 'unrealized_pnl' in positions.columns else 0
        
        # Inventory
        inventory = {}
        if not positions.empty and 'symbol' in positions.columns:
            for _, row in positions.iterrows():
                if row['quantity'] != 0:
                    inventory[row['symbol'].upper()] = {
                        'qty': row['quantity'],
                        'entry': row.get('avg_entry_price', 0),
                        'mark': row.get('mark_price', row.get('avg_entry_price', 0)),
                        'pnl': row.get('unrealized_pnl', 0)
                    }
        
        return {
            "balance": settings.INITIAL_CAPITAL + pnl_total + unrealized,
            "realized_pnl": pnl_total,
            "unrealized_pnl": unrealized,
            "trades_count": trades_count,
            "win_rate": win_rate,
            "winners": winners,
            "losers": losers,
            "equity_curve": equity_df,
            "recent_trades": trades,
            "positions": positions,
            "best_trade": best_pnl,
            "worst_trade": worst_pnl,
            "best_info": best_info,
            "worst_info": worst_info,
            "inventory": inventory
        }
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

@st.cache_data(ttl=3)
def get_system_health():
    health = {
        "ws_latency": 12,
        "uptime": "0h 0m",
        "db_status": "OK",
        "risk_level": 15
    }
    
    if os.path.exists("health_status.json"):
        try:
            with open("health_status.json") as f:
                data = json.load(f)
                health.update(data)
                
                # Calculate Uptime if start_time is present
                if 'start_time' in data:
                    try:
                        start_ts = float(data['start_time'])
                        uptime_seconds = time.time() - start_ts
                        hours = int(uptime_seconds // 3600)
                        minutes = int((uptime_seconds % 3600) // 60)
                        health['uptime'] = f"{hours}h {minutes}m"
                    except:
                        pass
        except:
            pass
    
    return health



def get_strategy_state():
    if not os.path.exists("data/strategy_state.json"):
        return {}
    try:
        with open("data/strategy_state.json") as f:
            return json.load(f)
    except:
        return {}

def get_log_entries():
    if not os.path.exists("logs/dashboard_log.txt"):
        return []
    try:
        with open("logs/dashboard_log.txt", "r") as f:
            return f.readlines()[-15:]
    except:
        return []

# =============================================================================
# HEADER
# =============================================================================
def render_header():
    status, status_type = get_signal_status()
    health = get_system_health()
    
    status_colors = {"active": "#22c55e", "warning": "#f59e0b", "error": "#ef4444"}
    status_color = status_colors.get(status_type, "#666")
    
    latency = health.get('ws_latency', 0)
    lat_color = "#22c55e" if latency < 50 else "#f59e0b" if latency < 100 else "#ef4444"
    
    risk = health.get('risk_level', 0)
    risk_color = "#22c55e" if risk < 30 else "#f59e0b" if risk < 70 else "#ef4444"
    
    utc_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
    
    st.markdown(f"""
<div class="serqet-header">
<div class="serqet-logo">
{SCORPION_SVG}
<span class="serqet-logo-text">SERQET</span>
</div>
<div class="header-stats">
<div class="header-stat">
<span class="header-stat-label">LATENCY</span>
<span class="header-stat-value" style="color: {lat_color}">{latency} MS</span>
</div>
<div class="header-stat">
<span class="header-stat-label">RISK</span>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 60px; height: 4px; background: #1a1a1a; border-radius: 2px; overflow: hidden;">
<div style="width: {risk}%; height: 100%; background: {risk_color};"></div>
</div>
</div>
</div>
<div class="header-stat">
<span class="status-dot {status_type}"></span>
<span style="color: {status_color}">{status}</span>
</div>
</div>
<div class="header-clock">
<span class="header-clock-time">{utc_time}</span> UTC
</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# CARD COMPONENT
# =============================================================================
def card(title, icon=""):
    st.markdown(textwrap.dedent(f"""
    <div class="terminal-card">
        <div class="terminal-card-header">
            <div class="terminal-card-title">
                <span class="terminal-card-title-icon">{icon}</span> {title}
            </div>
        </div>
        <div class="terminal-card-body">
    """), unsafe_allow_html=True)

def card_end():
    st.markdown("</div></div>", unsafe_allow_html=True)

# =============================================================================
# LEFT COLUMN - EMERGENCY CONTROLS & RISK METRICS
# =============================================================================
def render_left_column(metrics, health):
    # Emergency Controls
    card("EMERGENCY CONTROLS", "⚡")
    
    status, _ = get_signal_status()
    is_paused = status == "PAUSED"
    
    # Pause Toggle
    pause_col1, pause_col2 = st.columns([3, 1])
    with pause_col1:
        st.markdown(f"<div style='font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-primary);'>PAUSE</div>", unsafe_allow_html=True)
    with pause_col2:
        if st.button("ON" if not is_paused else "OFF", key="pause_toggle"):
            if is_paused:
                if os.path.exists(PAUSE_SIGNAL_FILE):
                    os.remove(PAUSE_SIGNAL_FILE)
            else:
                send_pause_signal()
            st.rerun()
    
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    
    # Flatten Button
    if st.session_state.flatten_armed:
        st.markdown(textwrap.dedent("""
        <div class="flatten-button armed">
            FLATTEN
            <span class="flatten-armed-text">ARMED - PRESS TO CONFIRM</span>
        </div>
        """), unsafe_allow_html=True)
        if st.button("CONFIRM FLATTEN", key="confirm_flatten", type="primary"):
            send_flatten_signal()
            st.session_state.flatten_armed = False
            st.rerun()
        if st.button("CANCEL", key="cancel_flatten"):
            st.session_state.flatten_armed = False
            st.rerun()
    else:
        if st.button("FLATTEN", key="arm_flatten", use_container_width=True):
            st.session_state.flatten_armed = True
            st.rerun()
    
    card_end()
    
    # Risk Metrics
    card("RISK METRICS", "≡")
    
    if metrics:
        balance = metrics.get('balance', 0)
        realized = metrics.get('realized_pnl', 0)
        win_rate = metrics.get('win_rate', 0)
        best = metrics.get('best_trade', 0)
        worst = metrics.get('worst_trade', 0)
        
        pnl_pct = (realized / settings.INITIAL_CAPITAL * 100) if settings.INITIAL_CAPITAL > 0 else 0
        pnl_color = "positive" if realized >= 0 else "negative"
        
        st.markdown(textwrap.dedent(f"""
        <div class="risk-metrics">
            <div class="metric-row">
                <span class="metric-label">BALANCE</span>
                <span class="metric-value large">${balance:,.2f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">REAL. PnL</span>
                <span class="metric-value {pnl_color}">
                    {'+' if realized >= 0 else ''}${realized:,.2f}
                    <span class="metric-delta {pnl_color}">{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%</span>
                </span>
            </div>
            <div class="metric-row">
                <span class="metric-label">WIN RATE</span>
                <span class="metric-value">{win_rate:.1f}%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">BEST TRADE</span>
                <span class="metric-value positive">+${best:,.2f} <span style="color: var(--text-muted); font-size: 0.7rem;">{metrics.get('best_info', '')}</span></span>
            </div>
            <div class="metric-row">
                <span class="metric-label">WORST TRADE</span>
                <span class="metric-value negative">${worst:,.2f} <span style="color: var(--text-muted); font-size: 0.7rem;">{metrics.get('worst_info', '')}</span></span>
            </div>
        </div>
        """), unsafe_allow_html=True)
    
    card_end()

# =============================================================================
# CENTER COLUMN - EQUITY CURVE, POSITIONS, STRATEGY
# =============================================================================
def render_center_column(metrics):
    # Equity Curve
    card("EQUITY CURVE", "≡")
    
    if metrics and not metrics['equity_curve'].empty:
        df = metrics['equity_curve'].copy()
        realized = metrics.get('realized_pnl', 0)
        pnl_pct = (realized / settings.INITIAL_CAPITAL * 100) if settings.INITIAL_CAPITAL > 0 else 0
        pnl_color = "#22c55e" if realized >= 0 else "#ef4444"
        
        st.markdown(textwrap.dedent(f"""
        <div style="display: flex; align-items: baseline; gap: 16px; margin-bottom: 16px;">
            <span style="font-family: var(--font-mono); font-size: 1.5rem; font-weight: 700; color: {pnl_color};">
                {'+' if realized >= 0 else ''}${realized:,.2f}
            </span>
            <span style="font-family: var(--font-mono); font-size: 0.9rem; color: {pnl_color};">
                {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%
            </span>
        </div>
        """), unsafe_allow_html=True)
        
        # Time filter
        time_window = st.radio("", ["1H", "4H", "24H", "ALL"], horizontal=True, label_visibility="collapsed", key="eq_time")
        
        now = pd.Timestamp.now(tz='UTC')
        if time_window == "1H":
            df = df[df['timestamp'] >= now - pd.Timedelta(hours=1)]
        elif time_window == "4H":
            df = df[df['timestamp'] >= now - pd.Timedelta(hours=4)]
        elif time_window == "24H":
            df = df[df['timestamp'] >= now - pd.Timedelta(hours=24)]
        
        if not df.empty:
            line_color = '#d4af37'
            fill_color = 'rgba(212, 175, 55, 0.08)'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['equity'],
                mode='lines',
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor=fill_color,
                hovertemplate='<b>$%{y:,.2f}</b><extra></extra>'
            ))
            
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=10, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#1a1a1a', zeroline=False, showline=False),
                yaxis=dict(showgrid=True, gridcolor='#1a1a1a', side='left', zeroline=False, showline=False, tickprefix='$'),
                showlegend=False,
                font=dict(family="JetBrains Mono", size=9, color="#666")
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.markdown("<div style='color: var(--text-muted); text-align: center; padding: 40px;'>No equity data</div>", unsafe_allow_html=True)
    
    card_end()
    
    # Open Positions
    card("OPEN POSITIONS", "◎")
    
    st.markdown(textwrap.dedent("""
    <div class="positions-header">
        <span>SYMBOL</span>
        <span>POSITION</span>
        <span style="text-align: right">ENTRY PRICE</span>
        <span style="text-align: right">MARK PRICE</span>
        <span style="text-align: right">UNREALIZED PnL</span>
    </div>
    """), unsafe_allow_html=True)
    
    if metrics and metrics.get('inventory'):
        html = ""
        for sym, data in metrics['inventory'].items():
            qty = data['qty']
            entry = data.get('entry', 0)
            mark = data.get('mark', entry)
            pnl = data.get('pnl', 0)
            
            pos_type = "long" if qty > 0 else "short"
            pnl_class = "positive" if pnl >= 0 else "negative"
            
            html += textwrap.dedent(f"""
            <div class="position-row">
                <div class="position-symbol">
                    <span class="position-symbol-dot"></span>
                    {sym}
                </div>
                <div><span class="position-badge {pos_type}">{pos_type.upper()}</span></div>
                <div style="text-align: right; color: var(--text-secondary);">${entry:,.2f}</div>
                <div style="text-align: right; color: var(--text-primary);">${mark:,.2f}</div>
                <div style="text-align: right;" class="metric-value {pnl_class}">{'+' if pnl >= 0 else ''}${pnl:,.2f}</div>
            </div>
            """)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: var(--text-muted); text-align: center; padding: 20px;'>No open positions</div>", unsafe_allow_html=True)
    
    card_end()
    
    # Strategy State
    card("STRATEGY STATE", "◈")
    
    state = get_strategy_state()
    if state:
        html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px;">'
        for sym, data in list(state.items())[:6]:
            regime = data.get('regime', 'UNKNOWN')
            vol = data.get('volatility', 0)
            kappa = data.get('kappa', 0)
            
            regime_colors = {"TRENDING": "#22c55e", "VOLATILE": "#f59e0b", "RANGING": "#666"}
            regime_color = regime_colors.get(regime, "#666")
            
            html += textwrap.dedent(f"""
            <div style="background: var(--bg-secondary); padding: 12px; border-radius: 4px; border-left: 2px solid var(--gold);">
                <div style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">{sym}</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 4px;">
                    <span style="color: var(--text-muted);">REGIME</span>
                    <span style="color: {regime_color};">{regime}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 4px;">
                    <span style="color: var(--text-muted);">VOL</span>
                    <span style="color: var(--text-secondary);">{vol:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem;">
                    <span style="color: var(--text-muted);">KAPPA</span>
                    <span style="color: var(--text-secondary);">{kappa:.2f}</span>
                </div>
            </div>
            """)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: var(--text-muted); text-align: center; padding: 20px;'>No strategy data</div>", unsafe_allow_html=True)
    
    card_end()

# =============================================================================
# RIGHT COLUMN - TRADE HISTORY, EVENT LOG, SYSTEM HEALTH
# =============================================================================
def render_right_column(metrics, health, sort_mode="Newest"):
    # Trade History
    card("TRADE HISTORY", "≡")
    
    if metrics and not metrics['recent_trades'].empty:
        df = metrics['recent_trades'].copy()
        
        # Apply Sorting
        if sort_mode == "Best PnL":
            df = df.sort_values("pnl", ascending=False)
        elif sort_mode == "Worst PnL":
            df = df.sort_values("pnl", ascending=True)
        else: # Newest
            if 'timestamp' in df.columns:
                df = df.sort_values("timestamp", ascending=False)
        
        df = df.head(15)
        
        html = '<div class="trades-feed">'
        for _, row in df.iterrows():
            ts = pd.to_datetime(row.get('timestamp', 0), unit='s').strftime('%H:%M:%S') if 'timestamp' in row else "--:--"
            sym = str(row.get('symbol', 'N/A')).upper()
            side = str(row.get('side', 'N/A')).upper()
            qty = float(row.get('quantity', 0))
            pnl = float(row.get('pnl', 0))
            
            side_class = "buy" if side == "BUY" else "sell"
            pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
            pnl_hue = "positive-hue" if pnl > 0 else "negative-hue" if pnl < 0 else ""
            
            html += textwrap.dedent(f"""
            <div class="trade-row {pnl_hue}">
                <span class="trade-time">{ts}</span>
                <span class="trade-symbol">{sym}</span>
                <span class="trade-badge {side_class}">{side}</span>
                <span class="trade-qty">{qty:.3f}</span>
                <span class="trade-pnl {pnl_class}">{'+' if pnl > 0 else ''}${pnl:.2f}</span>
            </div>
            """)
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: var(--text-muted); text-align: center; padding: 20px;'>No trades yet</div>", unsafe_allow_html=True)
    
    card_end()
    
    # Event Log
    card("EVENT LOG", "◉")
    
    entries = get_log_entries()
    html = '<div class="event-log">'
    if entries:
        for entry in reversed(entries):
            entry = entry.strip()
            if not entry:
                continue
            
            log_type = "info"
            badge_text = "INFO"
            if "ERROR" in entry or "CRITICAL" in entry:
                log_type = "error"
                badge_text = "ERROR"
            elif "WARNING" in entry or "WARN" in entry:
                log_type = "warn"
                badge_text = "WARN"
            
            display_text = entry[:80] + "..." if len(entry) > 80 else entry
            
            html += textwrap.dedent(f"""
            <div class="log-entry {log_type}">
                <span class="log-badge {log_type}">{badge_text}</span>
                <span>{display_text}</span>
            </div>
            """)
    else:
        html += '<div class="log-entry" style="font-style: italic; color: var(--text-muted);">No recent events</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    
    card_end()
    
    # System Health
    card("SYSTEM HEALTH", "⬡")
    
    latency = health.get('ws_latency', 0)
    uptime = health.get('uptime', '0h 0m')
    
    lat_pct = min(100, (100 - latency) if latency < 100 else 20)
    lat_class = "" if latency < 50 else "warning" if latency < 100 else "danger"
    
    st.markdown(textwrap.dedent(f"""
    <div class="health-panel">
        <div class="health-metric">
            <span class="health-label">UPTIME</span>
            <span class="health-value">{uptime}</span>
        </div>
        <div class="health-metric">
            <span class="health-label">LATENCY</span>
            <span class="health-value">{latency} ms</span>
            <div class="health-bar">
                <div class="health-bar-fill {lat_class}" style="width: {lat_pct}%;"></div>
            </div>
        </div>
    </div>
    <div class="heartbeat-container">
        <div class="heartbeat-line"></div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--font-mono); font-size: 0.7rem;">
        <span style="color: var(--text-muted);">STATUS:</span>
        <span style="color: #22c55e;">OPTIMAL</span>
    </div>
    """), unsafe_allow_html=True)
    
    card_end()

# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# MAIN EXECUTION
# =============================================================================
@st.fragment(run_every=2)
def auto_update_fragment(header_ph, left_ph, center_ph, right_ph):
    metrics = get_metrics()
    health = get_system_health()
    sort_mode = st.session_state.get('trade_sort', "Newest")
    
    with header_ph.container():
        render_header()
        
    with left_ph.container():
        render_left_column(metrics, health)
        
    with center_ph.container():
        render_center_column(metrics)
        
    with right_ph.container():
        render_right_column(metrics, health, sort_mode)

def main():
    # 1. Setup Layout (Once)
    header_container = st.empty()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        left_container = st.empty()
    
    with col2:
        center_container = st.empty()
        
    with col3:
        # Static Controls
        st.selectbox(
            "Sort Trades", 
            ["Newest", "Best PnL", "Worst PnL"], 
            key="trade_sort",
            label_visibility="collapsed"
        )
        right_container = st.empty()

    # 2. Run Auto-Update Fragment in-place
    # This will rerun every 2s, updating ONLY the containers above
    auto_update_fragment(header_container, left_container, center_container, right_container)

if __name__ == "__main__":
    main()
