import os
import sys

# Add project root to sys.path to allow imports from config/core/etc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import json
import asyncio
from datetime import datetime, timezone
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardServer")

from contextlib import asynccontextmanager

# Background Broadcast Loop
async def broadcast_loop():
    while True:
        try:
            if manager.active_connections:
                data = get_metrics()
                # Debug print to confirm data size
                # print(f"DEBUG: Broadcasting {len(str(data))} bytes")
                await manager.broadcast(json.dumps(data))
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(1.0) # 1s update rate (eased off to prevent lock contention)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 STARTING BROADCAST LOOP")
    task = asyncio.create_task(broadcast_loop())
    yield
    # Shutdown (if needed)
    task.cancel()

app = FastAPI(title="Serqet Command Center", lifespan=lifespan)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Constants
DB_PATH = "data/trading_data.db"
STATE_FILE = "data/strategy_state.json"
LOG_FILE = "logs/dashboard_log.txt"

def get_db_connection():
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"DB Connection error: {e}")
        return None

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 CLIENT CONNECTED: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("🔌 CLIENT DISCONNECTED")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass # Connection might be closed

manager = ConnectionManager()

# =============================================================================
# SIGNAL HANDLING
# =============================================================================
from pydantic import BaseModel

class ActionRequest(BaseModel):
    action: str

STOP_SIGNAL_FILE = "data/STOP_SIGNAL"
PAUSE_SIGNAL_FILE = "data/PAUSE_SIGNAL"

def get_signal_status():
    if os.path.exists(STOP_SIGNAL_FILE):
        return "STOPPED"
    elif os.path.exists(PAUSE_SIGNAL_FILE):
        return "PAUSED"
    return "ACTIVE"

@app.post("/api/action/{action}")
async def handle_action(action: str):
    timestamp = datetime.now().isoformat()
    if action == "pause":
        with open(PAUSE_SIGNAL_FILE, "w") as f:
            f.write(f"PAUSE:{timestamp}")
        return {"status": "paused"}
    elif action == "resume":
        for f in [STOP_SIGNAL_FILE, PAUSE_SIGNAL_FILE]:
            if os.path.exists(f): os.remove(f)
        return {"status": "active"}
    elif action == "flatten":
        with open(STOP_SIGNAL_FILE, "w") as f:
            f.write(f"STOP:{timestamp}")
        return {"status": "stopped"}
    return {"error": "Invalid action"}

# =============================================================================
# CONFIGURATION API (Dynamic Tuning)
# =============================================================================
CONFIG_FILE = "data/runtime_config.json"
from config.settings import settings

@app.get("/api/config")
async def get_config():
    """Return default settings merged with runtime overrides"""
    base_config = {
        "AS_GAMMA": settings.AS_GAMMA,
        "AS_KAPPA": settings.AS_KAPPA,
        "RISK_MAX_POSITION_PER_SYMBOL_USD": settings.RISK_MAX_POSITION_PER_SYMBOL_USD,
        "RISK_MAX_POSITION_TOTAL_USD": settings.RISK_MAX_POSITION_TOTAL_USD,
        "MIN_PROFIT_BPS": settings.MIN_PROFIT_BPS,
        "MAKER_FEE_BPS": settings.MAKER_FEE_BPS,
    }
    
    # Merge runtime overrides
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                overrides = json.load(f)
                base_config.update(overrides)
        except Exception as e:
            logger.error(f"Failed to load runtime config: {e}")
            
    return base_config

@app.post("/api/config")
async def update_config(request: Request):
    """Save new config values to runtime file"""
    try:
        new_values = await request.json()
        
        # Load existing
        current = {}
        if os.path.exists(CONFIG_FILE):
             with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                current = json.load(f)
        
        # Update
        current.update(new_values)
        
        # Write back
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=4)
            
        return {"status": "ok", "config": current}
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =============================================================================
# DATA LOGIC
# =============================================================================
def get_system_health():
    health = {
        "ws_status": "OK",
        "ws_latency": 12,
        "last_reconcile": "2s ago",
        "db_status": "OK",
        "router_mode": "LIMIT_CHASE",
        "gtx_rejections": 0
    }
    if os.path.exists("health_status.json"):
        try:
            with open("health_status.json", encoding='utf-8') as f:
                health.update(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load health_status.json: {e}")
    return health

def get_metrics() -> Dict[str, Any]:
    """Calculate all key metrics (Titan v3 Logic)"""
    # Default State
    metrics = {
        "balance": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "win_rate_total": 0.0,
        "win_rate_decision": 0.0,
        "winners": 0,
        "losers": 0,
        "breakevens": 0,
        "equity_curve": [],
        "recent_trades": [],
        "logs": [],
        "maker_rate": 0,
        "best_trade": 0,
        "worst_trade": 0,
        "system_health": get_system_health(),
        "status": get_signal_status(),
        "inventory": {},
        "strategy_state": {}
    }
    
    conn = get_db_connection()
    if not conn:
        return metrics
        
    try:
        # 1. TRADES & PNL
        # Check if table exists first to avoid crashes if DB initialized but empty
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not cursor.fetchone():
             conn.close()
             return metrics

        trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        
        # Check positions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
        if cursor.fetchone():
            positions = pd.read_sql("SELECT * FROM positions", conn)
        else:
            positions = pd.DataFrame()
        
        # Balance (Mock initial + Realized + Unrealized)
        initial_capital = settings.INITIAL_CAPITAL
        realized = trades_df['pnl'].sum() if not trades_df.empty else 0.0
        unrealized = positions['unrealized_pnl'].sum() if not positions.empty and 'unrealized_pnl' in positions.columns else 0.0
        
        metrics['balance'] = initial_capital + realized + unrealized
        metrics['realized_pnl'] = realized
        metrics['unrealized_pnl'] = unrealized
        
        # 2. WIN RATES & STATS
        if not trades_df.empty:
            winners = len(trades_df[trades_df['pnl'] > 0])
            losers = len(trades_df[trades_df['pnl'] < 0])
            breakevens = len(trades_df[trades_df['pnl'] == 0])
            total = len(trades_df)
            decisive = winners + losers
            
            metrics['winners'] = winners
            metrics['losers'] = losers
            metrics['breakevens'] = breakevens
            metrics['win_rate_total'] = (winners / total * 100) if total > 0 else 0
            metrics['win_rate_decision'] = (winners / decisive * 100) if decisive > 0 else 0
            
            # Maker Rate
            if 'is_maker' in trades_df.columns:
                maker_count = trades_df['is_maker'].sum()
                metrics['maker_rate'] = (maker_count / total * 100)
                
            # Best/Worst
            metrics['best_trade'] = trades_df['pnl'].max()
            metrics['worst_trade'] = trades_df['pnl'].min()
            
            # 3. EQUITY CURVE
            # Sort ascending for curve calc
            sorted_trades = trades_df.sort_values('timestamp')
            equity = initial_capital
            curve = []
            
            # Resample or just every trade? Titan does every trade.
            for _, row in sorted_trades.iterrows():
                equity += row['pnl']
                curve.append({"time": row['timestamp'], "equity": equity})
            metrics['equity_curve'] = curve
            
            # 4. RECENT TRADES LIST
            recent = trades_df.head(50)
            trades_list = []
            for _, row in recent.iterrows():
                trades_list.append({
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "quantity": row['quantity'],
                    "pnl": row['pnl'],
                    "timestamp": row['timestamp']
                })
            metrics['recent_trades'] = trades_list

        # 5. INVENTORY / POSITIONS
        inventory = {}
        if not positions.empty:
            for _, row in positions.iterrows():
                # Checking columns existence to be safe
                if 'quantity' in row and row['quantity'] != 0:
                    inventory[row['symbol']] = {
                        "qty": row['quantity'],
                        "pnl": row.get('unrealized_pnl', 0),
                        "entry": row.get('avg_entry_price', 0)
                    }
        metrics['inventory'] = inventory

        # 6. STRATEGY STATE
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    metrics['strategy_state'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load strategy_state.json: {e}")

        # 7. LOGS
        if os.path.exists(LOG_FILE):
             with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-30:]
                metrics['logs'] = [l.strip() for l in reversed(lines)]
                
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        # Add error to logs so client sees it
        metrics['logs'].insert(0, f"SERVER METRIC ERROR: {str(e)}")
    finally:
        conn.close()
        
    return metrics

@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(get_metrics())

@app.get("/api/latency")
async def api_latency():
    """
    Get latency metrics from strategy state for dashboard visualization.
    
    Returns latency histogram data and statistics.
    """
    latency_data = {
        "history": [],
        "avg_ms": 0.0,
        "p99_ms": 0.0,
        "max_ms": 0.0,
        "min_ms": 0.0
    }
    
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Extract latency history from any symbol's state
            for symbol, symbol_state in state.items():
                if isinstance(symbol_state, dict) and 'latency_history' in symbol_state:
                    history = symbol_state['latency_history']
                    if history:
                        latency_data['history'] = history[-100:]  # Last 100 samples
                        latency_data['avg_ms'] = sum(history) / len(history)
                        latency_data['max_ms'] = max(history)
                        latency_data['min_ms'] = min(history)
                        # P99 approximation
                        sorted_hist = sorted(history)
                        p99_idx = int(len(sorted_hist) * 0.99)
                        latency_data['p99_ms'] = sorted_hist[p99_idx] if sorted_hist else 0
                    break
                    
    except Exception as e:
        logger.warning(f"Failed to load latency data: {e}")
    
    return JSONResponse(latency_data)


@app.get("/api/equity_history")
async def api_equity_history():
    """
    Get realized equity history for charting.
    Reconstructs the equity curve from the trades table.
    """
    history = []
    
    conn = get_db_connection()
    if conn:
        try:
            # Check table existence (redundant but safe)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if cursor.fetchone():
                # Get all trades sorted by time
                df = pd.read_sql("SELECT timestamp, pnl FROM trades ORDER BY timestamp ASC", conn)
                
                if not df.empty:
                    initial_capital = settings.INITIAL_CAPITAL
                    cumulative = initial_capital
                    
                    # Optimization: Don't return every single trade if we have thousands
                    # Resample to 1 minute or max 500 points
                    
                    for _, row in df.iterrows():
                        cumulative += row['pnl']
                        history.append({
                            "time": int(row['timestamp']),
                            "value": float(cumulative)
                        })
                        
        except Exception as e:
            logger.warning(f"Failed to fetch equity history: {e}")
        finally:
            conn.close()
            
    return JSONResponse(history)


@app.get("/api/candles/{symbol}")
async def api_candles(symbol: str, interval: str = "1m", limit: int = 100):
    """
    Get OHLCV candle data for price chart.
    Fetches from Binance public API.
    """
    import aiohttp
    
    symbol = symbol.upper()
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    candles = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    candles = [{
                        "time": int(k[0] / 1000),  # Convert to seconds
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    } for k in data]
    except Exception as e:
        logger.warning(f"Failed to fetch candles: {e}")
    
    # Get recent trades for markers
    markers = []
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql(f"""
                SELECT timestamp, side, price, quantity 
                FROM trades 
                WHERE symbol = '{symbol.lower()}'
                ORDER BY timestamp DESC 
                LIMIT 20
            """, conn)
            markers = df.to_dict('records')
        except Exception as e:
            logger.debug(f"No trade markers: {e}")
        finally:
            conn.close()
    
    return JSONResponse({"candles": candles, "markers": markers})


@app.get("/api/fees")
async def api_fees():
    """
    Get total fees paid breakdown.
    Priority 4.2: Fee analysis for HFT.
    """
    fees = {
        "total_fees": 0.0,
        "maker_fees": 0.0,
        "taker_fees": 0.0,
        "fee_count": 0
    }
    
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if cursor.fetchone():
                result = pd.read_sql("""
                    SELECT 
                        SUM(commission) as total,
                        SUM(CASE WHEN is_maker = 1 THEN commission ELSE 0 END) as maker,
                        SUM(CASE WHEN is_maker = 0 THEN commission ELSE 0 END) as taker,
                        COUNT(*) as count
                    FROM trades
                """, conn)
                if not result.empty:
                    fees['total_fees'] = float(result.iloc[0]['total'] or 0)
                    fees['maker_fees'] = float(result.iloc[0]['maker'] or 0)
                    fees['taker_fees'] = float(result.iloc[0]['taker'] or 0)
                    fees['fee_count'] = int(result.iloc[0]['count'] or 0)
        except Exception as e:
            logger.warning(f"Failed to calculate fees: {e}")
        finally:
            conn.close()
    
    return JSONResponse(fees)


@app.get("/api/orderbook/{symbol}")
async def api_orderbook(symbol: str, limit: int = 20):
    """
    Get order book depth data for depth chart visualization.
    Critical for market makers to see where quotes sit vs liquidity.
    """
    import aiohttp
    
    symbol = symbol.upper()
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit={limit}"
    
    depth_data = {
        "symbol": symbol,
        "bids": [],
        "asks": [],
        "mid_price": 0,
        "spread_bps": 0
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Process bids (cumulative)
                    cum_qty = 0
                    for price, qty in data.get('bids', []):
                        cum_qty += float(qty)
                        depth_data['bids'].append({
                            "price": float(price),
                            "qty": float(qty),
                            "cumulative": cum_qty
                        })
                    
                    # Process asks (cumulative)
                    cum_qty = 0
                    for price, qty in data.get('asks', []):
                        cum_qty += float(qty)
                        depth_data['asks'].append({
                            "price": float(price),
                            "qty": float(qty),
                            "cumulative": cum_qty
                        })
                    
                    # Calculate mid price and spread
                    if depth_data['bids'] and depth_data['asks']:
                        best_bid = depth_data['bids'][0]['price']
                        best_ask = depth_data['asks'][0]['price']
                        depth_data['mid_price'] = (best_bid + best_ask) / 2
                        depth_data['spread_bps'] = ((best_ask - best_bid) / depth_data['mid_price']) * 10000
                        
    except Exception as e:
        logger.warning(f"Failed to fetch order book: {e}")
    
    return JSONResponse(depth_data)


@app.get("/api/rate_limit")
async def api_rate_limit():
    """
    Get API rate limit status from Binance.
    Helps avoid getting banned by the exchange.
    """
    import aiohttp
    
    rate_data = {
        "used_weight": 0,
        "weight_limit": 1200,  # Binance default per minute
        "order_count": 0,
        "order_limit": 10,  # Per second
        "pct_used": 0,
        "status": "OK"
    }
    
    try:
        # Make a lightweight request to check rate limits
        url = "https://fapi.binance.com/fapi/v1/time"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                # Binance returns rate limit info in headers
                used_weight = resp.headers.get('X-MBX-USED-WEIGHT-1M', '0')
                rate_data['used_weight'] = int(used_weight)
                rate_data['pct_used'] = (rate_data['used_weight'] / rate_data['weight_limit']) * 100
                
                if rate_data['pct_used'] > 80:
                    rate_data['status'] = "WARNING"
                elif rate_data['pct_used'] > 95:
                    rate_data['status'] = "CRITICAL"
                    
    except Exception as e:
        logger.warning(f"Failed to fetch rate limit: {e}")
        rate_data['status'] = "UNKNOWN"
    
    return JSONResponse(rate_data)

@app.get("/api/open_orders")
async def api_open_orders():
    """
    Get open (pending) orders from database.
    Priority 1.2: Shows working limit orders not yet filled.
    """
    orders = []
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
            if cursor.fetchone():
                df = pd.read_sql("""
                    SELECT client_order_id, symbol, side, quantity, price, status, created_at 
                    FROM orders 
                    WHERE status IN ('NEW', 'PENDING', 'PARTIALLY_FILLED')
                    ORDER BY created_at DESC
                    LIMIT 50
                """, conn)
                orders = df.to_dict('records')
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")
        finally:
            conn.close()
    return JSONResponse({"orders": orders})


@app.get("/api/decision_log")
async def api_decision_log():
    """
    Get strategy decision log explaining WHY trades were skipped/taken.
    Priority 1.3: Solves the "black box problem".
    """
    decisions = []
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            for symbol, symbol_state in state.items():
                if isinstance(symbol_state, dict):
                    # Extract decision context
                    decision = {
                        "symbol": symbol,
                        "inventory": symbol_state.get("inventory", 0),
                        "paused": symbol_state.get("paused", False),
                        "regime": symbol_state.get("regime", "UNKNOWN"),
                        "last_quote_time": symbol_state.get("last_quote_time", 0),
                        "skip_reason": symbol_state.get("skip_reason", None),
                        "volatility": symbol_state.get("volatility", 0),
                        "spread_bps": symbol_state.get("current_spread_bps", 0),
                    }
                    decisions.append(decision)
    except Exception as e:
        logger.warning(f"Failed to load decision log: {e}")
    
    return JSONResponse({"decisions": decisions})


@app.post("/api/close/{symbol}")
async def close_position(symbol: str):
    """
    Close a single position (not FLATTEN all).
    Priority 2.1: Granular control per symbol.
    """
    symbol = symbol.upper()
    signal_file = f"data/CLOSE_{symbol}"
    
    try:
        with open(signal_file, "w") as f:
            f.write(f"CLOSE:{datetime.now().isoformat()}")
        
        logger.info(f"📤 Close signal created for {symbol}")
        return {"status": "close_requested", "symbol": symbol}
    except Exception as e:
        logger.error(f"Failed to create close signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/cancel_order/{client_order_id}")
async def cancel_single_order(client_order_id: str):
    """
    Cancel a single order by ID.
    Priority 1.2: Granular control.
    """
    signal_file = f"data/CANCEL_ORDER_{client_order_id}"
    
    try:
        with open(signal_file, "w") as f:
            f.write(f"CANCEL:{datetime.now().isoformat()}")
        
        logger.info(f"📤 Cancel signal created for order {client_order_id}")
        return {"status": "cancel_requested", "order_id": client_order_id}
    except Exception as e:
        logger.error(f"Failed to create cancel signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/cancel_orders")
async def cancel_all_orders():
    """
    Cancel all open orders without selling holdings.
    Priority 2.1: Granular control - cancel orders, keep positions.
    """
    signal_file = "data/CANCEL_ALL_ORDERS"
    
    try:
        with open(signal_file, "w") as f:
            f.write(f"CANCEL:{datetime.now().isoformat()}")
        
        logger.info("📤 Cancel all orders signal created")
        return {"status": "cancel_requested"}
    except Exception as e:
        logger.error(f"Failed to create cancel signal: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/exposure")
async def api_exposure():
    """
    Get portfolio exposure breakdown (invested vs liquid).
    Priority 3.1: Capital utilization at a glance.
    """
    exposure = {
        "total_value": 0.0,
        "invested_value": 0.0,
        "liquid_value": 0.0,
        "invested_pct": 0.0,
        "per_symbol": {}
    }
    
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
            if cursor.fetchone():
                positions = pd.read_sql("SELECT * FROM positions", conn)
                
                for _, row in positions.iterrows():
                    if 'quantity' in row and row['quantity'] != 0:
                        qty = abs(float(row['quantity']))
                        price = float(row.get('mark_price', row.get('avg_entry_price', 0)))
                        value = qty * price
                        exposure['invested_value'] += value
                        exposure['per_symbol'][row['symbol']] = {
                            "qty": row['quantity'],
                            "value": value
                        }
            
            # Try to get wallet balance for liquid
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='account'")
            if cursor.fetchone():
                account = pd.read_sql("SELECT * FROM account LIMIT 1", conn)
                if not account.empty and 'available_balance' in account.columns:
                    exposure['liquid_value'] = float(account.iloc[0]['available_balance'])
            
            exposure['total_value'] = exposure['invested_value'] + exposure['liquid_value']
            if exposure['total_value'] > 0:
                exposure['invested_pct'] = (exposure['invested_value'] / exposure['total_value']) * 100
                
        except Exception as e:
            logger.warning(f"Failed to calculate exposure: {e}")
        finally:
            conn.close()
    
    return JSONResponse(exposure)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from client, just keep connection alive
            # The broadcasting happens in the background task
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== BACKTEST API ====================
import subprocess
import threading

# Global backtest state
backtest_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "last_report": None,
    "output": []
}

@app.post("/api/backtest/run")
async def api_backtest_run(request: Request):
    """
    Start a new backtest simulation.
    """
    global backtest_state
    
    if backtest_state["running"]:
        return JSONResponse({"error": "Backtest already running"}, status_code=400)
    
    body = await request.json()
    symbol = body.get("symbol", "BTCUSDT")
    duration = body.get("duration_minutes", 60)
    
    backtest_state = {
        "running": True,
        "progress": 0,
        "status": "starting",
        "last_report": None,
        "output": []
    }
    
    def run_backtest():
        global backtest_state
        try:
            backtest_state["status"] = "running"
            
            # Run the backtest script
            proc = subprocess.Popen(
                ["python", "scripts/run_backtest_simulation.py"],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            for line in proc.stdout:
                backtest_state["output"].append(line.strip())
                # Parse progress from output
                if "Tick " in line:
                    try:
                        parts = line.split("/")
                        if len(parts) >= 2:
                            current = int(parts[0].split()[-1])
                            total = int(parts[1].split()[0])
                            backtest_state["progress"] = int((current / total) * 100)
                    except:
                        pass
                        
            proc.wait()
            
            # Find report file
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
            if os.path.exists(reports_dir):
                reports = sorted([f for f in os.listdir(reports_dir) if f.endswith('.html')])
                if reports:
                    backtest_state["last_report"] = os.path.join(reports_dir, reports[-1])
            
            backtest_state["status"] = "completed"
            backtest_state["progress"] = 100
            
        except Exception as e:
            backtest_state["status"] = f"error: {e}"
        finally:
            backtest_state["running"] = False
    
    thread = threading.Thread(target=run_backtest)
    thread.start()
    
    return JSONResponse({"message": "Backtest started", "status": "running"})


@app.get("/api/backtest/status")
async def api_backtest_status():
    """
    Get current backtest status and progress.
    """
    return JSONResponse({
        "running": backtest_state["running"],
        "progress": backtest_state["progress"],
        "status": backtest_state["status"],
        "has_report": backtest_state["last_report"] is not None,
        "output_lines": len(backtest_state["output"]),
        "last_output": backtest_state["output"][-5:] if backtest_state["output"] else []
    })


@app.get("/api/backtest/report")
async def api_backtest_report():
    """
    Get the last backtest report content.
    """
    if backtest_state["last_report"] and os.path.exists(backtest_state["last_report"]):
        with open(backtest_state["last_report"], 'r') as f:
            return HTMLResponse(f.read())
    return JSONResponse({"error": "No report available"}, status_code=404)


# Serve Static Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print("🦂 SERQET COMMAND CENTER LAUNCHING... [WebSockets Active]")
    print("👉 Open http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
