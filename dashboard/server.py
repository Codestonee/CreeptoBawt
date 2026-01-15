import os
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

app = FastAPI(title="Serqet Command Center")

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

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

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
            with open("health_status.json") as f:
                health.update(json.load(f))
        except: pass
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
        trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        positions = pd.read_sql("SELECT * FROM positions", conn)
        
        # Balance (Mock initial + Realized + Unrealized)
        initial_capital = 500.0 # From Settings
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
                if row['quantity'] != 0:
                    inventory[row['symbol']] = {
                        "qty": row['quantity'],
                        "pnl": row.get('unrealized_pnl', 0),
                        "entry": row.get('avg_entry_price', 0)
                    }
        metrics['inventory'] = inventory

        # 6. STRATEGY STATE
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    metrics['strategy_state'] = json.load(f)
            except: pass

        # 7. LOGS
        if os.path.exists(LOG_FILE):
             with open(LOG_FILE, 'r') as f:
                lines = f.readlines()[-30:]
                metrics['logs'] = [l.strip() for l in reversed(lines)]
                
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    finally:
        conn.close()
        
    return metrics

@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(get_metrics())

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

# Background Broadcast Loop
async def broadcast_loop():
    while True:
        try:
            if manager.active_connections:
                data = get_metrics()
                await manager.broadcast(json.dumps(data))
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(0.5) # 500ms update rate

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())

# Serve Static Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    print("🦂 SERQET COMMAND CENTER LAUNCHING... [WebSockets Active]")
    print("👉 Open http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
