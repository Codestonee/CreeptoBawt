"""FastAPI Dashboard Backend."""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from pathlib import Path

from dashboard.backend.routers import balances, positions, orders, trades, pnl, system
from dashboard.backend.websocket.manager import WebSocketManager
from dashboard.backend.config import DashboardConfig

# Create config
config = DashboardConfig()

# Create FastAPI app
app = FastAPI(
    title="CreeptoBawt Dashboard",
    version="1.0.0",
    description="Web dashboard for CreeptoBawt trading system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
ws_manager = WebSocketManager()

# Include routers
app.include_router(balances.router, prefix=config.api_prefix, tags=["balances"])
app.include_router(positions.router, prefix=config.api_prefix, tags=["positions"])
app.include_router(orders.router, prefix=config.api_prefix, tags=["orders"])
app.include_router(trades.router, prefix=config.api_prefix, tags=["trades"])
app.include_router(pnl.router, prefix=config.api_prefix, tags=["pnl"])
app.include_router(system.router, prefix=config.api_prefix, tags=["system"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - in production, handle commands
            await ws_manager.send_message(websocket, {"echo": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "dashboard",
        "version": "1.0.0"
    }


# Mount static files if they exist (for production)
# Look for static directory relative to this file's location
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


def main():
    """Run the server."""
    uvicorn.run(
        "dashboard.backend.main:app",
        host=config.host,
        port=config.port,
        reload=True
    )


if __name__ == "__main__":
    main()
