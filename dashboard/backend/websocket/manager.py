"""WebSocket connection manager."""
import asyncio
import json
from typing import List
from fastapi import WebSocket


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_update(self, update_type: str, data: dict):
        """Send a typed update to all clients."""
        message = {
            "type": update_type,
            "data": data
        }
        await self.broadcast(message)
