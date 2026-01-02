"""
Health Check Server - HTTP endpoint for system health monitoring.

Provides:
- /health - System health status
- /metrics - Prometheus metrics
- /ready - Readiness probe for Kubernetes
- /live - Liveness probe for Kubernetes
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

from utils.metrics import get_metrics

log = structlog.get_logger()


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # healthy, degraded, unhealthy
    mode: str
    uptime_seconds: float
    trading_allowed: bool
    kill_switch_active: bool = False
    components: Dict[str, str] = field(default_factory=dict)
    open_orders: int = 0
    errors_last_hour: int = 0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "mode": self.mode,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "trading_allowed": self.trading_allowed,
            "kill_switch_active": self.kill_switch_active,
            "components": self.components,
            "open_orders": self.open_orders,
            "errors_last_hour": self.errors_last_hour,
            "timestamp": self.timestamp,
        }


# Health check callbacks
_health_callbacks: List[Callable[[], HealthStatus]] = []


def register_health_callback(callback: Callable[[], HealthStatus]) -> None:
    """Register a health check callback."""
    _health_callbacks.append(callback)


def get_health_status() -> HealthStatus:
    """Get aggregated health status from all callbacks."""
    if not _health_callbacks:
        return HealthStatus(
            status="unknown",
            mode="unknown",
            uptime_seconds=0,
            trading_allowed=False,
        )
    
    # Use first callback as primary (typically the engine)
    return _health_callbacks[0]()


class HealthCheckServer:
    """
    Async HTTP server for health checks and metrics.
    
    Endpoints:
    - GET /health - Full health status JSON
    - GET /metrics - Prometheus metrics
    - GET /ready - Readiness probe (200 if healthy/degraded)
    - GET /live - Liveness probe (200 always, unless crashed)
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.host = host
        self.port = port
        self._server = None
        self._running = False
    
    async def start(self) -> None:
        """Start the health check server."""
        if self._running:
            return
        
        try:
            from aiohttp import web
            
            app = web.Application()
            app.router.add_get("/health", self._health_handler)
            app.router.add_get("/metrics", self._metrics_handler)
            app.router.add_get("/ready", self._ready_handler)
            app.router.add_get("/live", self._live_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            self._server = web.TCPSite(runner, self.host, self.port)
            await self._server.start()
            
            self._running = True
            log.info(
                "health_check_server_started",
                host=self.host,
                port=self.port,
            )
            
        except ImportError:
            log.warning("aiohttp_not_installed", message="Health check server disabled")
        except Exception as e:
            log.error("health_check_server_error", error=str(e))
    
    async def stop(self) -> None:
        """Stop the health check server."""
        if self._server:
            await self._server.stop()
            self._running = False
            log.info("health_check_server_stopped")
    
    async def _health_handler(self, request) -> Any:
        """Handle /health endpoint."""
        from aiohttp import web
        
        health = get_health_status()
        
        # Determine HTTP status code
        if health.status == "healthy":
            status_code = 200
        elif health.status == "degraded":
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service unavailable
        
        return web.json_response(health.to_dict(), status=status_code)
    
    async def _metrics_handler(self, request) -> Any:
        """Handle /metrics endpoint."""
        from aiohttp import web
        
        metrics = get_metrics()
        content = metrics.generate_metrics()
        
        return web.Response(
            body=content,
            content_type=metrics.get_content_type(),
        )
    
    async def _ready_handler(self, request) -> Any:
        """Handle /ready endpoint for Kubernetes."""
        from aiohttp import web
        
        health = get_health_status()
        
        if health.status in ("healthy", "degraded"):
            return web.json_response({"ready": True}, status=200)
        
        return web.json_response({"ready": False}, status=503)
    
    async def _live_handler(self, request) -> Any:
        """Handle /live endpoint for Kubernetes."""
        from aiohttp import web
        
        # Always return 200 unless server is crashed
        return web.json_response({"alive": True}, status=200)
