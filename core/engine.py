"""
Main Trading Engine - Orchestrates all system components.

Uses uvloop for high-performance async I/O on Linux/macOS.
Falls back to standard asyncio on Windows.
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from core.event_bus import EventBus, create_event_bus
from core.events import HealthEvent, HealthStatus
from core.mode_controller import ModeController, TradingMode

# Configure structured logging
log = structlog.get_logger()

# Try to use uvloop (Linux/macOS) or winloop (Windows) for better performance
HAS_FAST_LOOP = False
if sys.platform == "win32":
    try:
        import winloop
        HAS_FAST_LOOP = True
        log.info("winloop_available", message="Using winloop for faster async I/O on Windows")
    except ImportError:
        log.warning("winloop_not_available", message="Using standard asyncio event loop")
else:
    try:
        import uvloop
        HAS_FAST_LOOP = True
        log.info("uvloop_available", message="Using uvloop for faster async I/O")
    except ImportError:
        log.warning("uvloop_not_available", message="Using standard asyncio event loop")


@dataclass
class EngineConfig:
    """Engine configuration."""
    # Mode settings
    initial_mode: TradingMode = TradingMode.PAPER
    
    # Event bus settings
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379"
    
    # Health check settings
    health_check_interval: float = 30.0
    health_check_port: int = 8080
    
    # Shutdown settings
    shutdown_timeout: float = 30.0
    
    # Component settings
    enabled_exchanges: List[str] = field(default_factory=lambda: ["binance"])
    enabled_strategies: List[str] = field(default_factory=list)


class ComponentManager:
    """Manages lifecycle of system components."""
    
    def __init__(self) -> None:
        self._components: Dict[str, Any] = {}
        self._startup_order: List[str] = []
        self._running = False
    
    def register(self, name: str, component: Any, startup_order: int = 100) -> None:
        """Register a component with startup priority."""
        self._components[name] = {
            "instance": component,
            "order": startup_order,
            "status": "registered",
        }
        self._startup_order = sorted(
            self._components.keys(),
            key=lambda k: self._components[k]["order"],
        )
        log.info("component_registered", name=name, order=startup_order)
    
    def get(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        entry = self._components.get(name)
        return entry["instance"] if entry else None
    
    async def start_all(self) -> None:
        """Start all components in order."""
        if self._running:
            log.warning("components_already_running")
            return
        
        log.info("starting_components", count=len(self._startup_order))
        
        for name in self._startup_order:
            entry = self._components[name]
            component = entry["instance"]
            
            try:
                if hasattr(component, "start"):
                    await component.start()
                elif hasattr(component, "connect"):
                    await component.connect()
                
                entry["status"] = "running"
                log.info("component_started", name=name)
                
            except Exception as e:
                entry["status"] = "failed"
                log.error("component_start_failed", name=name, error=str(e))
                raise
        
        self._running = True
        log.info("all_components_started")
    
    async def stop_all(self) -> None:
        """Stop all components in reverse order."""
        if not self._running:
            return
        
        log.info("stopping_components", count=len(self._startup_order))
        
        # Stop in reverse order
        for name in reversed(self._startup_order):
            entry = self._components[name]
            component = entry["instance"]
            
            try:
                if hasattr(component, "stop"):
                    await component.stop()
                elif hasattr(component, "disconnect"):
                    await component.disconnect()
                
                entry["status"] = "stopped"
                log.info("component_stopped", name=name)
                
            except Exception as e:
                log.error("component_stop_error", name=name, error=str(e))
        
        self._running = False
        log.info("all_components_stopped")
    
    def get_health(self) -> Dict[str, str]:
        """Get health status of all components."""
        return {
            name: entry["status"]
            for name, entry in self._components.items()
        }


class TradingEngine:
    """
    Main trading system engine.
    
    Orchestrates all components: connectors, strategies, risk, execution.
    Handles graceful startup and shutdown.
    """
    
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._start_time: float = 0.0
        
        # Core components
        self.event_bus: EventBus = create_event_bus(
            use_redis=self.config.use_redis,
            redis_url=self.config.redis_url,
        )
        self.mode_controller = ModeController(self.config.initial_mode)
        self.components = ComponentManager()
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        log.info(
            "engine_initialized",
            mode=self.config.initial_mode,
            use_redis=self.config.use_redis,
            uvloop_available=HAS_UVLOOP,
        )
    
    @property
    def uptime(self) -> float:
        """Get engine uptime in seconds."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal(s)),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                # Use alternative approach
                signal.signal(sig, lambda s, f: asyncio.create_task(self._handle_signal(s)))
        
        log.info("signal_handlers_configured")
    
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        log.warning("shutdown_signal_received", signal=sig.name)
        await self.shutdown()
    
    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            log.warning("engine_already_running")
            return
        
        self._start_time = time.time()
        self._running = True
        
        log.info("engine_starting")
        
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Connect event bus
            await self.event_bus.connect()
            
            # Register core components
            self.components.register("event_bus", self.event_bus, startup_order=1)
            self.components.register("mode_controller", self.mode_controller, startup_order=2)
            
            # Start health check task
            self._tasks.append(
                asyncio.create_task(self._health_check_loop())
            )
            
            # Publish engine started event
            await self.event_bus.publish("system.engine", {
                "event": "started",
                "mode": self.config.initial_mode.value,
                "timestamp": int(time.time() * 1_000_000),
            })
            
            log.info("engine_started", mode=self.config.initial_mode)
            
        except Exception as e:
            log.error("engine_start_failed", error=str(e))
            await self.shutdown()
            raise
    
    async def run(self) -> None:
        """Run the engine until shutdown."""
        await self.start()
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        if not self._running:
            return
        
        log.info("engine_shutting_down")
        
        # Signal shutdown
        self._shutdown_event.set()
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Publish shutdown event
        try:
            await self.event_bus.publish("system.engine", {
                "event": "shutdown",
                "uptime": self.uptime,
                "timestamp": int(time.time() * 1_000_000),
            })
        except Exception:
            pass
        
        # Stop all components
        await self.components.stop_all()
        
        # Disconnect event bus
        await self.event_bus.disconnect()
        
        log.info("engine_shutdown_complete", uptime=self.uptime)
    
    async def _health_check_loop(self) -> None:
        """Periodic health check."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                health = self.get_health_status()
                
                await self.event_bus.publish("system.health", health)
                
                if health["status"] != "healthy":
                    log.warning("system_health_degraded", health=health)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("health_check_error", error=str(e))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        component_health = self.components.get_health()
        
        # Determine overall status
        if all(s == "running" for s in component_health.values()):
            status = "healthy"
        elif any(s == "failed" for s in component_health.values()):
            status = "unhealthy"
        else:
            status = "degraded"
        
        return {
            "status": status,
            "mode": self.mode_controller.current_mode.value,
            "uptime_seconds": self.uptime,
            "trading_allowed": self.mode_controller.is_trading_allowed(),
            "components": component_health,
            "timestamp": int(time.time() * 1_000_000),
        }


def setup_event_loop() -> asyncio.AbstractEventLoop:
    """Setup the best available event loop."""
    if HAS_FAST_LOOP:
        if sys.platform == "win32":
            import winloop
            winloop.install()
            log.info("winloop_installed")
        else:
            import uvloop
            uvloop.install()
            log.info("uvloop_installed")
    
    return asyncio.new_event_loop()


def main() -> None:
    """Main entry point."""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup event loop
    loop = setup_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create and run engine
    config = EngineConfig()
    engine = TradingEngine(config)
    
    try:
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        log.info("keyboard_interrupt_received")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    main()
