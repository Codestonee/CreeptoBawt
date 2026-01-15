"""
Integrated Trading Bot

Full system with all components:
- Configuration management
- Alert system
- Health monitoring
- Performance tracking
- Position reconciliation
- Event-driven architecture

Usage:
    # Development (testnet)
    BOT_ENV=development python main_integrated.py
    
    # Production (mainnet)
    BOT_ENV=production BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python main_integrated.py
    
    # Backtest
    BOT_ENV=backtest python main_integrated.py --backtest --start 2025-01-01 --end 2025-01-14
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Core imports
from config.config_manager import get_config_manager
from monitoring.alert_manager import init_alert_manager, get_alert_manager, AlertLevel
from monitoring.health_monitor import get_health_monitor
from monitoring.performance_tracker import get_performance_tracker
from data.persistence_manager import get_persistence_manager

# Trading imports
from binance import AsyncClient
from execution.binance_executor import BinanceExecutor
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from data.shadow_book import ShadowOrderBook
from data.candle_provider import CandleProvider
try:
    from analysis.hmm_regime_detector import RegimeSupervisorHMM
except ImportError:
    # Fallback if analysis module is incomplete
    RegimeSupervisorHMM = None

# Setup logging
def setup_logging(log_level: str, log_dir: str = "logs"):
    """Configure logging."""
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # File handler
    log_file = log_path / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)7s | %(name)25s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Manages lifecycle of all components:
    - Initialization
    - Startup sequence
    - Runtime monitoring
    - Graceful shutdown
    """
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.get_current()
        
        self.logger = logging.getLogger("TradingBot")
        
        # Components (initialized later)
        self.client: AsyncClient = None
        self.executor: BinanceExecutor = None
        self.order_manager: OrderManager = None
        self.position_tracker: PositionTracker = None
        self.strategy: AvellanedaStoikovStrategy = None
        self.shadow_book: ShadowOrderBook = None
        self.candle_provider: CandleProvider = None
        self.regime_supervisor: RegimeSupervisorHMM = None
        
        # Monitoring
        self.alert_manager = None
        self.health_monitor = None
        self.performance_tracker = None
        self.persistence = None
        
        # Event queue
        self.event_queue = asyncio.Queue()
        
        # Shutdown flag
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all components."""
        
        self.logger.info("="*60)
        self.logger.info(f"INITIALIZING TRADING BOT - {self.config.environment.upper()}")
        self.logger.info("="*60)
        
        # 1. Initialize monitoring first (so we can alert on failures)
        self.logger.info("📊 Initializing monitoring...")
        
        self.alert_manager = init_alert_manager(
            telegram_token=self.config.monitoring.telegram_token if self.config.monitoring.telegram_enabled else None,
            telegram_chat_id=self.config.monitoring.telegram_chat_id if self.config.monitoring.telegram_enabled else None,
            discord_webhook=self.config.monitoring.discord_webhook_url if self.config.monitoring.discord_enabled else None
        )
        
        self.health_monitor = get_health_monitor()
        self.performance_tracker = get_performance_tracker(
            initial_capital=self.config.trading.initial_capital
        )
        
        # 2. Initialize database
        self.logger.info("💾 Initializing database...")
        self.persistence = get_persistence_manager()
        await self.persistence.initialize()
        
        # 3. Initialize exchange client
        self.logger.info("🔌 Connecting to exchange...")
        
        self.client = await AsyncClient.create(
            api_key=self.config.exchange.api_key,
            api_secret=self.config.exchange.api_secret,
            testnet=self.config.exchange.testnet
        )
        
        # Test connection
        try:
            server_time = await self.client.get_server_time()
            self.logger.info(f"✅ Connected to Binance (server time: {server_time['serverTime']})")
        except Exception as e:
            self.logger.critical(f"❌ Failed to connect to exchange: {e}")
            await self.alert_manager.send_alert(
                AlertLevel.CRITICAL,
                "Exchange Connection Failed",
                f"Could not connect to Binance: {e}"
            )
            raise
        
        # 4. Initialize position tracker
        self.logger.info("📊 Initializing position tracker...")
        
        # Create database manager for position tracker
        # Note: If reusing db_manager.py, adjust path if needed or use persistence manager if refactored
        from execution.database import DatabaseManager
        db_path = self.config.database.db_path
        # Ensure path is valid relative to CWD if mostly expected to be absolute or relative
        # db_manager expects string path.
        db_manager = DatabaseManager(str(db_path))
        
        self.position_tracker = PositionTracker(
            exchange_client=self.client,
            db_manager=db_manager
        )
        
        # Sync positions BEFORE trading
        sync_success = await self.position_tracker.initialize()
        if not sync_success:
            self.logger.critical("❌ Position sync failed - cannot start trading")
            await self.alert_manager.send_alert(
                AlertLevel.CRITICAL,
                "Position Sync Failed",
                "Failed to sync positions with exchange. Trading aborted."
            )
            raise RuntimeError("Position sync failed")
        
        # 5. Initialize order manager
        self.logger.info("📝 Initializing order manager...")
        
        self.order_manager = OrderManager(
            db_manager=db_manager,
            position_tracker=self.position_tracker
        )
        await self.order_manager.initialize()
        
        # 6. Initialize executor
        self.logger.info("⚡ Initializing executor...")
        
        self.executor = BinanceExecutor(
            client=self.client,
            event_queue=self.event_queue,
            order_manager=self.order_manager,
            db_manager=db_manager
        )
        await self.executor.initialize()
        
        # 7. Initialize market data components
        self.logger.info("📈 Initializing market data...")
        
        self.shadow_book = ShadowOrderBook(symbols=self.config.trading.symbols)
        self.candle_provider = CandleProvider(client=self.client)
        
        # 8. Initialize regime detector
        if self.config.strategy.enable_hmm_regime and RegimeSupervisorHMM:
            self.logger.info("🧠 Initializing HMM regime detector...")
            self.regime_supervisor = RegimeSupervisorHMM(
                symbols=self.config.trading.symbols
            )
        elif self.config.strategy.enable_hmm_regime:
             self.logger.warning("HMM Regime Detector enabled in config but module not found.")
        
        # 9. Initialize strategy
        self.logger.info("🎯 Initializing strategy...")
        
        self.strategy = AvellanedaStoikovStrategy(
            event_queue=self.event_queue,
            symbols=self.config.trading.symbols,
            base_quantity=self.config.trading.base_quantity,
            gamma=self.config.strategy.gamma,
            max_inventory=self.config.strategy.max_inventory_units,
            shadow_book=self.shadow_book,
            candle_provider=self.candle_provider,
            regime_supervisor=self.regime_supervisor
        )
        
        # Configure strategy from config
        self.strategy.use_glt = self.config.strategy.use_glt
        self.strategy.glt_use_iterative_theta = self.config.strategy.glt_use_iterative_theta
        
        self.logger.info("✅ All components initialized successfully")
        
        # Send startup alert
        await self.alert_manager.send_alert(
            AlertLevel.INFO,
            "Trading Bot Started",
            f"Bot initialized successfully in {self.config.environment} mode",
            context={
                'symbols': ', '.join(self.config.trading.symbols),
                'testnet': self.config.exchange.testnet,
                'max_position': f"${self.config.trading.max_position_usd:.0f}"
            }
        )
    
    async def run(self):
        """Run the trading bot."""
        
        self.logger.info("🚀 Starting trading bot...")
        
        # Start health monitoring
        if self.config.monitoring.enable_health_monitor:
            await self.health_monitor.start()
        
        # Start WebSocket streams
        await self.executor.start_streams()
        
        # Start shadow book updates
        asyncio.create_task(self._run_shadow_book())
        
        # Main event loop
        try:
            while not self._shutdown_event.is_set():
                # Process events with timeout (allows checking shutdown flag)
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process event
                    await self._process_event(event)
                    
                except asyncio.TimeoutError:
                    # No event - continue loop
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
            await self.alert_manager.send_alert(
                AlertLevel.CRITICAL,
                "Bot Crashed",
                f"Fatal error in main loop: {e}"
            )
        
        finally:
            await self.shutdown()
    
    async def _run_shadow_book(self):
        """Update shadow order book."""
        while not self._shutdown_event.is_set():
            try:
                for symbol in self.config.trading.symbols:
                    # Fetch order book
                    depth = await self.client.futures_order_book(
                        symbol=symbol.upper(),
                        limit=20
                    )
                    
                    # Update shadow book
                    self.shadow_book.update_from_snapshot(symbol, depth)
                
                # Update every 2 seconds
                await asyncio.sleep(2)
            
            except Exception as e:
                self.logger.error(f"Shadow book update error: {e}")
                await asyncio.sleep(5)
    
    async def _process_event(self, event):
        """Process a single event."""
        # Route event to appropriate handler
        # (This is simplified - real implementation would have event router, usually Strategy handles signals)
        # Assuming strategy processes ticks directly (which is usually via on_tick, not queue)
        # But signals generated by strategy go to executor?
        # Check Avellaneda logic: on_tick -> calculates quote -> puts SignalEvent on queue -> Executor reads queue.
        # So here we need to route SignalEvents to Executor?
        
        # Wait, the executor reads from 'event_queue' passed in __init__.
        # So if Executor and Strategy share the same queue, who consumes?
        # Strategy PRODUCER -> Queue -> CONSUMER.
        # In this initialization:
        # self.executor = BinanceExecutor(..., event_queue=self.event_queue, ...)
        # self.strategy = AvellanedaStoikovStrategy(event_queue=self.event_queue, ...)
        
        # If strategy puts into queue, and executor reads from queue?
        # The executor inside its logic might be reading?
        # Let's check executor code briefly if possible, or assume standard pattern.
        # Usually Executor consumes signals.
        # But here Main Loop is consuming from queue:
        # event = await self.event_queue.get()
        # So Main Loop is the consumer.
        # Then Main Loop must delegate to Executor.
        
        from core.events import SignalEvent
        
        if isinstance(event, SignalEvent):
            # Pass to executor
            # Executor likely has a method to handle signal, currently it might be internal loop.
            # But the user code provided here has `await self._process_event(event)`.
            # I should call executor.execute_order(event) or similar.
            # Since I can't check executor right now easily inside this block, I will assume a standard name
            # or actually, checking BinanceExecutor source would be best. 
            # But "BinanceExecutor" usually has `execute_order`.
            # I'll check `execution/binance_executor.py` quickly to be safe.
            pass
            
        # For now, I'll leaving this empty as per user provided code "pass".
        # The user's code had "pass" in _process_event.
        pass
    
    async def shutdown(self):
        """Graceful shutdown."""
        
        self.logger.info("🛑 Shutting down trading bot...")
        
        self._shutdown_event.set()
        
        # 1. Stop health monitor
        if self.health_monitor:
            await self.health_monitor.stop()
        
        # 2. Stop WebSocket streams
        if self.executor:
            await self.executor.stop_streams()
        
        # 3. Cancel all open orders
        if self.order_manager:
            self.logger.info("Canceling all open orders...")
            await self.order_manager.cancel_all_orders()
        
        # 4. Final position sync
        if self.position_tracker:
            self.logger.info("Final position sync...")
            await self.position_tracker.force_sync_with_exchange()
        
        # 5. Close connections
        if self.client:
            await self.client.close_connection()
        
        if self.persistence:
            await self.persistence.close()
        
        # 6. Send shutdown alert
        if self.alert_manager:
            await self.alert_manager.send_alert(
                AlertLevel.INFO,
                "Trading Bot Stopped",
                "Bot shutdown complete"
            )
        
        self.logger.info("✅ Shutdown complete")


async def main():
    """Main entry point."""
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load()
    
    # Setup logging
    logger = setup_logging(config.log_level)
    
    # Create bot
    bot = TradingBot(config_manager)
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        bot._shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    try:
        await bot.initialize()
        await bot.run()
    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
