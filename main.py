import asyncio
import logging
import sys
import signal
import io
import gc
import aiosqlite # NEW: For async DB checks

# --------------------------------------------------------------------------
# PERFORMANCE: Tune Garbage Collection for HFT
# Default thresholds cause frequent "stop-the-world" pauses
# New thresholds: collect less often, reduce latency spikes
# --------------------------------------------------------------------------
gc.set_threshold(100000, 10, 10)  # (gen0, gen1, gen2)

# --------------------------------------------------------------------------
# FIX: Force Windows to use UTF-8 for stdout/stderr
# --------------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    import uvloop
except ImportError:
    uvloop = None

from config.settings import settings
from core.engine import TradingEngine
from connectors.binance_futures import BinanceFuturesConnector
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from strategies.funding_arb import FundingArbStrategy
from strategies.cross_exchange_arb import CrossExchangeArbStrategy
# NEW: Import ShadowBook for L2 order book data
from core.shadow_book import get_shadow_book
# NEW: Import HMM Regime Detector (replaces ADX-based detection)
from analysis.hmm_regime_detector import RegimeSupervisorHMM
# from strategies.funding_arb import FundingRateMonitor, CarryTradeManager, run_funding_arb_loop  <-- Removed
# NEW: Telegram Alerts for remote monitoring
from utils.telegram_alerts import get_telegram_alerter

# NEW: Centralized logging
from utils.logging_config import setup_logging

# Initialize logging immediately
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("Main")





async def shutdown(engine, shadow_book, regime_supervisor, loop, stop_event=None):
    """Graceful shutdown with ShadowBook and HMM cleanup."""
    logger.warning("Shutdown initiated. Cleaning up...")
    
    # Signal circuit breaker monitor to stop
    if stop_event:
        stop_event.set()
    
    # Send Telegram alert
    alerter = get_telegram_alerter()
    await alerter.alert_bot_stopped("Normal shutdown")
    
    # Stop HMM background processes
    if regime_supervisor:
        regime_supervisor.stop()
    
    # Stop ShadowBook first
    if shadow_book:
        await shadow_book.stop()
        
    await engine.stop()
    
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} pending tasks...")
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            pass
            
    # Clean up default executor (prevents atexit errors)
    try:
        await loop.shutdown_default_executor()
    except Exception:
        pass

    logger.info("Shutdown complete.")


async def monitor_circuit_breaker(stop_event: asyncio.Event):
    """
    SAFETY NET: Periodically check PnL and trigger graceful shutdown if drawdown > 5%.
    """
    import sqlite3
    import os
    from config.settings import settings
    
    logger.info("Circuit Breaker Monitor started")
    
    while not stop_event.is_set():
        try:
            await asyncio.sleep(10)
            
            async with aiosqlite.connect('data/trading_data.db') as db:
                async with db.execute("SELECT SUM(pnl) FROM trades") as cursor:
                    row = await cursor.fetchone()
                    total_pnl = row[0] if row and row[0] else 0.0

            limit_loss = settings.INITIAL_CAPITAL * -0.05
            
            if total_pnl < limit_loss:
                logger.critical("=" * 60)
                logger.critical(f"CIRCUIT BREAKER TRIGGERED! PnL ${total_pnl:.2f} < Limit ${limit_loss:.2f}")
                logger.critical("INITIATING GRACEFUL SHUTDOWN...")
                logger.critical("=" * 60)
                
                STOP_SIGNAL_FILE = "data/STOP_SIGNAL"
                # Ensure directory exists
                os.makedirs(os.path.dirname(STOP_SIGNAL_FILE), exist_ok=True)
                with open(STOP_SIGNAL_FILE, "w") as f:
                    f.write("CIRCUIT_BREAKER_TRIGGERED")
                
                logger.critical("🚨 Triggering graceful shutdown via stop event...")
                stop_event.set()
                
                # Give shutdown handlers time to run
                for _ in range(10):
                    if stop_event.is_set(): 
                        pass
                    await asyncio.sleep(1)
                
                logger.critical("🚨 Graceful shutdown timed out - Forcing EXIT")
                os._exit(1)
            
        except Exception as e:
            logger.error(f"Circuit Breaker Error: {e}")
            await asyncio.sleep(5)



def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if uvloop and sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Create loop for the application
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    
    # ==========================================================================
    
    # 1. Initialize Engine
    engine = TradingEngine(settings)
    
    # 2. Initialize Shadow Order Book (Local L2 book for true Mid Price)
    logger.info("📊 Initializing Shadow Order Book...")
    shadow_book = get_shadow_book(
        symbols=settings.TRADING_SYMBOLS,
        testnet=getattr(settings, 'TESTNET', True)
    )
    
    # 2b. Initialize HMM Regime Detector (replaces ADX)
    logger.info("🧠 Initializing HMM Regime Detector...")
    regime_supervisor = RegimeSupervisorHMM(settings.TRADING_SYMBOLS)
    regime_supervisor.start()  # Starts background retraining processes
    logger.info("✅ HMM Regime Detector started (background retrainer active)")

    # 3. Initialize Connector (Trades/Ticks to Engine)
    binance_connector = BinanceFuturesConnector(
        event_queue=engine.event_queue,
        symbols=settings.TRADING_SYMBOLS,
        ws_url=settings.BINANCE_WS_URL
    )
    engine.add_connector(binance_connector)

    # 4. Initialize Strategy (Avellaneda-Stoikov)
    # CRITICAL: Pass shadow_book for true mid price calculation!
    if settings.TRADING_SYMBOLS:
        as_strategy = AvellanedaStoikovStrategy(
            event_queue=engine.event_queue, 
            symbols=settings.TRADING_SYMBOLS,
            base_quantity=getattr(settings, 'GRID_BASE_QUANTITY', 0.01),
            gamma=0.1,  # Risk aversion
            max_inventory=getattr(settings, 'MAX_INVENTORY', 1.0),
            shadow_book=shadow_book,  # <-- THE CRITICAL CONNECTION!
            regime_supervisor=regime_supervisor  # <-- HMM regime updates
        )
        engine.add_strategy(as_strategy)
        logger.info(f"✅ Avellaneda-Stoikov Strategy activated for {settings.TRADING_SYMBOLS}")
    else:
        logger.warning("No trading symbols configured. Strategy not started.")

    # Signal handlers (Windows only - Unix handlers set up after stop_event is created)
    if sys.platform == 'win32':
        logger.info("Windows detected: Registering SIGTERM handler...")
        def handle_sigterm(signum, frame):
            logger.warning("Received SIGTERM. triggering shutdown...")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGTERM, handle_sigterm)

    # Run everything
    try:
        # Start ShadowBook separately (it runs its own WebSocket streams)
        loop.create_task(shadow_book.start())
        logger.info("📈 Shadow Order Book WebSocket started")
        
        # Start Funding Arbitrage Strategy (DISABLED - requires larger balance)
        # funding_arb = FundingArbStrategy(event_queue=engine.event_queue, symbols=settings.TRADING_SYMBOLS)
        # engine.add_strategy(funding_arb)
        # logger.info("💰 Funding Arbitrage Strategy activated")

        # Start Cross-Exchange Arb (if multiple exchanges)
        if len(getattr(settings, 'ACTIVE_EXCHANGES', [])) > 1:
            cross_arb = CrossExchangeArbStrategy(event_queue=engine.event_queue)
            engine.add_strategy(cross_arb)
            logger.info("🌐 Cross-Exchange Arbitrage Strategy activated")
        
        # Start Circuit Breaker Monitor
        stop_event = asyncio.Event()
        circuit_breaker_task = loop.create_task(monitor_circuit_breaker(stop_event))
        
        # Set up Unix signal handlers with stop_event available
        if sys.platform != 'win32':
            for sig in (signal.SIGINT, signal.SIGTERM):
                def make_handler(evt=stop_event):
                    return lambda: asyncio.create_task(shutdown(engine, shadow_book, regime_supervisor, loop, evt))
                loop.add_signal_handler(sig, make_handler())
        
        # Send Telegram startup alert
        alerter = get_telegram_alerter()
        loop.run_until_complete(alerter.alert_bot_started())
        
        loop.run_until_complete(engine.start())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        loop.run_until_complete(shutdown(engine, shadow_book, regime_supervisor, loop, stop_event))
    except Exception as e:
        logger.critical(f"Fatal startup error: {e}", exc_info=True)
    finally:
        loop.close()
        logger.info("System process ended.")


if __name__ == "__main__":
    main()