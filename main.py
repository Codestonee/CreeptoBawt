import asyncio
import logging
import sys
import signal
import io  # <--- NYTT: Krävs för fixen nedan

# --------------------------------------------------------------------------
# FIX: Tvinga Windows att använda UTF-8 för standardutmatning.
# Detta förhindrar krascher när loggar innehåller emojis (t.ex. ⚠️, ✅).
# --------------------------------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Försök ladda uvloop för prestanda (fungerar ej på Windows)
try:
    import uvloop
except ImportError:
    uvloop = None

from config.settings import settings
from core.engine import TradingEngine
from connectors.binance_futures import BinanceFuturesConnector
# NYTT: Importera Grid Strategy istället för Dummy
from strategies.grid import DynamicGridStrategy

# Konfigurera loggning
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_execution.log", encoding='utf-8') # Även filen bör vara utf-8
    ]
)
logger = logging.getLogger("Main")

async def shutdown(engine, loop):
    logger.warning("Shutdown initiated. Cleaning up...")
    await engine.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Shutdown complete.")

def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if uvloop and sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    engine = TradingEngine(settings)
    
    binance_connector = BinanceFuturesConnector(
        event_queue=engine.event_queue,
        symbols=settings.TRADING_SYMBOLS,
        ws_url=settings.BINANCE_WS_URL
    )
    engine.add_connector(binance_connector)

    # NYTT: Aktivera Dynamic Grid Strategy
    if settings.TRADING_SYMBOLS:
        grid_strat = DynamicGridStrategy(
            engine.event_queue, 
            symbol=settings.TRADING_SYMBOLS[0],
            base_quantity=settings.GRID_BASE_QUANTITY
        )
        engine.add_strategy(grid_strat)
        logger.info(f"Dynamic Grid Strategy activated for {settings.TRADING_SYMBOLS[0]}.")
    else:
        logger.warning("No trading symbols configured. Strategy not started.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(engine, loop)))
    else:
        logger.info("Windows detected: Using KeyboardInterrupt for shutdown.")

    try:
        loop.run_until_complete(engine.start())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
        loop.run_until_complete(shutdown(engine, loop))
    except Exception as e:
        logger.critical(f"Fatal startup error: {e}", exc_info=True)
    finally:
        loop.close()
        logger.info("System process ended.")

if __name__ == "__main__":
    main()