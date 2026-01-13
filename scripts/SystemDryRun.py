import asyncio
import logging
import sys
import os
import signal
from dotenv import load_dotenv

# Lägg till rotkatalogen i sökvägen så vi kan importera moduler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.engine import TradingEngine
from connectors.binance_futures import BinanceFuturesConnector
from connectors.okx_futures import OkxFuturesConnector
from strategies.avellaneda_stoikov import AvellanedaStoikovStrategy
from strategies.arbitrage import ArbitrageStrategy
from data.shadow_book import get_shadow_book
from execution.binance_executor import BinanceExecutionHandler
from risk_engine.risk_manager import RiskManager

# Konfigurera logging för testet
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DRY-RUN] - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SystemDryRun")

async def run_system_check():
    logger.info("🚀 STARTING MULTI-EXCHANGE DRY RUN (30s)...")
    logger.info("1. Checking Environment Variables...")
    
    if not os.getenv("BINANCE_TESTNET_API_KEY") or not os.getenv("OKX_API_KEY"):
        logger.critical("❌ API KEYS MISSING in .env")
        return
    
    # 1. Starta Motorn
    # Enable both exchanges
    settings.ACTIVE_EXCHANGES = ['binance', 'okx']
    # Disable Paper Trading to test connection (Executors will use Testnet keys)
    settings.PAPER_TRADING = False 
    
    engine = TradingEngine(settings)
    
    # 2. Risk Manager
    logger.info("2. Initializing Risk Manager...")
    
    # 3. Execution Handlers (Initialized automatically by Engine based on settings)
    logger.info(f"3. Execution Handlers initialized: {list(engine.executors.keys())}")

    # 4. Shadow Order Book
    logger.info("4. Initializing Shadow Order Book...")
    shadow_book = get_shadow_book(
        symbols=settings.TRADING_SYMBOLS,
        testnet=True
    )

    # 5. Connectors
    logger.info("5. Initializing Connectors (Binance & OKX)...")
    
    # Binance
    binance_c = BinanceFuturesConnector(
        event_queue=engine.event_queue,
        symbols=settings.TRADING_SYMBOLS,
        ws_url=settings.BINANCE_WS_URL
    )
    engine.add_connector(binance_c)
    
    # OKX
    okx_c = OkxFuturesConnector(
        event_queue=engine.event_queue,
        symbols=settings.TRADING_SYMBOLS
    )
    engine.add_connector(okx_c)

    # 6. Strategies
    logger.info("6. Initializing Strategies...")
    
    # Arbitrage
    arb_strategy = ArbitrageStrategy(engine.event_queue)
    engine.add_strategy(arb_strategy)
    logger.info("   -> ArbitrageStrategy added")
    
    # Avellaneda (Legacy)
    if settings.TRADING_SYMBOLS:
        as_strategy = AvellanedaStoikovStrategy(
            event_queue=engine.event_queue, 
            symbols=settings.TRADING_SYMBOLS,
            shadow_book=shadow_book
        )
        engine.add_strategy(as_strategy)
        logger.info("   -> AvellanedaStoikovStrategy added")
    
    # --- START SEQUENCE ---
    try:
        # Starta ShadowBook (WebSockets)
        asyncio.create_task(shadow_book.start())
        
        # Starta Engine (Startar connectors, executor connect, loops)
        # Vi kör detta som en task så vi kan avbryta den
        engine_task = asyncio.create_task(engine.start())
        
        logger.info("✅ All systems initialized. Running for 30 seconds...")
        
        # Låt systemet köra i 30 sekunder
        for i in range(30):
            await asyncio.sleep(1)
            if i % 10 == 0:
                logger.info(f"⏱️ System executing... {i}/30s")
                
                # Enkel kontroll: Har vi fått data i orderboken?
                if settings.TRADING_SYMBOLS:
                    sym = settings.TRADING_SYMBOLS[0]
                    mid = shadow_book.get_mid_price(sym)
                    if mid:
                        logger.info(f"   Using Data: {sym} Mid Price = {mid:.2f}")
                    else:
                        logger.warning(f"   ⚠️ Waiting for OrderBook data for {sym}...")

        logger.info("⏳ Time up. Shutting down...")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR DURING RUN: {e}", exc_info=True)
    
    finally:
        # Shutdown sequence
        logger.info("Stopping ShadowBook...")
        await shadow_book.stop()
        
        logger.info("Stopping Engine...")
        await engine.stop()
        
        # Cancel engine task
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass
            
        logger.info("✅ DRY RUN COMPLETE. Check logs for errors.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(run_system_check())
    except KeyboardInterrupt:
        pass