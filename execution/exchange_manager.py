import logging
import asyncio
from typing import Dict, Optional
from config.settings import settings
from execution.ccxt_executor import CCXTExecutor

logger = logging.getLogger("Execution.MultiManager")

class MultiExchangeManager:
    """
    Manages connections to multiple exchanges (Binance, OKX).
    Acts as a registry and router.
    """
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.executors: Dict[str, CCXTExecutor] = {}
        self.initialized = False

    async def connect(self):
        """Connect to all exchanges (Alias for initialize)."""
        await self.initialize()

    async def execute(self, signal) -> Optional[str]:
        """Route execution to the correct exchange."""
        target_exchange = getattr(signal, 'exchange', 'binance').lower()
        
        executor = self.executors.get(target_exchange)
        if not executor:
            logger.error(f"❌ No executor found for exchange: {target_exchange}")
            return None
            
        return await executor.execute(signal)

    async def initialize(self):
        """Initialize all configured exchanges."""
        logger.info("🌐 Initializing Multi-Exchange Manager...")
        
        # Mapping for special config fields (password, etc)
        # Default assumes {NAME}_API_KEY and {NAME}_SECRET_KEY
        
        valid_exchanges = getattr(settings, 'ACTIVE_EXCHANGES', [])
        
        # Explicit Binance (Base) - handled separately if needed, or included in loop
        # We keep Binance separate if we want to use the specialized BinanceExecutor?
        # No, MultiExchangeManager uses CCXT for everything.
        # But engine.py uses BinanceExecutor for 'binance'.
        # Manager is for the REST -> 'mexc', 'okx', etc.
        
        for exc_name in valid_exchanges:
            e_id = exc_name.lower()
            if e_id == 'binance': 
                continue # Handled by specialized BinanceExecutor in Engine
                
            api_key = getattr(settings, f"{e_id.upper()}_API_KEY", "")
            secret = getattr(settings, f"{e_id.upper()}_SECRET_KEY", "")
            password = getattr(settings, f"{e_id.upper()}_PASSPHRASE", None)
            
            if api_key and secret:
                try:
                    self.executors[e_id] = CCXTExecutor(
                        exchange_id=e_id,
                        api_key=api_key,
                        api_secret=secret,
                        password=password,
                        testnet=self.testnet
                    )
                except Exception as e:
                    logger.error(f"Failed to create executor for {e_id}: {e}")
            else:
                pass # Skip if no keys configured logic handled inside executor init check? 
                     # Actually we check keys before creating instance to avoid empty inits.
        
        # Connect all in parallel
        tasks = []
        for name, executor in self.executors.items():
            tasks.append(executor.initialize())
            
        if tasks:
            results = await asyncio.gather(*tasks)
            success_count = sum(results)
            logger.info(f"✅ Connected to {success_count}/{len(tasks)} exchanges.")
            self.initialized = True
        else:
            logger.warning("⚠️ No additional exchanges configured (waiting for keys).")

    async def shutdown(self):
        for name, executor in self.executors.items():
            await executor.close()

    def get_executor(self, exchange_id: str) -> Optional[CCXTExecutor]:
        return self.executors.get(exchange_id)

    async def get_all_tickers(self, symbol: str):
        """Get price map {exchange: {bid, ask}} for a symbol."""
        tasks = {}
        for name, exc in self.executors.items():
            tasks[name] = exc.fetch_ticker(symbol)
            
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))
