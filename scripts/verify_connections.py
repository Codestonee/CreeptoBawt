import asyncio
import sys
import os
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from config.settings import settings
from execution.ccxt_executor import CCXTExecutor

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("ConnectivityCheck")

async def verify_exchange(name: str):
    """Test connection for a specific exchange."""
    logger.info(f"🔌 Testing connection to {name.upper()}...")
    
    # 1. Get Credentials
    api_key = getattr(settings, f"{name.upper()}_API_KEY", "")
    secret = getattr(settings, f"{name.upper()}_SECRET_KEY", "")
    password = getattr(settings, f"{name.upper()}_PASSPHRASE", None)
    
    # DEBUG: Show what we loaded (Masked)
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    masked_secret = f"{secret[:4]}...{secret[-4:]}" if len(secret) > 8 else "***"
    logger.info(f"   [{name.upper()}] Loaded Key: {masked_key}, Secret: {masked_secret}")
    
    if not api_key or not secret:
        logger.warning(f"⚠️ {name.upper()}: Missing credentials (Key or Secret empty).")
        return False
        
    try:
        # 2. Init Executor (Try Testnet First)
        logger.info(f"   [{name.upper()}] Attempting connection (Testnet={settings.TESTNET})...")
        try:
            executor = CCXTExecutor(
                exchange_id=name.lower(),
                api_key=api_key,
                api_secret=secret,
                password=password,
                testnet=settings.TESTNET
            )
            await executor.initialize()
            
            # 3. Test Auth (Fetch Balance)
            logger.info(f"🔑 {name.upper()}: Verifying credentials...")
            balance = await executor.exchange.fetch_balance()
            
            usdt = balance.get('USDT', {}).get('free', 0.0)
            logger.info(f"✅ {name.upper()}: SUCCESS! (Testnet={settings.TESTNET}, USDT Free: {usdt})")
            await executor.close()
            return True
            
        except Exception as e:
            err_msg = str(e).lower()
            if "sandbox" in err_msg or "testnet" in err_msg or "not have a sandbox" in err_msg:
                if settings.TESTNET:
                    logger.info(f"   [{name.upper()}] Testnet not supported/failed. Retrying with MAINNET (ReadOnly)...")
                    # RETRY MAINNET
                    executor = CCXTExecutor(
                        exchange_id=name.lower(),
                        api_key=api_key,
                        api_secret=secret,
                        password=password,
                        testnet=False # Force Mainnet
                    )
                    await executor.initialize()
                    balance = await executor.exchange.fetch_balance()
                    usdt = balance.get('USDT', {}).get('free', 0.0)
                    logger.info(f"✅ {name.upper()}: SUCCESS! (MAINNET, USDT Free: {usdt})")
                    await executor.close()
                    return True
            raise e # Re-raise if not a sandbox error or retry failed

    except Exception as e:
        logger.error(f"❌ {name.upper()}: FAILED - {str(e)}")
        return False

async def main():
    logger.info("==========================================")
    logger.info(f"🔍 STARTING CONNECTIVITY CHECK (TESTNET={settings.TESTNET})")
    logger.info("==========================================")
    
    exchanges = getattr(settings, 'ACTIVE_EXCHANGES', [])
    logger.info(f"📋 configured exchanges: {exchanges}")
    
    # Check all
    results = []
    for exc in exchanges:
        if exc == 'binance': 
            # CCXTExecutor handles binance too if we init it that way
            # But usually we use BinanceExecutionHandler. 
            # For this test, let's use CCXTExecutor for uniformity.
            pass
        
        res = await verify_exchange(exc)
        results.append(res)
        
    logger.info("==========================================")
    if all(results):
        logger.info("✅ ALL SYSTEMS GO! Ready for deployment.")
    else:
        logger.warning("⚠️ SOME CHECKS FAILED. See logs above.")
    logger.info("==========================================")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
