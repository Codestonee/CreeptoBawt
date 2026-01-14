import asyncio
import sys
import os
import logging
import traceback
import ccxt.async_support as ccxt

# Ensure project root is in path
sys.path.append(os.getcwd())

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepDebug")

async def test_okx():
    logger.info("--- DEBUGGING OKX ---")
    api_key = settings.OKX_API_KEY
    secret = settings.OKX_SECRET_KEY
    password = settings.OKX_PASSPHRASE
    
    # Masked print
    logger.info(f"Loaded Settings -> Key: {api_key[:4]}..., Secret: {secret[:4]}..., Passphrase: {password[:2]}...")
    
    config = {
        'apiKey': api_key,
        'secret': secret,
        'password': password,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    }
    
    # Check for empty strings which might cause 'requires secret' error
    for k, v in config.items():
        if not v:
            logger.error(f"❌ Config field '{k}' is empty/None!")
            
    try:
        logger.info("Initializing ccxt.okx...")
        exchange = ccxt.okx(config)
        # Sandbox?
        if settings.TESTNET:
            exchange.set_sandbox_mode(True)
            
        logger.info("Fetching Balance...")
        bal = await exchange.fetch_balance()
        logger.info("✅ OKX Success!")
        await exchange.close()
    except Exception:
        traceback.print_exc()
        await exchange.close()

async def test_coinbase():
    logger.info("--- DEBUGGING COINBASE ---")
    api_key = settings.COINBASE_API_KEY
    secret = settings.COINBASE_SECRET_KEY
    
    logger.info(f"Loaded Settings -> Key: {api_key[:4]}..., Secret: {secret[:4]}...")
    
    config = {
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        # Coinbase often needs specific options for API interaction
    }
    
    try:
        logger.info("Initializing ccxt.coinbase...")
        # Note: 'coinbase' is usually generic. 'coinbasepro' or 'coinbaseadvanced' might be needed?
        # CCXT 'coinbase' often refers to v2 (Wallet API), while traders want 'coinbasepro' (Exchange).
        # But 'coinbasepro' is deprecated for 'Advanced Trade'. 
        # CCXT mapped 'coinbase' to v2, and has 'coinbaseadvanced'??
        # Let's try 'coinbase' first as per user config.
        exchange = ccxt.coinbase(config)
        
        if settings.TESTNET:
             # Coinbase testnet support varies
             exchange.set_sandbox_mode(True)
             
        logger.info("Fetching Balance...")
        bal = await exchange.fetch_balance()
        logger.info("✅ Coinbase Success!")
        await exchange.close()
    except Exception:
        traceback.print_exc()
        if 'exchange' in locals():
            await exchange.close()

async def main():
    await test_okx()
    print("\n" * 2)
    await test_coinbase()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
