from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Titan_HFT_Bot"
    LOG_LEVEL: str = "INFO"
    
    # Binance Config
    # VIKTIGT: Namnen här måste matcha din .env-fil (case-insensitive)
    # Din .env verkar använda 'binance_api_secret' istället för 'BINANCE_SECRET_KEY'
    BINANCE_WS_URL: str = "wss://fstream.binance.com/ws"
    BINANCE_API_KEY: str = Field(..., description="Binance API Key")
    BINANCE_API_SECRET: str = Field(..., description="Binance Secret Key") # Ändrat namn för att matcha din .env
    
    # Binance Testnet (Aliases for internal use if env vars differ)
    # Binance Testnet (Aliases for internal use if env vars differ)
    BINANCE_TESTNET_API_KEY: str = Field(default="")
    BINANCE_TESTNET_SECRET_KEY: str = Field(default="")

    # OKX Config
    OKX_API_KEY: str = Field(default="", description="OKX API Key")
    OKX_SECRET_KEY: str = Field(default="", description="OKX Secret Key")
    OKX_PASSPHRASE: str = Field(default="", description="OKX Passphrase")
    OKX_API_URL: str = Field(default="https://eea.okx.com", description="OKX API URL (eea.okx.com for EU)")
    
    # Environment
    TESTNET: bool = False # Global flag for Testnet vs Mainnet
    
    # Exchange Selection
    # Default exchange for legacy support
    EXCHANGE: str = Field(default="binance", description="binance or okx")
    ACTIVE_EXCHANGES: list[str] = ["binance", "okx"]
    
    # Arbitrage Config
    ARBITRAGE_MIN_SPREAD: float = 0.005  # 0.5% spread (was 0.2%, increased to cover fees + slippage)
    ARBITRAGE_PAPER_TRADING: bool = True
    
    # Trading Pairs
    # CMC Research + Gemini Analysis (Jan 2026):
    # ✅ BNB - Index behavior, organic volume, sometimes lower fees
    # ✅ BCH - High tick/price ratio, boring = good for MM, BTC correlation
    # ✅ LTC - "Silver to Bitcoin's gold", stable, high volume
    # ❌ ZEC/XMR - Privacy coins, regulatory/delisting risk - AVOID
    # ⚠️ SUI - Trends hard, needs well-calibrated HMM first
    # Current: Safe defaults. Add BNB/BCH after tick size validation.
    TRADING_SYMBOLS: list[str] = ["btcusdt", "ethusdt", "solusdt", "dogeusdt", "xrpusdt", "bnbusdt", "adausdt", "ltcusdt"]

    # Paper Trading Config
    PAPER_TRADING: bool = False       # True = Låtsaspengar, False = Riktiga pengar
    INITIAL_CAPITAL: float = 500.0  # Startkapital för simulering

    # Strategy Config
    # Avellaneda-Stoikov Parameters
    AS_GAMMA: float = 0.5            # Reduced from 1.0 (Calmer spreads)
    AS_KAPPA: float = 0.5            # Reduced from 1.0
    MIN_PROFIT_PER_TRADE_USD: float = 0.05  # Minimum profit target per trade (was 0.01)
    ESTIMATED_FEE_BPS: float = 7.0  # Estimated taker fee in basis points (Conservative 0.07%)

    # --------------------------------------------------------------------------
    # RISK MANAGEMENT
    # --------------------------------------------------------------------------
    MAX_POSITION_USD: float = 200.0  # Maximum notional size per position ($200)
    MIN_NOTIONAL_USD: float = 10.0  # Min order value (Binance Spot requires >$10)
    
    # Fee Structure (Binance Futures)
    MAKER_FEE_BPS: float = 2.0       # 0.02% maker fee
    TAKER_FEE_BPS: float = 5.0       # 0.05% taker fee
    MIN_PROFIT_BPS: float = 10.0     # Minimum profit margin after fees (was 5.0)
    
    # GLT (Guéant-Lehalle-Tapia) Parameters
    # These override GLT defaults for unified config
    GLT_GAMMA: float = 0.3          # Risk aversion for GLT (0.1 was too aggressive!)
    GLT_A: float = 1.0              # Intensity scale (fills per hour at mid)
    GLT_K: float = 0.3              # Intensity decay
    
    # Optimization / Throttling
    HMM_UPDATE_INTERVAL: float = 1.0  # Seconds between HMM predictions (prevents blocking loop)
    VPIN_COMMIT_INTERVAL: float = 60.0 # Seconds between VPIN DB logs
    DT_ROUTER_INTERVAL_MS: int = 300   # Router loop interval
    
    # Delta-Neutral Hedging
    ENABLE_AUTO_HEDGE: bool = True  # Auto-hedge when position exceeds threshold
    HEDGE_THRESHOLD_PCT: float = 0.8  # Trigger at 80% of MAX_POSITION_USD
    
    # Order Execution
    USE_POST_ONLY_ORDERS: bool = True  # GTX mode - Guarantees maker fee, rejects if would cross spread
    
    # Telegram Alerts (Optional - get from @BotFather)
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    GRID_BASE_QUANTITY: float = 0.001  # Reduced from 0.002 to prevent margin errors on BTC

    # Konfiguration för Pydantic
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # VIKTIGT: Detta gör att den ignorerar alla andra nycklar i din .env (OKX, Bybit etc)
    )

# Instansiera settings
try:
    settings = Settings()
except Exception as e:
    print(f"Failed to load settings: {e}")
    raise