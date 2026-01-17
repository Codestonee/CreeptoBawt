from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "SERQET"
    LOG_LEVEL: str = "INFO"
    
    # Binance Config
    # IMPORTANT: Names must match your .env file (case-insensitive)
    BINANCE_WS_URL: str = "wss://fstream.binance.com/ws"  # Futures WS (ignored in spot mode)
    BINANCE_SPOT_WS_URL: str = "wss://stream.binance.com:9443/ws"  # Spot WS
    BINANCE_API_KEY: str = Field(..., description="Binance API Key")
    BINANCE_API_SECRET: str = Field(..., description="Binance Secret Key")
    
    # Binance Testnet
    BINANCE_TESTNET_API_KEY: str = Field(default="")
    BINANCE_TESTNET_SECRET_KEY: str = Field(default="")

    # OKX Config
    OKX_API_KEY: str = Field(default="", description="OKX API Key")
    OKX_SECRET_KEY: str = Field(default="", description="OKX Secret Key")
    OKX_PASSPHRASE: str = Field(default="", description="OKX Passphrase")
    
    # MEXC Config
    MEXC_API_KEY: str = Field(default="", description="MEXC API Key")
    MEXC_SECRET_KEY: str = Field(default="", description="MEXC Secret Key", validation_alias="MEXC_API_SECRET")

    # Bitget Config
    BITGET_API_KEY: str = Field(default="", description="Bitget API Key")
    BITGET_SECRET_KEY: str = Field(default="", description="Bitget Secret Key", validation_alias="BITGET_API_SECRET")
    BITGET_PASSPHRASE: str = Field(default="", description="Bitget Passphrase")

    # Coinbase Config
    COINBASE_API_KEY: str = Field(default="", description="Coinbase API Key")
    COINBASE_SECRET_KEY: str = Field(default="", description="Coinbase Secret Key", validation_alias="COINBASE_API_SECRET")

    # Environment
    TESTNET: bool = False  # Changed to False for mainnet
    SPOT_MODE: bool = True  # True = Spot trading, False = Futures trading (EU users must use Spot)

    
    # Exchange Selection
    EXCHANGE: str = Field(default="binance", description="binance, okx, mexc, bitget, coinbase")
    ACTIVE_EXCHANGES: list = ["binance"] # Only Binance for verification
    
    # Arbitrage Config
    ARBITRAGE_MIN_SPREAD: float = 0.005  # 0.5% spread
    ARBITRAGE_PAPER_TRADING: bool = False
    
    # Funding Arbitrage Config
    FUNDING_ARB_CONFIG: dict = {
        "MIN_FUNDING_RATE_PCT": 0.01,  # 0.01% (Normal funding is 0.01%, so this targets elevated rates)
        "POSITION_SIZE_USD": 50.0,     # Small size for verified safety
        "EXIT_FUNDING_PCT": 0.002      # Exit when rate calms down
    }
    
    # Trading Pairs
    TRADING_SYMBOLS: list[str] = ["ltcusdc", "xrpusdc", "dogeusdc"]  # Reduced to 3 pairs

    # Paper Trading Config
    PAPER_TRADING: bool = False       # True = Simulated, False = Real Money
    INITIAL_CAPITAL: float = 50.0  # Initial capital for simulation

    # Strategy Config
    # Avellaneda-Stoikov Parameters
    AS_GAMMA: float = 0.5            # Risk aversion
    AS_KAPPA: float = 0.5            # Inventory aversion
    MIN_PROFIT_PER_TRADE_USD: float = 0.05  # Minimum profit target per trade
    ESTIMATED_FEE_BPS: float = 7.0   # Estimated taker fee in basis points

    # --------------------------------------------------------------------------
    # RISK MANAGEMENT
    # --------------------------------------------------------------------------
    MAX_POSITION_USD: float = 50.0  # Maximum notional size per position
    MIN_NOTIONAL_USD: float = 5.0   # Binance minimum notional
    
    # --------------------------------------------------------------------------
    # STRICT RISK GATEKEEPER (HARD LIMITS)
    # --------------------------------------------------------------------------
    RISK_MIN_NOTIONAL_USD: float = 5.0           # Binance minimum notional
    RISK_MAX_ORDER_USD: float = 100.0            # Fat finger protection (raised for BTC orders)
    
    # Position Limits
    RISK_MAX_POSITION_PER_SYMBOL_USD: float = 1000.0  # Increased to $1000
    RISK_MAX_POSITION_TOTAL_USD: float = 5000.0       # Increased to $5000
    RISK_MAX_OPEN_POSITIONS: int = 5                  # Max concurrent symbols
    
    # Daily Safety
    RISK_MAX_DAILY_LOSS_USD: float = 50.0            # Stop if down $50 in 24h
    RISK_MAX_DAILY_LOSS_USD: float = 50.0            # Stop if down $50 in 24h
    RISK_MAX_ORDERS_PER_MINUTE: int = 300            # Rate limit (5/sec avg)
    
    # Position PnL Limits (Death Spiral Prevention)
    POSITION_PNL_WARNING_PCT: float = -0.03   # Warn at -3% on position
    # Strategy Limits
    STRATEGY_COOLDOWN_SECONDS: float = 60.0
    
    # GLT Configuration

    GLT_USE_ITERATIVE_THETA: bool = True     # Use enhanced theta calculation (slower but more accurate)
    
    # Symbol Whitelist (Empty = All Allowed)
    APPROVED_SYMBOLS: list[str] = ['ltcusdc', 'xrpusdc', 'dogeusdc']  # Reduced to 3 pairs for $50 balance
    
    ADMIN_RESUME_CODE: str = "creep-resume-123"      # Simple code to unhalt
    # --------------------------------------------------------------------------
    
    # Fee Structure
    MAKER_FEE_BPS: float = 2.0       # 0.02% maker fee
    TAKER_FEE_BPS: float = 5.0       # 0.05% taker fee
    MIN_PROFIT_BPS: float = 10.0     # Minimum profit margin after fees
    
    # GLT (Guéant-Lehalle-Tapia) Parameters
    GLT_GAMMA: float = 0.3          # Risk aversion
    GLT_A: float = 1.0              # Intensity scale
    GLT_K: float = 0.3              # Intensity decay
    
    # Optimization / Throttling
    HMM_UPDATE_INTERVAL: float = 1.0   # Seconds between HMM predictions
    VPIN_COMMIT_INTERVAL: float = 60.0 # Seconds between VPIN DB logs
    DT_ROUTER_INTERVAL_MS: int = 300   # Router loop interval
    
    # Delta-Neutral Hedging
    ENABLE_AUTO_HEDGE: bool = True  # Auto-hedge when position exceeds threshold
    HEDGE_THRESHOLD_PCT: float = 0.8  # Trigger at 80% of MAX_POSITION_USD
    
    # Order Execution
    USE_POST_ONLY_ORDERS: bool = True  # GTX mode - Guarantees maker fee
    
    # Telegram Alerts
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    
    GRID_BASE_QUANTITY: float = 0.001

    # Pydantic Config
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
try:
    settings = Settings()
except Exception as e:
    print(f"Failed to load settings: {e}")
    raise