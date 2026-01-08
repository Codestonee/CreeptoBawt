from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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
    
    # Trading Pairs
    TRADING_SYMBOLS: list[str] = ["btcusdt", "ethusdt"]

    # Paper Trading Config
    PAPER_TRADING: bool = False       # True = Låtsaspengar, False = Riktiga pengar
    INITIAL_CAPITAL: float = 1000.0  # Startkapital för simulering

    # Strategy Config
    GRID_BASE_QUANTITY: float = 0.002  # Default quantity for grid orders

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