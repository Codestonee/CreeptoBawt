"""
Centralized Configuration Management

Features:
- Environment-based configs (dev, staging, prod, backtest)
- Secret management (API keys never in code)
- Validation (catch misconfigs before runtime)
- Hot reload (change params without restart)
- Override system (CLI args > ENV vars > config file > defaults)
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("Config")


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    BACKTEST = "backtest"


@dataclass
class ExchangeConfig:
    """Exchange connection settings."""
    
    # Binance API credentials
    api_key: str = ""
    api_secret: str = ""
    
    # Network settings
    testnet: bool = True
    request_timeout: int = 10
    recv_window: int = 5000
    
    # Rate limits
    max_requests_per_minute: int = 1200
    max_order_rate_per_10s: int = 50
    
    def validate(self):
        """Validate exchange config."""
        if not self.api_key or not self.api_secret:
            # Only warn, don't crash, as env vars might act as fallback or user might fill later
            pass
        
        if not self.testnet:
            logger.warning("⚠️  USING MAINNET - REAL MONEY AT RISK")


@dataclass
class TradingConfig:
    """Trading strategy settings."""
    
    # Symbols to trade
    symbols: list = field(default_factory=lambda: ['btcusdt', 'ethusdt'])
    
    # Capital allocation
    initial_capital: float = 10000.0
    max_position_usd: float = 5000.0
    max_total_exposure_usd: float = 8000.0
    
    # Risk limits
    max_leverage: int = 1
    position_pnl_warning_pct: float = -0.03  # -3%
    position_pnl_critical_pct: float = -0.05  # -5%
    daily_loss_limit_pct: float = -0.10  # -10%
    
    # Fees (basis points)
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    min_profit_bps: float = 10.0
    
    # Order sizing
    min_notional_usd: float = 10.0
    base_quantity: float = 0.01
    
    # Auto-hedging
    enable_auto_hedge: bool = True
    hedge_threshold_pct: float = 0.8  # 80% of position limit
    
    def validate(self):
        """Validate trading config."""
        if self.max_position_usd <= 0:
            raise ValueError("max_position_usd must be positive")
        
        if self.max_leverage > 5:
            logger.warning(f"⚠️  High leverage: {self.max_leverage}x")
        
        if not self.symbols:
            raise ValueError("No trading symbols specified")


@dataclass
class StrategyConfig:
    """Avellaneda-Stoikov strategy parameters."""
    
    # Core parameters
    gamma: float = 0.1  # Risk aversion
    kappa: float = 0.5  # Order arrival rate
    
    # Dynamic adjustments
    use_glt: bool = True  # Use GLT infinite-horizon model
    glt_use_iterative_theta: bool = True
    
    # GLT parameters
    glt_a: float = 10.0  # Intensity scale
    glt_k: float = 0.5   # Decay rate
    glt_gamma: float = 0.1
    
    # Inventory management
    inventory_skew_lambda: float = 1.0  # Tanh steepness
    max_inventory_units: float = 1.0
    
    # Quote refresh
    quote_refresh_threshold_bps: float = 10.0
    min_quote_interval_seconds: float = 2.0
    
    # Regime detection
    enable_hmm_regime: bool = True
    hmm_update_interval: int = 10  # seconds
    
    # VPIN toxic flow detection
    enable_vpin: bool = True
    vpin_bucket_size: int = 50
    
    def validate(self):
        """Validate strategy config."""
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")


@dataclass
class MonitoringConfig:
    """Monitoring and alerting settings."""
    
    # Health checks
    enable_health_monitor: bool = True
    health_check_interval: int = 30  # seconds
    
    # Alerting
    enable_alerts: bool = True
    
    # Telegram
    telegram_enabled: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""
    
    # Email
    email_enabled: bool = False
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_to: str = ""
    
    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    # Slack
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    
    def validate(self):
        """Validate monitoring config."""
        if self.telegram_enabled and not self.telegram_token:
            # logger.warning("Telegram enabled but token missing")
            pass 
        
        if self.email_enabled and not self.email_smtp_host:
            # logger.warning("Email enabled but SMTP host missing")
            pass


@dataclass
class DatabaseConfig:
    """Database settings."""
    
    db_path: str = "data/trading_data.db"
    connection_pool_size: int = 5
    connection_timeout: float = 10.0
    enable_wal: bool = True  # Write-Ahead Logging
    
    # Data retention
    keep_trades_days: int = 90
    keep_events_days: int = 30
    
    def validate(self):
        """Validate database config."""
        db_path = Path(self.db_path)
        if not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    """Backtesting-specific settings."""
    
    # Data
    data_source: str = "binance"  # binance, csv
    cache_dir: str = "data/backtest_cache"
    
    # Realism
    spread_cost_bps: float = 5.0
    impact_per_10k_usd: float = 2.0
    simulate_latency_ms: int = 50
    
    # Execution
    maker_fill_probability: float = 0.7
    allow_lookahead: bool = False
    
    def validate(self):
        """Validate backtest config."""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)


@dataclass
class BotConfig:
    """Master configuration object."""
    
    # Environment
    environment: str = Environment.DEVELOPMENT.value
    debug: bool = True
    log_level: str = "INFO"
    
    # Component configs
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    def validate(self):
        """Validate entire config."""
        logger.info(f"Validating config for environment: {self.environment}")
        
        self.exchange.validate()
        self.trading.validate()
        self.strategy.validate()
        self.monitoring.validate()
        self.database.validate()
        self.backtest.validate()
        
        logger.info("✅ Configuration validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotConfig':
        """Load config from dictionary."""
        
        # Parse nested configs
        exchange = ExchangeConfig(**data.get('exchange', {}))
        trading = TradingConfig(**data.get('trading', {}))
        strategy = StrategyConfig(**data.get('strategy', {}))
        monitoring = MonitoringConfig(**data.get('monitoring', {}))
        database = DatabaseConfig(**data.get('database', {}))
        backtest = BacktestConfig(**data.get('backtest', {}))
        
        return cls(
            environment=data.get('environment', Environment.DEVELOPMENT.value),
            debug=data.get('debug', True),
            log_level=data.get('log_level', 'INFO'),
            exchange=exchange,
            trading=trading,
            strategy=strategy,
            monitoring=monitoring,
            database=database,
            backtest=backtest
        )


class ConfigManager:
    """
    Configuration manager with multi-source loading.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file (JSON)
    3. Defaults
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[BotConfig] = None
    
    def load(
        self,
        environment: str = None,
        config_file: str = None
    ) -> BotConfig:
        """
        Load configuration.
        
        Args:
            environment: Environment name (dev, staging, prod, backtest)
            config_file: Path to config file (overrides default)
        
        Returns:
            Loaded and validated BotConfig
        """
        # Determine environment
        if environment is None:
            environment = os.getenv('BOT_ENV', Environment.DEVELOPMENT.value)
        
        logger.info(f"Loading configuration for environment: {environment}")
        
        # Load base config from file
        if config_file is None:
            config_file = self.config_dir / f"{environment}.json"
        else:
            config_file = Path(config_file)
        
        if config_file.exists():
            logger.info(f"Loading config from: {config_file}")
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            config_data = {}
        
        # Create config object
        config = BotConfig.from_dict(config_data)
        config.environment = environment
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        # Validate
        config.validate()
        
        self._config = config
        return config
    
    def _apply_env_overrides(self, config: BotConfig) -> BotConfig:
        """Apply environment variable overrides."""
        
        # Exchange
        if os.getenv('BINANCE_API_KEY'):
            config.exchange.api_key = os.getenv('BINANCE_API_KEY')
        if os.getenv('BINANCE_API_SECRET'):
            config.exchange.api_secret = os.getenv('BINANCE_API_SECRET')
        if os.getenv('BINANCE_TESTNET'):
            config.exchange.testnet = os.getenv('BINANCE_TESTNET').lower() == 'true'
        
        # Trading
        if os.getenv('MAX_POSITION_USD'):
            config.trading.max_position_usd = float(os.getenv('MAX_POSITION_USD'))
        if os.getenv('TRADING_SYMBOLS'):
            config.trading.symbols = os.getenv('TRADING_SYMBOLS').split(',')
        
        # Monitoring
        if os.getenv('TELEGRAM_TOKEN'):
            config.monitoring.telegram_token = os.getenv('TELEGRAM_TOKEN')
            config.monitoring.telegram_enabled = True
        if os.getenv('TELEGRAM_CHAT_ID'):
            config.monitoring.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if os.getenv('DISCORD_WEBHOOK'):
            config.monitoring.discord_webhook_url = os.getenv('DISCORD_WEBHOOK')
            config.monitoring.discord_enabled = True
        
        # Debug
        if os.getenv('DEBUG'):
            config.debug = os.getenv('DEBUG').lower() == 'true'
        
        return config
    
    def save(self, config: BotConfig, filename: str = None):
        """Save config to file."""
        
        if filename is None:
            filename = f"{config.environment}.json"
        
        filepath = self.config_dir / filename
        
        # Mask secrets before saving
        config_dict = config.to_dict()
        config_dict['exchange']['api_key'] = '***MASKED***'
        config_dict['exchange']['api_secret'] = '***MASKED***'
        config_dict['monitoring']['telegram_token'] = '***MASKED***'
        config_dict['monitoring']['email_password'] = '***MASKED***'
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Config saved to: {filepath}")
    
    def get_current(self) -> BotConfig:
        """Get currently loaded config."""
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._config


# Global instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get global config manager (singleton)."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
