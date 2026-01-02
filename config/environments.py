"""
Environment Configuration - Dev/Staging/Production settings.

Provides environment-specific configurations for:
- Risk limits
- Trading parameters
- Database connections
- Feature flags
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from core.mode_controller import TradingMode


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class RiskLimits:
    """Risk management limits."""
    # Loss limits
    max_daily_loss_usd: Decimal = Decimal("5000")
    max_drawdown_pct: Decimal = Decimal("0.15")  # 15%
    
    # Position limits
    max_inventory_btc: Decimal = Decimal("5.0")
    max_inventory_usd: Decimal = Decimal("100000")
    
    # Order limits
    max_order_size_usd: Decimal = Decimal("10000")
    max_open_orders: int = 20
    max_orders_per_minute: int = 60
    
    # Fat finger protection
    max_price_deviation_pct: Decimal = Decimal("0.05")  # 5% from mid
    
    # API limits
    max_api_errors_per_minute: int = 5
    max_latency_ms: int = 500
    
    # Circuit breaker
    halt_on_rapid_loss: bool = True
    rapid_loss_threshold_usd: Decimal = Decimal("1000")
    rapid_loss_window_seconds: int = 60


@dataclass
class DatabaseConfig:
    """Database connection settings."""
    redis_url: str = "redis://localhost:6379"
    postgres_url: str = "postgresql://localhost:5432/trading"
    use_redis: bool = True
    connection_pool_size: int = 10


@dataclass
class MonitoringConfig:
    """Monitoring and alerting settings."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    health_check_port: int = 8080
    
    # Notifications
    telegram_enabled: bool = False
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None


@dataclass
class Config:
    """
    Main application configuration.
    
    Loads settings based on environment (dev/staging/prod).
    """
    environment: Environment
    mode: TradingMode
    
    # Exchange settings
    exchanges: List[str] = field(default_factory=lambda: ["binance"])
    default_order_size_usd: Decimal = Decimal("100")
    
    # Risk
    risk: RiskLimits = field(default_factory=RiskLimits)
    
    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Enabled strategies
    enabled_strategies: List[str] = field(default_factory=list)
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ["BTC-USDT", "ETH-USDT"])
    
    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            env = Environment(env_name)
        except ValueError:
            env = Environment.DEVELOPMENT
        
        if env == Environment.PRODUCTION:
            return cls._production_config()
        elif env == Environment.STAGING:
            return cls._staging_config()
        else:
            return cls._development_config()
    
    @classmethod
    def _development_config(cls) -> Config:
        """Development environment configuration."""
        return cls(
            environment=Environment.DEVELOPMENT,
            mode=TradingMode.PAPER,
            exchanges=["binance"],
            default_order_size_usd=Decimal("10"),
            risk=RiskLimits(
                max_daily_loss_usd=Decimal("1000"),
                max_inventory_btc=Decimal("1.0"),
                max_order_size_usd=Decimal("1000"),
                max_drawdown_pct=Decimal("0.30"),
            ),
            database=DatabaseConfig(
                redis_url="redis://localhost:6379",
                postgres_url="postgresql://localhost:5432/trading_dev",
                use_redis=False,  # Use in-memory for dev
            ),
            monitoring=MonitoringConfig(
                enable_prometheus=False,
                telegram_enabled=False,
            ),
            symbols=["BTC-USDT"],
        )
    
    @classmethod
    def _staging_config(cls) -> Config:
        """Staging environment configuration."""
        return cls(
            environment=Environment.STAGING,
            mode=TradingMode.PAPER,
            exchanges=["binance", "coinbase"],
            default_order_size_usd=Decimal("50"),
            risk=RiskLimits(
                max_daily_loss_usd=Decimal("2500"),
                max_inventory_btc=Decimal("2.0"),
                max_order_size_usd=Decimal("5000"),
                max_drawdown_pct=Decimal("0.20"),
            ),
            database=DatabaseConfig(
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                postgres_url=os.getenv("DATABASE_URL", "postgresql://localhost:5432/trading_staging"),
                use_redis=True,
            ),
            monitoring=MonitoringConfig(
                enable_prometheus=True,
                telegram_enabled=True,
                telegram_token=os.getenv("TELEGRAM_TOKEN"),
                telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            ),
            symbols=["BTC-USDT", "ETH-USDT"],
        )
    
    @classmethod
    def _production_config(cls) -> Config:
        """Production environment configuration."""
        return cls(
            environment=Environment.PRODUCTION,
            mode=TradingMode.LIVE,
            exchanges=["binance", "coinbase", "bybit"],
            default_order_size_usd=Decimal("100"),
            risk=RiskLimits(
                max_daily_loss_usd=Decimal("5000"),
                max_inventory_btc=Decimal("5.0"),
                max_order_size_usd=Decimal("10000"),
                max_drawdown_pct=Decimal("0.15"),
            ),
            database=DatabaseConfig(
                redis_url=os.getenv("REDIS_URL", "redis://redis:6379"),
                postgres_url=os.getenv("DATABASE_URL", "postgresql://postgres:5432/trading"),
                use_redis=True,
                connection_pool_size=20,
            ),
            monitoring=MonitoringConfig(
                enable_prometheus=True,
                telegram_enabled=True,
                telegram_token=os.getenv("TELEGRAM_TOKEN"),
                telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
                discord_enabled=True,
                discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
            ),
            enabled_strategies=["market_maker"],
            symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "environment": self.environment.value,
            "mode": self.mode.value,
            "exchanges": self.exchanges,
            "default_order_size_usd": str(self.default_order_size_usd),
            "symbols": self.symbols,
            "max_daily_loss_usd": str(self.risk.max_daily_loss_usd),
            "max_inventory_btc": str(self.risk.max_inventory_btc),
        }
