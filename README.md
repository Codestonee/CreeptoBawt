A high-frequency trading system for cryptocurrency futures markets. Built with Python asyncio for low-latency execution and designed around proper risk management.
Architecture
Core Components:

Event-driven architecture with async execution
Real-time order book reconstruction (Shadow Book)
ACID-compliant order state management
Risk checks on every order before exchange submission

Trading Strategies:

Avellaneda-Stoikov market making
Funding rate arbitrage (experimental)

Market Analysis:

HMM-based regime detection
Real-time P&L tracking
SQLite trade database

Quick Start
Requirements:

Python 3.11 or higher
Binance Futures API credentials

Setup:
bash# Clone and enter directory
git clone https://github.com/yourusername/titan-hft.git
cd titan-hft

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Binance testnet credentials
Run:
bash# Start trading engine
python main.py

# Launch dashboard (separate terminal)
streamlit run dashboard/app.py
The bot connects to Binance testnet by default. Visit http://localhost:8501 to view the dashboard.
Configuration
Key settings in config/settings.py:

TRADING_SYMBOLS: Markets to trade (default: BTCUSDT, ETHUSDT)
MAX_POSITION_USD: Maximum position size per symbol
INITIAL_CAPITAL: Starting capital for P&L tracking
TESTNET: Set to False for live trading (use with caution)

Safety Features

Position checks on startup: Refuses to start if dangerous inherited positions exist
Circuit breaker: Auto-shutdown if drawdown exceeds 5%
Order validation: Risk gatekeeper blocks orders that violate limits
Reconciliation: Continuous sync between local state and exchange

Project Structure
├── config/           # Settings and configuration
├── core/             # Trading engine and event system
├── strategies/       # Trading strategy implementations
├── execution/        # Order management and exchange connectors
├── data/             # Order book and candle data providers
├── analysis/         # Market regime detection
├── dashboard/        # Streamlit monitoring interface
└── tests/            # Unit and integration tests
Docker Deployment
bashdocker-compose up -d
Includes the trading bot and dashboard. Configure environment variables in .env before building.
Important Notes
This is experimental software. Start with testnet. Never trade more than you can afford to lose.

Paper trading is enabled by default
Test thoroughly before using real funds
Monitor the dashboard during operation
Check logs in bot_execution.log

The circuit breaker will shut down the bot if realized P&L drops 5% below starting capital.
Development
Run tests:
bashpytest
The codebase uses async/await throughout. On Linux, uvloop is automatically used for better performance.
