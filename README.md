# 🚀 CreeptoBawt - Autonomous Crypto Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An institutional-grade autonomous cryptocurrency trading system designed for production deployment. Built with resilience, security, and performance as core principles.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading cryptocurrencies carries significant risk. Never trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL PLANE                            │
│  Telegram Bot │ Web UI │ Emergency SSH Access               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                         │
│  State Machine │ Mode Controller │ Health Checker            │
│  (Live/Paper/Shadow/Replay/Emergency-Stop)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────┬──────────────────┬──────────────────────────┐
│   STRATEGY   │   RISK ENGINE    │   REGIME DETECTOR        │
│   PLUGINS    │                  │                          │
├──────────────┤  - Kill Switches │  - Volatility Regime     │
│ - Market Maker  - Drawdown Limits  - Trend Detection       │
│ - Arbitrage  │  - Inventory Caps│  - Liquidity Regime      │
│ - Stat Arb   │  - Rate Limits   │  - Correlation Breaks    │
└──────────────┴──────────────────┴──────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              EXECUTION & ORDER MANAGEMENT                    │
│  Lifecycle FSM │ Smart Routing │ Slippage Controls          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────┬────────────────────────────────────────┐
│  CEX CONNECTIVITY  │        STATE & PERSISTENCE             │
├────────────────────┤────────────────────────────────────────┤
│ - Binance WS/REST  │ - TimescaleDB (tick data)              │
│ - Coinbase WS/REST │ - Redis (event bus, cache)             │
│ - Unified CCXT     │ - State Reconciliation                 │
└────────────────────┴────────────────────────────────────────┘
```

## ✨ Features

### Core Capabilities
- **Event-Driven Architecture** - WebSocket-first, async I/O with uvloop
- **Multiple Trading Strategies** - Market making, arbitrage, statistical arbitrage
- **Production Risk Management** - Kill switches, circuit breakers, position limits
- **State Reconciliation** - Automatic sync with exchange state
- **Crash Recovery** - Write-ahead logging for operation recovery

### Trading Modes
- **Live** - Real money trading (use with extreme caution)
- **Paper** - Simulated execution with live market data
- **Shadow** - Dual execution logging for validation
- **Replay** - Historical data backtesting

### Security
- **OS Keyring Integration** - Never store secrets in .env files
- **Encrypted Credentials** - At-rest encryption for API keys
- **Principle of Least Privilege** - Minimal API permissions

### Monitoring
- **Prometheus Metrics** - Real-time performance metrics
- **Grafana Dashboards** - Visual monitoring
- **Telegram/Discord Alerts** - Multi-channel notifications
- **Health Check Endpoints** - Kubernetes-ready

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Redis
- PostgreSQL/TimescaleDB (optional, for production)

### Installation

```bash
# Clone the repository
git clone https://github.com/Codestonee/CreeptoBawt.git
cd CreeptoBawt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Store API credentials securely
python scripts/setup_secrets.py

# Run in paper trading mode
python -m core.engine --mode paper
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

## 📁 Project Structure

```
CreeptoBawt/
├── config/                 # Configuration management
├── core/                   # Event loop, event bus, mode controller
├── connectors/             # Exchange WebSocket/REST connectors
├── strategies/             # Trading strategy implementations
├── execution/              # Order management, paper exchange
├── risk/                   # Risk guardian, circuit breakers
├── reconciliation/         # State sync, crash recovery
├── pnl/                    # P&L tracking, attribution
├── database/               # Database models and clients
├── utils/                  # Security, logging, metrics
├── monitoring/             # Health checks, notifications
├── tests/                  # Unit and integration tests
├── scripts/                # Setup and utility scripts
├── grafana/                # Dashboard configurations
├── Dockerfile              # Container build
├── docker-compose.yml      # Full stack deployment
└── requirements.txt        # Python dependencies
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Trading Mode
TRADING_MODE=paper  # paper, live, shadow, replay

# Exchange Selection
EXCHANGES=binance,coinbase

# Risk Limits
MAX_DAILY_LOSS_USD=5000
MAX_DRAWDOWN_PCT=0.15
MAX_INVENTORY_BTC=5.0

# Database
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://localhost/trading

# Monitoring
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Strategy Configuration

Edit `config/strategies.yaml`:

```yaml
market_maker:
  enabled: true
  symbols:
    - BTC-USDT
    - ETH-USDT
  risk_aversion: 0.5
  inventory_target: 0
  max_inventory: 5.0

arbitrage:
  enabled: false
  min_profit_bps: 50
  max_leg_size_usd: 10000
```

## 🛡️ Risk Management

### Kill Switches

The system includes multiple layers of protection:

1. **Daily Loss Limit** - Halt trading if daily losses exceed threshold
2. **Drawdown Limit** - Stop if portfolio drawdown exceeds limit
3. **Inventory Limits** - Prevent excessive position accumulation
4. **API Error Rate** - Halt on repeated exchange errors
5. **Manual Override** - Telegram /stop command

### Circuit Breakers

- **Volatility Regime** - Widen spreads or halt in extreme volatility
- **Trend Detection** - Disable market making in strong trends
- **Liquidity Detection** - Reduce size in thin markets

## 📊 Monitoring

### Telegram Commands

- `/status` - Current system status
- `/pnl` - P&L report
- `/positions` - Open positions
- `/stop` - Emergency stop (kill switch)
- `/start` - Resume trading

### Grafana Dashboards

Access at `http://localhost:3000`:
- Trading Performance
- Risk Metrics
- System Health

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_risk_guardian.py

# Run integration tests
pytest tests/integration/ -v
```

## 📈 Performance Targets

- **Uptime**: >99.9%
- **Order Latency**: <100ms (p95)
- **Event Processing**: >1000 events/second
- **State Reconciliation**: <0.1% discrepancy rate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) - Exchange connectivity
- [Freqtrade](https://github.com/freqtrade/freqtrade) - Inspiration for bot architecture
- [Hummingbot](https://github.com/hummingbot/hummingbot) - Market making concepts

---

**Remember: Always start with paper trading and thoroughly validate before considering live trading.**