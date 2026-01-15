# Complete System Architecture & Deployment Guide

## System Components

- **Configuration**:
  - `config/config_manager.py`: Centralized config loading.
  - `config/*.json`: Environment specific settings.
- **Monitoring**:
  - `monitoring/alert_manager.py`: Alerts (Telegram/Discord).
  - `monitoring/health_monitor.py`: Health checks.
  - `dashboard/live_monitor.py`: Streamlit Dashboard.
- **Data**:
  - `data/persistence_manager.py`: Async SQLite with connection pooling.
  - `backtesting/data_loader.py`: Historical data management.
- **Execution**:
  - `main_integrated.py`: Unified entry point.
  - `docker-compose.yml`: Multi-container orchestration.

## Windows Setup Guide with Docker

### Prerequisites

1. **Docker Desktop for Windows** (WSL 2 backend recommended).
2. **Git for Windows**.

### Quick Start

1. **Clone & Setup**:

   ```powershell
   git clone <repo-url>
   cd CreeptBaws
   ```

2. **Run Start Script**:
   ```powershell
   .\start.ps1
   ```
   This script will:
   - Check Docker status.
   - Create `.env` from example if missing.
   - build Docker images.
   - Start Bot and Dashboard.

### Manual Commands

**Start Services:**

```powershell
docker-compose up -d
```

**View Logs:**

```powershell
docker-compose logs -f trading-bot
```

**Stop Services:**

```powershell
docker-compose down
```

**Rebuild:**

```powershell
docker-compose build --no-cache
```

## Dashboard

Access the live dashboard at: **http://localhost:8501**

## Configuration

- Edit `.env` for secrets (API Keys).
- Edit `config/production.json` for strategy parameters.
- Restart bot to apply changes: `docker-compose restart trading-bot`

## Troubleshooting

- **Database Locked**: Run `docker-compose down` and verify no other process uses `data/trading_data.db`.
- **Sync Failed**: Check logs (`docker-compose logs trading-bot`). Ensure API keys are correct and Testnet setting matches your keys.
