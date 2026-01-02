# CreeptoBawt Dashboard

Web-based dashboard for real-time monitoring and control of the CreeptoBawt trading system.

## Features

- **Real-time Monitoring**: View balances, positions, orders, and trades
- **P&L Tracking**: Monitor total and daily profit/loss with statistics
- **Order Management**: View and cancel open orders
- **Kill Switch**: Emergency stop button to halt all trading
- **WebSocket Support**: Real-time updates (coming soon)
- **Dark Mode**: Built-in dark mode support
- **Responsive Design**: Works on desktop, tablet, and mobile

## Technology Stack

### Backend
- **FastAPI**: High-performance async API framework
- **WebSockets**: Real-time bidirectional communication
- **Pydantic**: Data validation and settings management

### Frontend
- **React 18**: Modern UI framework with hooks
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Data fetching and caching
- **Recharts**: Composable charting library
- **Vite**: Fast build tool and dev server

## Project Structure

```
dashboard/
├── backend/               # FastAPI backend
│   ├── main.py           # API entry point
│   ├── config.py         # Configuration
│   ├── routers/          # API endpoints
│   ├── schemas/          # Pydantic models
│   ├── services/         # Business logic
│   └── websocket/        # WebSocket manager
├── frontend/             # React frontend
│   ├── src/
│   │   ├── api/         # API client
│   │   ├── components/  # React components
│   │   ├── hooks/       # Custom hooks
│   │   ├── pages/       # Page components
│   │   ├── types/       # TypeScript types
│   │   └── utils/       # Utilities
│   └── public/          # Static files
├── Dockerfile           # Multi-stage build
└── README.md           # This file
```

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm or yarn

### Backend Development

1. **Install dependencies**:
```bash
cd dashboard/backend
pip install -r requirements.txt
```

2. **Run the backend server**:
```bash
# From the repository root
python -m dashboard.backend.main

# Or with uvicorn directly
uvicorn dashboard.backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

### Frontend Development

1. **Install dependencies**:
```bash
cd dashboard/frontend
npm install
```

2. **Run the development server**:
```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

The dev server will proxy API requests to `http://localhost:8000`

### Running Both Services

Open two terminal windows:

**Terminal 1 (Backend)**:
```bash
python -m dashboard.backend.main
```

**Terminal 2 (Frontend)**:
```bash
cd dashboard/frontend
npm run dev
```

## Production Build

### Build Frontend

```bash
cd dashboard/frontend
npm run build
```

This creates optimized production files in `frontend/dist/`

### Run Production Server

The FastAPI backend can serve the built frontend:

```bash
# Build frontend first
cd dashboard/frontend
npm run build

# Run backend (it will serve frontend from /static)
cd ../..
python -m dashboard.backend.main
```

Access the dashboard at `http://localhost:8000`

## Docker Deployment

### Build Docker Image

```bash
docker build -f dashboard/Dockerfile -t creeptobawt-dashboard .
```

### Run Container

```bash
docker run -p 8000:8000 creeptobawt-dashboard
```

Access the dashboard at `http://localhost:8000`

## API Endpoints

### Balances
- `GET /api/balances` - Get account balances

### Positions
- `GET /api/positions` - Get open positions

### Orders
- `GET /api/orders` - Get open orders
- `DELETE /api/orders/{order_id}` - Cancel an order

### Trades
- `GET /api/trades` - Get recent trades

### P&L
- `GET /api/pnl` - Get P&L statistics

### System
- `GET /api/status` - Get system status
- `POST /api/kill-switch` - Activate/deactivate kill switch

### WebSocket
- `WS /ws` - WebSocket endpoint for real-time updates

## Configuration

### Backend Configuration

Edit `dashboard/backend/config.py`:

```python
class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    allow_origins: list[str] = ["*"]
    api_prefix: str = "/api"
    use_mock_data: bool = True  # Set to False to use real trading engine
```

### Frontend Configuration

Create `.env` in `dashboard/frontend/`:

```
VITE_API_URL=http://localhost:8000/api
```

## Connecting to Trading Engine

Currently, the dashboard uses mock data. To connect to the real trading engine:

1. Set `use_mock_data = False` in `dashboard/backend/config.py`
2. Update `dashboard/backend/services/trading_bridge.py` to connect to your trading engine instance
3. Implement the actual methods to fetch data from the engine

Example integration:

```python
class TradingBridge:
    def __init__(self, use_mock_data: bool = False):
        self.use_mock_data = use_mock_data
        if not use_mock_data:
            # Connect to trading engine
            from core.engine import TradingEngine
            self.engine = TradingEngine()
```

## Troubleshooting

### Backend won't start

- Check Python version: `python --version` (needs 3.11+)
- Install dependencies: `pip install -r dashboard/backend/requirements.txt`
- Check port 8000 is not in use: `lsof -i :8000`

### Frontend won't build

- Check Node version: `node --version` (needs 20+)
- Clear cache: `rm -rf node_modules package-lock.json && npm install`
- Check for TypeScript errors: `npm run build`

### API requests failing

- Ensure backend is running on port 8000
- Check CORS settings in `dashboard/backend/main.py`
- Verify API URL in frontend `.env` file

### WebSocket connection issues

- Ensure WebSocket endpoint is enabled in backend
- Check browser console for connection errors
- Verify firewall settings allow WebSocket connections

## Security Notes

⚠️ **Important**: This dashboard is for development and testing. For production:

1. Enable authentication (JWT tokens recommended)
2. Use HTTPS/WSS for secure connections
3. Restrict CORS origins to your domain only
4. Add rate limiting to API endpoints
5. Implement proper error handling and logging
6. Use environment variables for sensitive configuration

## Future Enhancements

- [ ] Real-time WebSocket updates for all data
- [ ] Historical charts with Recharts
- [ ] Performance metrics and analytics
- [ ] Custom alerts and notifications
- [ ] Strategy parameter configuration
- [ ] Trade execution from dashboard
- [ ] Multi-exchange support
- [ ] User authentication
- [ ] Role-based access control

## License

MIT License - See repository LICENSE file

## Support

For issues and questions, please open an issue on GitHub.
