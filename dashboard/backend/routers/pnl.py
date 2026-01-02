"""PnL API router."""
from fastapi import APIRouter, Depends
from dashboard.backend.schemas.pnl import PnLResponse
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()


def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/pnl", response_model=PnLResponse)
async def get_pnl(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get P&L statistics."""
    return bridge.get_pnl()
