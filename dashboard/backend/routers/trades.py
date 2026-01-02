"""Trades API router."""
from fastapi import APIRouter, Depends
from dashboard.backend.schemas.trade import TradesResponse
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()


def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/trades", response_model=TradesResponse)
async def get_trades(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get recent trades."""
    return bridge.get_trades()
