"""Positions API router."""
from fastapi import APIRouter, Depends
from dashboard.backend.schemas.position import PositionsResponse
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()


def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get open positions."""
    return bridge.get_positions()
