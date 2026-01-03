"""Balances API router."""
from fastapi import APIRouter, Depends
from dashboard.backend.schemas.balance import BalancesResponse
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()

# Dependency to get trading bridge
def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/balances", response_model=BalancesResponse)
async def get_balances(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get account balances."""
    return bridge.get_balances()
