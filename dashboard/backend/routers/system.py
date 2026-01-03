"""System API router."""
from fastapi import APIRouter, Depends
from dashboard.backend.schemas.system import (
    SystemStatus,
    KillSwitchRequest,
    KillSwitchResponse
)
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()


def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/status", response_model=SystemStatus)
async def get_status(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get system status."""
    return bridge.get_status()


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def toggle_kill_switch(
    request: KillSwitchRequest,
    bridge: TradingBridge = Depends(get_trading_bridge)
):
    """Activate or deactivate kill switch."""
    return bridge.activate_kill_switch(request.activate)
