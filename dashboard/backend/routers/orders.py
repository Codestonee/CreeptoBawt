"""Orders API router."""
from fastapi import APIRouter, Depends, HTTPException
from dashboard.backend.schemas.order import OrdersResponse
from dashboard.backend.services.trading_bridge import TradingBridge

router = APIRouter()


def get_trading_bridge():
    """Get trading bridge instance."""
    return TradingBridge(use_mock_data=True)


@router.get("/orders", response_model=OrdersResponse)
async def get_orders(bridge: TradingBridge = Depends(get_trading_bridge)):
    """Get open orders."""
    return bridge.get_orders()


@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
    bridge: TradingBridge = Depends(get_trading_bridge)
):
    """Cancel an order."""
    success = bridge.cancel_order(order_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel order")
    return {"success": True, "order_id": order_id}
