"""Trading bridge service - connects dashboard to trading engine."""
from datetime import datetime, timezone
from typing import Any, Dict, List


class TradingBridge:
    """Bridge between dashboard and trading engine."""
    
    def __init__(self, use_mock_data: bool = True):
        """Initialize trading bridge."""
        self.use_mock_data = use_mock_data
        # TODO: Connect to actual trading engine when available
        self.engine = None
    
    # Mock data methods - replace with real data from trading engine
    
    def get_balances(self) -> Dict[str, Any]:
        """Get account balances."""
        if self.use_mock_data:
            return {
                "balances": [
                    {
                        "currency": "BTC",
                        "available": "1.5",
                        "locked": "0.5",
                        "total": "2.0",
                        "usd_value": "90000.00"
                    },
                    {
                        "currency": "USDT",
                        "available": "50000",
                        "locked": "10000",
                        "total": "60000",
                        "usd_value": "60000.00"
                    }
                ],
                "total_usd_value": "150000.00"
            }
        # TODO: Get from trading engine
        return {"balances": [], "total_usd_value": "0.00"}
    
    def get_positions(self) -> Dict[str, Any]:
        """Get open positions."""
        if self.use_mock_data:
            return {
                "positions": [
                    {
                        "symbol": "BTC-USDT",
                        "side": "long",
                        "quantity": "1.5",
                        "entry_price": "44000.00",
                        "current_price": "45000.00",
                        "unrealized_pnl": "1500.00",
                        "unrealized_pnl_pct": "2.27"
                    }
                ]
            }
        # TODO: Get from trading engine
        return {"positions": []}
    
    def get_orders(self) -> Dict[str, Any]:
        """Get open orders."""
        if self.use_mock_data:
            return {
                "orders": [
                    {
                        "id": "order_123456",
                        "symbol": "BTC-USDT",
                        "side": "buy",
                        "type": "limit",
                        "price": "44500.00",
                        "quantity": "0.1",
                        "filled_quantity": "0.0",
                        "status": "open",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                ]
            }
        # TODO: Get from trading engine
        return {"orders": []}
    
    def get_trades(self) -> Dict[str, Any]:
        """Get recent trades."""
        if self.use_mock_data:
            return {
                "trades": [
                    {
                        "id": "trade_123456",
                        "symbol": "BTC-USDT",
                        "side": "buy",
                        "price": "45000.00",
                        "quantity": "0.1",
                        "fee": "4.50",
                        "realized_pnl": "125.00",
                        "executed_at": datetime.now(timezone.utc).isoformat()
                    }
                ]
            }
        # TODO: Get from trading engine
        return {"trades": []}
    
    def get_pnl(self) -> Dict[str, Any]:
        """Get P&L statistics."""
        if self.use_mock_data:
            return {
                "total_pnl": "12450.32",
                "realized_pnl": "10200.00",
                "unrealized_pnl": "2250.32",
                "today_pnl": "1230.50",
                "today_pnl_pct": "2.1",
                "fees_paid": "450.00",
                "sharpe_ratio": "1.85",
                "max_drawdown": "8.2"
            }
        # TODO: Get from trading engine
        return {
            "total_pnl": "0.00",
            "realized_pnl": "0.00",
            "unrealized_pnl": "0.00",
            "today_pnl": "0.00",
            "today_pnl_pct": "0.0",
            "fees_paid": "0.00",
            "sharpe_ratio": "0.00",
            "max_drawdown": "0.0"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        if self.use_mock_data:
            return {
                "status": "running",
                "mode": "paper",
                "uptime_seconds": 86400,
                "kill_switch_active": False,
                "connected_exchanges": ["binance"],
                "active_strategies": ["market_maker"]
            }
        # TODO: Get from trading engine
        return {
            "status": "stopped",
            "mode": "paper",
            "uptime_seconds": 0,
            "kill_switch_active": False,
            "connected_exchanges": [],
            "active_strategies": []
        }
    
    def activate_kill_switch(self, activate: bool) -> Dict[str, Any]:
        """Activate or deactivate kill switch."""
        if self.use_mock_data:
            # Mock response
            return {
                "success": True,
                "kill_switch_active": activate,
                "orders_cancelled": 12 if activate else 0
            }
        # TODO: Call trading engine kill switch
        return {
            "success": False,
            "kill_switch_active": False,
            "orders_cancelled": 0
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.use_mock_data:
            return True
        # TODO: Cancel via trading engine
        return False
