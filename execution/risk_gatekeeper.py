import logging
from dataclasses import dataclass
from typing import Tuple, Dict
from config.settings import settings

logger = logging.getLogger("Execution.RiskGatekeeper")

@dataclass
class RiskCheckResult:
    is_allowed: bool
    reason: str

class RiskGatekeeper:
    """
    The 'Hard Stop' for all outgoing orders.
    Intercepts and validates every order before it hits the exchange.
    """
    def __init__(self):
        self.max_order_value = settings.RISK_MAX_ORDER_USD
        self.min_notional = settings.RISK_MIN_NOTIONAL_USD
        self.max_total_exposure = settings.RISK_MAX_POSITION_TOTAL_USD
        
        # Track simulated exposure (for this session)
        # Note: In production, we should fetch real balances, 
        # but for HFT speed, we track delta here and sync periodically.
        self._current_exposure: float = 0.0

    async def validate_order(self, symbol: str, quantity: float, price: float, side: str) -> RiskCheckResult:
        """
        Validate an order against hard risk limits.
        
        Args:
            symbol: Trading pair (e.g. 'BTCUSDT')
            quantity: Order quantity
            price: Order price
            side: 'BUY' or 'SELL'
            
        Returns:
            RiskCheckResult(is_allowed, reason)
        """
        try:
            order_value = quantity * price
            
            # 1. MIN NOTIONAL CHECK
            if order_value < self.min_notional:
                return RiskCheckResult(False, f"Value ${order_value:.2f} < Min ${self.min_notional}")

            # 2. MAX ORDER SIZE (Fat Finger)
            if order_value > self.max_order_value:
                return RiskCheckResult(False, f"Value ${order_value:.2f} > Max Order ${self.max_order_value}")

            # 3. MAX TOTAL EXPOSURE
            # If we are BUYING (Long exposure) or SELLING (Short exposure), 
            # we simply treat it as adding to "Gross Exposure" for safety.
            # A more complex version would track Net Exposure per symbol.
            projected_exposure = self._current_exposure + order_value
            if projected_exposure > self.max_total_exposure:
                 return RiskCheckResult(False, f"Exposure ${projected_exposure:.2f} > Limit ${self.max_total_exposure}")
            
            return RiskCheckResult(True, "OK")
            
        except Exception as e:
            logger.error(f"Risk Check Error: {e}")
            return RiskCheckResult(False, f"Validation Error: {e}")

    def update_exposure(self, value_delta: float):
        """Update tracked exposure (called after successful fill)."""
        self._current_exposure += value_delta
        # Keep non-negative
        self._current_exposure = max(0.0, self._current_exposure)
