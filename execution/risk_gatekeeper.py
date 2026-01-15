import logging
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
from datetime import datetime, timedelta
from config.settings import settings

logger = logging.getLogger("Execution.RiskGatekeeper")

@dataclass
class RiskCheckResult:
    is_allowed: bool
    reason: str
    severity: str = "INFO"  # INFO, WARNING, CRITICAL

class RiskGatekeeper:
    """
    The 'Hard Stop' for all outgoing orders.
    Intercepts and validates every order before it hits the exchange.
    Now Production-Grade: Aware of Real Positions, Margin, and Direction.
    """
    def __init__(self, position_tracker, exchange_client):
        # Basic limits
        self.max_order_value = settings.RISK_MAX_ORDER_USD
        self.min_notional = settings.RISK_MIN_NOTIONAL_USD
        self.max_total_exposure = settings.RISK_MAX_POSITION_TOTAL_USD
        self.max_position_per_symbol = settings.RISK_MAX_POSITION_PER_SYMBOL_USD
        
        # Dependencies
        self.position_tracker = position_tracker  # OrderManager
        self.exchange_client = exchange_client    # AsyncClient
        
        # Advanced limits
        self.max_daily_loss = settings.RISK_MAX_DAILY_LOSS_USD
        self.max_positions = settings.RISK_MAX_OPEN_POSITIONS
        self.max_order_rate = settings.RISK_MAX_ORDERS_PER_MINUTE
        
        # State tracking
        self._daily_pnl: float = 0.0
        self._daily_loss_start: datetime = datetime.now()
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        self._recent_orders: list = []  # For rate limiting
        self._consecutive_losses: int = 0

    async def validate_order(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: str
    ) -> RiskCheckResult:
        """
        Multi-layer risk validation with atomic position snapshot.
        """
        try:
            # LAYER 0: Emergency halt check
            if self._trading_halted:
                return RiskCheckResult(False, f"TRADING HALTED: {self._halt_reason}", "CRITICAL")
            
            symbol = symbol.lower()
            side = side.upper()
            order_value = abs(quantity * price)
            
            # LAYER 1: Order sanity checks (no async calls)
            sanity_check = self._check_order_sanity(symbol, quantity, price, order_value)
            if not sanity_check.is_allowed:
                return sanity_check
            
            # LAYER 2: Rate limiting
            rate_check = self._check_order_rate()
            if not rate_check.is_allowed:
                return rate_check
            
            # LAYER 3: Margin check (async API call if client available)
            if self.exchange_client:
                margin_check = await self._check_margin_availability(order_value)
                if not margin_check.is_allowed:
                    return margin_check
            
            # =====================================================================
            # CRITICAL FIX: Atomic position snapshot with version check
            # =====================================================================
            current_position = await self.position_tracker.get_position(symbol)
            
            if current_position:
                pos_qty = current_position.quantity
                pos_val = abs(pos_qty * current_position.avg_entry_price)
                expected_version = current_position.version
            else:
                pos_qty = 0.0
                pos_val = 0.0
                expected_version = 0
            
            # LAYER 4: Position limits (atomic)
            position_check = await self._check_position_limits_atomic(
                symbol, quantity, price, side, order_value, 
                pos_qty, pos_val, expected_version
            )
            if not position_check.is_allowed:
                return position_check
            
            # LAYER 5: Daily loss limit
            loss_check = self._check_daily_loss()
            if not loss_check.is_allowed:
                return loss_check
            
            # LAYER 6: Exposure limits
            exposure_check = await self._check_exposure_limits(order_value)
            if not exposure_check.is_allowed:
                return exposure_check
            
            # FINAL VERIFICATION: Re-check position version hasn't changed
            final_position = await self.position_tracker.get_position(symbol)
            final_version = final_position.version if final_position else 0
            
            if final_version != expected_version:
                return RiskCheckResult(
                    False, 
                    f"Race condition detected! Position version changed: {expected_version} -> {final_version}",
                    "CRITICAL"
                )
            
            # All checks passed
            self._record_order()
            return RiskCheckResult(True, "All risk checks passed", "INFO")
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}", exc_info=True)
            return RiskCheckResult(False, f"Validation error: {e}", "CRITICAL")

    def _check_order_sanity(self, symbol: str, quantity: float, price: float, order_value: float) -> RiskCheckResult:
        """Fast, synchronous sanity checks."""
        # Check for obviously bad values
        if quantity <= 0 or price <= 0:
            return RiskCheckResult(False, f"Invalid quantity or price", "WARNING")
        
        # MIN NOTIONAL
        if order_value < self.min_notional:
            return RiskCheckResult(False, 
                f"Value ${order_value:.2f} < Min ${self.min_notional}", "INFO")
        
        # MAX ORDER SIZE (Fat finger protection)
        if order_value > self.max_order_value:
            return RiskCheckResult(False, 
                f"Value ${order_value:.2f} > Max ${self.max_order_value} (FAT FINGER?)", "CRITICAL")
        
        # Check symbol is approved
        if not self._is_approved_symbol(symbol):
            return RiskCheckResult(False, f"{symbol} not in approved list", "WARNING")
        
        return RiskCheckResult(True, "Sanity checks passed", "INFO")

    def _check_order_rate(self) -> RiskCheckResult:
        """Prevent order spam."""
        now = datetime.now()
        # Clean old orders (older than 1 minute)
        self._recent_orders = [t for t in self._recent_orders if now - t < timedelta(minutes=1)]
        
        if len(self._recent_orders) >= self.max_order_rate:
            return RiskCheckResult(False, 
                f"Rate limit: {len(self._recent_orders)} orders in 1 min (max {self.max_order_rate})", "WARNING")
        
        return RiskCheckResult(True, "Rate limit OK", "INFO")

    async def _check_position_limits_atomic(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        order_value: float,
        current_qty: float,
        current_val: float,
        version: float
    ) -> RiskCheckResult:
        """
        Check position limits using atomic snapshot.
        Version parameter ensures position hasn't changed during validation.
        """
        # Calculate projected position EXACTLY once with snapshot
        new_value = 0.0
        
        if side == 'BUY':
            if current_qty < 0:
                # Closing short
                if quantity <= abs(current_qty):
                    return RiskCheckResult(True, "Closing short - OK", "INFO")
                remaining_long = quantity - abs(current_qty)
                new_value = remaining_long * price
            else:
                # Adding to long
                new_value = current_val + order_value
        else:
            if current_qty > 0:
                # Closing long
                if quantity <= current_qty:
                    return RiskCheckResult(True, "Closing long - OK", "INFO")
                remaining_short = quantity - current_qty
                new_value = remaining_short * price
            else:
                # Adding to short
                new_value = current_val + order_value
        
        # Check limit
        if new_value > self.max_position_per_symbol:
            return RiskCheckResult(
                False, 
                f"{symbol} position would be ${new_value:.2f} > limit ${self.max_position_per_symbol}",
                "WARNING"
            )
        
        return RiskCheckResult(True, "Position limits OK", "INFO")

    def _check_daily_loss(self) -> RiskCheckResult:
        """Halt trading if daily loss limit exceeded."""
        # Reset counter if new day
        if datetime.now().date() != self._daily_loss_start.date():
            self._daily_pnl = 0.0
            self._daily_loss_start = datetime.now()
            self._consecutive_losses = 0
        
        if self._daily_pnl < -self.max_daily_loss:
            self._halt_trading(f"Daily loss limit hit: ${abs(self._daily_pnl):.2f}")
            return RiskCheckResult(False, 
                f"DAILY LOSS LIMIT: ${abs(self._daily_pnl):.2f} > ${self.max_daily_loss}", "CRITICAL")
        
        return RiskCheckResult(True, "Daily loss check passed", "INFO")

    async def _check_exposure_limits(self, order_value: float) -> RiskCheckResult:
        """
        Check total exposure - FAIL-SAFE on error (was allowing on error).
        """
        try:
            total_exposure = await self.position_tracker.get_total_exposure()
            projected_exposure = total_exposure + order_value
            
            if projected_exposure > self.max_total_exposure:
                return RiskCheckResult(
                    False, 
                    f"Exposure ${projected_exposure:.2f} > limit ${self.max_total_exposure}",
                    "WARNING"
                )
            
            return RiskCheckResult(True, "Exposure OK", "INFO")
            
        except Exception as e:
            logger.error(f"Exposure check failed: {e}")
            # CRITICAL FIX: Reject on error (was allowing)
            return RiskCheckResult(
                False, 
                f"Exposure check failed: {e} - REJECTING FOR SAFETY",
                "CRITICAL"
            )

    def _is_approved_symbol(self, symbol: str) -> bool:
        """Check if symbol is in approved trading list."""
        approved = settings.APPROVED_SYMBOLS or []
        return len(approved) == 0 or symbol.lower() in [s.lower() for s in approved]
    
    def _record_order(self):
        """Record order for rate limiting."""
        self._recent_orders.append(datetime.now())
    
    def _halt_trading(self, reason: str):
        """Emergency halt all trading."""
        self._trading_halted = True
        self._halt_reason = reason
        logger.critical(f"🚨 TRADING HALTED: {reason}")
    
    def update_pnl(self, pnl_delta: float):
        """Update daily PnL tracking (Called by Executor on Fill)."""
        self._daily_pnl += pnl_delta
        if pnl_delta < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def update_exposure(self, value_delta: float):
        # Legacy/Unused if we rely on OrderManager
        pass
