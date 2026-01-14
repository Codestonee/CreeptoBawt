import logging
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
        Multi-layer risk validation with fail-fast approach.
        """
        try:
            # Normalize inputs
            symbol = symbol.lower()
            side = side.upper()
            order_value = abs(quantity * price)
            
            # LAYER 0: Emergency halt check (fastest)
            if self._trading_halted:
                return RiskCheckResult(False, f"TRADING HALTED: {self._halt_reason}", "CRITICAL")
            
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
            
            # LAYER 4: Position limits (async - uses local OrderManager state which is synced)
            position_check = await self._check_position_limits(symbol, quantity, price, side, order_value)
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

    async def _check_position_limits(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        side: str, 
        order_value: float
    ) -> RiskCheckResult:
        """Check per-symbol position limits using Real/Synced State."""
        try:
            # Get REAL position from position tracker (OrderManager)
            # Expecting Position object: quantity, avg_entry_price
            current_position = await self.position_tracker.get_position(symbol)
            
            if current_position:
                pos_qty = current_position.quantity
                pos_val = abs(pos_qty * current_position.avg_entry_price) if pos_qty != 0 else 0
            else:
                pos_qty = 0.0
                pos_val = 0.0
            
            # Calculate new position after this order
            # Note: This is an estimation. Fills might be partial.
            # But we validate based on "Worst Case" (Full Fill).
            
            new_value = 0.0
            
            if side == 'BUY':
                # BUY:
                # If Short (qty < 0): Reduces risk (Closing)
                # If Long (qty >= 0): Adds risk (Opening)
                if pos_qty < 0:
                    # Closing Short
                    # If buy qty <= abs(short qty), value goes down. OK.
                    if quantity <= abs(pos_qty):
                        return RiskCheckResult(True, "Closing short position - OK", "INFO")
                    # Flip to Long
                    remaining_long = quantity - abs(pos_qty)
                    new_value = remaining_long * price
                else:
                    # Adding to Long
                    new_value = pos_val + order_value
            else:
                # SELL:
                # If Long (qty > 0): Reduces risk (Closing)
                # If Short (qty <= 0): Adds risk (Opening)
                if pos_qty > 0:
                    # Closing Long
                    if quantity <= pos_qty:
                        return RiskCheckResult(True, "Closing long position - OK", "INFO")
                    # Flip to Short
                    remaining_short = quantity - pos_qty
                    new_value = remaining_short * price
                else:
                    # Adding to Short
                    new_value = pos_val + order_value
            
            # Check per-symbol limit (if increasing risk)
            if new_value > self.max_position_per_symbol:
                 # Allow if we are actually reducing from an even HIGHER state (Emergency Unwind) breaks this logic?
                 # No, if we are reducing, we caught it in the "Closing" blocks above.
                 # If we are here, we are INCREASING risk or FLIPPING.
                return RiskCheckResult(False, 
                    f"{symbol} position would be ${new_value:.2f} > limit ${self.max_position_per_symbol}", "WARNING")
            
            # Check total position count (unique symbols)
            # Not easily available from OrderManager without iterating all. 
            # Skipping for now to avoid perf hit, or add get_active_symbol_count() later.
            
            return RiskCheckResult(True, "Position limits OK", "INFO")
            
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            return RiskCheckResult(False, f"Position check error: {e}", "CRITICAL")

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
        """Check total exposure across all positions."""
        try:
            total_exposure = 0.0
            if hasattr(self.position_tracker, 'positions'):
                for pos in self.position_tracker.positions.values():
                    total_exposure += abs(pos.quantity * pos.avg_entry_price)
            
            projected_exposure = total_exposure + order_value
            
            if projected_exposure > self.max_total_exposure:
                return RiskCheckResult(False, 
                   f"Total exposure ${projected_exposure:.2f} > limit ${self.max_total_exposure}", "WARNING")
            
            return RiskCheckResult(True, "Exposure check passed", "INFO")
            
        except Exception as e:
            logger.error(f"Total exposure check error - REJECTING for safety: {e}")
            return RiskCheckResult(False, f"Exposure check failed: {e}", "CRITICAL")

    async def _check_margin_availability(self, order_value: float) -> RiskCheckResult:
        """Check if there's sufficient margin for the order."""
        try:
            if not self.exchange_client:
                return RiskCheckResult(True, "No exchange client - margin check skipped", "INFO")
            
            # Fetch account info for margin data
            account_info = await self.exchange_client.futures_account()
            
            available_margin = float(account_info.get('availableBalance', 0))
            total_margin = float(account_info.get('totalMarginBalance', 0))
            
            # Calculate margin utilization
            if total_margin > 0:
                utilization = 1 - (available_margin / total_margin)
            else:
                utilization = 0
            
            # Conservative margin requirement (assume 10x leverage = 10% margin)
            required_margin = order_value * 0.10
            
            # Check if order would exceed safe margin threshold (80%)
            if utilization > 0.80:
                return RiskCheckResult(False, 
                    f"Margin utilization too high: {utilization*100:.1f}%", "CRITICAL")
            
            if required_margin > available_margin:
                return RiskCheckResult(False, 
                    f"Insufficient margin: need ${required_margin:.2f}, have ${available_margin:.2f}", "CRITICAL")
            
            return RiskCheckResult(True, f"Margin OK (utilization: {utilization*100:.1f}%)", "INFO")
            
        except Exception as e:
            logger.warning(f"Margin check failed (proceeding with caution): {e}")
            return RiskCheckResult(True, "Margin check unavailable", "WARNING")

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
