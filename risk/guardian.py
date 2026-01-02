"""
Risk Guardian - Central risk management with kill switches.

Features:
- Pre-order validation
- Kill switch activation
- Drawdown monitoring
- Position limits
- Rate limiting
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

import structlog

from config.environments import RiskLimits

log = structlog.get_logger()


class OrderDecision(Enum):
    """Order validation result."""
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE = "reduce"  # Allow with reduced size


@dataclass
class OrderCheck:
    """Result of pre-order validation."""
    decision: OrderDecision
    reason: str = ""
    adjusted_quantity: Optional[Decimal] = None
    
    @staticmethod
    def allow() -> OrderCheck:
        return OrderCheck(decision=OrderDecision.ALLOW)
    
    @staticmethod
    def reject(reason: str) -> OrderCheck:
        return OrderCheck(decision=OrderDecision.REJECT, reason=reason)
    
    @staticmethod
    def reduce(reason: str, new_quantity: Decimal) -> OrderCheck:
        return OrderCheck(
            decision=OrderDecision.REDUCE,
            reason=reason,
            adjusted_quantity=new_quantity,
        )


class RiskGuardian:
    """
    Central risk management system.
    
    All orders must pass through check_pre_order() before submission.
    Can activate kill switch to halt all trading.
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        notification_callback: Optional[Callable] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self._notify = notification_callback
        
        # State
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self.kill_switch_time: Optional[float] = None
        
        # Tracking
        self._daily_pnl = Decimal("0")
        self._peak_equity = Decimal("0")
        self._current_equity = Decimal("0")
        
        self._inventory: Dict[str, Decimal] = {}  # symbol -> quantity
        self._open_orders: Dict[str, Any] = {}  # order_id -> order
        
        # Rate limiting
        self._order_timestamps: Deque[float] = deque(maxlen=1000)
        self._api_errors: Deque[float] = deque(maxlen=100)
        
        # Loss tracking for rapid loss detection
        self._recent_losses: Deque[tuple[float, Decimal]] = deque(maxlen=100)
        
        log.info("risk_guardian_initialized", limits=str(self.limits))
    
    # =========================================================================
    # Pre-Order Validation
    # =========================================================================
    
    async def check_pre_order(self, order: Any) -> OrderCheck:
        """
        Validate order before submission.
        
        Checks:
        1. Kill switch status
        2. Fat finger protection
        3. Inventory limits
        4. Open orders limit
        5. Daily loss limit
        6. Rate limiting
        
        Args:
            order: Order to validate
            
        Returns:
            OrderCheck with decision and reason
        """
        # 1. Kill switch check
        if self.kill_switch_active:
            return OrderCheck.reject(f"Kill switch active: {self.kill_switch_reason}")
        
        # 2. Fat finger check
        if hasattr(order, 'value_usd'):
            value = order.value_usd
        elif hasattr(order, 'price') and hasattr(order, 'quantity'):
            value = order.price * order.quantity
        else:
            value = Decimal("0")
        
        if value > self.limits.max_order_size_usd:
            return OrderCheck.reject(
                f"Order value ${value} exceeds max ${self.limits.max_order_size_usd}"
            )
        
        # 3. Inventory check
        symbol = getattr(order, 'symbol', 'UNKNOWN')
        side = getattr(order, 'side', 'buy')
        quantity = getattr(order, 'quantity', Decimal("0"))
        
        new_inventory = self._estimate_post_order_inventory(symbol, side, quantity)
        
        if abs(new_inventory) > self.limits.max_inventory_btc:
            return OrderCheck.reject(
                f"Would exceed inventory limit: {new_inventory} > {self.limits.max_inventory_btc}"
            )
        
        # 4. Open orders check
        if len(self._open_orders) >= self.limits.max_open_orders:
            return OrderCheck.reject(
                f"Too many open orders: {len(self._open_orders)}"
            )
        
        # 5. Daily loss check
        if self._daily_pnl < -self.limits.max_daily_loss_usd:
            await self.activate_kill_switch("Daily loss limit exceeded")
            return OrderCheck.reject("Daily loss limit exceeded")
        
        # 6. Rate limit check
        orders_last_minute = sum(
            1 for ts in self._order_timestamps
            if time.time() - ts < 60
        )
        if orders_last_minute >= self.limits.max_orders_per_minute:
            return OrderCheck.reject(
                f"Rate limit: {orders_last_minute} orders in last minute"
            )
        
        # 7. Drawdown check
        drawdown = self._calculate_drawdown()
        if drawdown > self.limits.max_drawdown_pct:
            await self.activate_kill_switch(
                f"Max drawdown exceeded: {drawdown:.2%}"
            )
            return OrderCheck.reject("Max drawdown exceeded")
        
        # All checks passed
        self._order_timestamps.append(time.time())
        return OrderCheck.allow()
    
    def _estimate_post_order_inventory(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
    ) -> Decimal:
        """Estimate inventory after order fills."""
        current = self._inventory.get(symbol, Decimal("0"))
        
        if side.lower() == "buy":
            return current + quantity
        else:
            return current - quantity
    
    def _calculate_drawdown(self) -> Decimal:
        """Calculate current drawdown from peak."""
        if self._peak_equity <= 0:
            return Decimal("0")
        
        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
        return max(Decimal("0"), drawdown)
    
    # =========================================================================
    # Kill Switch
    # =========================================================================
    
    async def activate_kill_switch(self, reason: str) -> None:
        """
        Activate emergency kill switch.
        
        This will:
        1. Halt all new orders
        2. Cancel all open orders
        3. Send critical alerts
        4. Log state for forensics
        """
        if self.kill_switch_active:
            return  # Already active
        
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_time = time.time()
        
        log.critical(
            "kill_switch_activated",
            reason=reason,
            daily_pnl=str(self._daily_pnl),
            inventory=self._inventory,
            open_orders=len(self._open_orders),
        )
        
        # Cancel all open orders
        await self._cancel_all_orders()
        
        # Send notifications
        if self._notify:
            try:
                await self._notify(
                    level="CRITICAL",
                    message=f"🚨 KILL SWITCH ACTIVATED: {reason}",
                )
            except Exception as e:
                log.error("notification_failed", error=str(e))
    
    def deactivate_kill_switch(self, reason: str = "manual") -> None:
        """
        Deactivate kill switch.
        
        Use with caution - only after root cause is addressed.
        """
        if not self.kill_switch_active:
            return
        
        self.kill_switch_active = False
        
        log.warning(
            "kill_switch_deactivated",
            reason=reason,
            was_active_for=time.time() - (self.kill_switch_time or time.time()),
        )
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        log.info("canceling_all_orders", count=len(self._open_orders))
        # Implementation would call exchange APIs to cancel orders
        self._open_orders.clear()
    
    # =========================================================================
    # State Updates
    # =========================================================================
    
    def update_pnl(self, realized_pnl: Decimal, unrealized_pnl: Decimal) -> None:
        """Update PnL tracking."""
        total = realized_pnl + unrealized_pnl
        
        # Update equity
        self._current_equity = total
        if total > self._peak_equity:
            self._peak_equity = total
        
        # Track daily PnL (simplified - would reset at midnight)
        self._daily_pnl = realized_pnl
    
    def update_inventory(self, symbol: str, quantity: Decimal) -> None:
        """Update inventory for a symbol."""
        self._inventory[symbol] = quantity
        
        log.debug("inventory_updated", symbol=symbol, quantity=str(quantity))
    
    def record_fill(self, order_id: str, pnl: Decimal) -> None:
        """Record a fill and its PnL impact."""
        # Remove from open orders
        self._open_orders.pop(order_id, None)
        
        # Track loss for rapid loss detection
        if pnl < 0:
            self._recent_losses.append((time.time(), pnl))
            
            # Check for rapid loss
            if self.limits.halt_on_rapid_loss:
                recent_loss = sum(
                    loss for ts, loss in self._recent_losses
                    if time.time() - ts < self.limits.rapid_loss_window_seconds
                )
                if abs(recent_loss) > self.limits.rapid_loss_threshold_usd:
                    asyncio.create_task(
                        self.activate_kill_switch(
                            f"Rapid loss: ${abs(recent_loss)} in {self.limits.rapid_loss_window_seconds}s"
                        )
                    )
    
    def record_api_error(self) -> None:
        """Record an API error for rate limiting."""
        self._api_errors.append(time.time())
        
        # Check error rate
        errors_last_minute = sum(
            1 for ts in self._api_errors
            if time.time() - ts < 60
        )
        
        if errors_last_minute >= self.limits.max_api_errors_per_minute:
            asyncio.create_task(
                self.activate_kill_switch(
                    f"Too many API errors: {errors_last_minute}/min"
                )
            )
    
    def add_open_order(self, order_id: str, order: Any) -> None:
        """Track a new open order."""
        self._open_orders[order_id] = order
    
    def remove_open_order(self, order_id: str) -> None:
        """Remove an order from tracking."""
        self._open_orders.pop(order_id, None)
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "daily_pnl": str(self._daily_pnl),
            "drawdown": str(self._calculate_drawdown()),
            "inventory": {k: str(v) for k, v in self._inventory.items()},
            "open_orders": len(self._open_orders),
            "orders_last_minute": sum(
                1 for ts in self._order_timestamps if time.time() - ts < 60
            ),
            "api_errors_last_minute": sum(
                1 for ts in self._api_errors if time.time() - ts < 60
            ),
        }
