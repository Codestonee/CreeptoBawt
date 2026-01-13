"""
Hedge Manager - Critical safety component for multi-leg arbitrage trades.

Ensures that if one leg of an arbitrage fills but the other fails,
we immediately close the position to prevent directional exposure.

Design decisions:
- Timeout: 1000ms (balance between speed and network latency)
- Emergency close: MARKET order (guaranteed fill, safety over price)
- Circuit breaker: Pause arbitrage after 3 consecutive failures
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime

# Telegram alerts for critical events
try:
    from utils.telegram_alerts import get_telegram_alerter
    _telegram = get_telegram_alerter()
except ImportError:
    _telegram = None

logger = logging.getLogger("Execution.HedgeManager")


class LegStatus(str, Enum):
    """Status of an arbitrage leg."""
    PENDING = "PENDING"       # Sent but not confirmed
    SUBMITTED = "SUBMITTED"   # Confirmed on exchange, waiting for fill
    FILLED = "FILLED"         # Successfully filled
    FAILED = "FAILED"         # Rejected or timed out
    EMERGENCY_CLOSED = "EMERGENCY_CLOSED"  # Closed due to legged position


class ArbStatus(str, Enum):
    """Status of an arbitrage attempt."""
    ACTIVE = "ACTIVE"
    SUCCESS = "SUCCESS"       # Both legs filled
    FAILED = "FAILED"         # One or both legs failed
    LEGGED = "LEGGED"        # One leg filled, other failed - DANGEROUS


@dataclass
class ArbLeg:
    """Single leg of an arbitrage attempt."""
    leg_id: str
    order_id: str
    exchange: str
    symbol: str
    side: str
    quantity: float
    expected_price: float
    status: LegStatus = LegStatus.PENDING
    filled_qty: float = 0.0
    filled_price: float = 0.0
    submitted_at: float = field(default_factory=time.time)
    filled_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class ArbitrageAttempt:
    """Tracks a complete arbitrage attempt (two legs)."""
    arb_id: str
    symbol: str
    leg1: ArbLeg
    leg2: ArbLeg
    status: ArbStatus = ArbStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    expected_profit_bps: float = 0.0
    realized_profit: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if both legs are in a terminal state."""
        terminal = {LegStatus.FILLED, LegStatus.FAILED, LegStatus.EMERGENCY_CLOSED}
        return self.leg1.status in terminal and self.leg2.status in terminal
    
    @property
    def is_legged(self) -> bool:
        """Check if we have a dangerous legged position."""
        one_filled = (self.leg1.status == LegStatus.FILLED) != (self.leg2.status == LegStatus.FILLED)
        one_failed = self.leg1.status == LegStatus.FAILED or self.leg2.status == LegStatus.FAILED
        return one_filled and one_failed


class HedgeManager:
    """
    Manages arbitrage attempts and ensures proper hedging.
    
    Key features:
    - Tracks both legs of each arbitrage attempt
    - Automatically closes legged positions with MARKET orders
    - Circuit breaker after consecutive failures
    - Profit/loss tracking
    """
    
    # Configuration
    LEG_TIMEOUT_MS = 1000      # Maximum time to wait for second leg (1 second)
    MAX_CONSECUTIVE_FAILURES = 3
    WATCHDOG_INTERVAL_MS = 100  # Check for timeouts every 100ms
    
    def __init__(self, event_queue):
        self.queue = event_queue
        self._attempts: Dict[str, ArbitrageAttempt] = {}
        self._order_to_arb: Dict[str, str] = {}  # order_id -> arb_id
        self._consecutive_failures = 0
        self._paused = False
        self._watchdog_task: Optional[asyncio.Task] = None
        self._stats = {
            'total_attempts': 0,
            'successful': 0,
            'failed': 0,
            'legged': 0,
            'emergency_closes': 0,
            'total_profit': 0.0
        }
        logger.info("HedgeManager initialized")
    
    async def start(self):
        """Start the watchdog task."""
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info("HedgeManager watchdog started")
    
    async def stop(self):
        """Stop the watchdog task."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
            logger.info("HedgeManager stopped")
    
    def is_paused(self) -> bool:
        """Check if arbitrage is paused due to failures."""
        return self._paused
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._paused = False
        self._consecutive_failures = 0
        logger.info("🔄 HedgeManager circuit breaker reset")
    
    def start_arbitrage(
        self,
        symbol: str,
        quantity: float,
        leg1_exchange: str,
        leg1_side: str,
        leg1_price: float,
        leg2_exchange: str,
        leg2_side: str,
        leg2_price: float,
        expected_profit_bps: float
    ) -> Optional[ArbitrageAttempt]:
        """
        Start tracking a new arbitrage attempt.
        
        Returns:
            ArbitrageAttempt with generated IDs, or None if paused
        """
        if self._paused:
            logger.warning("❌ Arbitrage paused due to circuit breaker")
            return None
        
        arb_id = f"arb_{uuid.uuid4().hex[:12]}"
        
        leg1 = ArbLeg(
            leg_id=f"{arb_id}_L1",
            order_id=f"c_{uuid.uuid4().hex[:16]}",
            exchange=leg1_exchange,
            symbol=symbol,
            side=leg1_side,
            quantity=quantity,
            expected_price=leg1_price
        )
        
        leg2 = ArbLeg(
            leg_id=f"{arb_id}_L2",
            order_id=f"c_{uuid.uuid4().hex[:16]}",
            exchange=leg2_exchange,
            symbol=symbol,
            side=leg2_side,
            quantity=quantity,
            expected_price=leg2_price
        )
        
        attempt = ArbitrageAttempt(
            arb_id=arb_id,
            symbol=symbol,
            leg1=leg1,
            leg2=leg2,
            expected_profit_bps=expected_profit_bps
        )
        
        self._attempts[arb_id] = attempt
        self._order_to_arb[leg1.order_id] = arb_id
        self._order_to_arb[leg2.order_id] = arb_id
        self._stats['total_attempts'] += 1
        
        logger.info(
            f"🎯 ARB STARTED: {arb_id} | {symbol} | "
            f"L1: {leg1_side} {leg1_exchange} | L2: {leg2_side} {leg2_exchange} | "
            f"Expected: {expected_profit_bps:.1f} bps"
        )
        
        return attempt
    
    async def on_order_submitted(self, order_id: str):
        """Mark order as submitted to exchange."""
        arb_id = self._order_to_arb.get(order_id)
        if not arb_id:
            return
        
        attempt = self._attempts.get(arb_id)
        if not attempt:
            return
        
        if attempt.leg1.order_id == order_id:
            attempt.leg1.status = LegStatus.SUBMITTED
        elif attempt.leg2.order_id == order_id:
            attempt.leg2.status = LegStatus.SUBMITTED
    
    async def on_fill(self, order_id: str, filled_qty: float, fill_price: float):
        """Handle fill event for an arbitrage leg."""
        arb_id = self._order_to_arb.get(order_id)
        if not arb_id:
            return  # Not an arbitrage order
        
        attempt = self._attempts.get(arb_id)
        if not attempt:
            return
        
        # Update the correct leg
        if attempt.leg1.order_id == order_id:
            leg = attempt.leg1
            leg_name = "LEG1"
        elif attempt.leg2.order_id == order_id:
            leg = attempt.leg2
            leg_name = "LEG2"
        else:
            return
        
        leg.status = LegStatus.FILLED
        leg.filled_qty = filled_qty
        leg.filled_price = fill_price
        leg.filled_at = time.time()
        
        logger.info(f"✅ {arb_id} {leg_name} FILLED: {filled_qty} @ {fill_price}")
        
        # Check if arbitrage is complete
        await self._check_completion(attempt)
    
    async def on_reject(self, order_id: str, error: str):
        """Handle rejection/failure of an arbitrage leg."""
        arb_id = self._order_to_arb.get(order_id)
        if not arb_id:
            return
        
        attempt = self._attempts.get(arb_id)
        if not attempt:
            return
        
        # Update the correct leg
        if attempt.leg1.order_id == order_id:
            leg = attempt.leg1
            leg_name = "LEG1"
        elif attempt.leg2.order_id == order_id:
            leg = attempt.leg2
            leg_name = "LEG2"
        else:
            return
        
        leg.status = LegStatus.FAILED
        leg.error_message = error
        
        logger.warning(f"❌ {arb_id} {leg_name} FAILED: {error}")
        
        # Check if we're legged
        await self._check_completion(attempt)
    
    async def _check_completion(self, attempt: ArbitrageAttempt):
        """Check if arbitrage attempt is complete and handle accordingly."""
        if not attempt.is_complete:
            return
        
        attempt.completed_at = time.time()
        duration_ms = (attempt.completed_at - attempt.created_at) * 1000
        
        # Both legs filled = SUCCESS
        if attempt.leg1.status == LegStatus.FILLED and attempt.leg2.status == LegStatus.FILLED:
            attempt.status = ArbStatus.SUCCESS
            
            # Calculate realized profit
            l1_value = attempt.leg1.filled_qty * attempt.leg1.filled_price
            l2_value = attempt.leg2.filled_qty * attempt.leg2.filled_price
            
            if attempt.leg1.side == 'BUY':
                profit = l2_value - l1_value  # Bought L1, sold L2
            else:
                profit = l1_value - l2_value  # Sold L1, bought L2
            
            attempt.realized_profit = profit
            self._stats['successful'] += 1
            self._stats['total_profit'] += profit
            self._consecutive_failures = 0
            
            logger.info(
                f"🎉 ARB SUCCESS: {attempt.arb_id} | "
                f"Profit: ${profit:+.2f} | Duration: {duration_ms:.0f}ms"
            )
        
        # Legged position = DANGER
        elif attempt.is_legged:
            attempt.status = ArbStatus.LEGGED
            self._stats['legged'] += 1
            self._consecutive_failures += 1
            
            logger.critical(f"🚨 LEGGED POSITION: {attempt.arb_id} - Initiating emergency close!")
            await self._emergency_close(attempt)
        
        # Both failed = just mark as failed
        else:
            attempt.status = ArbStatus.FAILED
            self._stats['failed'] += 1
            self._consecutive_failures += 1
            logger.warning(f"❌ ARB FAILED: {attempt.arb_id} - Both legs failed")
        
        # Check circuit breaker
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._paused = True
            reason = f"{self._consecutive_failures} consecutive failures"
            logger.critical(
                f"🛑 CIRCUIT BREAKER TRIGGERED: {reason}. "
                f"Arbitrage paused. Call reset_circuit_breaker() to resume."
            )
            # Send Telegram alert
            if _telegram:
                asyncio.create_task(_telegram.alert_circuit_breaker(reason))
    
    async def _emergency_close(self, attempt: ArbitrageAttempt):
        """
        Close a legged position immediately with a MARKET order.
        
        This is a safety mechanism - we accept potential slippage
        to avoid holding directional exposure.
        """
        from core.events import SignalEvent
        
        # Find the filled leg
        if attempt.leg1.status == LegStatus.FILLED:
            filled_leg = attempt.leg1
        else:
            filled_leg = attempt.leg2
        
        # Determine opposite side to close
        close_side = 'SELL' if filled_leg.side == 'BUY' else 'BUY'
        
        # Create emergency close signal
        emergency_signal = SignalEvent(
            strategy_id='hedge_manager_emergency',
            symbol=filled_leg.symbol,
            side=close_side,
            quantity=filled_leg.filled_qty,
            exchange=filled_leg.exchange,
            price=0.0,  # MARKET order - price ignored
            order_type='MARKET'
        )
        
        logger.critical(
            f"🚨 EMERGENCY CLOSE: {close_side} {filled_leg.filled_qty} "
            f"{filled_leg.symbol} on {filled_leg.exchange.upper()} (MARKET ORDER)"
        )
        
        await self.queue.put(emergency_signal)
        self._stats['emergency_closes'] += 1
        
        # Send Telegram alert
        if _telegram:
            asyncio.create_task(_telegram.alert_emergency_close(
                symbol=filled_leg.symbol,
                side=close_side,
                qty=filled_leg.filled_qty,
                exchange=filled_leg.exchange
            ))
        
        # Mark leg as emergency closed for tracking
        filled_leg.status = LegStatus.EMERGENCY_CLOSED
    
    async def _watchdog_loop(self):
        """
        Background task that monitors for timeout situations.
        
        Checks every 100ms for attempts where one leg is filled
        but the other is still pending after the timeout.
        """
        while True:
            try:
                await asyncio.sleep(self.WATCHDOG_INTERVAL_MS / 1000)
                now = time.time()
                timeout_seconds = self.LEG_TIMEOUT_MS / 1000
                
                for arb_id, attempt in list(self._attempts.items()):
                    if attempt.status != ArbStatus.ACTIVE:
                        continue
                    
                    age = now - attempt.created_at
                    if age < timeout_seconds:
                        continue  # Not timed out yet
                    
                    # Check if we're in a dangerous state
                    l1_filled = attempt.leg1.status == LegStatus.FILLED
                    l2_filled = attempt.leg2.status == LegStatus.FILLED
                    l1_pending = attempt.leg1.status in {LegStatus.PENDING, LegStatus.SUBMITTED}
                    l2_pending = attempt.leg2.status in {LegStatus.PENDING, LegStatus.SUBMITTED}
                    
                    # One filled, one still pending = TIMEOUT
                    if (l1_filled and l2_pending) or (l2_filled and l1_pending):
                        pending_leg = attempt.leg1 if l1_pending else attempt.leg2
                        pending_leg.status = LegStatus.FAILED
                        pending_leg.error_message = "Timeout waiting for fill"
                        
                        logger.critical(
                            f"⏱️ TIMEOUT: {arb_id} - "
                            f"{'LEG1' if l1_pending else 'LEG2'} timed out after {age*1000:.0f}ms"
                        )
                        
                        await self._check_completion(attempt)
                    
                    # Both pending after timeout = Just fail both
                    elif l1_pending and l2_pending:
                        attempt.leg1.status = LegStatus.FAILED
                        attempt.leg2.status = LegStatus.FAILED
                        attempt.leg1.error_message = "Timeout"
                        attempt.leg2.error_message = "Timeout"
                        await self._check_completion(attempt)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
    
    def get_stats(self) -> dict:
        """Get hedge manager statistics."""
        return {
            **self._stats,
            'active_attempts': len([a for a in self._attempts.values() if a.status == ArbStatus.ACTIVE]),
            'paused': self._paused,
            'consecutive_failures': self._consecutive_failures
        }
    
    def cleanup_old_attempts(self, max_age_seconds: float = 3600):
        """Remove completed attempts older than max_age."""
        now = time.time()
        to_remove = []
        
        for arb_id, attempt in self._attempts.items():
            if attempt.status != ArbStatus.ACTIVE:
                if attempt.completed_at and (now - attempt.completed_at) > max_age_seconds:
                    to_remove.append(arb_id)
        
        for arb_id in to_remove:
            attempt = self._attempts.pop(arb_id)
            self._order_to_arb.pop(attempt.leg1.order_id, None)
            self._order_to_arb.pop(attempt.leg2.order_id, None)


# Global instance
_hedge_manager: Optional[HedgeManager] = None


def get_hedge_manager(event_queue=None) -> HedgeManager:
    """Get or create the global hedge manager."""
    global _hedge_manager
    if _hedge_manager is None:
        if event_queue is None:
            raise ValueError("event_queue required for first initialization")
        _hedge_manager = HedgeManager(event_queue)
    return _hedge_manager
