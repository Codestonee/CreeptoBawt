import logging
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger("Execution.Router")

class FillStatus(str, Enum):
    """Fill confirmation status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"

@dataclass
class RouterResult:
    filled_qty: float
    avg_price: float
    total_cost_bps: float
    status: str = "FILLED"
    attempts: int = 0
    fill_events: list = None

class DeterministicOrderRouter:
    """
    Limit Chase Router with REAL fill confirmation.
    
    Strategy:
    1. Place aggressive limit order (GTX post-only)
    2. Wait for fill via WebSocket (up to reprice_interval)
    3. If not filled, cancel and reprice closer to market
    4. After max_repricings, fall back to market order
    
    Key Fix: Now actually waits for fills instead of assuming!
    """
    
    def __init__(
        self,
        reprice_interval_ms: int = 300,
        max_repricings: int = 5,
        min_update_interval_ms: int = 300,
        tick_tolerance: int = 1
    ):
        self.reprice_interval = reprice_interval_ms / 1000.0
        self.max_attempts = max_repricings
        self.min_update = min_update_interval_ms / 1000.0
        self.tick_tolerance = tick_tolerance
        
        self._stats = {
            'maker_fill_pct': 0,
            'limit_chase_count': 0,
            'avg_attempts': 0.0,
            'market_fallback_count': 0
        }
        
        # Fill tracking
        self._pending_fills: Dict[str, asyncio.Event] = {}
        self._fill_results: Dict[str, dict] = {}
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats
    
    def on_fill_confirmed(self, client_order_id: str, filled_qty: float, avg_price: float):
        """
        Call this from your execution handler when a fill is confirmed.
        This unblocks the waiting router.
        
        Example integration in binance_executor._handle_order_update:
        ```
        if self.order_router:
            self.order_router.on_fill_confirmed(
                client_order_id, last_filled_qty, last_filled_price
            )
        ```
        """
        if client_order_id in self._pending_fills:
            self._fill_results[client_order_id] = {
                'filled_qty': filled_qty,
                'avg_price': avg_price
            }
            self._pending_fills[client_order_id].set()
    
    async def _wait_for_fill(
        self,
        client_order_id: str,
        timeout: float
    ) -> Optional[dict]:
        """
        Wait for fill confirmation with timeout.
        
        Returns:
            {'filled_qty': float, 'avg_price': float} or None if timeout
        """
        # Create event for this order IF not pre-registered
        if client_order_id in self._pending_fills:
            fill_event = self._pending_fills[client_order_id]
        else:
            fill_event = asyncio.Event()
            self._pending_fills[client_order_id] = fill_event
        
        try:
            # Wait for fill or timeout
            await asyncio.wait_for(fill_event.wait(), timeout=timeout)
            
            # Fill received!
            result = self._fill_results.pop(client_order_id, None)
            return result
            
        except asyncio.TimeoutError:
            # Timeout - order not filled
            return None
            
        finally:
            # Cleanup
            if client_order_id in self._pending_fills:
                del self._pending_fills[client_order_id]
    
    async def fill_order(
        self,
        side: str,
        quantity: float,
        symbol: str,
        get_best_bid_ask_fn: Callable,
        place_order_fn: Callable,
        cancel_order_fn: Callable,
        max_wait_seconds: float = 3.0,
        client_order_id: Optional[str] = None
    ) -> RouterResult:
        """
        Execute order with limit chasing and fill confirmation.
        
        Returns:
            RouterResult with actual fill data (not assumptions!)
        """
        self._stats['limit_chase_count'] += 1
        
        attempt = 0
        total_filled = 0.0
        weighted_price_sum = 0.0
        fill_events = []
        
        remaining_qty = quantity
        
        while attempt < self.max_attempts and remaining_qty > 0.001:
            attempt += 1
            
            # Get current market
            best_bid, best_ask = await get_best_bid_ask_fn(symbol)
            
            if best_bid == 0 or best_ask == 0:
                logger.error(f"[{symbol}] Invalid orderbook: bid={best_bid}, ask={best_ask}")
                break
            
            # Calculate aggressive price
            if side.upper() == "BUY":
                # Start at best ask (most aggressive maker)
                # Each attempt gets more aggressive
                aggression = 1.0 + (attempt * 0.0001)  # 0.01% more aggressive per attempt
                price = best_ask * aggression
            else:
                aggression = 1.0 - (attempt * 0.0001)
                price = best_bid * aggression
            
            logger.info(
                f"[{symbol}] Router attempt {attempt}/{self.max_attempts}: "
                f"{side} {remaining_qty:.4f} @ ${price:.4f}"
            )
            
            # Generate unique order ID for this attempt
            order_cid = f"{client_order_id}_a{attempt}" if client_order_id else None
            
            # CRITICAL FIX: Pre-register to capture instant fills (Race Condition)
            if order_cid:
                self._pending_fills[order_cid] = asyncio.Event()
            
            try:
                # Place order
                order_result = await place_order_fn(
                    symbol=symbol,
                    side=side,
                    quantity=remaining_qty,
                    price=price,
                    order_type="LIMIT",
                    client_order_id=order_cid
                )
                
                order_id = order_result.get('order_id')
                if not order_id:
                    logger.warning(f"[{symbol}] Order rejected (likely GTX): {order_result}")
                    # Cleanup pre-registration
                    if order_cid in self._pending_fills:
                        del self._pending_fills[order_cid]
                        
                    continue
                
                # Wait for fill confirmation
                fill_result = await self._wait_for_fill(
                    order_cid or order_id,
                    timeout=self.reprice_interval
                )
                
                if fill_result:
                    # Fill confirmed!
                    filled = fill_result['filled_qty']
                    fill_price = fill_result['avg_price']
                    
                    total_filled += filled
                    weighted_price_sum += filled * fill_price
                    remaining_qty -= filled
                    
                    fill_events.append({
                        'attempt': attempt,
                        'qty': filled,
                        'price': fill_price
                    })
                    
                    logger.info(
                        f"✅ [{symbol}] Filled {filled:.4f} @ ${fill_price:.4f} "
                        f"(remaining: {remaining_qty:.4f})"
                    )
                    
                    if remaining_qty < 0.001:
                        break  # Fully filled!
                else:
                    # Not filled within timeout - cancel
                    logger.debug(f"[{symbol}] Order not filled in {self.reprice_interval}s - canceling")
                    
                    cancel_success = await cancel_order_fn(order_cid or order_id)
                    if not cancel_success:
                        logger.warning(f"[{symbol}] Failed to cancel order {order_id}")
                        # Wait a bit to see if it fills anyway
                        await asyncio.sleep(0.5)
                        
                        # Check one more time
                        late_fill = await self._wait_for_fill(
                            order_cid or order_id,
                            timeout=0.5
                        )
                        if late_fill:
                            filled = late_fill['filled_qty']
                            fill_price = late_fill['avg_price']
                            total_filled += filled
                            weighted_price_sum += filled * fill_price
                            remaining_qty -= filled
                            fill_events.append({
                                'attempt': attempt,
                                'qty': filled,
                                'price': fill_price,
                                'late': True
                            })
                    
            except Exception as e:
                logger.error(f"[{symbol}] Router error on attempt {attempt}: {e}")
                await asyncio.sleep(0.2)
                continue
        
        # Check if we filled enough (>90% is acceptable)
        fill_ratio = total_filled / quantity if quantity > 0 else 0
        
        if fill_ratio < 0.9:
            # Less than 90% filled - use market order for remainder
            logger.warning(
                f"[{symbol}] Only {fill_ratio:.1%} filled after {attempt} attempts - "
                f"using MARKET for remaining {remaining_qty:.4f}"
            )
            
            try:
                market_result = await place_order_fn(
                    symbol=symbol,
                    side=side,
                    quantity=remaining_qty,
                    price=None,
                    order_type="MARKET",
                    client_order_id=f"{client_order_id}_mkt" if client_order_id else None
                )
                
                # Market orders fill immediately
                if market_result.get('filled_qty', 0) > 0:
                    total_filled += market_result['filled_qty']
                    weighted_price_sum += market_result['filled_qty'] * market_result['avg_price']
                    fill_events.append({
                        'attempt': 'MARKET',
                        'qty': market_result['filled_qty'],
                        'price': market_result['avg_price']
                    })
                    self._stats['market_fallback_count'] += 1
                    
            except Exception as e:
                logger.error(f"[{symbol}] Market order fallback failed: {e}")
        
        # Calculate results
        if total_filled > 0:
            avg_price = weighted_price_sum / total_filled
            
            # Estimate cost in bps (vs mid price)
            mid = (best_bid + best_ask) / 2
            if side.upper() == "BUY":
                cost_bps = ((avg_price / mid) - 1) * 10000
            else:
                cost_bps = (1 - (avg_price / mid)) * 10000
            
            # Update stats
            maker_fills = sum(1 for e in fill_events if e.get('attempt') != 'MARKET')
            maker_pct = (maker_fills / len(fill_events) * 100) if fill_events else 0
            self._stats['maker_fill_pct'] = (
                self._stats['maker_fill_pct'] * 0.9 + maker_pct * 0.1  # EWMA
            )
            self._stats['avg_attempts'] = (
                self._stats['avg_attempts'] * 0.9 + attempt * 0.1
            )
            
            return RouterResult(
                filled_qty=total_filled,
                avg_price=avg_price,
                total_cost_bps=cost_bps,
                status="FILLED" if total_filled >= quantity * 0.99 else "PARTIAL",
                attempts=attempt,
                fill_events=fill_events
            )
        else:
            logger.error(f"[{symbol}] Router failed completely - no fills")
            return RouterResult(
                filled_qty=0,
                avg_price=0,
                total_cost_bps=0,
                status="FAILED",
                attempts=attempt,
                fill_events=[]
            )
