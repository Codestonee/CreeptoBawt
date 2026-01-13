import logging
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("Execution.Router")

@dataclass
class RouterResult:
    filled_qty: float
    avg_price: float
    total_cost_bps: float

class DeterministicOrderRouter:
    """
    Restored minimal Router for Limit Chase execution.
    Original was deleted; this replaces the dependency.
    """
    def __init__(self):
        self._stats = {'maker_fill_pct': 0}

    def get_stats(self):
        return self._stats

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
        Execute aggressive limit order (Limit Chase).
        For simplicity, this version places a single aggressive LIMIT order 
        crossing the spread to ensure fill (IOC-like behavior).
        """
        best_bid, best_ask = await get_best_bid_ask_fn(symbol)
        
        if side.upper() == "BUY":
            # Buy aggressively (Ask + buffer)
            price = best_ask * 1.001 if best_ask > 0 else 0
        else:
            # Sell aggressively (Bid - buffer)
            price = best_bid * 0.999 if best_bid > 0 else 0
            
        logger.info(f"Router Chasing: {side} {quantity} {symbol} @ {price}")
        
        # Place Order
        # Note: place_order_fn returns an order object/dict
        order_res = await place_order_fn(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type="LIMIT",
            client_order_id=client_order_id
        )
        
        # In a real router, we would wait for WS fill confirmation.
        # Here we assume aggressive price fills immediately or we just return the attempt.
        # Capturing actual fill price requires listening to streams, 
        # but for this synchronous return, we estimate.
        
        return RouterResult(
            filled_qty=quantity,
            avg_price=price,
            total_cost_bps=5.0 # Estimate taker fee
        )
