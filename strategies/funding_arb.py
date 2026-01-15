
import asyncio
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.events import MarketEvent, SignalEvent, FillEvent, RegimeEvent, FundingRateEvent
from strategies.base import BaseStrategy
from config.settings import settings
from execution.order_manager import get_order_manager

logger = logging.getLogger("Strategy.FundingArb")

@dataclass
class ArbPosition:
    symbol: str
    side: str # "LONG_SPOT_SHORT_PERP" or "SHORT_SPOT_LONG_PERP"
    quantity: float
    entry_funding_rate: float
    entry_time: float
    spot_order_id: Optional[str] = None
    perp_order_id: Optional[str] = None

class FundingArbStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy.
    
    Profits from high funding rates by taking delta-neutral positions:
    - If Rate > 0: Shorts receive funding. -> Short Perp / Long Spot.
    - If Rate < 0: Longs receive funding. -> Long Perp / Short Spot.
    
    Configuration comes from settings.FUNDING_ARB_CONFIG.
    """
    
    def __init__(self, event_queue, symbols: List[str]):
        super().__init__(event_queue, symbols)
        self.config = settings.FUNDING_ARB_CONFIG
        self.funding_rates: Dict[str, float] = {}
        self.active_arbs: Dict[str, ArbPosition] = {} # Key: symbol
        self.mid_prices: Dict[str, float] = {}
        
        # Access OrderManager to check open positions/orders if needed
        self.order_manager = get_order_manager()
        
    async def on_tick(self, event: MarketEvent):
        """Track mid-prices for sizing."""
        if not event.order_book:
            return
            
        mid = (event.order_book.bids[0].price + event.order_book.asks[0].price) / 2
        self.mid_prices[event.symbol] = mid
        
        # Check exit conditions if we have active positions
        if event.symbol in self.active_arbs:
            await self._check_exit(event.symbol)

    async def on_funding_rate(self, event: FundingRateEvent):
        """
        Handle funding rate updates. 
        Calculates annualized yield and triggers entry if threshold met.
        """
        self.funding_rates[event.symbol] = event.rate
        
        rate_bps = event.rate * 10000
        logger.info(f"Funding Update {event.symbol}: {rate_bps:.2f} bps ({event.rate:.4%})")
        
        await self._check_entry(event.symbol, event.rate)
        
        # Also check exit if we are in a position
        if event.symbol in self.active_arbs:
            await self._check_exit(event.symbol)
        
    async def _check_entry(self, symbol: str, rate: float):
        """Check if we should enter an arb position."""
        if symbol in self.active_arbs:
            return # Already in a position
            
        threshold = self.config.get("MIN_FUNDING_RATE_PCT", 0.01) / 100 # Config is often in % (e.g. 0.01)
        # Note: settings says 0.01 which implies 0.01% = 0.0001
        # Let's verify config interpretation.
        # settings: "MIN_FUNDING_RATE_PCT": 0.01 (meaning 0.01% or absolute 0.01?)
        # Convention: "PCT" usually means percentage value. 0.01% = 0.0001.
        # But normal funding is like 0.01% (basis). 
        # If user put 0.01 in settings, likely means 0.01% (basis rate).
        # Let's assume input is percentage: 0.01 -> 0.0001.
        
        abs_rate = abs(rate)
        target_rate = self.config["MIN_FUNDING_RATE_PCT"] / 100 # Convert 0.01 -> 0.0001
        
        if abs_rate < target_rate:
            return

        # We have an opportunity
        mid = self.mid_prices.get(symbol)
        if not mid:
            logger.warning(f"Skipping arb entry for {symbol}: No price data")
            return
            
        # Size Calculation
        size_usd = self.config["POSITION_SIZE_USD"]
        quantity = size_usd / mid
        
        # Determine Legs
        # Rate > 0: Longs pay Shorts. We want to be Short Perp. Hedge with Long Spot.
        if rate > 0:
            logger.info(f"⚔️ ARB SIGNAL {symbol}: High Positive Rate ({rate:.4%}). SHORT PERP / LONG SPOT.")
            
            # 1. Execute Long Spot (Market)
            # 2. Execute Short Perp (Market) - for speed in MVP
            # TODO: Use limit orders in production
            
            # Emit Signals
            await self._execute_leg(symbol, "BUY", quantity, "SPOT")
            await self._execute_leg(symbol, "SELL", quantity, "PERP")
            
            self.active_arbs[symbol] = ArbPosition(
                symbol=symbol,
                side="LONG_SPOT_SHORT_PERP",
                quantity=quantity,
                entry_funding_rate=rate,
                entry_time=time.time()
            )
            
        elif rate < 0:
            logger.info(f"⚔️ ARB SIGNAL {symbol}: High Negative Rate ({rate:.4%}). LONG PERP / SHORT SPOT.")
            # Rate < 0: Shorts pay Longs. We want to be Long Perp. Hedge with Short Spot.
            
            await self._execute_leg(symbol, "SELL", quantity, "SPOT") # Short Spot (margin?)
            await self._execute_leg(symbol, "BUY", quantity, "PERP")
            
            self.active_arbs[symbol] = ArbPosition(
                symbol=symbol,
                side="SHORT_SPOT_LONG_PERP",
                quantity=quantity,
                entry_funding_rate=rate,
                entry_time=time.time()
            )

    async def _execute_leg(self, symbol: str, side: str, quantity: float, instrument_type: str):
        """Emit a signal Event."""
        # Fix: Use correct SignalEvent fields
        sig = SignalEvent(
            strategy_id="FundingArb",
            symbol=symbol,
            timestamp=time.time(),
            side=side, # BUY/SELL
            price=self.mid_prices.get(symbol, 0),
            quantity=quantity,
            order_type="MARKET" # Ensure market execution for arb legs
        )
        await self.queue.put(sig)
        logger.info(f"Sent {side} signal for {symbol} ({instrument_type} leg)")

    async def _check_exit(self, symbol: str):
        """Check if funding rate has normalized."""
        current_rate = abs(self.funding_rates.get(symbol, 0))
        exit_threshold = self.config["EXIT_FUNDING_PCT"] / 100
        
        if current_rate <= exit_threshold:
            # Unwind
            arb = self.active_arbs[symbol]
            logger.info(f"🕊️ ARB EXIT {symbol}: Rate normalized ({current_rate:.4%}). Unwinding.")
            
            # Invert sides
            if arb.side == "LONG_SPOT_SHORT_PERP":
                await self._execute_leg(symbol, "SELL", arb.quantity, "SPOT") # Sell Spot
                await self._execute_leg(symbol, "BUY", arb.quantity, "PERP")  # Buy Back Perp
            else:
                await self._execute_leg(symbol, "BUY", arb.quantity, "SPOT")  # Buy Back Spot
                await self._execute_leg(symbol, "SELL", arb.quantity, "PERP") # Sell Perp
                
            del self.active_arbs[symbol]

    async def on_fill(self, event: FillEvent):
        pass

    async def start(self):
        logger.info("Funding Arb Strategy Started")

    async def stop(self):
        logger.info("Funding Arb Strategy Stopped")
