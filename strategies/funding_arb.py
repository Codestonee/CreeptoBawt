"""
Delta-Neutral Funding Rate Arbitrage Strategy.

Captures perpetual futures funding rates via delta-neutral positions:
- Long Spot + Short Perpetual (when funding > 0)
- Short Spot + Long Perpetual (when funding < 0)

Real Implementation using FundingRateEvent and Spot+Futures Execution.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from core.events import SignalEvent, FundingRateEvent
# from config.settings import APISettings  <-- Removed

logger = logging.getLogger("Strategy.FundingArb")


class TradeStatus(str, Enum):
    """Status of a carry trade."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass
class FundingArbConfig:
    min_funding_rate: float = 0.0001  # 0.01% per 8h
    min_profit_usd: float = 5.0
    position_size_usd: float = 100.0
    max_positions: int = 1
    max_leverage: float = 1.0  # Safe 1x leverage
    exit_threshold_rate: float = 0.00005  # Exit if rate drops below 0.005%


@dataclass
class CarryTrade:
    """An active carry trade position."""
    trade_id: str
    symbol: str
    direction: str              # "LONG_SPOT_SHORT_PERP"
    spot_quantity: float
    perp_quantity: float
    entry_funding_rate: float
    
    # Execution Tracking
    spot_order_id: Optional[str] = None
    perp_order_id: Optional[str] = None
    spot_filled: bool = False
    perp_filled: bool = False
    
    status: TradeStatus = TradeStatus.PENDING
    entry_time: float = field(default_factory=time.time)


class FundingArbStrategy:
    def __init__(self, event_queue: asyncio.Queue):
        self.queue = event_queue
        self.config = FundingArbConfig()
        
        # State
        self.active_trades: Dict[str, CarryTrade] = {}
        self.funding_rates: Dict[str, float] = {}
        self.mark_prices: Dict[str, float] = {}
        self.last_scan: float = 0
        
        logger.info("FundingArbStrategy initialized (Real Mode)")

    async def on_tick(self, event):
        """Ignore individual ticks, we only care about Funding Rates."""
        pass

    async def on_fill(self, event):
        """Handle execution fills."""
        # Find the trade for this fill
        # Note: In V1 we are "Fire and Forget" on Market Orders, 
        # allowing the CarryTrade object to just track status.
        # But we could update 'filled' status here.
        pass

    async def on_funding_rate(self, event: FundingRateEvent):
        """Handle real-time funding rate updates."""
        self.funding_rates[event.symbol] = event.rate
        self.mark_prices[event.symbol] = event.mark_price
        
        # Check for entry opportunities
        await self._check_entry(event.symbol, event.rate, event.mark_price)
        
        # Check for exit opportunities
        await self._check_exit0(event.symbol, event.rate)

    async def _check_entry(self, symbol: str, rate: float, price: float):
        """Check if we should enter a trade."""
        # 1. Basic Filters
        if rate < self.config.min_funding_rate:
            return
            
        # 2. Already invested in this symbol?
        if any(t.symbol == symbol for t in self.active_trades.values()):
            return
            
        # 3. Max positions limit
        if len(self.active_trades) >= self.config.max_positions:
            return

        # 4. Calculate Size
        size_usd = self.config.position_size_usd
        quantity = size_usd / price
        
        # 5. Execute Dual Leg
        logger.info(f"💰 Opportunity found: {symbol} Funding={rate*100:.4f}%")
        await self._execute_entry(symbol, quantity, rate)

    async def _execute_entry(self, symbol: str, quantity: float, rate: float):
        """Execute Long Spot and Short Perp simultaneously."""
        from config.settings import settings  # Lazy import to avoid circular dep
        
        trade_id = f"arb_{uuid.uuid4().hex[:8]}"
        is_paper = settings.ARBITRAGE_PAPER_TRADING
        
        trade = CarryTrade(
            trade_id=trade_id,
            symbol=symbol,
            direction="LONG_SPOT_SHORT_PERP",
            spot_quantity=quantity,
            perp_quantity=quantity,
            entry_funding_rate=rate,
            status=TradeStatus.ACTIVE  # Mark active for tracking
        )
        self.active_trades[trade_id] = trade
        
        if is_paper:
            logger.info(f"📝 [PAPER] Simulate ARB Entry: Long Spot / Short Perp {quantity} {symbol}")
            return

        # 1. Send Spot BUY (Market for speed, or aggressive Limit)
        # Using separate signals for Spot and Futures
        spot_signal = SignalEvent(
            strategy_id="funding_arb",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=None,     # Market Order
            order_type="MARKET",
            exchange="binance_spot", # Targets _execute_spot_order
            arb_id=trade_id
        )
        
        # 2. Send Perp SELL
        perp_signal = SignalEvent(
            strategy_id="funding_arb",
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            price=None,     # Market Order
            order_type="MARKET",
            exchange="binance_futures", # Targets _execute_futures_order
            arb_id=trade_id
        )
        
        logger.info(f"🚀 Executing ARB {trade_id}: Long Spot / Short Perp {quantity} {symbol}")
        await self.queue.put(spot_signal)
        await self.queue.put(perp_signal)
        
        trade.status = TradeStatus.ACTIVE

    async def _check_exit0(self, symbol: str, current_rate: float):
        """Check if we should exit active trades."""
        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol != symbol or trade.status != TradeStatus.ACTIVE:
                continue
                
            # Exit if funding rate drops too low
            if current_rate < self.config.exit_threshold_rate:
                logger.info(f"📉 Funding dropped ({current_rate*100:.4f}%), closing {trade.symbol}")
                await self._execute_exit(trade)

    async def _execute_exit(self, trade: CarryTrade):
        """Close both legs."""
        from config.settings import settings
        is_paper = settings.ARBITRAGE_PAPER_TRADING

        trade.status = TradeStatus.CLOSING
        
        if is_paper:
            logger.info(f"📝 [PAPER] Simulate ARB Exit: Close {trade.symbol}")
            trade.status = TradeStatus.CLOSED
            del self.active_trades[trade.trade_id]
            return
        
        # 1. Close Spot (Sell)
        spot_signal = SignalEvent(
            strategy_id="funding_arb",
            symbol=trade.symbol,
            side="SELL",
            quantity=trade.spot_quantity,
            price=None,
            order_type="MARKET",
            exchange="binance_spot"
        )
        
        # 2. Close Perp (Buy)
        perp_signal = SignalEvent(
            strategy_id="funding_arb",
            symbol=trade.symbol,
            side="BUY",
            quantity=trade.perp_quantity,
            price=None,
            order_type="MARKET",
            exchange="binance_futures"
        )
        
        await self.queue.put(spot_signal)
        await self.queue.put(perp_signal)
        
        trade.status = TradeStatus.CLOSED
        del self.active_trades[trade.trade_id]
        logger.info(f"✅ Arb {trade.trade_id} closed.")
