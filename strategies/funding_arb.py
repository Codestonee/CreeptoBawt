"""
Delta-Neutral Funding Rate Arbitrage Strategy.

Captures perpetual futures funding rates via delta-neutral positions:
- Long Spot + Short Perpetual (when funding > 0)
- Short Spot + Long Perpetual (when funding < 0)

Strategy flow:
1. Monitor funding rates across exchanges
2. Enter when funding > threshold (covers costs)
3. Maintain delta-neutral hedge
4. Exit when funding drops or risk triggers hit

Risk controls:
- Basis deviation monitoring
- Margin ratio checks
- Maximum hold time
- Breakeven calculation

References:
- Binance Futures Funding Rate: https://www.binance.com/en/support/faq/360033525031
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger("Strategy.FundingArb")


class TradeStatus(str, Enum):
    """Status of a carry trade."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass
class FundingOpportunity:
    """A funding rate arbitrage opportunity."""
    symbol: str
    perp_exchange: str          # Exchange with the perp position
    spot_exchange: str          # Exchange with the spot position
    funding_rate: float         # Current funding rate (per 8h)
    predicted_rate: float       # Predicted next funding rate
    spot_price: float
    perp_price: float
    basis: float                # (perp - spot) / spot
    expected_carry_8h: float    # Expected carry in USD per $100k notional
    timestamp: float = field(default_factory=time.time)
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (365 * 3 funding periods per day)."""
        return self.funding_rate * 3 * 365 * 100  # Percentage
    
    def is_profitable(self, min_rate: float = 0.0001) -> bool:
        """Check if opportunity is profitable after fees."""
        return self.funding_rate > min_rate


@dataclass
class CarryTrade:
    """An active carry trade position."""
    trade_id: str
    symbol: str
    direction: str              # "LONG_SPOT_SHORT_PERP" or "SHORT_SPOT_LONG_PERP"
    spot_exchange: str
    perp_exchange: str
    entry_spot_price: float
    entry_perp_price: float
    entry_basis: float
    quantity: float             # Base asset quantity
    notional_usd: float
    entry_time: float
    expected_daily_yield: float
    status: TradeStatus = TradeStatus.PENDING
    
    # Tracking
    funding_collected: float = 0.0
    exit_spot_price: Optional[float] = None
    exit_perp_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: float = 0.0
    
    def age_hours(self) -> float:
        """Hours since entry."""
        return (time.time() - self.entry_time) / 3600
    
    def age_str(self) -> str:
        """Human-readable age."""
        hours = self.age_hours()
        if hours < 1:
            return f"{int(hours * 60)}m"
        return f"{hours:.1f}h"


@dataclass
class FundingArbConfig:
    """Configuration for funding rate arbitrage."""
    # Entry thresholds
    min_funding_rate: float = 0.00025      # 0.025% per 8h minimum
    min_profit_8h_usd: float = 10.0        # Minimum $10 profit per 8h
    
    # Exit thresholds
    exit_funding_rate: float = 0.00015     # Exit if funding drops below this
    max_basis_deviation: float = 0.005     # 0.5% basis risk tolerance
    min_margin_ratio: float = 0.10         # 10% margin ratio safety
    max_hold_hours: float = 48.0           # Maximum hold time
    
    # Position sizing
    max_positions: int = 5
    position_size_usd: float = 50000.0     # Default position size
    max_total_exposure_usd: float = 200000.0
    
    # Cost assumptions
    maker_fee: float = 0.0002              # 0.02%
    taker_fee: float = 0.0005              # 0.05%
    expected_slippage: float = 0.0003      # 0.03%
    
    @property
    def entry_cost_pct(self) -> float:
        """Total entry cost as percentage."""
        return (self.maker_fee + self.taker_fee + self.expected_slippage) * 2  # Both legs
    
    @property
    def breakeven_funding(self) -> float:
        """Minimum funding rate to break even after 1 cycle."""
        return self.entry_cost_pct


class FundingRateMonitor:
    """
    Monitors funding rates across exchanges and identifies opportunities.
    
    Usage:
        monitor = FundingRateMonitor(config)
        
        # Periodic scan
        opps = await monitor.scan_opportunities()
        
        for opp in opps:
            if opp.is_profitable(config.min_funding_rate):
                await trade_manager.execute(opp)
    """
    
    def __init__(
        self, 
        config: Optional[FundingArbConfig] = None,
        symbols: Optional[List[str]] = None
    ):
        self.config = config or FundingArbConfig()
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        
        # Rate cache
        self._funding_cache: Dict[str, Dict[str, float]] = {}
        self._price_cache: Dict[str, Dict[str, float]] = {}
        self._last_update: float = 0
        
        logger.info(f"FundingRateMonitor initialized for {self.symbols}")
    
    async def scan_opportunities(self) -> List[FundingOpportunity]:
        """
        Scan all symbols for funding arbitrage opportunities.
        
        Returns list of opportunities sorted by expected yield.
        """
        opportunities = []
        
        for symbol in self.symbols:
            try:
                opp = await self._check_symbol(symbol)
                if opp and opp.is_profitable(self.config.min_funding_rate):
                    opportunities.append(opp)
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
        
        # Sort by expected carry (highest first)
        opportunities.sort(key=lambda x: x.expected_carry_8h, reverse=True)
        
        return opportunities
    
    async def _check_symbol(self, symbol: str) -> Optional[FundingOpportunity]:
        """Check a single symbol for opportunity."""
        # Get funding rates
        binance_funding = await self._get_funding_rate("BINANCE", symbol)
        
        if binance_funding is None:
            return None
        
        # Get prices
        spot_price = await self._get_spot_price(symbol)
        perp_price = await self._get_perp_price("BINANCE", symbol)
        
        if spot_price is None or perp_price is None:
            return None
        
        # Calculate basis
        basis = (perp_price - spot_price) / spot_price if spot_price > 0 else 0
        
        # Calculate expected carry on $100k notional
        expected_carry_8h = binance_funding * 100000
        
        return FundingOpportunity(
            symbol=symbol,
            perp_exchange="BINANCE",
            spot_exchange="BINANCE",
            funding_rate=binance_funding,
            predicted_rate=binance_funding,  # Could fetch predicted rate
            spot_price=spot_price,
            perp_price=perp_price,
            basis=basis,
            expected_carry_8h=expected_carry_8h
        )
    
    async def _get_funding_rate(self, exchange: str, symbol: str) -> Optional[float]:
        """
        Get current funding rate from exchange.
        
        Note: In production, replace with actual API call.
        """
        # Placeholder - replace with actual exchange API
        # For Binance: GET /fapi/v1/premiumIndex
        
        # Return cached or simulated value
        cache_key = f"{exchange}:{symbol}"
        if cache_key in self._funding_cache:
            return self._funding_cache[cache_key].get("rate")
        
        # Simulate rate for testing
        import random
        rate = random.uniform(-0.0005, 0.001)
        self._funding_cache[cache_key] = {"rate": rate, "time": time.time()}
        
        return rate
    
    async def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price."""
        # Placeholder - replace with actual API
        if symbol.upper() == "BTCUSDT":
            return 42000.0
        elif symbol.upper() == "ETHUSDT":
            return 2500.0
        return None
    
    async def _get_perp_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get current perpetual price."""
        # Placeholder - replace with actual API
        spot = await self._get_spot_price(symbol)
        if spot:
            # Simulate small basis
            import random
            return spot * (1 + random.uniform(-0.001, 0.002))
        return None
    
    def update_prices(self, symbol: str, spot: float, perp: float) -> None:
        """Update cached prices (can be called from websocket handler)."""
        self._price_cache[symbol] = {
            "spot": spot,
            "perp": perp,
            "time": time.time()
        }
    
    def update_funding_rate(self, symbol: str, rate: float, exchange: str = "BINANCE") -> None:
        """Update cached funding rate."""
        cache_key = f"{exchange}:{symbol}"
        self._funding_cache[cache_key] = {"rate": rate, "time": time.time()}


class CarryTradeManager:
    """
    Manages delta-neutral carry trade positions.
    
    Handles:
    - Position entry (spot + perp legs)
    - Risk monitoring (basis, margin, time)
    - Position exit
    - Funding collection tracking
    """
    
    def __init__(
        self,
        config: Optional[FundingArbConfig] = None,
        monitor: Optional[FundingRateMonitor] = None
    ):
        self.config = config or FundingArbConfig()
        self.monitor = monitor or FundingRateMonitor(self.config)
        
        # Active trades
        self.active_trades: Dict[str, CarryTrade] = {}
        
        # Statistics
        self.total_funding_collected: float = 0.0
        self.total_trades: int = 0
        self.profitable_trades: int = 0
        
        logger.info("CarryTradeManager initialized")
    
    async def execute_trade(
        self, 
        opportunity: FundingOpportunity,
        size_usd: Optional[float] = None
    ) -> Optional[CarryTrade]:
        """
        Execute a carry trade based on opportunity.
        
        Args:
            opportunity: The funding opportunity to exploit
            size_usd: Position size in USD (defaults to config)
        
        Returns:
            CarryTrade if successful, None otherwise
        """
        size_usd = size_usd or self.config.position_size_usd
        
        # Check active symbols to prevent duplicates
        active_symbols = {t.symbol for t in self.active_trades.values()}
        if opportunity.symbol in active_symbols:
            logger.debug(f"Already have position in {opportunity.symbol}, skipping.")
            return None
            
        # Check limits
        if len(self.active_trades) >= self.config.max_positions:
            logger.warning("Max positions reached, skipping opportunity")
            return None
        
        total_exposure = sum(t.notional_usd for t in self.active_trades.values())
        if total_exposure + size_usd > self.config.max_total_exposure_usd:
            logger.warning("Max exposure reached, skipping opportunity")
            return None
        
        # Calculate quantity
        quantity = size_usd / opportunity.spot_price
        
        # Determine direction
        if opportunity.funding_rate > 0:
            direction = "LONG_SPOT_SHORT_PERP"
        else:
            direction = "SHORT_SPOT_LONG_PERP"
        
        # Create trade record
        trade = CarryTrade(
            trade_id=f"carry_{uuid.uuid4().hex[:8]}",
            symbol=opportunity.symbol,
            direction=direction,
            spot_exchange=opportunity.spot_exchange,
            perp_exchange=opportunity.perp_exchange,
            entry_spot_price=opportunity.spot_price,
            entry_perp_price=opportunity.perp_price,
            entry_basis=opportunity.basis,
            quantity=quantity,
            notional_usd=size_usd,
            entry_time=time.time(),
            expected_daily_yield=abs(opportunity.funding_rate) * 3 * size_usd
        )
        
        logger.info(
            f"Executing carry trade: {trade.trade_id} | "
            f"{opportunity.symbol} | {direction} | "
            f"${size_usd:,.0f} | funding={opportunity.funding_rate:.4%}"
        )
        
        # Execute legs (placeholder - integrate with actual execution)
        try:
            # In production: await self._execute_spot_leg(trade)
            # In production: await self._execute_perp_leg(trade)
            
            trade.status = TradeStatus.ACTIVE
            self.active_trades[trade.trade_id] = trade
            self.total_trades += 1
            
            logger.info(f"Carry trade {trade.trade_id} ACTIVE")
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute carry trade: {e}")
            trade.status = TradeStatus.FAILED
            return None
    
    async def check_exit_conditions(self) -> List[str]:
        """
        Check all active trades for exit conditions.
        
        Returns list of trade_ids that should be closed.
        """
        trades_to_close = []
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade.status != TradeStatus.ACTIVE:
                continue
            
            should_close, reason = await self._should_close(trade)
            if should_close:
                logger.info(f"Trade {trade_id} exit triggered: {reason}")
                trades_to_close.append(trade_id)
        
        return trades_to_close
    
    async def _should_close(self, trade: CarryTrade) -> Tuple[bool, str]:
        """Check if a trade should be closed."""
        # 1. Max hold time
        if trade.age_hours() > self.config.max_hold_hours:
            return True, f"max_hold_time ({trade.age_str()})"
        
        # 2. Funding rate dropped
        current_funding = await self._get_current_funding(trade.symbol)
        if current_funding is not None:
            if abs(current_funding) < self.config.exit_funding_rate:
                return True, f"funding_dropped ({current_funding:.4%})"
            
            # Check if funding flipped sign (bad for our position)
            if trade.direction == "LONG_SPOT_SHORT_PERP" and current_funding < 0:
                return True, "funding_flipped_negative"
            if trade.direction == "SHORT_SPOT_LONG_PERP" and current_funding > 0:
                return True, "funding_flipped_positive"
        
        # 3. Basis deviation
        current_basis = await self._get_current_basis(trade.symbol)
        if current_basis is not None:
            basis_deviation = abs(current_basis - trade.entry_basis)
            if basis_deviation > self.config.max_basis_deviation:
                return True, f"basis_deviation ({basis_deviation:.4%})"
        
        # 4. Margin ratio (placeholder - integrate with exchange)
        # margin_ratio = await self._get_margin_ratio(trade)
        # if margin_ratio < self.config.min_margin_ratio:
        #     return True, f"low_margin ({margin_ratio:.1%})"
        
        return False, ""
    
    async def close_trade(self, trade_id: str) -> bool:
        """
        Close a carry trade.
        
        Returns True if successful.
        """
        trade = self.active_trades.get(trade_id)
        if trade is None:
            logger.warning(f"Trade {trade_id} not found")
            return False
        
        trade.status = TradeStatus.CLOSING
        
        try:
            # Get exit prices (placeholder)
            exit_spot = await self.monitor._get_spot_price(trade.symbol)
            exit_perp = await self.monitor._get_perp_price(
                trade.perp_exchange, trade.symbol
            )
            
            # In production: await self._close_spot_leg(trade)
            # In production: await self._close_perp_leg(trade)
            
            trade.exit_spot_price = exit_spot
            trade.exit_perp_price = exit_perp
            trade.exit_time = time.time()
            trade.status = TradeStatus.CLOSED
            
            # Calculate PnL
            trade.pnl = self._calculate_pnl(trade)
            
            if trade.pnl > 0:
                self.profitable_trades += 1
            
            self.total_funding_collected += trade.funding_collected
            
            # Remove from active
            del self.active_trades[trade_id]
            
            logger.info(
                f"Closed trade {trade_id}: PnL=${trade.pnl:,.2f}, "
                f"hold_time={trade.age_str()}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False
    
    def _calculate_pnl(self, trade: CarryTrade) -> float:
        """Calculate trade PnL including funding collected."""
        if trade.exit_spot_price is None or trade.exit_perp_price is None:
            return 0.0
        
        # Spot leg PnL
        if trade.direction == "LONG_SPOT_SHORT_PERP":
            spot_pnl = (trade.exit_spot_price - trade.entry_spot_price) * trade.quantity
            perp_pnl = (trade.entry_perp_price - trade.exit_perp_price) * trade.quantity
        else:
            spot_pnl = (trade.entry_spot_price - trade.exit_spot_price) * trade.quantity
            perp_pnl = (trade.exit_perp_price - trade.entry_perp_price) * trade.quantity
        
        # Total = position PnL + funding collected - fees
        fees = trade.notional_usd * self.config.entry_cost_pct
        
        return spot_pnl + perp_pnl + trade.funding_collected - fees
    
    async def _get_current_funding(self, symbol: str) -> Optional[float]:
        """Get current funding rate for symbol."""
        return await self.monitor._get_funding_rate("BINANCE", symbol)
    
    async def _get_current_basis(self, symbol: str) -> Optional[float]:
        """Get current basis for symbol."""
        spot = await self.monitor._get_spot_price(symbol)
        perp = await self.monitor._get_perp_price("BINANCE", symbol)
        if spot and perp and spot > 0:
            return (perp - spot) / spot
        return None
    
    def record_funding_payment(self, trade_id: str, amount: float) -> None:
        """Record a funding payment received for a trade."""
        trade = self.active_trades.get(trade_id)
        if trade:
            trade.funding_collected += amount
            logger.debug(f"Funding payment recorded for {trade_id}: ${amount:.2f}")
    
    def get_stats(self) -> dict:
        """Get trading statistics."""
        active_exposure = sum(t.notional_usd for t in self.active_trades.values())
        
        return {
            "active_trades": len(self.active_trades),
            "active_exposure_usd": active_exposure,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "win_rate": self.profitable_trades / max(self.total_trades, 1),
            "total_funding_collected": self.total_funding_collected
        }


async def run_funding_arb_loop(
    manager: CarryTradeManager,
    scan_interval: float = 60.0,
    check_interval: float = 30.0
):
    """
    Main loop for funding arbitrage strategy.
    
    This can be run as an asyncio task alongside the market making strategy.
    """
    logger.info("Funding arbitrage loop started")
    
    last_scan = 0
    
    while True:
        try:
            now = time.time()
            
            # Check exit conditions frequently
            trades_to_close = await manager.check_exit_conditions()
            for trade_id in trades_to_close:
                await manager.close_trade(trade_id)
            
            # Scan for new opportunities less frequently
            if now - last_scan > scan_interval:
                opportunities = await manager.monitor.scan_opportunities()
                
                for opp in opportunities:
                    if (opp.is_profitable(manager.config.min_funding_rate) and
                        opp.expected_carry_8h > manager.config.min_profit_8h_usd):
                        
                        await manager.execute_trade(opp)
                
                last_scan = now
            
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Funding arb loop error: {e}")
            await asyncio.sleep(60)
