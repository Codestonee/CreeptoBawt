import logging
import time
from typing import Dict, Optional
from core.events import MarketEvent, SignalEvent, FillEvent
from config.settings import settings

logger = logging.getLogger("Strategy.Arbitrage")


class ArbitrageStrategy:
    """
    Multi-Exchange Arbitrage Strategy with proper hedge management.
    
    Exploits price discrepancies between Binance and OKX.
    Uses HedgeManager to ensure both legs fill or emergency close.
    """
    
    def __init__(self, event_queue):
        self.queue = event_queue
        # Price Cache: {exchange: {symbol: {bid: float, ask: float}}}
        self.prices: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        self.min_spread = settings.ARBITRAGE_MIN_SPREAD
        self.trade_qty = 0.005  # Default trade size (BTC)
        self.last_trade_time = 0
        self.cooldown = 5  # Seconds between trades
        
        # Hedge Manager for leg tracking
        self._hedge_manager = None
        
        logger.info(f"ArbitrageStrategy initialized. Min Spread: {self.min_spread*100}%")
    
    def set_hedge_manager(self, hedge_manager):
        """Set the hedge manager for leg tracking."""
        self._hedge_manager = hedge_manager
        logger.info("✅ HedgeManager connected to ArbitrageStrategy")

    async def on_tick(self, event: MarketEvent):
        """Process market data updates."""
        if event.event_type != 'TICK':
            return
            
        # Update Price Cache
        if event.exchange not in self.prices:
            self.prices[event.exchange] = {}
        if event.symbol not in self.prices[event.exchange]:
            self.prices[event.exchange][event.symbol] = {'bid': 0.0, 'ask': 0.0}
            
        self.prices[event.exchange][event.symbol]['last'] = event.price
        
        await self._check_arbitrage(event.symbol)

    async def _check_arbitrage(self, symbol: str):
        """Check for arbitrage opportunity for a symbol."""
        now = time.time()
        if now - self.last_trade_time < self.cooldown:
            return
        
        # Check if paused by circuit breaker
        if self._hedge_manager and self._hedge_manager.is_paused():
            return

        p_binance = self._get_price('binance', symbol)
        p_okx = self._get_price('okx', symbol)
        
        if p_binance == 0 or p_okx == 0:
            return
            
        # Check Spread
        # 1. Binance > OKX (Sell Binance, Buy OKX)
        spread_1 = (p_binance / p_okx) - 1
        if spread_1 > self.min_spread:
            logger.info(f"⚡ ARB OPPORTUNITY: Binance ({p_binance}) > OKX ({p_okx}) | Spread: {spread_1*100:.2f}%")
            await self._execute_arb(
                symbol=symbol,
                leg1_side='SELL', leg1_exchange='binance', leg1_price=p_binance,
                leg2_side='BUY', leg2_exchange='okx', leg2_price=p_okx,
                spread_bps=spread_1 * 10000
            )
            return

        # 2. OKX > Binance (Sell OKX, Buy Binance)
        spread_2 = (p_okx / p_binance) - 1
        if spread_2 > self.min_spread:
            logger.info(f"⚡ ARB OPPORTUNITY: OKX ({p_okx}) > Binance ({p_binance}) | Spread: {spread_2*100:.2f}%")
            await self._execute_arb(
                symbol=symbol,
                leg1_side='SELL', leg1_exchange='okx', leg1_price=p_okx,
                leg2_side='BUY', leg2_exchange='binance', leg2_price=p_binance,
                spread_bps=spread_2 * 10000
            )
            return

    def _get_price(self, exchange, symbol):
        return self.prices.get(exchange, {}).get(symbol, {}).get('last', 0.0)

    async def _execute_arb(
        self, symbol: str,
        leg1_side: str, leg1_exchange: str, leg1_price: float,
        leg2_side: str, leg2_exchange: str, leg2_price: float,
        spread_bps: float
    ):
        """Execute dual-leg arbitrage with hedge tracking."""
        
        # Paper Trading Check
        if getattr(settings, 'ARBITRAGE_PAPER_TRADING', True):
            logger.info(
                f"📝 PAPER ARB: {leg1_side} {leg1_exchange} / {leg2_side} {leg2_exchange} | "
                f"Spread: {spread_bps:.1f} bps"
            )
            return

        qty = self.trade_qty
        
        # Start tracked arbitrage attempt
        if self._hedge_manager:
            attempt = self._hedge_manager.start_arbitrage(
                symbol=symbol,
                quantity=qty,
                leg1_exchange=leg1_exchange,
                leg1_side=leg1_side,
                leg1_price=leg1_price,
                leg2_exchange=leg2_exchange,
                leg2_side=leg2_side,
                leg2_price=leg2_price,
                expected_profit_bps=spread_bps
            )
            
            if attempt is None:
                return  # Circuit breaker active
            
            # Create signals with tracking IDs
            sig1 = SignalEvent(
                strategy_id="arbitrage",
                symbol=symbol,
                side=leg1_side,
                quantity=qty,
                price=leg1_price,
                exchange=leg1_exchange,
                arb_id=attempt.arb_id
            )
            sig1.client_order_id = attempt.leg1.order_id  # Link to hedge manager
            
            sig2 = SignalEvent(
                strategy_id="arbitrage",
                symbol=symbol,
                side=leg2_side,
                quantity=qty,
                price=leg2_price,
                exchange=leg2_exchange,
                arb_id=attempt.arb_id
            )
            sig2.client_order_id = attempt.leg2.order_id
        else:
            # Fallback without hedge manager (NOT RECOMMENDED)
            logger.warning("⚠️ No HedgeManager - executing WITHOUT leg tracking!")
            sig1 = SignalEvent(
                strategy_id="arbitrage",
                symbol=symbol,
                side=leg1_side,
                quantity=qty,
                price=leg1_price,
                exchange=leg1_exchange
            )
            sig2 = SignalEvent(
                strategy_id="arbitrage",
                symbol=symbol,
                side=leg2_side,
                quantity=qty,
                price=leg2_price,
                exchange=leg2_exchange
            )
        
        logger.info(f"🚀 EXECUTING ARBITRAGE: {leg1_side} {leg1_exchange} / {leg2_side} {leg2_exchange}")
        
        # Send BOTH legs simultaneously
        await self.queue.put(sig1)
        await self.queue.put(sig2)
        
        self.last_trade_time = time.time()

    async def on_fill(self, event: FillEvent):
        """Handle fill events - forward to hedge manager."""
        if self._hedge_manager and hasattr(event, 'client_order_id'):
            await self._hedge_manager.on_fill(
                order_id=event.client_order_id,
                filled_qty=event.quantity,
                fill_price=event.price
            )
        
    async def on_regime_change(self, event):
        pass

