"""
Avellaneda-Stoikov Market Making Strategy.

Dynamic bid/ask quoting based on:
- Inventory risk (penalize holding)
- Volatility (EWMA estimated)
- Order arrival rate (kappa)
- Spread optimization
"""

import asyncio
import logging
import time
import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from core.events import MarketEvent, SignalEvent, FillEvent, RegimeEvent
from data.shadow_book import ShadowOrderBook
from data.candle_provider import CandleProvider
from config.settings import settings  # Import settings
from analysis.vpin_calculator import VPINCalculator, VPINState
from strategies.glt_quote_engine import GLTQuoteEngine, GLTParams
from analysis.hmm_regime_detector import RegimeSupervisorHMM
from execution.order_manager import get_order_manager
from execution.inventory_hedger import get_inventory_hedger

logger = logging.getLogger("Strategy.AvellanedaStoikov")


@dataclass
class Quote:
    """A two-sided quote."""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: float


def inventory_skew_tanh(
    inventory: float,
    max_inventory: float,
    lambda_param: float = 2.5
) -> float:
    """
    Calculate inventory skew using hyperbolic tangent.
    
    Tanh provides stronger pressure at extremes than linear skew:
    - At 10% inventory: Linear=0.10, Tanh=-0.90 (strong buy pressure)
    - At 50% inventory: Linear=0.50, Tanh= 0.00 (neutral)
    - At 90% inventory: Linear=0.90, Tanh=+0.97 (aggressive sell)
    
    Args:
        inventory: Current signed inventory (-max to +max)
        max_inventory: Maximum inventory limit
        lambda_param: Steepness parameter (2.0-3.0 recommended)
        
    Returns:
        Skew in [-1, 1]: Positive = sell pressure, Negative = buy pressure
    """
    if max_inventory <= 0:
        return 0.0
    
    # Normalize to [-1, 1] range
    normalized_q = inventory / max_inventory
    normalized_q = max(-1.0, min(1.0, normalized_q))  # Clamp
    
    # Apply tanh transformation
    # tanh(λ * q) gives S-curve with steepness controlled by λ
    return math.tanh(lambda_param * normalized_q)


def get_regime_spread_multiplier(regime: str) -> float:
    """
    Get spread multiplier based on market regime.
    
    Regime-aware spreads from research:
    - Mean Reversion: 0.70x (tighten - predictable, win on fills)
    - Trending: 1.50x (widen - adverse selection risk)
    - Choppy/Uncertain: 1.10x (slight widen - moderate uncertainty)
    
    Args:
        regime: Current regime string from HMM
        
    Returns:
        Spread multiplier (0.7 to 1.5)
    """
    regime_multipliers = {
        'MEAN_REVERTING': 0.70,
        'TRENDING': 1.50,
        'CHOPPY': 1.10,
        'UNCERTAIN': 1.10,
        'HIGH_VOLATILITY': 1.30,
        'LOW_VOLATILITY': 0.80,
        'WARMUP': 1.00
    }
    return regime_multipliers.get(regime, 1.0)


class AvellanedaStoikovStrategy:
    """
    Avellaneda-Stoikov optimal market making strategy.
    
    Features:
    - EWMA volatility estimation (regime-conditional)
    - Inventory skew (quote adjustment based on position)
    - Dynamic spread based on volatility and arrival rate
    - Adverse selection filter (pause on volume spikes)
    - Hysteresis to avoid API spam (not fixed timer)
    """
    
    # Configuration
    DEFAULT_GAMMA = settings.AS_GAMMA
    DEFAULT_KAPPA = settings.AS_KAPPA
    EWMA_SPAN = 20           # EWMA span for volatility
    MAX_INVENTORY = 10.0     # Max position (in base asset)
    QUOTE_REFRESH_THRESHOLD = 0.0010  # 10 bps price move to refresh quotes
    # Fee-aware minimum spread: 2 * (maker_fee + taker_fee) + profit margin
    # This ensures we profit even if we pay taker fee on close (worst case)
    MIN_SPREAD_BPS = (2 * settings.MAKER_FEE_BPS) + (2 * settings.TAKER_FEE_BPS) + settings.MIN_PROFIT_BPS
    # Add 10% buffer to Min Notional to avoid rejections
    MIN_NOTIONAL_USD = settings.MIN_NOTIONAL_USD * 1.1
    
    def __init__(
        self,
        event_queue,
        symbols: List[str] = None, # Optional, defaults to settings.TRADING_SYMBOLS
        base_quantity: float = 0.01,
        gamma: float = DEFAULT_GAMMA,
        max_inventory: float = MAX_INVENTORY,
        shadow_book: Optional[ShadowOrderBook] = None,
        candle_provider: Optional[CandleProvider] = None,
        regime_supervisor: Optional[RegimeSupervisorHMM] = None
    ):
        self.queue = event_queue
        self.symbols = [s.lower() for s in (symbols or settings.TRADING_SYMBOLS)]
        self.base_quantity = base_quantity
        self.gamma = gamma
        self.max_inventory = max_inventory
        
        # Dynamic Kappa Config
        self.target_fills_per_min = 2.0  # Target: 2 fills/min
        self.min_kappa = 0.05            # High aggression (tight spreads)
        self.max_kappa = 5.0             # Low aggression (wide spreads)
        self.shadow_book = shadow_book
        self.candle_provider = candle_provider
        self.regime_supervisor = regime_supervisor
        
        # GLT Quote Engine (optional, enable via use_glt=True)
        self.use_glt = True  # GLT ENABLED!
        self.glt_engine = GLTQuoteEngine()
        
        # VPIN logging timer
        self._last_vpin_log: Dict[str, float] = {}
        
        # OrderManager for accurate position tracking
        self._order_manager = None  # Lazy init to avoid async in __init__
        
        # State Manager for persistence (Analytics, HMM, VPIN)
        from utils.state_manager import StateManager
        self.state_manager = StateManager()
        
        # State per symbol
        self._state: Dict[str, dict] = {}
        for sym in self.symbols:
            # 1. Default State
            default_state = {
                'inventory': 0.0,
                'last_mid': 0.0,
                'volatility': 0.001,  # Initial estimate
                'kappa': self.DEFAULT_KAPPA,
                'regime': 'UNCERTAIN',
                'last_quote_time': 0.0,
                'last_quote_mid': 0.0,
                'last_fill_time': 0.0,  # NEW: For fill cooldown
                'returns': deque(maxlen=100),
                'fill_times': deque(maxlen=50),  # For kappa estimation
                'paused': False,  # Adverse selection pause
                'vpin': VPINCalculator(sym),  # VPIN toxic flow detector
                'last_warning_time': 0.0,  # Log rate limiter
                'latency_history': deque(maxlen=20), # Latency tracking (ms)
                # Visualization Data
                'my_bid': 0.0,
                'my_ask': 0.0,
                'market_bid': 0.0,
                'market_ask': 0.0
            }
            
            # 2. Limit Restore (Overlay persisted metrics)
            saved = self.state_manager.get_symbol_state(sym)
            if saved:
                logger.info(f"[{sym}] Restoring state: Vol={saved.get('volatility')}, Kappa={saved.get('kappa')}")
                if 'volatility' in saved: default_state['volatility'] = saved['volatility']
                if 'kappa' in saved: default_state['kappa'] = saved['kappa']
                if 'regime' in saved: default_state['regime'] = saved['regime']
                # Restore Lists (converted to deque)
                if 'returns' in saved: default_state['returns'] = deque(saved['returns'], maxlen=100)
                if 'fill_times' in saved: default_state['fill_times'] = deque(saved['fill_times'], maxlen=50)

            self._state[sym] = default_state
            
        # Flag for background tasks
        self._tasks_started = False
    
    # Cooldown period after a fill (seconds) - prevents revenge trading
    FILL_COOLDOWN_SECONDS = 2.0
    
    async def on_tick(self, event: MarketEvent):
        """Process each tick - recalculate quotes if needed."""
        tick_start_time = time.perf_counter() # Latency Start
        
        # Lazy start background tasks (ensure loop is running)
        if not self._tasks_started:
            asyncio.create_task(self.state_manager.start_auto_save())
            self._tasks_started = True

        symbol = event.symbol.lower()
        if symbol not in self.symbols:
            return
        
        state = self._state[symbol]
        
        # Feed VPIN calculator with trade data
        if event.volume > 0 and event.side:
            vpin = state['vpin']
            vpin.on_trade(event.price, event.volume, event.side)
            # Sync volatility for BVC
            vpin.set_volatility(state['volatility'])
        
        # Update regime from HMM (throttled to avoid blocking loop)
        now = time.time()
        if self.regime_supervisor and state['volatility'] > 0:
            if now - state.get('last_hmm_check', 0) > settings.HMM_UPDATE_INTERVAL:
                log_ret = math.log(event.price / state['last_mid']) if state['last_mid'] > 0 else 0
                
                # This predict() call can be CPU heavy - hence the throttle
                regime, conf = self.regime_supervisor.detectors[symbol].predict(
                    log_ret, state['volatility'], 1.0
                )
                
                if regime.value not in ['WARMUP']:
                    state['regime'] = regime.value
                    
                state['last_hmm_check'] = now
        
        # Update returns for volatility
        if state['last_mid'] > 0:
            ret = math.log(event.price / state['last_mid'])
            state['returns'].append(ret)
        
        state['last_mid'] = event.price
        
        # Update volatility estimate (EWMA)
        self._update_volatility(symbol)
        
        # Check fill cooldown (prevent revenge trading)
        if time.time() - state['last_fill_time'] < self.FILL_COOLDOWN_SECONDS:
            return  # Wait before quoting again after a fill
        
        # Check if quote refresh needed (hysteresis)
        if not self._should_refresh_quote(symbol, event.price):
            return
        
        # Check adverse selection filters
        if self._check_adverse_selection(symbol):
            if not state['paused']:
                logger.info(f"[{symbol}] Adverse selection detected - pausing quotes")
                state['paused'] = True
            return
        
        state['paused'] = False
        
        # Staleness check - don't quote on stale order book data
        if self.shadow_book and self.shadow_book.is_stale(symbol, max_age_seconds=2.0):
            logger.warning(f"[{symbol.upper()}] Order book stale - pausing quotes")
            return
        
        # Warmup check - need enough data for reliable volatility estimate
        if len(state['returns']) < 20:
            logger.debug(f"[{symbol}] Warming up ({len(state['returns'])}/20 samples)")
            return
        
        # ===================================================================
        # CRITICAL: Get inventory from EXCHANGE (OrderManager), not local state!
        # Local state['inventory'] can be corrupted by ghost fills from previous sessions.
        # ===================================================================
        exchange_inventory = state['inventory']  # Fallback to local
        if self._order_manager is None:
            self._order_manager = get_order_manager()
        
        if self._order_manager:
            try:
                position = await self._order_manager.get_position(symbol)
                if position:
                    exchange_inventory = position.quantity
                    # Also sync local state to prevent drift
                    state['inventory'] = exchange_inventory
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to get exchange position: {e}")
        
        # Update Persistence (every 10s)
        self._update_persistence(symbol, state)
        
        # EMERGENCY KILL SWITCH: Stop quoting if position is dangerously large
        # This kicks in at 1000x the normal limit to allow recovery of inherited positions
        # Standard limits (1x) are handled smoothly in _calculate_sizes via reduce-only quoting
        emergency_limit_usd = settings.MAX_POSITION_USD * 1000
        inventory_usd = abs(exchange_inventory) * event.price
        
        # AUTO-HEDGE: Before kill switch, try to hedge the position
        if settings.ENABLE_AUTO_HEDGE and inventory_usd > settings.MAX_POSITION_USD * settings.HEDGE_THRESHOLD_PCT:
            try:
                hedger = get_inventory_hedger(self.queue)
                await hedger.check_and_hedge(symbol, exchange_inventory, event.price)
            except Exception as e:
                logger.error(f"Auto-hedge failed: {e}")
        
        if inventory_usd > emergency_limit_usd:
             logger.critical(f"🚨 KILL SWITCH TRIGGERED for {symbol}: Position ${inventory_usd:,.0f} > Limit ${emergency_limit_usd:,.0f}")
             state['paused'] = True
             return

        elif inventory_usd > settings.MAX_POSITION_USD * 1.5:
             logger.warning(
                f"⚠️ [{symbol.upper()}] Heavy Position ${inventory_usd:.0f} > Limit ${settings.MAX_POSITION_USD:.0f} - REDUCE ONLY MODE ACTIVE"
             )
        
        # Calculate optimal quote
        quote = await self._calculate_quote(symbol, event.price)
        
        if quote:
            await self._submit_quote(quote)
            state['last_quote_time'] = time.time()
            state['last_quote_mid'] = event.price
            state['my_bid'] = quote.bid_price
            state['my_ask'] = quote.ask_price
            # Approximate market spread from event (if we had L2 here it would be better, but price is mid)
            # We'll rely on the visualizer to use mid+/-spread or just capture what we can.
            # Actually, let's grab the best bid/ask if we have them from the book?
            # For now, store the mid.
            state['market_mid'] = event.price
            
        # Record Latency
        latency_ms = (time.perf_counter() - tick_start_time) * 1000
        state['latency_history'].append(latency_ms)
        
    def _update_kappa(self, symbol: str):
        """
        Dynamic Kappa Adjustment (The 'Brain').
        
        Logic:
        - Kappa controls risk aversion (Higher Kappa = Wider Spreads = Less Fills).
        - If Fill Rate > Target: Market is active/we are too tight -> INCREASE Kappa (Slow down).
        - If Fill Rate < Target: We are missing moves -> DECREASE Kappa (Speed up).
        """
        state = self._state[symbol]
        fill_times = state['fill_times']
        
        # 1. Calculate Fill Rate (fills per minute)
        now = time.time()
        # Remove fills older than 60s from calculation view (but keep in deque for history)
        recent_fills = [t for t in fill_times if now - t <= 60.0]
        fill_rate = len(recent_fills)
        
        # 2. Adjust Kappa
        current_kappa = state['kappa']
        adjustment = 0.0
        
        if fill_rate > self.target_fills_per_min:
             # Too many fills -> Increase Kappa (Widen)
             adjustment = 0.05
        elif fill_rate < self.target_fills_per_min:
             # Too few fills -> Decrease Kappa (Tighten)
             adjustment = -0.05
             
        # Apply & Clamp
        new_kappa = max(self.min_kappa, min(self.max_kappa, current_kappa + adjustment))
        
        if abs(new_kappa - current_kappa) > 0.01:
             state['kappa'] = new_kappa
             # Log only significant changes (e.g. crossing integer boundaries) to avoid spam
             if int(new_kappa * 10) != int(current_kappa * 10):
                 logger.info(f"[{symbol}] Dynamic Kappa: {current_kappa:.2f} -> {new_kappa:.2f} (Rate: {fill_rate}/min)")
    
    def _update_persistence(self, symbol: str, state: dict):
        """Update persistent state manager with latest volatile metrics."""
        now = time.time()
        # Throttle updates to once per 10s
        if now - state.get('_last_persist_time', 0) < 10.0:
            return
            
        try:
            # Snapshot simple types
            snapshot = {
                'volatility': state['volatility'],
                'gamma': self.gamma,  # Added for dashboard visibility
                'kappa': state['kappa'],
                'regime': state['regime'],
                # Convert deques to lists for JSON serialization
                'returns': list(state['returns']),
                'fill_times': list(state['fill_times']),
                'latency_last': state['latency_history'][-1] if state['latency_history'] else 0,
                'latency_avg': sum(state['latency_history'])/len(state['latency_history']) if state['latency_history'] else 0,
                'my_bid': state.get('my_bid', 0.0),
                'my_ask': state.get('my_ask', 0.0),
                'market_mid': state.get('market_mid', 0.0)
            }
            
            self.state_manager.update_symbol_state(symbol, snapshot)
            state['_last_persist_time'] = now
        except Exception as e:
            logger.warning(f"Failed to persist state for {symbol}: {e}")

    async def on_fill(self, event: FillEvent):
        """Handle fill events - update inventory."""
        symbol = event.symbol.lower()
        if symbol not in self.symbols:
            return
        
        state = self._state[symbol]
        
        # Update inventory
        if event.side == 'BUY':
            state['inventory'] += event.quantity
        else:
            state['inventory'] -= event.quantity
        
        # Record fill time for kappa estimation AND cooldown
        now = time.time()
        state['fill_times'].append(now)
        state['last_fill_time'] = now  # For cooldown
        self._update_kappa(symbol)
        
        logger.info(
            f"[{symbol}] Fill: {event.side} {event.quantity} @ {event.price}, "
            f"Inventory: {state['inventory']:.4f}"
        )
    
    async def on_regime_change(self, event: RegimeEvent):
        """Adjust strategy for regime changes."""
        symbol = event.symbol.lower()
        if symbol not in self.symbols:
            return
        
        state = self._state[symbol]
        old_regime = state['regime']
        state['regime'] = event.regime
        
        logger.info(f"[{symbol}] Regime: {old_regime} -> {event.regime}")
        
        # Pause in strong trends (inventory death spiral risk)
        # UPDATE: We no longer PAUSE globally. We use asymmetric quoting in _calculate_quote
        # to cut only the dangerous side.
        if event.regime == 'TRENDING':
            # state['paused'] = True  <-- DISABLED, we want to trade the safe side
            logger.info(f"[{symbol}] Trending market detected - Switching to Asymmetric Quoting")
            
            if abs(state['inventory']) > 0.0001:
                direction = 'LONG' if state['inventory'] > 0 else 'SHORT'
                logger.warning(
                    f"⚠️ [{symbol}] Trending with {direction} inventory={state['inventory']:.4f}. "
                    f"Monitoring for safe exit opportunities."
                )
        
        elif event.regime in ('MEAN_REVERTING', 'UNCERTAIN'):
            state['paused'] = False
            logger.info(f"[{symbol}] Resuming MM in {event.regime} regime")
    
    async def _calculate_quote(self, symbol: str, mid_price: float) -> Optional[Quote]:
        """Calculate optimal bid/ask using Avellaneda-Stoikov."""
        
        # Get state first (needed for volatility in imbalance calculation)
        state = self._state[symbol]
        
        # --- CRITICAL: Use true Mid Price from ShadowBook if available ---
        # aggTrade gives last trade price, but we need the L2 mid price
        # for accurate reservation price calculation
        true_mid = mid_price
        imbalance_skew = 0.0
        
        if self.shadow_book:
            book_mid = self.shadow_book.get_mid_price(symbol)
            if book_mid:
                true_mid = book_mid
                logger.debug(f"[{symbol}] Using book mid: {book_mid:.2f} (trade: {mid_price:.2f})")
                
                # --- ORDER BOOK IMBALANCE ALPHA ---
                # If bids >> asks, price likely to rise -> shift our quotes UP
                # If asks >> bids, price likely to fall -> shift our quotes DOWN
                imbalance = self.shadow_book.get_imbalance(symbol, levels=5)
                if imbalance is not None:
                    # Scale imbalance (-1 to +1) by a fraction of current volatility
                    # This shifts reservation price towards the "heavy" side
                    imbalance_skew = imbalance * state['volatility'] * true_mid * 0.5
                    logger.debug(f"[{symbol}] Imbalance: {imbalance:.2f}, skew: ${imbalance_skew:.2f}")
        # --------------------------------------------------------------------------
        # 3. Calculate Optimal Spread (Avellaneda-Stoikov)
        # --------------------------------------------------------------------------
        # Symmetric spread around reservation price
        # --------------------------------------------------------------------------
        # 3. Calculate Optimal Spread (Avellaneda-Stoikov or GLT)
        # --------------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        # Model parameters
        sigma = state['volatility']  # Estimated volatility
        kappa = state['kappa']       # Order arrival rate
        
        # Get inventory from OrderManager (synced with exchange) if available
        # Fall back to local state if OrderManager not initialized
        if self._order_manager is None:
            self._order_manager = get_order_manager()
        
        try:
            # Read from OrderManager - this is the source of truth after reconciliation
            position = await self._order_manager.get_position(symbol)
            
            if position:
                q = position.quantity
            else:
                q = 0.0 # Default to 0 if no position found
            # Sync local state with OrderManager
            if abs(q - state['inventory']) > 0.001:
                logger.debug(f"[{symbol}] Syncing inventory: local={state['inventory']:.4f} -> OM={q:.4f}")
                state['inventory'] = q
        except Exception as e:
            logger.debug(f"[{symbol}] Using local inventory (OM error: {e})")
            q = state['inventory']  # Fallback to local state
        
        # Skip quoting the side that would increase inventory beyond limit
        # Calculate USD-based max units for this symbol (dynamic per symbol price)
        max_units = settings.MAX_POSITION_USD / true_mid if true_mid > 0 else self.max_inventory
        
        # Use the USD-based maximum units as the effective limit
        # This fixes the issue where legacy 'max_inventory=1.0' (from main.py default) was capping XRP/SOL at 1 unit
        effective_limit = max_units
        
        # CRITICAL FIX: Logic was inverted!
        # If Inventory > Limit (Too Long) -> Stop Buying -> Skip Bid
        skip_bid = state['inventory'] >= effective_limit
        # If Inventory < -Limit (Too Short) -> Stop Selling -> Skip Ask
        skip_ask = state['inventory'] <= -effective_limit
        
        # CRITICAL: Hard enforcement of inventory limits
        if skip_bid or skip_ask:
            direction = 'LONG' if skip_bid else 'SHORT'
            usd_value = abs(q) * true_mid
            # Log what we are keeping (Opposite of what we skip)
            # If skip_bid (Stop Buy), we are quoting ASK (Sell) to reduce Long
            quote_side = 'ASK' if skip_bid else 'BID'
            
            # Rate limit warnings to avoid log spam (every 5s)
            now = time.time()
            if now - state.get('last_warning_time', 0) > 5.0:
                logger.warning(
                    f"🛑 [{symbol}] INVENTORY LIMIT HIT! {q:.2f} units (${usd_value:.0f}) "
                    f"max={effective_limit:.1f} units (${settings.MAX_POSITION_USD:.0f}) "
                    f"- Only quoting {quote_side} to reduce {direction}"
                )
                state['last_warning_time'] = now
        
        # --- QUOTE CALCULATION ---
        # 1. FEE FLOOR (The "Don't Pay to Trade" Rule)
        # Ensure spread covers 2x Maker Fee + Min Profit
        # We calculate this first to ensure it applies to all logic
        maker_fee_pct = settings.MAKER_FEE_BPS / 10000
        min_profit_pct = settings.MIN_PROFIT_BPS / 10000
        
        # Minimum half-spread to break even + profit
        # If we capture spread, we pay maker fee on entry and exit (2x)
        min_half_spread = true_mid * (maker_fee_pct + min_profit_pct)

        if self.use_glt:
            # ========== NEW GLT ENGINE (Infinite Horizon) ==========
            reserv_bid, reserv_ask = self.glt_engine.compute_quotes(
                symbol=symbol,
                mid_price=true_mid,
                inventory=q,
                volatility=sigma
            )
            # Center price before spread application
            reservation_price = (reserv_bid + reserv_ask) / 2
            optimal_half_spread = (reserv_ask - reserv_bid) / 2
        else:
            # ========== LEGACY LINEAR SKEW (Finite Horizon) - FIXED ==========
            dynamic_gamma = self.gamma * (1 + sigma * 50)
            optimal_spread = dynamic_gamma * (sigma ** 2) * (1.0) + (2 / dynamic_gamma) * math.log(1 + dynamic_gamma / kappa)
            optimal_half_spread = optimal_spread / 2
            reservation_price = true_mid  # Will be adjusted by skew below

        # 2. TANH SKEW (Aggressive Inventory Management) - FIXED TO BE ASYMMETRIC & CALMER
        # Apply skew to Reservation Price to shift quotes away from heavy side
        max_units = settings.MAX_POSITION_USD / true_mid if true_mid > 0 else 10.0
        
        # FIX: Reduced lambda from 2.5 to 1.0 (Less aggressive)
        lambda_inv = 1.0
        inventory_pct = q / max_units
        skew_factor = math.tanh(lambda_inv * inventory_pct)
        
        # FIX: Standardized Reservation Price Shift (Smooth & Classic)
        # Instead of hacking margins with 0.2 multipliers, we shift the entire bracket.
        # If LONG (skew > 0) -> Shift price DOWN -> Lower Bid (Buy less), Lower Ask (Sell faster)
        # If SHORT (skew < 0) -> Shift price UP -> Higher Bid (Buy faster), Higher Ask (Sell less)
        
        skew_intensity = 1.0  # 100% of half-spread at max skew
        reservation_shift = -skew_factor * optimal_half_spread * skew_intensity
        reservation_price += reservation_shift

        # We remove the old manual adjustments
        bid_adjustment = 0.0
        ask_adjustment = 0.0

        # 3. APPLY SPREADS & FLOOR
        
        # --- PROFIT PROTECTION: USD Minimum Profit Floor ---
        # Ensure we cover fees + fixed USD profit per trade (conservative logic using base_qty)
        est_fee_rate = getattr(settings, 'ESTIMATED_FEE_BPS', 5) / 10000.0
        # Cost to open + close 1 UNIT (approx)
        fees_per_unit = true_mid * est_fee_rate * 2
        min_profit_target = getattr(settings, 'MIN_PROFIT_PER_TRADE_USD', 0.05)
        
        
        # Required Spread per unit = Fees_per_unit + (Profit_Target / Quantity)
        # FIX: Use MIN_NOTIONAL sized trade, not base_quantity
        # This reflects the true cost of the smallest allowed trade
        min_trade_qty = self.MIN_NOTIONAL_USD / true_mid if true_mid > 0 else self.base_quantity
        expected_trade_size = min_trade_qty
        
        required_spread_abs = fees_per_unit + (min_profit_target / expected_trade_size)
        required_half_spread = required_spread_abs / 2
        
        if required_half_spread > max(optimal_half_spread, min_half_spread):
             logger.debug(f"[{symbol}] Widening spread: {max(optimal_half_spread, min_half_spread)*2:.4f} -> {required_spread_abs:.4f} (Profit Protect ${min_profit_target})")
        
        final_half_spread = max(optimal_half_spread, min_half_spread, required_half_spread)
        
        # Toxic/Volatile Market Adjustment (VPIN-Based)
        # Continuous scaling: Spread Multiplier = 1.0 + VPIN
        # VPIN 0.0 (Calm) -> 1.0x Spread
        # VPIN 0.5 (Active) -> 1.5x Spread
        # VPIN 1.0 (Toxic) -> 2.0x Spread
        current_vpin = state['vpin'].get_vpin() or 0.0
        spread_multiplier = 1.0 + current_vpin
        if spread_multiplier > 1.05:
             logger.debug(f"[{symbol}] Volatility Widening: {spread_multiplier:.2f}x (VPIN {current_vpin:.2f})")
        final_half_spread *= spread_multiplier
        
        # Apply asymmetric adjustments
        bid_price = reservation_price - final_half_spread + bid_adjustment
        ask_price = reservation_price + final_half_spread + ask_adjustment
        
        # --- CRITICAL FIX: Fee/Profit Floor Enforcement ---
        # Skew can push prices too close to mid, eating into fees/profit.
        # We enforce that quotes must be at least 'min_half_spread' away from mid.
        # This prevents "dust trades" where we essentially scalp 0 or negative after fees.
        # Exception: If inventory is extremely critical (>80%), we allow aggressive closing.
        
        # Check inventory stress
        inventory_stress = abs(q) / effective_limit
        is_emergency = inventory_stress > 0.8
        
        if not is_emergency:
            # Clamp Bid: Must be <= Mid - MinSpread
            # ie. cannot be closer to mid than the fee floor
            bid_ceiling = true_mid - min_half_spread
            if bid_price > bid_ceiling:
                logger.debug(f"[{symbol}] Clamping Bid to Profit Floor: {bid_price:.2f} -> {bid_ceiling:.2f}")
                bid_price = bid_ceiling
                
            # Clamp Ask: Must be >= Mid + MinSpread
            # ie. cannot be closer to mid than the fee floor
            ask_floor = true_mid + min_half_spread
            if ask_price < ask_floor:
                 logger.debug(f"[{symbol}] Clamping Ask to Profit Floor: {ask_price:.2f} -> {ask_floor:.2f}")
                 ask_price = ask_floor

        # SANITY CHECK: Prevent negative prices
        if bid_price <= 0:
            logger.warning(
                f"[{symbol}] Negative Bid Price calculated (${bid_price:.2f}) - "
                f"Res: {reservation_price:.2f}, HalfSpread: {final_half_spread:.2f}. "
                f"Clamping to 0.01"
            )
            bid_price = 0.01

        # 4. ADJUST SIZES (Legacy Skew + Min Notional)
        bid_size, ask_size = self._calculate_sizes(symbol, q, true_mid)
        
        # 5. ENFORCE INVENTORY LIMITS - skip sides that would increase position
        if skip_bid:
            bid_size = 0  # At max SHORT - don't sell more (would increase short)
        if skip_ask:
            ask_size = 0  # At max LONG - don't buy more (would increase long)
        
        # 6. ASYMMETRIC QUOTING (Toxic Flow/Trend Protection)
        # Instead of pausing, just cut the dangerous side
        if self.shadow_book:
            imbalance = self.shadow_book.get_imbalance(symbol, levels=5) or 0
            
            # Strong UP Trend (Bids >> Asks) -> Don't Sell (Ask)
            if imbalance > 0.6 or (state['regime'] == 'TRENDING' and imbalance > 0.2):
                ask_size = 0
                logger.debug(f"[{symbol}] Trend UP (Imbal {imbalance:.2f}) - Cutting ASK")
            
            # Strong DOWN Trend (Asks >> Bids) -> Don't Buy (Bid)
            elif imbalance < -0.6 or (state['regime'] == 'TRENDING' and imbalance < -0.2):
                bid_size = 0
        if bid_size == 0 and ask_size == 0:
            return None
        
        # Prevent negative/zero prices
        if bid_price <= 0 or ask_price <= 0:
            logger.error(f"[{symbol}] Invalid quote prices: Bid={bid_price}, Ask={ask_price}")
            return None
            
        return Quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=time.time()
        )
    
    # Tanh skew parameters (from research: λ=2.5 optimal)
    TANH_LAMBDA = 2.5
    TANH_AGGRESSION = 0.5  # How much skew affects size (0.5 = up to 50% adjustment)
    
    def _calculate_sizes(self, symbol: str, inventory: float, price: float) -> Tuple[float, float]:
        """Calculate bid/ask sizes with tanh inventory skew and min notional enforcement."""
        
        # Calculate minimum quantity to meet MIN_NOTIONAL_USD
        min_qty = self.MIN_NOTIONAL_USD / price if price > 0 else self.base_quantity
        
        # Use the larger of base_quantity or min_qty
        base = max(self.base_quantity, min_qty)
        
        # Check for Inventory Excess (Emergency Unwind Sizing)
        inventory_usd = abs(inventory * price)
        max_position_usd = settings.MAX_POSITION_USD
        excess_ratio = inventory_usd / max_position_usd if max_position_usd > 0 else 0
        
        # If we are significantly over the limit (e.g. > 110%), we need to dump faster
        unwind_qty = 0.0
        if excess_ratio > 1.1:
            excess_usd = inventory_usd - max_position_usd
            # Target clearing 10% of the excess per trade, or at least $50 USD
            target_unwind_usd = max(excess_usd * 0.10, 50.0) 
            unwind_qty = target_unwind_usd / price if price > 0 else 0
            
            # Cap unwind size to avoid market impact (e.g. max $500 per clip)
            max_clip_usd = 500.0
            if target_unwind_usd > max_clip_usd:
                unwind_qty = max_clip_usd / price
                
            logger.info(f"[{symbol}] Emergency Unwind Active (Excess ${excess_usd:.0f}). Boosting size to {unwind_qty:.4f}")

        # Determine Precision / Lot Size Steps
        # Heuristic based on price bucket (should ideally come from exchange info)
        # TODO: Get actual precision from ExchangeManager
        if price > 1000:
            precision_step = 0.001 
        elif price > 10:
            precision_step = 0.01 
        else:
            precision_step = 1.0 if not symbol.endswith('usdt') else 0.1 # Fallback
            
        # FIX: Remove hardcoded 'solusdt' check which was dangerous
        # Sol tick size is 0.01 or 0.001 usually, not 1.0! 
        # For now, consistent small steps are safer (exchange will round it)
        precision_step = 0.001 # Safe default for most crypto
            
        # Smart Rounding for Min Notional
        # We need strictly >= RISK_MIN_NOTIONAL_USD (safe buffer). 
        target_min_usd = settings.RISK_MIN_NOTIONAL_USD
        min_raw = target_min_usd / price if price > 0 else self.base_quantity
        # Ceiling division to next lot step
        min_qty = math.ceil(min_raw / precision_step) * precision_step
        
        # Calculate Base Size (Larger of config base or min_notional)
        base = max(self.base_quantity, min_qty)
        
        # Round the base to precision (it should be already, but to be safe)
        base = math.ceil(base / precision_step) * precision_step
        
        # ========== TANH-BASED SKEW (replaces linear) ==========
        # Dynamic MAX units based on USD limit
        max_units = settings.MAX_POSITION_USD / price if price > 0 else 10.0
        
        # Tanh skew provides stronger pressure at extremes
        skew = inventory_skew_tanh(inventory, max_units, self.TANH_LAMBDA)
        
        bid_size = base * (1 - skew * 0.5)  # Reduce when long
        ask_size = base * (1 + skew * 0.5)  # Reduce when short
        
        # BOOST UNWIND SIDE
        if excess_ratio > 1.1:
            if inventory > 0: # Long -> Boost Ask
                ask_size = max(ask_size, unwind_qty)
            else:             # Short -> Boost Bid
                bid_size = max(bid_size, unwind_qty)
        
        # Ensure minimum notional is met
        bid_size = max(bid_size, min_qty)
        ask_size = max(ask_size, min_qty)
        
        # HARD INVENTORY LIMIT: Disable one side when position too large
        if inventory_usd > max_position_usd:
            if inventory > 0:
                # Too LONG - disable bids, only quote asks (to reduce position)
                bid_size = 0
                logger.debug(f"[{symbol}] Inventory limit reached (LONG ${inventory_usd:.0f}) - BID disabled")
            else:
                # Too SHORT - disable asks, only quote bids (to reduce position)
                ask_size = 0
                logger.debug(f"[{symbol}] Inventory limit reached (SHORT ${inventory_usd:.0f}) - ASK disabled")
        
        # Round to appropriate precision
        if price > 1000:
            bid_size = round(bid_size, 3)
            ask_size = round(ask_size, 3)
        # Special case for SOL which often rejects decimals if step size is 1
        elif symbol == 'solusdt':
            bid_size = int(bid_size)
            ask_size = int(ask_size)
        elif price > 10:
            bid_size = round(bid_size, 2)
            ask_size = round(ask_size, 2)
        else:
            bid_size = round(bid_size, 1)
            ask_size = round(ask_size, 1)
        
        return bid_size, ask_size
    
    def _update_volatility(self, symbol: str):
        """Update EWMA volatility estimate."""
        state = self._state[symbol]
        returns = list(state['returns'])
        
        if len(returns) < 5:
            return
        
        # EWMA variance
        alpha = 2 / (self.EWMA_SPAN + 1)
        variance = returns[0] ** 2
        
        for r in returns[1:]:
            variance = alpha * (r ** 2) + (1 - alpha) * variance
        
        state['volatility'] = math.sqrt(variance)
    
    def _update_kappa(self, symbol: str):
        """Estimate order arrival rate from fill history."""
        state = self._state[symbol]
        fill_times = list(state['fill_times'])
        
        if len(fill_times) < 3:
            return
        
        # Calculate average inter-arrival time
        intervals = [
            fill_times[i] - fill_times[i-1]
            for i in range(1, len(fill_times))
        ]
        
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval > 0:
            state['kappa'] = 1 / avg_interval
    
    def _should_refresh_quote(self, symbol: str, current_price: float) -> bool:
        """Check if quote should be refreshed (hysteresis, not timer)."""
        state = self._state[symbol]
        
        # CRITICAL: Don't quote faster than 2.0s to prevent order flooding (-2025 error)
        # 1. THROTTLE: Do not quote faster than once every 2 seconds
        # This prevents API bans and order limit errors
        if time.time() - state['last_quote_time'] < 2.0:
            return False
        
        # First quote
        if state['last_quote_mid'] == 0:
            return True
        
        # Price moved enough
        price_change = abs(current_price - state['last_quote_mid']) / state['last_quote_mid']
        if price_change >= self.QUOTE_REFRESH_THRESHOLD:
            return True
        
        # Time-based fallback (60s max)
        if time.time() - state['last_quote_time'] > 60:
            return True
        
        return False
    
    def _check_adverse_selection(self, symbol: str) -> bool:
        """Check for adverse selection conditions."""
        state = self._state[symbol]
        
        # Pause in trending markets
        if state['regime'] == 'TRENDING':
            return True
        
        # Check for unusual volatility spike
        if state['volatility'] > 0.01:  # 1% instant volatility is unusual
            return True
        
        # VPIN-based toxic flow detection
        vpin: VPINCalculator = state['vpin']
        vpin_state = vpin.get_state()
        
        # Log VPIN status periodically (every 60s) for dashboard
        vpin_value = vpin.get_vpin()
        if time.time() - self._last_vpin_log.get(symbol, 0) > 60:
            if vpin_value is not None:
                logger.info(f"[{symbol.upper()}] VPIN: {vpin_state.value} ({vpin_value:.3f})")
            else:
                logger.info(f"[{symbol.upper()}] VPIN: {vpin_state.value} (warming up)")
            self._last_vpin_log[symbol] = time.time()

        # TOXIC: Full pause
        if vpin_state == VPINState.TOXIC and vpin_value is not None:
            logger.warning(f"[{symbol.upper()}] VPIN TOXIC ({vpin_value:.3f}) - pausing quotes")
            return True
        
        # WARNING: Don't return True (continue quoting), but flag for wider spreads
        # The actual spread widening is handled in _calculate_quote via state
        if vpin_state == VPINState.WARNING and vpin_value is not None:
            logger.warning(f"[{symbol.upper()}] VPIN WARNING ({vpin_value:.3f}) - widening spreads 50%")
            state['vpin_warning'] = True
        else:
            state['vpin_warning'] = False
        
        return False
    
    async def _submit_quote(self, quote: Quote):
        """Submit bid and ask orders."""
        sym = quote.symbol.upper()
        bid_str = f"BID {quote.bid_size}@${quote.bid_price:,.2f}" if quote.bid_size > 0 else "---"
        ask_str = f"ASK {quote.ask_size}@${quote.ask_price:,.2f}" if quote.ask_size > 0 else "---"
        logger.info(f"📊 {sym}: {bid_str} | {ask_str}")
        
        # Submit bid
        if quote.bid_size > 0:
            bid_signal = SignalEvent(
                symbol=quote.symbol,
                side='BUY',
                price=quote.bid_price,
                quantity=quote.bid_size,
                strategy_id='avellaneda_stoikov'
            )
            await self.queue.put(bid_signal)
        
        # Submit ask
        if quote.ask_size > 0:
            ask_signal = SignalEvent(
                symbol=quote.symbol,
                side='SELL',
                price=quote.ask_price,
                quantity=quote.ask_size,
                strategy_id='avellaneda_stoikov'
            )
            await self.queue.put(ask_signal)
    
    def set_shadow_book(self, shadow_book: ShadowOrderBook):
        """Set shadow book for deeper market analysis."""
        self.shadow_book = shadow_book
    
    def set_candle_provider(self, candle_provider: CandleProvider):
        """Set candle provider for historical volatility."""
        self.candle_provider = candle_provider
