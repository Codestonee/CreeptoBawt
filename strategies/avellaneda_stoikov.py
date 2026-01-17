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
import os
import json
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from core.events import MarketEvent, SignalEvent, FillEvent, RegimeEvent, FundingRateEvent
from strategies.base import BaseStrategy
from core.shadow_book import ShadowOrderBook
from core.candle_provider import CandleProvider
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


class AvellanedaStoikovStrategy(BaseStrategy):
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
        super().__init__(event_queue, symbols or settings.TRADING_SYMBOLS)
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
        
        # Configure to use enhanced theta calculation
        self.glt_use_iterative_theta = getattr(settings, 'GLT_USE_ITERATIVE_THETA', True)
        
        # Initialize GLT params for each symbol
        for sym in self.symbols:
            glt_params = self._get_symbol_glt_params(sym)
            
            self.glt_engine.set_params(
                sym,
                glt_params,
                use_iterative_theta=self.glt_use_iterative_theta
            )
            
            logger.info(
                f"[{sym.upper()}] GLT initialized with "
                f"{'iterative' if self.glt_use_iterative_theta else 'quadratic'} theta"
            )
        
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
                'kappa': settings.AS_KAPPA,
                'skip_reason': None,
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
                'trade_pnls': [],  # For gamma adjustment (win rate tracking)
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
    
    def _get_symbol_glt_params(self, symbol: str) -> GLTParams:
        """
        Get tailored GLT parameters based on asset price class.
        
        Adjusts 'delta' (trade unit) and 'k' (spread sensitivity) to prevent
        huge spreads on low-priced assets (like XRP).
        """
        s = symbol.lower()
        
        # Defaults from settings
        glt_A = getattr(settings, 'GLT_A', 10.0)
        glt_gamma = getattr(settings, 'GLT_GAMMA', 0.1)
        
        # Tailored Params based on approximate price
        if 'btc' in s:
            k = 0.3      # ~$100k
            delta = 0.001 # $95
        elif 'eth' in s:
            k = 10.0     # ~$3k
            delta = 0.01  # $30
        elif 'sol' in s or 'bnb' in s or 'bch' in s:
            k = 100.0    # ~$200-600
            delta = 0.1   # $20-60
        elif 'ltc' in s or 'aave' in s:
            k = 500.0    # ~$70-100
            delta = 0.1   # $7-10
        elif 'link' in s or 'uni' in s:
            k = 2000.0   # ~$10-20
            delta = 1.0   # $10-20
        elif 'xrp' in s or 'ada' in s or 'doge' in s or 'matic' in s:
            k = 15000.0  # ~$0.5-2.0
            delta = 10.0  # $5-20
        else:
            # Conservative default for unknown mid-range assets
            k = 50.0
            delta = 0.01
            
        return GLTParams(
            A=glt_A,
            k=k,
            gamma=glt_gamma,
            delta=delta,
            min_spread_bps=self.MIN_SPREAD_BPS
        )

    # Cooldown period after a fill (seconds) - prevents revenge trading
    FILL_COOLDOWN_SECONDS = 2.0
    
    def _check_config_update(self):
        """Check for runtime config updates from Dashboard."""
        now = time.time()
        # Check every 2 seconds
        if now - getattr(self, '_last_config_check', 0) < 2.0:
            return

        self._last_config_check = now
        config_file = "data/runtime_config.json"
        
        if not os.path.exists(config_file):
            return
            
        try:
            # Check modification time
            mtime = os.path.getmtime(config_file)
            if mtime <= getattr(self, '_last_config_mtime', 0):
                return
                
            self._last_config_mtime = mtime
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Apply Updates
            if 'AS_GAMMA' in data: 
                self.gamma = float(data['AS_GAMMA'])
                settings.AS_GAMMA = self.gamma # Sync global
                
            # Kappa is per-symbol in self._state, but we can update defaults
            # or update all symbols if they are close to default?
            # Actually, _update_kappa runs dynamically. 
            # If user manually sets KAPPA in config, we should override the dynamic value?
            # Let's say: Config overwrites dynamic value and resets it.
            if 'AS_KAPPA' in data:
                new_kappa = float(data['AS_KAPPA'])
                # Only update if significantly different (manual override)
                if abs(new_kappa - self.DEFAULT_KAPPA) > 0.01:
                    logger.info(f"🔧 Config Update: Reseting Kappa to {new_kappa}")
                    for sym in self.symbols:
                        self._state[sym]['kappa'] = new_kappa
                    self.DEFAULT_KAPPA = new_kappa
            
            if 'RISK_MAX_POSITION_PER_SYMBOL_USD' in data:
                settings.RISK_MAX_POSITION_PER_SYMBOL_USD = float(data['RISK_MAX_POSITION_PER_SYMBOL_USD'])
            
            if 'RISK_MAX_POSITION_TOTAL_USD' in data:
                settings.RISK_MAX_POSITION_TOTAL_USD = float(data['RISK_MAX_POSITION_TOTAL_USD'])
                
            logger.info("🔧 Runtime Config Reloaded")
            
        except Exception as e:
            logger.error(f"Config reload failed: {e}")

    async def on_tick(self, event: MarketEvent):
        """Process each tick - recalculate quotes if needed."""
        self._check_config_update()
        
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

        # Upgrade to iterative theta after warmup (one-time)
        if (len(state['returns']) >= 50 and 
            self.glt_use_iterative_theta and 
            not state.get('theta_upgraded', False)):
            
            logger.info(f"[{symbol}] Upgrading to iterative theta (warmup complete)")
            
            glt_params = self.glt_engine.params.get(symbol)
            if glt_params:
                self.glt_engine._compute_theta_table_iterative(
                    symbol,
                    glt_params,
                    volatility_estimate=state['volatility']
                )
                state['theta_upgraded'] = True
        
        # Check fill cooldown (prevent revenge trading)
        if time.time() - state['last_fill_time'] < self.FILL_COOLDOWN_SECONDS:
            state['skip_reason'] = f"Fill cooldown ({self.FILL_COOLDOWN_SECONDS}s)"
            return  # Wait before quoting again after a fill
        
        # Check if quote refresh needed (hysteresis)
        if not self._should_refresh_quote(symbol, event.price):
            state['skip_reason'] = "Quote refresh not needed (hysteresis)"
            return
        
        # Check adverse selection filters
        if self._check_adverse_selection(symbol):
            if not state['paused']:
                logger.info(f"[{symbol}] Adverse selection detected - pausing quotes")
                state['paused'] = True
            state['skip_reason'] = "Adverse selection detected"
            return
        
        state['paused'] = False
        state['skip_reason'] = None  # Clear skip reason when quoting
        
        # Staleness check - don't quote on stale order book data
        if self.shadow_book and self.shadow_book.is_stale(symbol, max_age_seconds=2.0):
            logger.warning(f"[{symbol.upper()}] Order book stale - pausing quotes")
            state['skip_reason'] = "Order book stale (>2s)"
            return
        
        # Warmup check - need enough data for reliable volatility estimate
        if len(state['returns']) < 20:
            logger.debug(f"[{symbol}] Warming up ({len(state['returns'])}/20 samples)")
            state['skip_reason'] = f"Warming up ({len(state['returns'])}/20)"
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
    
    def _update_gamma(self, symbol: str):
        """
        Dynamic Gamma Adjustment based on Win Rate.
        
        Logic:
        - Gamma controls spread width (Higher Gamma = Wider Spreads = Less Adverse Selection).
        - If Win Rate > 55%: We're capturing spread well -> DECREASE Gamma (Tighter spreads, more volume).
        - If Win Rate < 45%: We're getting picked off -> INCREASE Gamma (Wider spreads, less adverse selection).
        
        Win is defined as: Total PnL from last N round-trips > 0 (includes fees).
        """
        state = self._state[symbol]
        trade_pnls = state.get('trade_pnls', [])
        
        # Need at least 20 trades to make a meaningful adjustment
        if len(trade_pnls) < 20:
            return
        
        # Calculate win rate from last 50 trades
        recent_pnls = trade_pnls[-50:]
        wins = sum(1 for pnl in recent_pnls if pnl > 0)
        win_rate = wins / len(recent_pnls)
        
        # Adjustment parameters
        target_win_rate_high = 0.55  # Above this, we can be more aggressive
        target_win_rate_low = 0.45   # Below this, we need to widen
        gamma_step = 0.02            # Adjustment step
        min_gamma = 0.1              # Floor (can't go too tight)
        max_gamma = 2.0              # Ceiling (can't go too wide)
        
        current_gamma = self.gamma
        adjustment = 0.0
        
        if win_rate > target_win_rate_high:
            # High win rate -> Decrease gamma (tighter spreads)
            adjustment = -gamma_step
        elif win_rate < target_win_rate_low:
            # Low win rate -> Increase gamma (wider spreads)
            adjustment = gamma_step
        
        # Apply & Clamp
        new_gamma = max(min_gamma, min(max_gamma, current_gamma + adjustment))
        
        if abs(new_gamma - current_gamma) > 0.001:
            self.gamma = new_gamma
            settings.AS_GAMMA = new_gamma  # Sync global
            
            direction = "↓ TIGHTER" if adjustment < 0 else "↑ WIDER"
            logger.info(
                f"[{symbol}] Dynamic Gamma: {current_gamma:.2f} -> {new_gamma:.2f} "
                f"({direction}, WinRate: {win_rate:.1%})"
            )
    
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
                'gamma': self.gamma,
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
        
        # Record realized PnL for gamma adjustment (if available in event)
        realized_pnl = getattr(event, 'realized_pnl', None)
        if realized_pnl is not None:
            trade_pnls = state.get('trade_pnls', [])
            trade_pnls.append(realized_pnl)
            # Keep only last 100 trades
            if len(trade_pnls) > 100:
                state['trade_pnls'] = trade_pnls[-100:]
            else:
                state['trade_pnls'] = trade_pnls
            
            # Update gamma based on win rate
            self._update_gamma(symbol)
        
        logger.info(
            f"[{symbol}] Fill: {event.side} {event.quantity} @ {event.price}, "
            f"Inventory: {state['inventory']:.4f}"
            + (f", PnL: ${realized_pnl:.4f}" if realized_pnl else "")
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
    
    def _check_underwater_position(
        self,
        symbol: str,
        current_inventory: float,
        current_price: float,
        avg_entry_price: float
    ) -> tuple[bool, float]:
        """
        Check if current position is significantly underwater.
        
        Returns:
            (is_underwater, unrealized_pnl_pct)
        """
        if abs(current_inventory) < 0.001 or avg_entry_price <= 0:
            return False, 0.0
        
        # Calculate unrealized PnL percentage
        if current_inventory > 0:  # Long
            pnl_pct = (current_price / avg_entry_price) - 1
        else:  # Short
            pnl_pct = (avg_entry_price / current_price) - 1
        
        # Thresholds from settings
        warning_threshold = getattr(settings, 'POSITION_PNL_WARNING_PCT', -0.03)  # -3%
        critical_threshold = getattr(settings, 'POSITION_PNL_CRITICAL_PCT', -0.05)  # -5%
        
        is_underwater = pnl_pct < critical_threshold
        
        if pnl_pct < warning_threshold:
            logger.warning(
                f"[{symbol.upper()}] Position underwater: "
                f"{current_inventory:+.4f} @ ${avg_entry_price:.2f}, "
                f"Mark: ${current_price:.2f}, PnL: {pnl_pct:.2%}"
            )
        
        return is_underwater, pnl_pct

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
        
        # ========== UNDERWATER POSITION CHECK (Death Spiral Prevention) ==========
        # If position is significantly underwater, switch to emergency exit mode

        if self._order_manager:
            try:
                position = await self._order_manager.get_position(symbol)
                if position and abs(position.quantity) > 0.001:
                    is_underwater, pnl_pct = self._check_underwater_position(
                        symbol=symbol,
                        current_inventory=position.quantity,
                        current_price=true_mid,
                        avg_entry_price=position.avg_entry_price
                    )
                    
                    if is_underwater:
                        logger.critical(
                            f"🚨 [{symbol.upper()}] UNDERWATER POSITION DETECTED! "
                            f"PnL: {pnl_pct:.2%} - EMERGENCY EXIT MODE"
                        )
                        
                        # Calculate emergency exit price (aggressive but not stupid)
                        if position.quantity > 0:  # Long - need to sell
                            # Sell aggressively but within reason
                            emergency_exit_price = true_mid * 0.998  # 0.2% below mid
                            emergency_qty = abs(position.quantity)
                            
                            return Quote(
                                symbol=symbol,
                                bid_price=0,  # Don't buy more!
                                ask_price=emergency_exit_price,
                                bid_size=0,
                                ask_size=emergency_qty,
                                timestamp=time.time()
                            )
                            
                        else:  # Short - need to buy
                            emergency_exit_price = true_mid * 1.002  # 0.2% above mid
                            emergency_qty = abs(position.quantity)
                            
                            return Quote(
                                symbol=symbol,
                                bid_price=emergency_exit_price,
                                ask_price=0,  # Don't sell more!
                                bid_size=emergency_qty,
                                ask_size=0,
                                timestamp=time.time()
                            )
                            
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to check underwater position: {e}")
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

        # 2. TANH SKEW - DISABLED HERE (Applied later as Asymmetric Spreads)
        # Old code removed to avoid double application.
        pass

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
        
        # ========== INVENTORY SKEW (FIXED - NO RESERVATION SHIFT) ==========
        # 
        # OLD BROKEN LOGIC (Commented out):
        # reservation_shift = -skew_factor * optimal_half_spread * skew_intensity
        # reservation_price += reservation_shift  # ← This caused bid/ask on same side!
        #
        # NEW LOGIC: Asymmetric spread widening instead of reservation shift
        # - Long position: Widen bid (discourage buying), narrow ask (encourage selling)
        # - Short position: Narrow bid (encourage buying), widen ask (discourage selling)

        # Calculate inventory pressure
        max_units = settings.MAX_POSITION_USD / true_mid if true_mid > 0 else 10.0
        lambda_inv = 1.0  # Skew strength (1.0 = moderate)
        inventory_pct = q / max_units
        skew_factor = math.tanh(lambda_inv * inventory_pct)  # Range: [-1, 1]

        # Asymmetric spread adjustment
        # Max adjustment = 2x at extreme inventory
        max_spread_multiplier = 2.0

        if skew_factor > 0:  # Long position (q > 0)
            # Widen bid (less aggressive buying), narrow ask (more aggressive selling)
            bid_spread_mult = 1.0 + (skew_factor * max_spread_multiplier)
            ask_spread_mult = 1.0 - (skew_factor * 0.5)  # Half the narrowing
            
        elif skew_factor < 0:  # Short position (q < 0)
            # Narrow bid (more aggressive buying), widen ask (less aggressive selling)
            bid_spread_mult = 1.0 - (abs(skew_factor) * 0.5)
            ask_spread_mult = 1.0 + (abs(skew_factor) * max_spread_multiplier)
            
        else:  # Neutral
            bid_spread_mult = 1.0
            ask_spread_mult = 1.0

        # Apply asymmetric spreads (around un-shifted mid)
        reservation_price = true_mid  # Keep reservation at mid!
        bid_half_spread = final_half_spread * bid_spread_mult
        ask_half_spread = final_half_spread * ask_spread_mult

        # Calculate final quotes
        bid_price = reservation_price - bid_half_spread
        ask_price = reservation_price + ask_half_spread

        logger.debug(
            f"[{symbol}] Skew: {skew_factor:+.2f} | "
            f"Bid spread: {bid_spread_mult:.2f}x, Ask spread: {ask_spread_mult:.2f}x"
        )
        
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
        # Using MIN_NOTIONAL_USD (typically $10) gives a good safety margin over the $5 limit.
        target_min_usd = settings.MIN_NOTIONAL_USD
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
        
        # Round to appropriate precision using CEILING to meet minimum notional
        # Standard round() can bring values below minimum (0.074 -> 0.07)
        def ceil_to_precision(value, decimals):
            if value == 0:
                return 0
            multiplier = 10 ** decimals
            return math.ceil(value * multiplier) / multiplier
        
        if price > 1000:
            bid_size = ceil_to_precision(bid_size, 3)
            ask_size = ceil_to_precision(ask_size, 3)
        elif price > 10:
            bid_size = ceil_to_precision(bid_size, 2)
            ask_size = ceil_to_precision(ask_size, 2)
        elif price > 1:
            bid_size = ceil_to_precision(bid_size, 1)
            ask_size = ceil_to_precision(ask_size, 1)
        else:
            # For very cheap coins like DOGE, use whole numbers
            bid_size = math.ceil(bid_size)
            ask_size = math.ceil(ask_size)
        
        # ========== CRITICAL FIX: CAP SELL SIZE TO ACTUAL INVENTORY ==========
        # Prevent "insufficient balance" errors by only selling what we have
        if inventory > 0:
            # We're long - cap ask_size to what we actually own
            # Leave 5% buffer for rounding/precision issues
            max_sellable = inventory * 0.95
            if ask_size > max_sellable:
                ask_size = max_sellable
                logger.debug(f"[{symbol}] Ask size capped to inventory: {ask_size:.4f}")
        else:
            # We're flat or short - can't sell what we don't have (in spot mode)
            ask_size = 0
        
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
    
    def _estimate_kappa_from_arrivals(self, symbol: str) -> float:
        """Estimate order arrival rate from fill history (used as baseline)."""
        state = self._state[symbol]
        fill_times = list(state['fill_times'])
        
        if len(fill_times) < 3:
            return state['kappa']  # Return current kappa if insufficient data
        
        intervals = [
            fill_times[i] - fill_times[i-1]
            for i in range(1, len(fill_times))
        ]
        
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval > 0:
            return 1 / avg_interval
        return state['kappa']
    
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
        
        # 0. CANCEL EXISTING (Cancel-Replace Logic)
        # We emit a special CANCEL signal to wipe old orders for this symbol
        cancel_signal = SignalEvent(
            symbol=quote.symbol,
            side='CANCEL',
            quantity=0.0,
            strategy_id='avellaneda_stoikov'
        )
        await self.queue.put(cancel_signal)
        
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
