"""
GLT (Guéant-Lehalle-Tapia) Quote Engine.

Implements infinite-horizon optimal market making with non-linear inventory
risk. Unlike standard Avellaneda-Stoikov which uses finite horizon T and
creates artificial "inventory dumping" as t→T, GLT removes time dependency
entirely - ideal for 24/7 crypto markets.

Key formulas (infinite horizon, exponential intensities):
- Base half-spread: φ = (1/γδ) * ln(1 + γδ/k)
- Skew from theta differences: uses precomputed value function θ(q)

References:
- Guéant, Lehalle, Fernandez-Tapia (2012, 2013)
- "Dealing with the Inventory Risk" arXiv:1105.3115
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("Strategy.GLT")

from config.settings import settings

@dataclass
class GLTParams:
    """Parameters for GLT quote engine per symbol."""
    # Defaults now read from settings for unified config
    A: float = None       # Intensity scale (fills per hour at mid)
    k: float = None       # Intensity decay (how fast fills drop with depth)
    gamma: float = None   # Risk aversion (higher = dump inventory faster)
    delta: float = 0.001  # Trade size in base units
    min_spread_bps: float = 15.0  # Minimum spread in basis points
    
    def __post_init__(self):
        """Apply defaults from settings and validate."""
        # Use settings values if not explicitly provided
        if self.A is None:
            self.A = settings.GLT_A
        if self.k is None:
            self.k = settings.GLT_K
        if self.gamma is None:
            self.gamma = settings.GLT_GAMMA
        
        assert self.A > 0, "A must be positive"
        assert self.k > 0, "k must be positive"
        assert self.gamma > 0, "gamma must be positive"
        assert self.delta > 0, "delta must be positive"


class GLTQuoteEngine:
    """
    GLT-based optimal quote calculator.
    
    Provides non-linear inventory skew using the infinite horizon solution.
    Can be used as a drop-in replacement for linear A-S skew.
    
    Usage:
        engine = GLTQuoteEngine()
        engine.set_params("btcusdt", GLTParams(A=10, k=0.5, gamma=0.1, delta=0.001))
        
        bid, ask = engine.compute_quotes(
            symbol="btcusdt",
            mid_price=42000,
            inventory=0.05,
            volatility=0.001
        )
    """
    
    def __init__(self, inventory_limit: float = 10.0):
        """
        Initialize GLT engine with multi-asset correlation support.
        
        Args:
            inventory_limit: Max inventory for theta table computation.
        """
        self.params: Dict[str, GLTParams] = {}
        self.theta_tables: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.inventory_limit = inventory_limit
        self._precomputed: Dict[str, dict] = {}  # Cached constants
        
        # Multi-asset correlation support (Gemini research integration)
        # Default correlations based on empirical crypto data
        self._asset_order = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt', 'bchusdt']
        self._correlation_matrix = np.array([
            [1.00, 0.85, 0.72, 0.75, 0.90],  # BTC
            [0.85, 1.00, 0.78, 0.70, 0.82],  # ETH
            [0.72, 0.78, 1.00, 0.65, 0.70],  # SOL
            [0.75, 0.70, 0.65, 1.00, 0.72],  # BNB
            [0.90, 0.82, 0.70, 0.72, 1.00],  # BCH
        ])
        
        # Portfolio state: track all positions for cross-asset adjustment
        self._portfolio_inventory: Dict[str, float] = {}
        self._portfolio_volatility: Dict[str, float] = {}
    
    def set_params(self, symbol: str, params: GLTParams) -> None:
        """
        Set or update parameters for a symbol.
        
        This also triggers precomputation of constants and theta table.
        """
        symbol = symbol.lower()
        self.params[symbol] = params
        
        # Precompute base spread constant
        xi = params.gamma
        delta = params.delta
        k = params.k
        
        # Base half-spread (infinite horizon, no time dependence)
        # φ = (1/γδ) * ln(1 + γδ/k)
        phi = (1.0 / (xi * delta)) * math.log(1.0 + xi * delta / k)
        
        self._precomputed[symbol] = {
            'phi': phi,
            'xi': xi
        }
        
        # Compute theta table for this symbol
        self._compute_theta_table(symbol, params)
        
        logger.info(
            f"[{symbol.upper()}] GLT params set: A={params.A:.2f}, k={params.k:.3f}, "
            f"γ={params.gamma:.3f}, base_spread={phi*10000:.1f}bps"
        )
    
    def update_from_calibrator(self, symbol: str, A: float, k: float) -> None:
        """
        Update A and k parameters from intensity calibrator.
        
        This allows dynamic adaptation to market conditions
        without changing gamma or other risk parameters.
        
        Args:
            symbol: Trading symbol
            A: Calibrated intensity scale
            k: Calibrated decay rate
        """
        symbol = symbol.lower()
        if symbol not in self.params:
            # Initialize with defaults if not exists
            self.set_params(symbol, GLTParams(A=A, k=k))
            return
        
        params = self.params[symbol]
        
        # Only update if significantly different (>5% change)
        if abs(params.A - A) / params.A > 0.05 or abs(params.k - k) / params.k > 0.05:
            # Update params
            params.A = A
            params.k = k
            
            # Recompute constants
            xi = params.gamma
            delta = params.delta
            phi = (1.0 / (xi * delta)) * math.log(1.0 + xi * delta / k)
            
            self._precomputed[symbol] = {
                'phi': phi,
                'xi': xi
            }
            
            logger.info(
                f"[{symbol.upper()}] GLT params updated from calibrator: "
                f"A={A:.3f}, k={k:.4f}, spread={phi*10000:.1f}bps"
            )
    
    def _compute_theta_table(self, symbol: str, params: GLTParams) -> None:
        """
        Compute theta value function table for discrete inventory levels.
        
        For simplicity, we use a quadratic approximation:
        θ(q) ≈ (1/2) * γ * σ² * q²
        
        This can be replaced with a numerical ODE solution for higher accuracy.
        """
        # Discretize inventory into steps of delta
        delta = params.delta
        max_q = int(self.inventory_limit / delta)
        
        theta = {}
        for i in range(-max_q, max_q + 1):
            q = i * delta
            # Quadratic approximation (can upgrade to ODE solution later)
            # θ(q) = (1/2) * γ * q² (normalized, actual uses σ² at runtime)
            theta[i] = 0.5 * params.gamma * (q ** 2)
        
        self.theta_tables[symbol] = theta
        logger.debug(f"[{symbol.upper()}] Computed theta table with {len(theta)} entries")
    
    def _get_theta(self, symbol: str, inventory: float, volatility: float) -> float:
        """
        Get theta value for given inventory level.
        
        Interpolates if inventory is not exactly on grid.
        """
        symbol = symbol.lower()
        params = self.params.get(symbol)
        if params is None:
            return 0.0
        
        theta_table = self.theta_tables.get(symbol, {})
        delta = params.delta
        
        # Discretize inventory to grid index
        idx = int(round(inventory / delta))
        
        # Clamp to table bounds
        max_idx = int(self.inventory_limit / delta)
        idx = max(-max_idx, min(max_idx, idx))
        
        # Get base theta and scale by volatility
        base_theta = theta_table.get(idx, 0.5 * params.gamma * (inventory ** 2))
        
        # Scale by actual volatility squared
        return base_theta * (volatility ** 2)
    
    def update_portfolio_state(
        self,
        symbol: str,
        inventory: float,
        volatility: float
    ) -> None:
        """
        Update portfolio state for multi-asset correlation adjustment.
        
        Call this before compute_quotes() for each symbol to enable
        cross-asset inventory pressure.
        
        Args:
            symbol: Trading symbol
            inventory: Current inventory
            volatility: Current volatility estimate
        """
        self._portfolio_inventory[symbol.lower()] = inventory
        self._portfolio_volatility[symbol.lower()] = volatility
    
    def set_correlation_matrix(
        self,
        matrix: np.ndarray,
        asset_order: list
    ) -> None:
        """
        Update correlation matrix for multi-asset adjustments.
        
        Args:
            matrix: Square correlation matrix
            asset_order: List of symbols matching matrix rows/cols
        """
        self._correlation_matrix = matrix
        self._asset_order = [s.lower() for s in asset_order]
        logger.info(f"Updated correlation matrix for {len(asset_order)} assets")
    
    def _get_cross_asset_adjustment(self, symbol: str, params: GLTParams) -> float:
        """
        Calculate reservation price adjustment from correlated assets.
        
        When holding BTC, ETH quotes should also shift because
        BTC/ETH correlation is ~85% - they move together.
        
        Formula: adj = -γ * Σ(ρ_ij * σ_i * σ_j * q_j) for all j ≠ i
        
        Returns:
            Adjustment to add to reservation price
        """
        symbol = symbol.lower()
        
        # Check if we have multi-asset data
        if len(self._portfolio_inventory) < 2:
            return 0.0
        
        # Get index for this symbol
        try:
            idx_i = self._asset_order.index(symbol)
        except ValueError:
            return 0.0  # Symbol not in correlation matrix
        
        sigma_i = self._portfolio_volatility.get(symbol, 0.01)
        adjustment = 0.0
        
        for other_symbol, q_j in self._portfolio_inventory.items():
            if other_symbol == symbol:
                continue  # Skip self
            
            try:
                idx_j = self._asset_order.index(other_symbol)
            except ValueError:
                continue  # Other symbol not in matrix
            
            sigma_j = self._portfolio_volatility.get(other_symbol, 0.01)
            rho = self._correlation_matrix[idx_i, idx_j]
            
            # Add cross-asset pressure
            adjustment -= params.gamma * rho * sigma_i * sigma_j * q_j
        
        return adjustment
    
    def compute_quotes(
        self,
        symbol: str,
        mid_price: float,
        inventory: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Compute optimal bid/ask prices using GLT infinite horizon model.
        
        Now includes multi-asset correlation adjustment from research.
        
        Args:
            symbol: Trading symbol
            mid_price: Current mid price
            inventory: Current inventory in base units
            volatility: Volatility estimate (annualized or per-period)
        
        Returns:
            (bid_price, ask_price) tuple
        """
        symbol = symbol.lower()
        params = self.params.get(symbol)
        
        if params is None:
            # Fallback to simple quotes if no params set
            spread = mid_price * 0.001  # 10 bps default
            return mid_price - spread/2, mid_price + spread/2
        
        # Update portfolio state for this symbol
        self.update_portfolio_state(symbol, inventory, volatility)
        
        precomp = self._precomputed.get(symbol, {})
        phi = precomp.get('phi', 0.001)
        delta = params.delta
        
        # Get theta values for inventory ± delta
        theta_q = self._get_theta(symbol, inventory, volatility)
        theta_q_plus = self._get_theta(symbol, inventory + delta, volatility)
        theta_q_minus = self._get_theta(symbol, inventory - delta, volatility)
        
        # Compute theta differences (gradients)
        dq_plus = (theta_q - theta_q_plus) / delta if delta > 0 else 0
        dq_minus = (theta_q - theta_q_minus) / delta if delta > 0 else 0
        
        # Compute bid/ask depths (distance from mid)
        delta_b = phi + dq_plus / (mid_price + 1e-8)
        delta_a = phi - dq_minus / (mid_price + 1e-8)
        
        # Ensure minimum spread
        min_half_spread = mid_price * (params.min_spread_bps / 10000) / 2
        delta_b = max(delta_b, min_half_spread)
        delta_a = max(delta_a, min_half_spread)
        
        # Compute reservation price (mid adjusted for inventory risk)
        # r = mid - q * γ * σ² (simplified single-asset)
        reservation = mid_price - inventory * params.gamma * (volatility ** 2) * mid_price * 0.5
        
        # ========== MULTI-ASSET CORRELATION ADJUSTMENT (NEW) ==========
        # Adjust reservation based on correlated asset inventories
        # If holding BTC and ETH corr=0.85, BTC inventory affects ETH quotes
        cross_asset_adj = self._get_cross_asset_adjustment(symbol, params)
        reservation += cross_asset_adj * mid_price
        # ==============================================================
        
        # Final quotes
        bid_price = reservation - delta_b
        ask_price = reservation + delta_a
        
        # Ensure bid < ask
        if bid_price >= ask_price:
            spread = ask_price - bid_price if ask_price > bid_price else min_half_spread * 2
            center = (bid_price + ask_price) / 2
            bid_price = center - abs(spread) / 2 - min_half_spread
            ask_price = center + abs(spread) / 2 + min_half_spread
        
        # CRITICAL: Ensure positive prices (prevent negative bid bug)
        min_price = mid_price * 0.5  # Never go below 50% of mid
        bid_price = max(bid_price, min_price)
        ask_price = max(ask_price, bid_price + min_half_spread * 2)
        
        return bid_price, ask_price
    
    def get_stats(self, symbol: str) -> dict:
        """Get current GLT statistics for a symbol."""
        symbol = symbol.lower()
        params = self.params.get(symbol)
        precomp = self._precomputed.get(symbol, {})
        
        return {
            "symbol": symbol,
            "params": {
                "A": params.A if params else None,
                "k": params.k if params else None,
                "gamma": params.gamma if params else None,
                "delta": params.delta if params else None
            } if params else None,
            "base_spread_bps": precomp.get('phi', 0) * 10000,
            "theta_table_size": len(self.theta_tables.get(symbol, {}))
        }


def calibrate_lambda_exponential(
    depth_to_counts: Dict[float, int],
    total_hours: float
) -> Tuple[float, float]:
    """
    Calibrate intensity parameters A and k from empirical fill data.
    
    Fits the model: Λ(δ) = A * exp(-k * δ)
    where δ is depth (distance from mid in bps) and Λ is fill rate.
    
    Args:
        depth_to_counts: Mapping from depth (bps) to number of fills observed
        total_hours: Total observation time in hours
    
    Returns:
        (A, k) tuple for intensity function
    
    Note:
        In production, consider using RANSAC or robust regression
        to handle outliers from empty depth levels.
    """
    if not depth_to_counts or total_hours <= 0:
        logger.warning("Insufficient data for calibration, using defaults")
        return 1.0, 0.3
    
    depths = np.array(list(depth_to_counts.keys()), dtype=float)
    counts = np.array(list(depth_to_counts.values()), dtype=float)
    
    # Convert counts to intensity (fills per hour)
    lambdas = counts / max(total_hours, 1e-6)
    
    # Avoid log(0) - clip to small positive value
    lambdas = np.clip(lambdas, 1e-6, None)
    
    # Filter out zero-count depths for stability
    mask = counts > 0
    if mask.sum() < 2:
        logger.warning("Too few non-zero depth levels for calibration")
        return 1.0, 0.3
    
    depths = depths[mask]
    lambdas = lambdas[mask]
    
    # Linear regression in log-space: log(Λ) = log(A) - k * δ
    try:
        # np.polyfit returns [slope, intercept] for degree 1
        coeffs = np.polyfit(depths, np.log(lambdas), 1)
        slope, intercept = coeffs[0], coeffs[1]
        
        k_est = -slope
        A_est = float(np.exp(intercept))
        
        # Sanity checks
        if k_est <= 0 or A_est <= 0:
            logger.warning(f"Invalid calibration results: A={A_est}, k={k_est}")
            return 1.0, 0.3
        
        logger.info(f"Calibrated: A={A_est:.2f}, k={k_est:.4f}")
        return A_est, k_est
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1.0, 0.3


def multi_asset_inventory_adjustment(
    q_btc: float,
    q_eth: float,
    sigma_btc: float,
    sigma_eth: float,
    rho: float,
    gamma: float
) -> Tuple[float, float]:
    """
    Compute reservation price adjustments for correlated assets.
    
    When trading BTC and ETH simultaneously, inventory in one affects
    the optimal quotes for the other due to correlation.
    
    Args:
        q_btc: BTC inventory
        q_eth: ETH inventory
        sigma_btc: BTC volatility
        sigma_eth: ETH volatility
        rho: BTC/ETH correlation (-1 to 1)
        gamma: Risk aversion
    
    Returns:
        (adj_btc, adj_eth) adjustments to add to reservation prices
    """
    adj_btc = -gamma * (q_btc * sigma_btc**2 + rho * sigma_btc * sigma_eth * q_eth)
    adj_eth = -gamma * (q_eth * sigma_eth**2 + rho * sigma_btc * sigma_eth * q_btc)
    return adj_btc, adj_eth
