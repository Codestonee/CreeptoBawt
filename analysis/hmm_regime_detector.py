"""
HMM Regime Detector with Async-Safe Background Retraining.

Uses Hidden Markov Models to detect market regimes (MEAN_REVERTING, TRENDING,
VOLATILE) with better lead time than ADX-based detection.

Architecture:
- Main Process: Fast inference (<0.2ms) using frozen model
- Background Process: Retrains model every N minutes using EM algorithm
- Communication: multiprocessing.Queue for model updates

Features:
- Gaussian HMM with 3 states
- Parkinson volatility, log returns, volume ratio as emissions
- Hysteresis wrapper to prevent state flickering
- Thread-safe observation buffer

References:
- Bishop (2006) "Pattern Recognition and Machine Learning" Ch. 13
- Hamilton (1989) "A New Approach to Economic Analysis of Nonstationary
  Time Series and the Business Cycle"
"""

import logging
import math
import pickle
import time
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from threading import Lock

import numpy as np

logger = logging.getLogger("Analysis.HMMRegime")


class MarketRegime(str, Enum):
    """Market regime states."""
    WARMUP = "WARMUP"              # Not enough observations
    MEAN_REVERTING = "MEAN_REVERTING"  # Low vol, good for tight spreads
    TRENDING = "TRENDING"          # Directional, skew against trend
    VOLATILE = "VOLATILE"          # High vol, widen or pause


@dataclass
class HMMConfig:
    """Configuration for HMM regime detector."""
    n_states: int = 3
    retrain_interval_minutes: int = 60
    min_observations: int = 200      # Min obs before first prediction
    warmup_observations: int = 500   # Min obs before first training
    buffer_size: int = 2000          # ~1.5 days @ 1-min candles
    hysteresis_count: int = 3        # Confirmations before state change
    covariance_type: str = "diag"    # Cheaper than full covariance
    n_iter: int = 50                 # EM iterations per training


class HMMRegimeDetector:
    """
    HMM-based regime detector with background retraining.
    
    Main loop gets fast inference (<0.2ms per call).
    Heavy EM training is offloaded to a background process.
    
    Usage:
        detector = HMMRegimeDetector()
        detector.start()  # Start background retrainer
        
        # On each 1-minute candle:
        regime, confidence = detector.predict(log_ret, vol, vol_ratio)
        
        # Cleanup:
        detector.stop()
    """
    
    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        
        # Observation buffer (thread-safe)
        self._buffer_lock = Lock()
        self._observation_buffer: deque = deque(maxlen=self.config.buffer_size)
        
        # Current model (None until first training)
        self._model = None
        self._model_lock = Lock()
        
        # State tracking
        self._current_regime: MarketRegime = MarketRegime.WARMUP
        self._confidence: float = 0.0
        self._last_regime: Optional[MarketRegime] = None
        self._regime_count: int = 0
        
        # Multiprocessing components
        self._model_queue: Optional[mp.Queue] = None
        self._obs_queue: Optional[mp.Queue] = None
        self._retrainer_process: Optional[mp.Process] = None
        self._running = mp.Value('b', False)
        
        # State mapping (will be determined by training)
        self._state_map: Dict[int, MarketRegime] = {
            0: MarketRegime.MEAN_REVERTING,
            1: MarketRegime.TRENDING,
            2: MarketRegime.VOLATILE
        }
        
        # Regime flip tracking for sanity check
        self._regime_flip_times: deque = deque(maxlen=10)
        self._max_flips_per_minute: int = 5
        
        logger.debug(f"HMMRegimeDetector initialized: n_states={self.config.n_states}")
    
    def start(self) -> None:
        """Start background retrainer process."""
        if self._retrainer_process is not None and self._retrainer_process.is_alive():
            logger.warning("Retrainer already running")
            return
        
        # Create queues for IPC
        self._model_queue = mp.Queue(maxsize=1)
        self._obs_queue = mp.Queue(maxsize=100)
        
        # Start background process
        self._running.value = True
        self._retrainer_process = mp.Process(
            target=self._retrainer_worker,
            args=(
                self._model_queue,
                self._obs_queue,
                self._running,
                self.config
            ),
            daemon=True
        )
        self._retrainer_process.start()
        logger.debug("HMM background retrainer started")
    
    def stop(self) -> None:
        """Stop background retrainer process."""
        self._running.value = False
        if self._retrainer_process is not None:
            # Fast shutdown: wait briefly then kill
            self._retrainer_process.join(timeout=0.2)
            if self._retrainer_process.is_alive():
                self._retrainer_process.terminate()
            self._retrainer_process = None
        logger.debug("HMM background retrainer stopped")
    
    def update(self, log_return: float, volatility: float, volume_ratio: float) -> None:
        """
        Add new observation to buffer.
        
        This should be called on each candle close.
        
        Args:
            log_return: log(close/open)
            volatility: Parkinson volatility sqrt(log(high/low)) / sqrt(2*log(2))
            volume_ratio: volume / SMA(volume, 20)
        """
        obs = [log_return, volatility, volume_ratio]
        
        with self._buffer_lock:
            self._observation_buffer.append(obs)
        
        # Send to retrainer (non-blocking)
        if self._obs_queue is not None:
            try:
                self._obs_queue.put_nowait(obs)
            except:
                pass  # Queue full, skip
    
    def predict(
        self,
        log_return: float,
        volatility: float,
        volume_ratio: float
    ) -> Tuple[MarketRegime, float]:
        """
        Predict regime from current observation.
        
        This is the fast path (<0.2ms) called on the main asyncio loop.
        
        Args:
            log_return: log(close/open)
            volatility: Parkinson volatility
            volume_ratio: volume / SMA(volume, 20)
        
        Returns:
            (regime, confidence) tuple
        """
        # Also add observation to buffer
        self.update(log_return, volatility, volume_ratio)
        
        # Check for new model from retrainer (non-blocking)
        self._check_for_new_model()
        
        # Check if in warmup
        if len(self._observation_buffer) < self.config.min_observations:
            return MarketRegime.WARMUP, 0.0
        
        # Check if model available
        with self._model_lock:
            model = self._model
        
        if model is None:
            return MarketRegime.WARMUP, 0.0
        
        # Fast inference
        try:
            obs = np.array([[log_return, volatility, volume_ratio]])
            regime_idx = model.predict(obs)[0]
            log_prob = model.score(obs)
            
            # Map to regime
            raw_regime = self._state_map.get(regime_idx, MarketRegime.VOLATILE)
            
            # Apply hysteresis
            regime = self._apply_hysteresis(raw_regime)
            
            # Convert log_prob to confidence (0-1 scale, normalized)
            # Higher log_prob = better fit
            confidence = min(1.0, max(0.0, (log_prob + 10) / 10))
            
            self._current_regime = regime
            self._confidence = confidence
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"HMM prediction error: {e}")
            return self._current_regime, 0.0
    
    def _apply_hysteresis(self, new_regime: MarketRegime) -> MarketRegime:
        """Apply hysteresis to prevent state flickering."""
        if new_regime == self._last_regime:
            self._regime_count = min(self._regime_count + 1, self.config.hysteresis_count)
        else:
            self._regime_count = 1
            self._last_regime = new_regime
        
        # Only change if seen enough consecutive times
        if self._regime_count >= self.config.hysteresis_count:
            # SANITY CHECK: Detect rapid regime flipping
            if new_regime != self._current_regime:
                now = time.time()
                self._regime_flip_times.append(now)
                
                # Count flips in last 60 seconds
                recent_flips = sum(1 for t in self._regime_flip_times if now - t < 60)
                
                if recent_flips > self._max_flips_per_minute:
                    logger.warning(
                        f"HMM SANITY CHECK: {recent_flips} regime flips in 60s! "
                        f"Staying in {self._current_regime.value} (ignoring {new_regime.value})"
                    )
                    return self._current_regime
            
            return new_regime
        
        return self._current_regime
    
    def _check_for_new_model(self) -> None:
        """Check if retrainer has produced a new model."""
        if self._model_queue is None:
            return
        
        try:
            model_data = self._model_queue.get_nowait()
            with self._model_lock:
                self._model = model_data['model']
                if 'state_map' in model_data:
                    old_map = self._state_map.copy()
                    self._state_map = model_data['state_map']
                    
                    # LOG STATE MAPPING CHANGES
                    if old_map != self._state_map:
                        logger.warning(
                            f"HMM STATE MAPPING CHANGED: {old_map} -> {self._state_map}"
                        )
            logger.info("New HMM model received from background retrainer")
        except:
            pass  # No new model
    
    @staticmethod
    def _retrainer_worker(
        model_queue: mp.Queue,
        obs_queue: mp.Queue,
        running: mp.Value,
        config: HMMConfig
    ) -> None:
        """
        Background worker that retrains the HMM periodically.
        
        Runs in a separate process to avoid blocking the main asyncio loop.
        """
        # Import inside process to avoid pickling issues
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed, HMM retrainer disabled")
            return
        
        # Ignore SIGINT in child process so only parent handles it
        import signal
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except AttributeError:
            # Windows might not have all signals, but SIGINT usually works
            pass

        # Local observation buffer
        observations = deque(maxlen=config.buffer_size)
        last_train = 0
        
        while running.value:
            try:
                # Collect observations from queue
                while True:
                    try:
                        obs = obs_queue.get_nowait()
                        observations.append(obs)
                    except:
                        break
                
                # Check if time to retrain
                now = time.time()
                should_train = (
                    len(observations) >= config.warmup_observations and
                    (now - last_train) >= config.retrain_interval_minutes * 60
                )
                
                if should_train:
                    logger.info(f"HMM retraining with {len(observations)} observations...")
                    
                    # Convert to numpy array
                    X = np.array(list(observations))
                    
                    # Train new model
                    model = GaussianHMM(
                        n_components=config.n_states,
                        covariance_type=config.covariance_type,
                        n_iter=config.n_iter,
                        random_state=42
                    )
                    model.fit(X)
                    
                    # Determine state mapping based on means
                    # Lower volatility state = MEAN_REVERTING
                    # Higher volatility state = VOLATILE
                    # Middle or high directional = TRENDING
                    state_map = HMMRegimeDetector._infer_state_mapping(model.means_)
                    
                    # Send to main process
                    try:
                        # Clear queue first
                        try:
                            model_queue.get_nowait()
                        except:
                            pass
                        
                        model_queue.put_nowait({
                            'model': model,
                            'state_map': state_map
                        })
                        logger.info("New HMM model sent to main process")
                    except:
                        logger.warning("Could not send model to main process")
                    
                    last_train = now
                
                # Sleep with check (responsiveness)
                for _ in range(10):
                    if not running.value:
                        break
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"HMM retrainer error: {e}")
                time.sleep(60)
    
    @staticmethod
    def _infer_state_mapping(means: np.ndarray) -> Dict[int, MarketRegime]:
        """
        Infer state mapping from HMM means.
        
        Uses volatility (second column) to classify states.
        """
        # means shape: (n_states, n_features)
        # features: [log_return, volatility, volume_ratio]
        
        n_states = means.shape[0]
        
        if n_states != 3:
            # Fallback for non-3-state models
            return {i: MarketRegime.VOLATILE for i in range(n_states)}
        
        # Sort states by volatility (column 1)
        vol_order = np.argsort(means[:, 1])
        
        state_map = {
            vol_order[0]: MarketRegime.MEAN_REVERTING,  # Lowest vol
            vol_order[1]: MarketRegime.TRENDING,        # Medium vol
            vol_order[2]: MarketRegime.VOLATILE         # Highest vol
        }
        
        return state_map
    
    def get_regime(self) -> MarketRegime:
        """Get current regime without new prediction."""
        return self._current_regime
    
    def get_stats(self) -> dict:
        """Get current HMM statistics."""
        with self._model_lock:
            has_model = self._model is not None
        
        return {
            "regime": self._current_regime.value,
            "confidence": self._confidence,
            "observations": len(self._observation_buffer),
            "has_model": has_model,
            "retrainer_running": (
                self._retrainer_process is not None and
                self._retrainer_process.is_alive()
            )
        }


class RegimeSupervisorHMM:
    """
    Drop-in replacement for ADX-based RegimeSupervisor.
    
    Wraps HMMRegimeDetector with the same interface as the original.
    """
    
    def __init__(self, symbols: List[str], config: Optional[HMMConfig] = None):
        self.symbols = [s.lower() for s in symbols]
        self.config = config or HMMConfig()
        
        # One detector per symbol
        self.detectors: Dict[str, HMMRegimeDetector] = {}
        for sym in self.symbols:
            self.detectors[sym] = HMMRegimeDetector(self.config)
        
        # Volume MA for volume ratio calculation
        self._volume_ma: Dict[str, deque] = {
            sym: deque(maxlen=20) for sym in self.symbols
        }
    
    def start(self) -> None:
        """Start all background retrainers."""
        for detector in self.detectors.values():
            detector.start()
    
    def stop(self) -> None:
        """Stop all background retrainers."""
        for detector in self.detectors.values():
            detector.stop()
    
    def analyze(
        self,
        symbol: str,
        open_price: float,
        close_price: float,
        high_price: float,
        low_price: float,
        volume: float
    ) -> Tuple[str, float]:
        """
        Analyze candle and return regime.
        
        Compatible interface with original RegimeSupervisor.
        
        Returns:
            (regime_name, confidence) tuple
        """
        symbol = symbol.lower()
        if symbol not in self.detectors:
            return "UNCERTAIN", 0.0
        
        detector = self.detectors[symbol]
        
        # Calculate features
        log_return = math.log(close_price / open_price) if open_price > 0 else 0.0
        
        # Parkinson volatility
        if high_price > low_price > 0:
            parkinson_vol = math.sqrt(math.log(high_price / low_price)**2) / math.sqrt(2 * math.log(2))
        else:
            parkinson_vol = 0.001
        
        # Volume ratio
        self._volume_ma[symbol].append(volume)
        vol_ma = sum(self._volume_ma[symbol]) / len(self._volume_ma[symbol]) if self._volume_ma[symbol] else volume
        volume_ratio = volume / vol_ma if vol_ma > 0 else 1.0
        
        # Predict
        regime, confidence = detector.predict(log_return, parkinson_vol, volume_ratio)
        
        return regime.value, confidence
