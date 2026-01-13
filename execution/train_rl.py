"""
Train RL Agent for Smart Order Routing.
Uses Stable Baselines3 and the MarketReplayEnv.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.smart_router import MarketReplayEnv, generate_synthetic_snapshots

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:
    print("❌ Stable Baselines3 not installed. Please run: pip install stable-baselines3 shimmy gymnasium")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL.Train")

def train(output_path="models/smart_router.zip", total_timesteps=100_000):
    """Train PPO agent on synthetic market data."""
    logger.info("🤖 Generating synthetic order book snapshots...")
    snapshots = generate_synthetic_snapshots(n_snapshots=5000, symbol="BTCUSDT")
    logger.info(f"✅ Generated {len(snapshots)} snapshots.")
    
    logger.info("🎮 Initializing Environment...")
    
    def make_env():
        return MarketReplayEnv(
            snapshots=snapshots,
            max_steps=60,  # 1 minute max execution
            urgency=0.5,
            initial_qty=1.0
        )
    
    # Vectorized environment
    env = DummyVecEnv([make_env])
    
    # Initialize Agent
    logger.info("🧠 Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Train
    logger.info(f"🏋️ Starting training for {total_timesteps} steps...")
    start_time = datetime.now()
    
    model.learn(total_timesteps=total_timesteps)
    
    duration = datetime.now() - start_time
    logger.info(f"🏁 Training complete in {duration}!")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    logger.info(f"💾 Model saved to {output_path}")
    
    # Verify load
    loaded = PPO.load(output_path)
    logger.info("✅ Verification load successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--out", type=str, default="models/smart_router", help="Output model path")
    args = parser.parse_args()
    
    train(output_path=args.out, total_timesteps=args.steps)
