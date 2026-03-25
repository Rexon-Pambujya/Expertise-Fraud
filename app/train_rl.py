from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .synthetic_data import build_dataset, save_dataset
from .env import ExpertiseFraudEnv
from .utils import ensure_dir, save_json


def train_model(total_timesteps: int = 20_000, n_samples: int = 500, model_path: str = "artifacts/fraud_ppo"):
    ensure_dir("artifacts")
    save_dataset("data/synthetic_candidates.jsonl", n=n_samples)
    data = build_dataset(n=n_samples)
    env = DummyVecEnv([lambda: ExpertiseFraudEnv(data)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    save_json("artifacts/train_config.json", {
        "total_timesteps": total_timesteps,
        "n_samples": n_samples,
        "model_path": model_path,
    })
    return model


if __name__ == "__main__":
    train_model()
