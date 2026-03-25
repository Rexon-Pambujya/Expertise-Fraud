from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .synthetic_data import build_dataset
from .env import ExpertiseFraudEnv
from .utils import ensure_dir, save_json
from stable_baselines3 import PPO


def evaluate(n_samples: int = 300):
    data = build_dataset(n=n_samples)
    env = ExpertiseFraudEnv(data)
    model = PPO.load("artifacts/fraud_ppo")

    y_true = []
    y_pred = []
    rewards = []

    for _ in range(n_samples):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, terminated, truncated, step_info = env.step(action)

        y_true.append(int(step_info["label"]))
        y_pred.append(1 if action == 1 else 0)
        rewards.append(reward)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "confusion_matrix": cm,
        "classification_report": report,
        "avg_reward": float(np.mean(rewards)),
        "fraud_recall": float(report["1"]["recall"]) if "1" in report else 0.0,
        "pass_precision": float(report["0"]["precision"]) if "0" in report else 0.0,
    }

    ensure_dir("artifacts")
    save_json("artifacts/eval_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
