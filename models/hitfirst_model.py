"""HITFIRST model: 3-class softmax barrier prediction."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

import miner.config as mcfg
from miner.models.base import BaseModel


class HitfirstModel(BaseModel):
    """LightGBM multiclass for ETH-HITFIRST.

    Output: [P(up first), P(down first), P(neither)] — must sum to 1, all in (0, 1).
    """

    challenge_name = "hitfirst"

    def __init__(self):
        self.model: lgb.LGBMClassifier | None = None

    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> dict:
        """Train 3-class classifier.

        Args:
            features: (N, F) feature matrix
            labels: (N,) int labels (0=up, 1=down, 2=neither)

        Returns:
            metrics dict
        """
        n = len(labels)
        weights = self.recency_weights(n)
        split = int(n * 0.9)

        params = {**mcfg.LIGHTGBM_PARAMS, "objective": "multiclass", "num_class": 3}
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            features[:split], labels[:split],
            sample_weight=weights[:split],
            eval_set=[(features[split:], labels[split:])],
            callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
        )

        y_pred = self.model.predict(features[split:])
        acc = np.mean(y_pred == labels[split:])

        return {"accuracy": acc, "n_samples": n}

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate 3-way probability vector."""
        if self.model is None:
            return np.array([0.333, 0.333, 0.334])

        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        # Add noise to avoid stale filter
        proba = proba + np.random.normal(0, 0.005, size=3)
        proba = np.clip(proba, 0.01, 0.99)
        proba = proba / proba.sum()

        return proba

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
