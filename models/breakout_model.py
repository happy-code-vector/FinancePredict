"""MULTI-BREAKOUT model: per-asset binary classifiers (33 assets)."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import miner.config as mcfg
from miner.models.base import BaseModel


class BreakoutModel(BaseModel):
    """One LightGBM binary classifier per asset for breakout continuation prediction.

    Output: {asset: [P(continuation), P(reversal)]} for all 33 assets.
    """

    challenge_name = "breakout"

    def __init__(self):
        self.models: dict[str, lgb.LGBMClassifier] = {}

    def train(
        self,
        features: dict[str, np.ndarray],
        labels: dict[str, np.ndarray],
        **kwargs,
    ) -> dict:
        """Train per-asset models.

        Args:
            features: {asset: (N, F)} feature matrices
            labels: {asset: (N,)} binary labels (1=continuation, 0=reversal)

        Returns:
            metrics dict
        """
        metrics = {}
        for asset in features:
            X = features[asset]
            y = labels[asset]

            if len(y) < 50 or len(np.unique(y)) < 2:
                continue

            n = len(y)
            weights = self.recency_weights(n)
            split = int(n * 0.9)

            params = {**mcfg.LIGHTGBM_PARAMS, "objective": "binary"}
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X[:split], y[:split],
                sample_weight=weights[:split],
                eval_set=[(X[split:], y[split:])],
                callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
            )

            y_pred = model.predict_proba(X[split:])[:, 1]
            auc = roc_auc_score(y[split:], y_pred)
            metrics[asset] = {"auc": auc, "n_samples": n}
            self.models[asset] = model

        return metrics

    def predict(self, features: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """Generate breakout predictions for all assets.

        Returns: {asset: [P(continuation), P(reversal)]}
        """
        result = {}
        for asset, x in features.items():
            if asset in self.models:
                p_cont = self.models[asset].predict_proba(x.reshape(1, -1))[0, 1]
                # Add noise
                p_cont = np.clip(p_cont + np.random.normal(0, 0.005), 0.01, 0.99)
            else:
                p_cont = 0.5
            result[asset] = [float(p_cont), float(1.0 - p_cont)]
        return result

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)
        return path

    def load(self, path: Path) -> None:
        self.models = joblib.load(path)
