"""TRADE-MIX model: position sizing for BTC, ETH, TAO, SOL."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

import miner.config as mcfg
from miner.models.base import BaseModel


class TradeMixModel(BaseModel):
    """Per-asset LightGBM regressors predicting forward returns → positions.

    Output: {asset: position in [-1, 1]}.
    """

    challenge_name = "trade_mix"

    def __init__(self):
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def train(
        self,
        features: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        **kwargs,
    ) -> dict:
        """Train per-asset regression models.

        Args:
            features: {asset: (N, F)} feature matrices
            targets: {asset: (N,)} signed return targets in [-1, 1]
        """
        metrics = {}
        for asset in features:
            X = features[asset]
            y = targets[asset]

            if len(y) < 100:
                continue

            n = len(y)
            weights = self.recency_weights(n)
            split = int(n * 0.9)

            params = {**mcfg.LIGHTGBM_PARAMS, "objective": "regression"}
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X[:split], y[:split],
                sample_weight=weights[:split],
                eval_set=[(X[split:], y[split:])],
                callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
            )

            y_pred = model.predict(X[split:])
            # Directional accuracy
            dir_acc = np.mean(np.sign(y_pred) == np.sign(y[split:]))
            # Sharpe of predictions (simplified)
            pnl = y_pred * y[split:]
            sharpe = np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(525600)

            metrics[asset] = {"direction_accuracy": dir_acc, "sharpe": sharpe, "n_samples": n}
            self.models[asset] = model

        return metrics

    def predict(self, features: dict[str, np.ndarray]) -> dict[str, float]:
        """Generate position signals for all 4 assets."""
        result = {}
        for asset, x in features.items():
            if asset in self.models:
                pred = self.models[asset].predict(x.reshape(1, -1))[0]
                # Clip to [-1, 1] with noise
                pred = np.clip(pred + np.random.normal(0, 0.005), -1.0, 1.0)
                result[asset] = float(pred)
            else:
                result[asset] = 0.0
        return result

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)
        return path

    def load(self, path: Path) -> None:
        self.models = joblib.load(path)
