"""XSEC-RANK model: pooled 33-asset cross-sectional rank model."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import miner.config as mcfg
from miner.models.base import BaseModel


class XsecModel(BaseModel):
    """Single LightGBM binary classifier trained on pooled 33-asset data.

    Output: {asset: score in [-1, 1]} where positive = above-median expected.
    """

    challenge_name = "xsec_rank"

    def __init__(self):
        self.model: lgb.LGBMClassifier | None = None

    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> dict:
        """Train pooled cross-sectional model.

        Args:
            features: (M, F) pooled feature matrix (all assets stacked)
            labels: (M,) binary labels

        Returns:
            metrics dict
        """
        n = len(labels)
        weights = self.recency_weights(n)
        split = int(n * 0.9)

        params = {**mcfg.LIGHTGBM_PARAMS, "objective": "binary"}
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            features[:split], labels[:split],
            sample_weight=weights[:split],
            eval_set=[(features[split:], labels[split:])],
            callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
        )

        y_pred = self.model.predict_proba(features[split:])[:, 1]
        auc = roc_auc_score(labels[split:], y_pred)

        return {"auc": auc, "n_samples": n}

    def predict(self, features: dict[str, np.ndarray]) -> dict[str, float]:
        """Generate cross-sectional scores for all assets.

        Args:
            features: {asset: (F,)} feature vectors

        Returns:
            {asset: score in [-1, 1]}
        """
        result = {}
        if self.model is None:
            return {a: 0.0 for a in features}

        for asset, x in features.items():
            proba = self.model.predict_proba(x.reshape(1, -1))[0, 1]
            # Map probability to [-1, 1]
            score = 2.0 * proba - 1.0 + np.random.normal(0, 0.005)
            result[asset] = float(np.clip(score, -1.0, 1.0))

        return result

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
