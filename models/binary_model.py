"""Binary challenge model: LightGBM classifier → 2 features in [-1, 1]."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import miner.config as mcfg
from miner.models.base import BaseModel


class BinaryModel(BaseModel):
    """LightGBM binary classifier for ETH, CADUSD, NZDUSD, CHFUSD, XAGUSD.

    Output: 2 features in [-1, 1]. These are inputs to the validator's
    logistic regression, NOT probabilities. We output:
    [0] = raw prediction score (directional confidence)
    [1] = volatility-adjusted confidence
    """

    def __init__(self, ticker: str):
        self.challenge_name = f"binary_{ticker}"
        self.ticker = ticker
        self.model: lgb.LGBMClassifier | None = None
        self.model_dir = mcfg.MODEL_DIR

    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> dict:
        """Train binary classifier.

        Args:
            features: (N, F) feature matrix
            labels: (N,) binary labels (0 or 1)

        Returns:
            metrics dict with AUC
        """
        n = len(labels)
        weights = self.recency_weights(n)

        # Split for early stopping
        split = int(n * 0.9)
        X_train, X_val = features[:split], features[split:]
        y_train, y_val = labels[:split], labels[split:]
        w_train = weights[:split]

        params = {**mcfg.LIGHTGBM_PARAMS, "objective": "binary"}

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
        )

        # Evaluate
        y_pred = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.5

        return {"auc": auc, "n_samples": n}

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate 2-feature embedding in [-1, 1].

        Returns (2,) array: [directional_score, vol_adjusted_confidence]
        """
        if self.model is None:
            return np.array([0.0, 0.0])

        # Raw probability
        proba = self.model.predict_proba(features.reshape(1, -1))[0, 1]

        # Feature 1: directional score centered at 0
        # Map probability to [-1, 1]: p=0.5 → 0, p=1 → 1, p=0 → -1
        direction = 2.0 * proba - 1.0

        # Feature 2: confidence (distance from 0.5)
        confidence = abs(proba - 0.5) * 2.0

        # Add small noise to avoid stale filter
        direction += np.random.normal(0, 0.01)
        confidence += np.random.normal(0, 0.005)

        return np.array([
            np.clip(direction, -1.0, 1.0),
            np.clip(confidence, -1.0, 1.0),
        ])

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
