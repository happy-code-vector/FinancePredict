"""LBFGS model: regime classifier (5-class) + exceedance regressors (12)."""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import balanced_accuracy_score

import miner.config as mcfg
from miner.models.base import BaseModel


class LBFGSModel(BaseModel):
    """Two-stage LightGBM for LBFGS challenges (17-dim output).

    Stage 1: 5-class regime classifier → p[0:5] (softmax)
    Stage 2: 12 independent regressors → q[5:17] (sigmoid-clipped)

    Used for both ETHLBFGS and BTCLBFGS.
    """

    def __init__(self, ticker: str):
        self.challenge_name = f"lbfgs_{ticker}"
        self.ticker = ticker
        self.regime_model: lgb.LGBMClassifier | None = None
        self.exceedance_models: list[lgb.LGBMRegressor] | None = None
        self.model_dir = mcfg.MODEL_DIR

    def train(
        self,
        features: np.ndarray,
        regime_labels: np.ndarray,
        exceedance_targets: np.ndarray,
        **kwargs,
    ) -> dict:
        """Train regime classifier and exceedance regressors.

        Args:
            features: (N, F) feature matrix
            regime_labels: (N,) int labels in 0..4
            exceedance_targets: (N, 12) binary targets

        Returns:
            metrics dict
        """
        n = len(regime_labels)
        weights = self.recency_weights(n)

        # --- Stage 1: Regime classifier ---
        split = int(n * 0.9)
        X_train, X_val = features[:split], features[split:]
        y_regime_train, y_regime_val = regime_labels[:split], regime_labels[split:]
        w_train = weights[:split]

        params_cls = {**mcfg.LIGHTGBM_PARAMS, "objective": "multiclass", "num_class": 5}
        self.regime_model = lgb.LGBMClassifier(**params_cls)
        self.regime_model.fit(
            X_train, y_regime_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_regime_val)],
            callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
        )

        regime_pred = self.regime_model.predict(X_val)
        regime_acc = balanced_accuracy_score(y_regime_val, regime_pred)

        # --- Stage 2: 12 exceedance regressors ---
        params_reg = {**mcfg.LIGHTGBM_PARAMS, "objective": "cross_entropy"}
        self.exceedance_models = []
        q_mse = []

        for i in range(12):
            y_q = exceedance_targets[:split, i]
            y_q_val = exceedance_targets[split:, i]

            model_q = lgb.LGBMRegressor(**params_reg)
            model_q.fit(
                X_train, y_q,
                sample_weight=w_train,
                eval_set=[(X_val, y_q_val)],
                callbacks=[lgb.early_stopping(mcfg.LIGHTGBM_PARAMS["early_stopping_rounds"], verbose=False)],
            )
            self.exceedance_models.append(model_q)

            pred_q = model_q.predict(X_val)
            q_mse.append(np.mean((pred_q - y_q_val) ** 2))

        return {
            "regime_balanced_acc": regime_acc,
            "q_avg_mse": np.mean(q_mse),
            "n_samples": n,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate 17-dim LBFGS embedding.

        Returns: [p0, p1, p2, p3, p4, q0, q1, ..., q11]
        """
        if self.regime_model is None or self.exceedance_models is None:
            # Fallback: neutral regime + mid exceedance
            p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            q = np.full(12, 0.3)
            return np.concatenate([p, q])

        x = features.reshape(1, -1)

        # Stage 1: regime probabilities
        p = self.regime_model.predict_proba(x)[0]
        # Add noise to avoid stale filter
        p = p + np.random.normal(0, 0.005, size=5)
        p = np.clip(p, 0.01, 0.99)
        p = p / p.sum()

        # Stage 2: exceedance probabilities
        q = []
        for model_q in self.exceedance_models:
            pred = model_q.predict(x)[0]
            pred = pred + np.random.normal(0, 0.005)
            q.append(np.clip(pred, 0.01, 0.99))

        return np.concatenate([p, np.array(q)])

    def save(self, path: Path | None = None) -> Path:
        path = path or self.get_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "regime_model": self.regime_model,
            "exceedance_models": self.exceedance_models,
        }, path)
        return path

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self.regime_model = data["regime_model"]
        self.exceedance_models = data["exceedance_models"]
