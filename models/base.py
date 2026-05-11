"""Base model class for all challenge models."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

import miner.config as mcfg


class BaseModel(ABC):
    """Abstract base for challenge prediction models."""

    challenge_name: str = ""
    model_dir: Path = mcfg.MODEL_DIR

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray, **kwargs) -> dict:
        """Train the model. Returns metrics dict."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for the given features."""
        ...

    @abstractmethod
    def save(self, path: Path | None = None) -> Path:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        ...

    def get_model_path(self, version: str = "latest") -> Path:
        """Get path for a model version."""
        return self.model_dir / f"{self.challenge_name}_{version}.joblib"

    @staticmethod
    def recency_weights(n: int, half_life_days: float = 15, samples_per_day: int = 1440) -> np.ndarray:
        """Compute exponential decay sample weights."""
        days = np.arange(n, dtype=float) / samples_per_day
        gamma = 0.5 ** (1.0 / half_life_days)
        w = gamma ** (days.max() - days)
        return (w / w.sum() * n).astype(np.float32)
