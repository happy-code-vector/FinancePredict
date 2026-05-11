"""Base class for feature engines."""

from abc import ABC, abstractmethod

import numpy as np


class FeatureEngine(ABC):
    """Abstract base for computing feature vectors."""

    warmup: int = 200  # minimum rows before features are valid

    @abstractmethod
    def compute(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Compute feature vector from input data.

        Args:
            data: 1-D or 2-D array of price/metric history

        Returns:
            1-D feature vector
        """
        ...
