"""Funding rate features."""

import numpy as np

from miner.features.base import FeatureEngine


class FundingFeatures(FeatureEngine):
    """Computes funding rate features for a single asset."""

    warmup = 30

    def compute(
        self,
        rates: np.ndarray,
        oi: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute funding rate features.

        Args:
            rates: 1-D funding rate history (most recent last)
            oi: optional 1-D open interest history

        Returns:
            1-D feature vector
        """
        eps = 1e-12
        features = []

        # Current level
        features.append(rates[-1] if len(rates) > 0 else 0.0)

        # Z-scored level
        if len(rates) > 30:
            features.append((rates[-1] - np.mean(rates[-30:])) / (np.std(rates[-30:]) + eps))
        else:
            features.append(0.0)

        # Rate changes over last 1, 2, 3 periods
        for lag in [1, 2, 3]:
            if len(rates) > lag:
                features.append(rates[-1] - rates[-1 - lag])
            else:
                features.append(0.0)

        # Mean reversion signal
        if len(rates) > 30:
            med = np.median(rates[-30:])
            features.append(rates[-1] - med)
        else:
            features.append(0.0)

        # Rate momentum (sum of last 5 changes)
        if len(rates) > 5:
            features.append(np.sum(np.diff(rates[-5:])))
        else:
            features.append(0.0)

        # OI change rate
        if oi is not None and len(oi) > 1:
            features.append((oi[-1] - oi[-2]) / (oi[-2] + eps))
        else:
            features.append(0.0)

        # OI z-score
        if oi is not None and len(oi) > 30:
            features.append((oi[-1] - np.mean(oi[-30:])) / (np.std(oi[-30:]) + eps))
        else:
            features.append(0.0)

        return np.array(features, dtype=float)
