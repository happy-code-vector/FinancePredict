"""FX and metals features from yfinance data."""

import numpy as np

from miner.features.base import FeatureEngine


class FXMetalsFeatures(FeatureEngine):
    """Computes features for FX pairs and precious metals."""

    warmup = 60

    def compute(self, prices: np.ndarray) -> np.ndarray:
        """Compute FX/metal features.

        Args:
            prices: 1-D price history (most recent last)

        Returns:
            1-D feature vector
        """
        eps = 1e-12
        features = []

        if len(prices) < self.warmup:
            return np.zeros(15, dtype=float)

        log_ret = np.diff(np.log(prices + eps))

        # Returns at multiple horizons
        for h in [1, 5, 15, 60]:
            if len(log_ret) >= h:
                features.append(np.sum(log_ret[-h:]))
            else:
                features.append(0.0)

        # Rolling stats
        for w in [15, 60]:
            window = log_ret[-w:]
            features.append(np.mean(window))
            features.append(np.std(window))

        # RSI
        delta = np.diff(prices[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gains) + eps
        avg_loss = np.mean(losses) + eps
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        features.append(rsi / 100 - 0.5)

        # Momentum percentile
        if len(log_ret) > 60:
            features.append(np.mean(log_ret[-60:] < log_ret[-1]))
        else:
            features.append(0.5)

        # Mean reversion
        if len(prices) > 20:
            ma20 = np.mean(prices[-20:])
            features.append((prices[-1] - ma20) / (np.std(prices[-20:]) + eps))
        else:
            features.append(0.0)

        return np.array(features, dtype=float)
