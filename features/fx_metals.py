"""FX and metals features from yfinance + FRED + GNews data."""

import numpy as np

from miner.features.base import FeatureEngine


class FXMetalsFeatures(FeatureEngine):
    """Computes features for FX pairs and precious metals.

    Combines price features with FRED macro data and GNews sentiment.
    """

    warmup = 60

    # Fixed output size: 11 (price) + 8 (FRED) + 3 (GNews) = 22
    OUTPUT_DIM = 22

    def compute(
        self,
        prices: np.ndarray,
        fred: dict[str, float] | None = None,
        gnews: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute FX/metal features with macro and news context.

        Args:
            prices: 1-D price history (most recent last)
            fred: dict of FRED macro values (from store)
            gnews: dict of GNews sentiment scores (from store)

        Returns:
            1-D feature vector (26 dims)
        """
        eps = 1e-12
        features = []

        if len(prices) < self.warmup:
            return np.zeros(self.OUTPUT_DIM, dtype=float)

        log_ret = np.diff(np.log(prices + eps))

        # --- Price features (15 dims) ---
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

        # --- FRED macro features (8 dims) ---
        fred = fred or {}
        fred_keys = [
            "fed_funds_rate",
            "treasury_spread_10y2y",
            "dxy_broad",
            "cpi",
            "unemployment",
            "breakeven_inflation",
            "usd_cad",
            "usd_chf",
        ]
        # Normalize FRED values: store raw, model learns scaling
        for key in fred_keys:
            features.append(fred.get(key, 0.0))

        # --- GNews sentiment features (3 dims) ---
        gnews = gnews or {}
        for topic in ["crypto", "forex", "commodities"]:
            features.append(gnews.get(topic, 0.0))

        return np.array(features, dtype=float)
