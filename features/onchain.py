"""On-chain features from CryptoQuant."""

import numpy as np

from miner.features.base import FeatureEngine


class OnchainFeatures(FeatureEngine):
    """Computes on-chain flow features."""

    warmup = 30

    def compute(
        self,
        inflows: np.ndarray,
        outflows: np.ndarray,
        reserves: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute on-chain features.

        Args:
            inflows: 1-D exchange inflow history
            outflows: 1-D exchange outflow history
            reserves: optional 1-D exchange reserve history

        Returns:
            1-D feature vector
        """
        eps = 1e-12
        features = []

        # Net flow (outflow - inflow, positive = bullish)
        net = outflows[-1] - inflows[-1] if len(outflows) > 0 else 0.0
        features.append(net)

        # Net flow z-score
        if len(inflows) > 30:
            net_series = outflows[-30:] - inflows[-30:]
            features.append(net / (np.std(net_series) + eps))
        else:
            features.append(0.0)

        # Net flow momentum (3-period)
        if len(inflows) > 3:
            net_3 = (outflows[-3:] - inflows[-3:]).sum()
            features.append(net_3)
        else:
            features.append(0.0)

        # Inflow spike detection
        if len(inflows) > 30:
            features.append((inflows[-1] - np.mean(inflows[-30:])) / (np.std(inflows[-30:]) + eps))
        else:
            features.append(0.0)

        # Outflow spike detection
        if len(outflows) > 30:
            features.append((outflows[-1] - np.mean(outflows[-30:])) / (np.std(outflows[-30:]) + eps))
        else:
            features.append(0.0)

        # Reserve change
        if reserves is not None and len(reserves) > 1:
            features.append(reserves[-1] - reserves[-2])
        else:
            features.append(0.0)

        # Reserve z-score
        if reserves is not None and len(reserves) > 30:
            features.append((reserves[-1] - np.mean(reserves[-30:])) / (np.std(reserves[-30:]) + eps))
        else:
            features.append(0.0)

        return np.array(features, dtype=float)
