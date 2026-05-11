"""Cross-sectional features: rank across assets."""

import numpy as np

from miner.features.base import FeatureEngine


class CrossSectionFeatures(FeatureEngine):
    """Computes cross-sectional rank features comparing one asset to the group."""

    warmup = 60

    def compute(
        self,
        asset_returns: np.ndarray,
        all_returns: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-sectional features for one asset.

        Args:
            asset_returns: 1-D returns for this asset (recent first)
            all_returns: (N, T) returns for all assets in the group

        Returns:
            1-D feature vector
        """
        features = []

        # Return rank (percentile within cross-section)
        latest_returns = all_returns[:, -1]
        rank = np.mean(latest_returns < asset_returns[-1])
        features.append(rank * 2 - 1)  # scale to [-1, 1]

        # 5-bar momentum rank
        if all_returns.shape[1] >= 5:
            mom5 = all_returns[:, -5:].sum(axis=1)
            asset_mom5 = asset_returns[-5:].sum()
            features.append(np.mean(mom5 < asset_mom5) * 2 - 1)
        else:
            features.append(0.0)

        # Volatility rank
        if all_returns.shape[1] >= 60:
            vols = np.std(all_returns[:, -60:], axis=1)
            asset_vol = np.std(asset_returns[-60:])
            features.append(np.mean(vols < asset_vol) * 2 - 1)
        else:
            features.append(0.0)

        # Mean reversion signal (distance from cross-sectional mean)
        cross_mean = np.mean(latest_returns)
        features.append(asset_returns[-1] - cross_mean)

        # Relative strength (asset vs group average over 15 bars)
        if all_returns.shape[1] >= 15:
            group_avg = all_returns[:, -15:].mean()
            asset_avg = asset_returns[-15:].mean()
            features.append(asset_avg - group_avg)
        else:
            features.append(0.0)

        return np.array(features, dtype=float)
