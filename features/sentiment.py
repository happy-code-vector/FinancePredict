"""Sentiment features from LunarCrush and Alternative.me."""

import numpy as np

from miner.features.base import FeatureEngine


class SentimentFeatures(FeatureEngine):
    """Computes sentiment features."""

    warmup = 10

    def compute(
        self,
        social_volume: np.ndarray | None = None,
        engagement: np.ndarray | None = None,
        galaxy_score: np.ndarray | None = None,
        fear_greed: float | None = None,
    ) -> np.ndarray:
        """Compute sentiment features.

        Args:
            social_volume: 1-D social mention count history
            engagement: 1-D engagement rate history
            galaxy_score: 1-D LunarCrush galaxy score history
            fear_greed: latest Fear & Greed Index value (0-100)

        Returns:
            1-D feature vector
        """
        eps = 1e-12
        features = []

        # Social volume z-score
        if social_volume is not None and len(social_volume) > 10:
            features.append(
                (social_volume[-1] - np.mean(social_volume[-10:]))
                / (np.std(social_volume[-10:]) + eps)
            )
        else:
            features.append(0.0)

        # Social volume trend
        if social_volume is not None and len(social_volume) > 5:
            features.append(np.sum(np.diff(social_volume[-5:])))
        else:
            features.append(0.0)

        # Engagement change
        if engagement is not None and len(engagement) > 2:
            features.append(engagement[-1] - engagement[-2])
        else:
            features.append(0.0)

        # Galaxy score change
        if galaxy_score is not None and len(galaxy_score) > 2:
            features.append(galaxy_score[-1] - galaxy_score[-2])
        else:
            features.append(0.0)

        # Fear & Greed (normalized to [-1, 1])
        if fear_greed is not None:
            features.append(fear_greed / 50 - 1.0)
        else:
            features.append(0.0)

        return np.array(features, dtype=float)
