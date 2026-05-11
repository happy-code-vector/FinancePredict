"""LBFGS labels: 5 regime buckets + 12 exceedance probability targets.

Uses the exact same make_bins_from_price() from utils.py for regime labels.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import make_bins_from_price, sigma_from_price


def compute_regime_labels(
    prices: np.ndarray,
    horizon_steps: int = 300,
    vol_window: int = 7200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 5-class regime labels using make_bins_from_price.

    Buckets:
        0: z <= -2.0  (large loss)
        1: -2.0 < z <= -1.0
        2: -1.0 <= z <= 1.0  (neutral)
        3: 1.0 < z < 2.0
        4: z >= 2.0  (large gain)

    Returns:
        y: int array of regime labels (0-4)
        valid_idx: indices where label is valid
    """
    return make_bins_from_price(
        prices,
        horizon_steps=horizon_steps,
        vol_window=vol_window,
    )


def compute_exceedance_targets(
    prices: np.ndarray,
    horizon_steps: int = 300,
    vol_window: int = 7200,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0),
    tail_buckets: tuple[int, ...] = (0, 1, 3, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 12 binary exceedance probability targets.

    For each tail bucket (0, 1, 3, 4) and threshold (0.5σ, 1.0σ, 2.0σ),
    compute whether the return magnitude exceeds that threshold.

    Returns:
        q_targets: (N, 12) float array of binary targets
        valid_idx: indices where targets are valid
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)

    if T <= horizon_steps:
        return np.zeros((0, 12), dtype=np.float32), np.zeros(0, dtype=int)

    eps = 1e-12
    log_ret = np.log(prices[horizon_steps:] + eps) - np.log(prices[: T - horizon_steps] + eps)
    sigma = sigma_from_price(prices, return_horizon_steps=horizon_steps, vol_window=vol_window)
    sigma_start = sigma[: len(log_ret)]

    valid_mask = np.isfinite(sigma_start) & (sigma_start > 0)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return np.zeros((0, 12), dtype=np.float32), np.zeros(0, dtype=int)

    z = log_ret[valid_mask] / (sigma_start[valid_mask] + eps)

    # Regime labels for each valid index
    regime = np.zeros_like(z, dtype=int)
    regime[z <= -2.0] = 0
    regime[(z > -2.0) & (z < -1.0)] = 1
    regime[(z >= -1.0) & (z <= 1.0)] = 2
    regime[(z > 1.0) & (z < 2.0)] = 3
    regime[z >= 2.0] = 4

    # For each tail bucket and threshold, compute exceedance target
    q_targets = np.zeros((len(valid_idx), 12), dtype=np.float32)
    col = 0
    for bucket in tail_buckets:
        for threshold in thresholds:
            # Whether |return| exceeds threshold * sigma
            if bucket in (0, 1):
                # Negative tail: whether return < -threshold * sigma
                q_targets[:, col] = (z < -threshold).astype(np.float32)
            else:
                # Positive tail: whether return > threshold * sigma
                q_targets[:, col] = (z > threshold).astype(np.float32)
            col += 1

    return q_targets, valid_idx
