"""TRADE-MIX labels: forward returns as position targets."""

import numpy as np


def compute_trade_mix_labels(
    price_matrix: np.ndarray,
    assets: list[str],
    horizon_bars: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward return targets for TRADE-MIX.

    The target is the signed forward return, clipped to [-1, 1],
    used as a proxy for optimal position size.

    Args:
        price_matrix: (T, N) array of prices for N assets (BTC, ETH, TAO, SOL)
        assets: list of 4 asset names
        horizon_bars: forward horizon (default 60 = ~1 hour)

    Returns:
        targets: (M, N) array of signed position targets in [-1, 1]
        valid_idx: 1-D array of valid start indices
    """
    T, N = price_matrix.shape
    eps = 1e-12

    if T <= horizon_bars:
        return np.zeros((0, N), dtype=np.float32), np.zeros(0, dtype=int)

    n = T - horizon_bars
    fwd_ret = (price_matrix[horizon_bars:] - price_matrix[:n]) / (
        price_matrix[:n] + eps
    )

    # Clip to [-1, 1] as position targets
    targets = np.clip(fwd_ret, -1.0, 1.0).astype(np.float32)

    # Filter out rows with any NaN
    valid_mask = ~np.any(np.isnan(targets), axis=1)
    valid_idx = np.where(valid_mask)[0]
    targets = targets[valid_mask]

    return targets, valid_idx
