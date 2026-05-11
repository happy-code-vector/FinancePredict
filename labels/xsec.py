"""Cross-sectional rank labels: y = 1[return > cross-sectional median]."""

import numpy as np


def compute_xsec_labels(
    price_matrix: np.ndarray,
    assets: list[str],
    blocks_ahead: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-sectional return rank labels.

    Args:
        price_matrix: (T, N) array of prices for N assets
        assets: list of asset names (length N)
        blocks_ahead: forward horizon in samples

    Returns:
        y: (M,) binary labels pooled across all assets
        valid_idx: (M, 2) array of (time_idx, asset_idx) pairs
    """
    T, N = price_matrix.shape
    eps = 1e-12

    if T <= blocks_ahead:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 2), dtype=int)

    n = T - blocks_ahead

    # Forward returns: (n, N)
    fwd_ret = (price_matrix[blocks_ahead:] - price_matrix[:n]) / (
        price_matrix[:n] + eps
    )

    # Cross-sectional median at each timestep: (n, 1)
    med = np.nanmedian(fwd_ret, axis=1, keepdims=True)

    # Binary label: 1 if above cross-sectional median
    y_2d = np.where(
        np.isnan(fwd_ret) | np.isnan(med),
        np.nan,
        (fwd_ret > med).astype(np.float32),
    )

    # Pool: flatten valid entries
    valid_mask = ~np.isnan(y_2d)
    valid_t, valid_a = np.where(valid_mask)
    y_pooled = y_2d[valid_t, valid_a]
    valid_idx = np.column_stack([valid_t, valid_a])

    return y_pooled, valid_idx
