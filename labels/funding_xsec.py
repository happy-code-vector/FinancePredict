"""Funding cross-sectional labels: y = 1[funding_change > median]."""

import numpy as np


def compute_funding_xsec_labels(
    funding_matrix: np.ndarray,
    assets: list[str],
    blocks_ahead: int = 2400,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute funding rate cross-sectional change labels.

    Args:
        funding_matrix: (T, N) array of funding rates for N assets
        assets: list of asset names (length N)
        blocks_ahead: forward horizon in samples

    Returns:
        y: (M,) binary labels pooled across all assets
        valid_idx: (M, 2) array of (time_idx, asset_idx) pairs
    """
    T, N = funding_matrix.shape

    if T <= blocks_ahead:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 2), dtype=int)

    n = T - blocks_ahead

    # Funding rate change: Δf = f[t+h] - f[t]
    delta_fr = funding_matrix[blocks_ahead:] - funding_matrix[:n]

    # Cross-sectional median of changes: (n, 1)
    med_delta = np.nanmedian(delta_fr, axis=1, keepdims=True)

    # Binary label
    y_2d = np.where(
        np.isnan(delta_fr) | np.isnan(med_delta),
        np.nan,
        (delta_fr > med_delta).astype(np.float32),
    )

    # Pool valid entries
    valid_mask = ~np.isnan(y_2d)
    valid_t, valid_a = np.where(valid_mask)
    y_pooled = y_2d[valid_t, valid_a]
    valid_idx = np.column_stack([valid_t, valid_a])

    return y_pooled, valid_idx
