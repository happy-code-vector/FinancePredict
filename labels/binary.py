"""Binary challenge labels: y = 1[forward_return > 0]."""

import numpy as np


def compute_binary_labels(
    prices: np.ndarray,
    blocks_ahead: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary direction labels from a 1-D price series.

    Returns:
        y: binary labels (1 = price went up, 0 = price went down)
        valid_idx: indices where the label is valid (enough future data)
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)

    if T <= blocks_ahead:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    # Forward returns
    fwd_ret = (prices[blocks_ahead:] - prices[: T - blocks_ahead]) / (
        prices[: T - blocks_ahead] + 1e-12
    )

    # Binary label: 1 if positive return
    y = (fwd_ret > 0).astype(np.float32)
    valid_idx = np.arange(len(y), dtype=int)

    return y, valid_idx
