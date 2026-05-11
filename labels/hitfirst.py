"""HITFIRST labels: 3-class barrier-hit prediction.

Label 0: up barrier hit first
Label 1: down barrier hit first
Label 2: neither barrier hit within horizon
"""

import numpy as np

from utils import sigma_from_price


def compute_hitfirst_labels(
    prices: np.ndarray,
    blocks_ahead: int = 500,
    vol_window: int = 7200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3-class hitfirst labels.

    Barriers are set at +/- 1σ of recent returns.

    Returns:
        y: int array of labels (0=up, 1=down, 2=neither)
        valid_idx: indices where the label is valid
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    horizon = blocks_ahead

    if T <= horizon:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    eps = 1e-12
    log_price = np.log(prices + eps)
    sigma = sigma_from_price(prices, return_horizon_steps=horizon, vol_window=vol_window)
    n = T - horizon

    labels = np.full(n, 2, dtype=int)  # default: neither
    valid = np.zeros(n, dtype=bool)

    for t in range(n):
        sig_t = sigma[t]
        if not np.isfinite(sig_t) or sig_t <= 0:
            continue
        valid[t] = True

        base_lp = log_price[t]
        seg = log_price[t + 1 : t + 1 + horizon] - base_lp

        idx_up = np.where(seg >= sig_t)[0]
        idx_dn = np.where(seg <= -sig_t)[0]

        has_up = idx_up.size > 0
        has_dn = idx_dn.size > 0

        if has_up and has_dn:
            labels[t] = 0 if idx_up[0] < idx_dn[0] else 1
        elif has_up:
            labels[t] = 0
        elif has_dn:
            labels[t] = 1
        # else: neither (already 2)

    valid_idx = np.where(valid)[0]
    return labels[valid_idx], valid_idx
