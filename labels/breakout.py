"""Breakout labels: continuation vs reversal after range breach.

Reuses the RangeBreakoutTracker from the validator's range_breakout.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from range_breakout import RangeBreakoutTracker
from config import RANGE_LOOKBACK_BLOCKS


def compute_breakout_labels(
    prices: np.ndarray,
    lookback: int = 28800,  # 4 days of blocks
    barrier_pct: float = 25.0,
    min_range_pct: float = 1.0,
) -> tuple[list[dict], int]:
    """Run the breakout state machine on a price series and extract labeled episodes.

    Returns:
        episodes: list of dicts with keys:
            - trigger_idx: index where breakout triggered
            - direction: 'up' or 'down'
            - label: 1 (continuation) or 0 (reversal)
            - range_width: width of the pre-breakout range
        n_pending: number of unresolved breakouts
    """
    tracker = RangeBreakoutTracker(
        lookback_sidx=lookback,
        barrier_pct=barrier_pct,
        min_range_pct=min_range_pct,
    )

    episodes = []
    n_pending = 0

    for i, price in enumerate(prices):
        if not np.isfinite(price) or price <= 0:
            continue

        result = tracker.update(i, float(price))
        if result is None:
            continue

        if result.status == "resolved":
            episodes.append({
                "trigger_idx": result.trigger_idx,
                "direction": result.direction,
                "label": 1 if result.outcome == "continuation" else 0,
                "range_width": result.range_width,
            })
        elif result.status == "triggered":
            n_pending += 1

    return episodes, n_pending
