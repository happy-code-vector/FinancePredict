"""Walk-forward validation: expanding-window segmentation matching the validator."""

import numpy as np


def walk_forward_split(
    n_samples: int,
    chunk_size: int = 12000,
    lag: int = 60,
    min_train: int = 2000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window walk-forward splits.

    Matches the validator's protocol: train on all data up to (segment_start - lag),
    predict on segment. Each segment is chunk_size samples.

    Args:
        n_samples: total number of training samples
        chunk_size: samples per segment
        lag: embargo between train end and segment start
        min_train: minimum training samples for first segment

    Yields:
        list of (train_indices, test_indices) tuples
    """
    splits = []
    seg_start = min_train

    while seg_start + chunk_size <= n_samples:
        train_end = max(0, seg_start - lag)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(seg_start, min(seg_start + chunk_size, n_samples))

        if len(train_idx) >= min_train and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        seg_start += chunk_size

    return splits


def walk_forward_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    predict_fn,
    chunk_size: int = 12000,
    lag: int = 60,
    metric_fn=None,
) -> list[dict]:
    """Run walk-forward evaluation.

    Args:
        features: (N, F) feature matrix
        labels: (N,) label array
        predict_fn: callable(features_train, labels_train, features_test) -> predictions
        chunk_size: segment size
        lag: embargo
        metric_fn: callable(y_true, y_pred) -> float

    Returns:
        list of segment metrics dicts
    """
    splits = walk_forward_split(len(labels), chunk_size, lag)
    results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = features[train_idx], labels[train_idx]
        X_test, y_test = features[test_idx], labels[test_idx]

        y_pred = predict_fn(X_train, y_train, X_test)

        result = {"segment": i, "n_train": len(train_idx), "n_test": len(test_idx)}
        if metric_fn:
            result["metric"] = metric_fn(y_test, y_pred)
        results.append(result)

    return results
