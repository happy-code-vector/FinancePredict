"""Training orchestrator: coordinates label construction, feature computation, and model training."""

import logging
from pathlib import Path

import numpy as np

import miner.config as mcfg
from miner.data.store import Store
from miner.labels.binary import compute_binary_labels
from miner.labels.lbfgs import compute_regime_labels, compute_exceedance_targets
from miner.labels.hitfirst import compute_hitfirst_labels
from miner.features.price import PriceFeatures
from miner.features.fx_metals import FXMetalsFeatures
from miner.models.binary_model import BinaryModel
from miner.models.lbfgs_model import LBFGSModel
from miner.models.hitfirst_model import HitfirstModel

logger = logging.getLogger(__name__)


def train_binary_models(store: Store, tickers: list[str]) -> dict:
    """Train all binary challenge models."""
    results = {}
    price_fe = PriceFeatures()
    fx_fe = FXMetalsFeatures()

    for ticker in tickers:
        logger.info("Training binary model for %s ...", ticker)
        rows = store.get_prices(ticker, limit=500000)
        if len(rows) < 2000:
            logger.warning("Insufficient data for %s (%d rows)", ticker, len(rows))
            continue

        prices = np.array([r[1] for r in reversed(rows)], dtype=float)

        # Labels
        y, valid_idx = compute_binary_labels(prices, blocks_ahead=300)
        if len(y) < 1000:
            logger.warning("Too few labels for %s", ticker)
            continue

        # Features (compute for each valid index)
        fe = fx_fe if ticker in ("CADUSD", "NZDUSD", "CHFUSD", "XAGUSD") else price_fe
        feature_rows = []
        for idx in valid_idx:
            start = max(0, idx - fe.warmup)
            f = fe.compute(prices[start : idx + 1])
            feature_rows.append(f)

        X = np.array(feature_rows)
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X, y = X[:min_len], y[:min_len]

        model = BinaryModel(ticker)
        metrics = model.train(X, y)
        model.save()

        results[ticker] = metrics
        logger.info("  %s: AUC=%.4f (%d samples)", ticker, metrics.get("auc", 0), len(y))

    return results


def train_lbfgs_models(store: Store, configs: list[dict]) -> dict:
    """Train LBFGS models for ETH and BTC."""
    results = {}
    price_fe = PriceFeatures()

    for cfg in configs:
        ticker = cfg["ticker"]
        price_key = cfg.get("price_key", ticker)
        horizon = cfg["blocks_ahead"]

        logger.info("Training LBFGS model for %s (horizon=%d) ...", ticker, horizon)
        rows = store.get_prices(price_key, limit=500000)
        if len(rows) < 10000:
            logger.warning("Insufficient data for %s", ticker)
            continue

        prices = np.array([r[1] for r in reversed(rows)], dtype=float)

        # Regime labels
        regime_y, regime_valid = compute_regime_labels(prices, horizon_steps=horizon)
        # Exceedance targets
        q_targets, q_valid = compute_exceedance_targets(prices, horizon_steps=horizon)

        if len(regime_y) < 5000:
            logger.warning("Too few regime labels for %s", ticker)
            continue

        # Features
        feature_rows = []
        for idx in regime_valid:
            start = max(0, idx - price_fe.warmup)
            f = price_fe.compute(prices[start : idx + 1])
            feature_rows.append(f)

        X = np.array(feature_rows)
        min_len = min(len(X), len(regime_y))
        X, regime_y = X[:min_len], regime_y[:min_len]
        q_targets = q_targets[:min_len] if len(q_targets) >= min_len else q_targets

        model = LBFGSModel(ticker)
        metrics = model.train(X, regime_y, q_targets)
        model.save()

        results[ticker] = metrics
        logger.info(
            "  %s: regime_acc=%.4f, q_mse=%.6f",
            ticker, metrics.get("regime_balanced_acc", 0), metrics.get("q_avg_mse", 0),
        )

    return results


def train_hitfirst_model(store: Store) -> dict:
    """Train the HITFIRST model."""
    logger.info("Training HITFIRST model ...")
    price_fe = PriceFeatures()

    rows = store.get_prices("ETH", limit=500000)
    if len(rows) < 5000:
        logger.warning("Insufficient ETH data for HITFIRST")
        return {}

    prices = np.array([r[1] for r in reversed(rows)], dtype=float)
    labels, valid_idx = compute_hitfirst_labels(prices, blocks_ahead=500)

    if len(labels) < 2000:
        logger.warning("Too few HITFIRST labels")
        return {}

    feature_rows = []
    for idx in valid_idx:
        start = max(0, idx - price_fe.warmup)
        f = price_fe.compute(prices[start : idx + 1])
        feature_rows.append(f)

    X = np.array(feature_rows)
    min_len = min(len(X), len(labels))
    X, labels = X[:min_len], labels[:min_len]

    model = HitfirstModel()
    metrics = model.train(X, labels)
    model.save()

    logger.info("  HITFIRST: acc=%.4f (%d samples)", metrics.get("accuracy", 0), len(labels))
    return metrics


def train_all(store: Store | None = None) -> dict:
    """Run full training pipeline."""
    store = store or Store()

    logger.info("=== Starting full training pipeline ===")
    all_results = {}

    # Binary challenges
    binary_tickers = ["ETH", "CADUSD", "NZDUSD", "CHFUSD", "XAGUSD"]
    all_results["binary"] = train_binary_models(store, binary_tickers)

    # LBFGS challenges
    lbfgs_configs = [
        {"ticker": "ETHLBFGS", "price_key": "ETH", "blocks_ahead": 300},
        {"ticker": "BTCLBFGS", "price_key": "BTC", "blocks_ahead": 1800},
    ]
    all_results["lbfgs"] = train_lbfgs_models(store, lbfgs_configs)

    # HITFIRST
    all_results["hitfirst"] = train_hitfirst_model(store)

    logger.info("=== Training complete ===")
    return all_results
