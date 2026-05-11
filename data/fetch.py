"""Live data fetching from all sources."""

import asyncio
import json
import logging
from datetime import datetime, timezone

import numpy as np
import requests
import yfinance as yf

import miner.config as mcfg
from miner.data.store import Store

logger = logging.getLogger(__name__)

# Canonical price feed from the validator
PRICE_FEED_URL = "https://pub-ba8c1b8edb8046edaccecbd26b5ca7f8.r2.dev/latest_prices.json"
ALTERNATIVE_ME_URL = "https://api.alternative.me/fng/"

# Assets we need from yfinance (ticker -> yfinance symbol mapping)
YF_TICKERS = {
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    "XAGUSD": "XAGUSD=X",
}


def fetch_canonical_prices() -> dict:
    """Fetch the canonical price feed from the validator's R2 bucket."""
    resp = requests.get(PRICE_FEED_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_fear_greed() -> tuple[int, str] | None:
    """Fetch the Fear & Greed Index from Alternative.me."""
    try:
        resp = requests.get(ALTERNATIVE_ME_URL, params={"limit": 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entry = data["data"][0]
        return int(entry["value"]), entry["value_classification"]
    except Exception:
        logger.exception("Failed to fetch Fear & Greed")
        return None


def fetch_yfinance_rates() -> dict[str, float]:
    """Fetch latest FX and metals rates from yfinance."""
    rates = {}
    for asset, symbol in YF_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                rates[asset] = float(hist["Close"].iloc[-1])
        except Exception:
            logger.warning("Failed to fetch yfinance rate for %s", asset)
    return rates


async def price_fetch_loop(store: Store) -> None:
    """Fetch canonical prices every 60 seconds and store them."""
    while True:
        try:
            data = fetch_canonical_prices()
            now = datetime.now(timezone.utc)
            rows = []

            prices = data.get("prices", {})
            if isinstance(prices, dict):
                for asset, price in prices.items():
                    if price is not None:
                        rows.append((asset, now, None, None, None, float(price), None))

            if rows:
                store.insert_prices(rows)
                logger.debug("Stored %d prices at %s", len(rows), now.isoformat())

        except Exception:
            logger.exception("Price fetch failed")

        await asyncio.sleep(60)


async def fear_greed_loop(store: Store) -> None:
    """Fetch Fear & Greed every hour."""
    while True:
        result = fetch_fear_greed()
        if result:
            value, classification = result
            now = datetime.now(timezone.utc)
            store.insert_fear_greed([(now, value, classification)])
            logger.debug("Fear & Greed: %d (%s)", value, classification)

        await asyncio.sleep(3600)


async def yfinance_loop(store: Store) -> None:
    """Fetch FX/metal rates every 5 minutes."""
    while True:
        rates = fetch_yfinance_rates()
        if rates:
            now = datetime.now(timezone.utc)
            rows = [(asset, now, None, None, None, rate, None) for asset, rate in rates.items()]
            store.insert_prices(rows)
            logger.debug("Stored %d FX/metal rates", len(rows))

        await asyncio.sleep(300)
