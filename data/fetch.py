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


# ---------------------------------------------------------------------------
# FRED (Federal Reserve Economic Data) — macro features for FX binary
# ---------------------------------------------------------------------------
FRED_BASE = "https://api.stlouisfed.org/fred"

# Key series for forex prediction
FRED_SERIES = {
    "DFF": "fed_funds_rate",          # Federal Funds Effective Rate
    "T10Y2Y": "treasury_spread_10y2y", # 10Y-2Y Treasury Spread
    "DEXCAUS": "usd_cad",             # Canada/US Exchange Rate
    "DEXUSNZ": "usd_nzd",             # US/New Zealand Exchange Rate
    "DEXSZUS": "usd_chf",             # Switzerland/US Exchange Rate
    "DTWEXBGS": "dxy_broad",          # Trade Weighted USD Index
    "CPIAUCSL": "cpi",                # Consumer Price Index
    "UNRATE": "unemployment",          # Unemployment Rate
    "T10YIE": "breakeven_inflation",  # 10-Yr Breakeven Inflation
}


def fetch_fred_series() -> dict[str, float]:
    """Fetch latest values from FRED for key macro series."""
    if not mcfg.FRED_API_KEY:
        return {}

    values = {}
    for series_id, name in FRED_SERIES.items():
        try:
            resp = requests.get(
                f"{FRED_BASE}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": mcfg.FRED_API_KEY,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            obs = resp.json()["observations"]
            if obs and obs[0]["value"] != ".":
                values[name] = float(obs[0]["value"])
        except Exception:
            logger.debug("FRED series %s fetch failed", series_id)

    return values


async def fred_loop(store: Store) -> None:
    """Fetch FRED macro data every 4 hours (daily publishing schedule)."""
    while True:
        values = fetch_fred_series()
        if values:
            now = datetime.now(timezone.utc)
            # Store as synthetic "asset" rows for feature computation
            for name, val in values.items():
                store.execute(
                    "INSERT OR REPLACE INTO prices VALUES (?, ?, NULL, NULL, NULL, ?, NULL)",
                    [(f"FRED_{name}", now, val)],
                )
            logger.debug("Stored %d FRED series", len(values))

        await asyncio.sleep(14400)


# ---------------------------------------------------------------------------
# GNews — news sentiment for all challenges
# ---------------------------------------------------------------------------
GNEWS_BASE = "https://gnews.io/api/v4"

GNEWS_TOPICS = {
    "crypto": ["bitcoin", "ethereum", "cryptocurrency"],
    "forex": ["forex", "dollar", "federal reserve"],
    "commodities": ["silver", "gold", "commodities"],
}


def fetch_gnews_sentiment() -> dict[str, float]:
    """Fetch recent news and compute simple sentiment scores per topic.

    Returns a dict of topic -> sentiment score in [-1, 1].
    Uses headline keyword counting as a lightweight proxy.
    """
    if not mcfg.GNEWS_API_KEY:
        return {}

    scores = {}
    bullish_kw = {"rise", "surge", "rally", "gain", "bull", "high", "up", "grow", "boost", "positive"}
    bearish_kw = {"fall", "drop", "crash", "decline", "bear", "low", "down", "sink", "loss", "negative", "fear", "risk"}

    for topic, keywords in GNEWS_TOPICS.items():
        try:
            resp = requests.get(
                f"{GNEWS_BASE}/search",
                params={
                    "q": " OR ".join(keywords),
                    "lang": "en",
                    "max": 10,
                    "apikey": mcfg.GNEWS_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

            if not articles:
                scores[topic] = 0.0
                continue

            pos, neg = 0, 0
            for a in articles:
                title = (a.get("title", "") + " " + a.get("description", "")).lower()
                pos += sum(1 for kw in bullish_kw if kw in title)
                neg += sum(1 for kw in bearish_kw if kw in title)

            total = pos + neg
            if total > 0:
                scores[topic] = (pos - neg) / total  # [-1, 1]
            else:
                scores[topic] = 0.0

        except Exception:
            logger.debug("GNews topic %s fetch failed", topic)
            scores[topic] = 0.0

    return scores


async def gnews_loop(store: Store) -> None:
    """Fetch news sentiment every 30 minutes."""
    while True:
        scores = fetch_gnews_sentiment()
        if scores:
            now = datetime.now(timezone.utc)
            for topic, score in scores.items():
                store.execute(
                    "INSERT OR REPLACE INTO sentiment VALUES (?, ?, NULL, NULL, NULL, NULL)",
                    [(f"GNEWS_{topic}", now, score)],
                )
            logger.debug("Stored %d GNews sentiment scores", len(scores))

        await asyncio.sleep(1800)
