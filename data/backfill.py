"""Historical data backfill from datalog archive, CCXT, and yfinance."""

import logging
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yfinance as yf

import miner.config as mcfg
from miner.data.store import Store

logger = logging.getLogger(__name__)

DATALOG_URL = "https://pub-879ad825983e43529792665f4f510cd6.r2.dev/datalog.db"

# All crypto assets across challenges
ALL_CRYPTO_ASSETS = list(set(
    [
        "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
        "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
        "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
        "NEAR", "ICP", "ETC", "ONDO", "SKY",
    ]
))

# yfinance symbols for fiat/commodity
YF_TICKERS = {
    "CADUSD": "CADUSD=X",
    "NZDUSD": "NZDUSD=X",
    "CHFUSD": "CHFUSD=X",
    "XAGUSD": "SI=F",  # Silver futures
}


def download_datalog(dest: Path | None = None) -> Path:
    """Download the validator's datalog archive."""
    dest = dest or mcfg.DATA_DIR / "datalog.db"
    if dest.exists():
        logger.info("Datalog already exists at %s, skipping download", dest)
        return dest

    logger.info("Downloading datalog archive from %s ...", DATALOG_URL)
    resp = requests.get(DATALOG_URL, stream=True, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Datalog saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def extract_datalog_prices(datalog_path: Path, store: Store) -> int:
    """Extract historical prices from the validator's datalog SQLite DB into DuckDB."""
    logger.info("Extracting prices from datalog %s ...", datalog_path)
    con = sqlite3.connect(str(datalog_path), check_same_thread=False)

    # The datalog stores prices in challenge_data table
    # Schema: ticker, sidx, price, price_data, hotkeys, embeddings
    try:
        rows = con.execute(
            "SELECT DISTINCT ticker, sidx, price FROM challenge_data WHERE price IS NOT NULL ORDER BY sidx"
        ).fetchall()
    except Exception:
        logger.warning("Could not read challenge_data from datalog — trying blocks table")
        rows = []

    if not rows:
        logger.info("No price rows found in datalog")
        con.close()
        return 0

    # sidx * SAMPLE_EVERY * 12 seconds = approximate timestamp
    # SAMPLE_EVERY = 5 blocks, ~12s per block → sidx * 60s
    price_rows = []
    for ticker, sidx, price in rows:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sidx * 60)
        price_rows.append((ticker, ts, None, None, None, float(price), None))

    if price_rows:
        store.insert_prices(price_rows)
        logger.info("Extracted %d price rows from datalog", len(price_rows))

    con.close()
    return len(price_rows)


def backfill_yfinance(store: Store, period: str = "2y") -> int:
    """Fetch 2 years of FX/metal data from yfinance."""
    logger.info("Fetching yfinance data for %s ...", list(YF_TICKERS.keys()))
    total = 0

    for asset, symbol in YF_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1h")
            if hist.empty:
                logger.warning("No yfinance data for %s", asset)
                continue

            rows = []
            for ts, row in hist.iterrows():
                ts_utc = ts.to_pydatetime().replace(tzinfo=timezone.utc)
                rows.append((
                    asset, ts_utc,
                    float(row["Open"]), float(row["High"]),
                    float(row["Low"]), float(row["Close"]),
                    float(row["Volume"]),
                ))

            store.insert_prices(rows)
            total += len(rows)
            logger.info("  %s: %d rows", asset, len(rows))

        except Exception:
            logger.exception("Failed to fetch yfinance data for %s", asset)

    logger.info("yfinance backfill: %d total rows", total)
    return total


def backfill_ccxt(store: Store, days: int = 730) -> int:
    """Fetch historical OHLCV from exchanges via CCXT.

    Requires ccxt to be installed. Falls back gracefully if not available.
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("ccxt not installed — skipping CCXT backfill")
        return 0

    logger.info("Fetching CCXT OHLCV for %d assets (%d days) ...", len(ALL_CRYPTO_ASSETS), days)
    exchange = ccxt.binance({"enableRateLimit": True})
    total = 0
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )

    for asset in ALL_CRYPTO_ASSETS:
        symbol = f"{asset}/USDT"
        if symbol not in exchange.markets:
            # Try /BUSD or skip
            continue
        try:
            all_ohlcv = []
            cursor = since
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", since=cursor, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                cursor = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break

            rows = []
            for candle in all_ohlcv:
                ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                rows.append((
                    asset, ts,
                    candle[1], candle[2], candle[3], candle[4], candle[5],
                ))

            if rows:
                store.insert_prices(rows)
                total += len(rows)
                logger.info("  %s: %d rows", asset, len(rows))

        except Exception:
            logger.warning("Failed to fetch CCXT data for %s", asset)

    logger.info("CCXT backfill: %d total rows", total)
    return total


def run_backfill(store: Store | None = None) -> None:
    """Run the full backfill pipeline."""
    store = store or Store()
    logger.info("=== Starting data backfill ===")

    # 1. Datalog archive
    try:
        datalog_path = download_datalog()
        extract_datalog_prices(datalog_path, store)
    except Exception:
        logger.exception("Datalog backfill failed")

    # 2. CCXT historical OHLCV (1-2 years)
    try:
        backfill_ccxt(store, days=730)
    except Exception:
        logger.exception("CCXT backfill failed")

    # 3. yfinance FX + metals (2 years)
    try:
        backfill_yfinance(store, period="2y")
    except Exception:
        logger.exception("yfinance backfill failed")

    # 4. Export to Parquet for backup
    try:
        store.export_parquet("prices")
    except Exception:
        logger.exception("Parquet export failed")

    logger.info("=== Backfill complete ===")
