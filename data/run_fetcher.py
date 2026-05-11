"""Entry point for the fetcher container.

Runs all data fetch loops concurrently:
- R2 price feed (60s)
- CoinGlass funding/OI (5m)
- yfinance FX/metals (5m)
- Alternative.me Fear & Greed (1h)
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from miner.data.store import Store
from miner.data.fetch import price_fetch_loop, fear_greed_loop, yfinance_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fetcher")


async def run():
    store = Store()
    logger.info("Fetcher starting — running all data loops")

    await asyncio.gather(
        price_fetch_loop(store),
        yfinance_loop(store),
        fear_greed_loop(store),
    )


if __name__ == "__main__":
    asyncio.run(run())
