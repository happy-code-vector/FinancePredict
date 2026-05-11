"""Entry point for the trainer container.

Runs initial backfill if needed, then periodic retraining.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from miner.data.store import Store
from miner.data.backfill import run_backfill
from miner.training.retrain import RetrainScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trainer")


async def run():
    store = Store()

    # Run backfill on first start
    logger.info("Checking if backfill is needed ...")
    row_count = store.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    if row_count < 10000:
        logger.info("Insufficient data (%d rows) — running backfill", row_count)
        run_backfill(store)
    else:
        logger.info("Data sufficient (%d rows) — skipping backfill", row_count)

    # Start retraining loop
    scheduler = RetrainScheduler(store)
    await scheduler.run()


if __name__ == "__main__":
    asyncio.run(run())
