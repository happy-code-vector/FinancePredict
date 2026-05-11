"""Scheduled retraining: full retrain every 6h, incremental every 30m."""

import asyncio
import logging
import time

from miner.data.store import Store
from miner.training.trainer import train_all

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """Manages periodic model retraining."""

    def __init__(self, store: Store):
        self.store = store
        self.last_full_retrain = 0.0
        self.last_incremental = 0.0

    async def run(self) -> None:
        """Run the retraining loop indefinitely."""
        while True:
            now = time.time()

            # Full retrain every 6 hours
            if now - self.last_full_retrain > 6 * 3600:
                logger.info("Starting full retrain ...")
                try:
                    results = train_all(self.store)
                    self.last_full_retrain = now
                    logger.info("Full retrain complete: %s", {k: type(v).__name__ for k, v in results.items()})
                except Exception:
                    logger.exception("Full retrain failed")

            await asyncio.sleep(300)  # Check every 5 minutes
