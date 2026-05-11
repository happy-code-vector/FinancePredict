"""Entry point for the submitter container.

Loads trained models, runs the 60s submission loop.
Falls back to dummy embeddings if no models are trained yet.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import bittensor as bt
import miner.config as mcfg
from miner.data.store import Store
from miner.submission.commit import commit_url_once
from miner.submission.loop import submission_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("submitter")


def get_hotkey() -> str:
    wallet = bt.wallet(name=mcfg.WALLET_NAME, hotkey=mcfg.HOTKEY_NAME)
    return str(wallet.hotkey.ss58_address)


async def run():
    hotkey = get_hotkey()
    logger.info("Submitter starting — hotkey: %s", hotkey)

    # One-time on-chain commit
    try:
        await commit_url_once(hotkey)
    except Exception:
        logger.exception("On-chain commit failed — may already be committed")

    # Start submission (uses dummy embeddings until models exist)
    await submission_loop(hotkey=hotkey)


if __name__ == "__main__":
    asyncio.run(run())
