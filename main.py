"""MANTIS Miner entry point."""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so config.py / generate_and_encrypt.py are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bittensor as bt

import miner.config as mcfg
from miner.submission.commit import commit_url_once
from miner.submission.loop import submission_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("miner")


def get_hotkey() -> str:
    """Load the hotkey string from the bittensor wallet."""
    wallet = bt.wallet(name=mcfg.WALLET_NAME, hotkey=mcfg.HOTKEY_NAME)
    return str(wallet.hotkey.ss58_address)


async def run() -> None:
    """Main async entry point."""
    hotkey = get_hotkey()
    logger.info("Miner hotkey: %s", hotkey)

    # One-time on-chain commit of R2 URL
    try:
        await commit_url_once(hotkey)
    except Exception:
        logger.exception("On-chain commit failed — may already be committed")

    # Start the submission loop (dummy embeddings until models are trained)
    await submission_loop(hotkey=hotkey)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
