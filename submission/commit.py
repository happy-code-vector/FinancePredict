"""One-time on-chain commit of the R2 URL."""

import logging

import bittensor as bt

import miner.config as mcfg

logger = logging.getLogger(__name__)


async def commit_url_once(hotkey: str) -> None:
    """Commit the R2 public URL on-chain so validators know where to find payloads."""
    r2_url = f"{mcfg.R2_PUBLIC_URL.rstrip('/')}/{hotkey}"
    wallet = bt.wallet(name=mcfg.WALLET_NAME, hotkey=mcfg.HOTKEY_NAME)
    subtensor = bt.subtensor(network=mcfg.SUBTENSOR_NETWORK)

    logger.info("Committing R2 URL on-chain: %s", r2_url)
    subtensor.commit(wallet=wallet, netuid=mcfg.NETUID, data=r2_url)
    logger.info("Commit successful.")
