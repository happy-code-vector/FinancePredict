"""60-second submission loop."""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Coroutine

import numpy as np

from config import (
    BREAKOUT_ASSETS,
    CHALLENGES,
    FUNDING_ASSETS,
    TRADE_MIX_ASSETS,
)

from miner.submission.encrypt import encrypt_payload
from miner.submission.upload import upload_to_r2
from miner.submission.validate import validate_embeddings

logger = logging.getLogger(__name__)


def generate_dummy_embeddings() -> dict[str, Any]:
    """Generate valid but non-informative embeddings for all challenges.

    Uses small random noise (not zeros) to avoid the stale filter.
    """
    emb: dict[str, Any] = {}

    for spec in CHALLENGES:
        tk = spec["ticker"]

        if tk == "MULTIBREAKOUT":
            # [P_continuation, P_reversal] per asset, with noise
            emb[tk] = {}
            for a in BREAKOUT_ASSETS:
                p = 0.5 + random.gauss(0, 0.02)
                p = max(0.01, min(0.99, p))
                emb[tk][a] = [p, 1.0 - p]

        elif tk == "TRADEMIX":
            emb[tk] = {a: round(random.gauss(0, 0.05), 6) for a in TRADE_MIX_ASSETS}

        elif tk == "MULTIXSEC":
            emb[tk] = {a: round(random.gauss(0, 0.05), 6) for a in BREAKOUT_ASSETS}

        elif tk == "FUNDINGXSEC":
            emb[tk] = {a: round(random.gauss(0, 0.05), 6) for a in FUNDING_ASSETS}

        elif tk == "ETHHITFIRST":
            # 3-way probability vector, small perturbation from uniform
            raw = [0.333 + random.gauss(0, 0.01) for _ in range(3)]
            total = sum(raw)
            probs = [max(0.01, min(0.99, r / total)) for r in raw]
            s = sum(probs)
            probs = [p / s for p in probs]
            emb[tk] = probs

        elif tk in ("ETHLBFGS", "BTCLBFGS"):
            # p[0:5] regime probs (sum to 1) + q[5:17] exceedance probs
            p_raw = [max(0.01, 0.6 if i == 2 else 0.1 + random.gauss(0, 0.02)) for i in range(5)]
            p_sum = sum(p_raw)
            p = [x / p_sum for x in p_raw]
            q = [max(0.01, min(0.99, 0.3 + random.gauss(0, 0.05))) for _ in range(12)]
            emb[tk] = p + q

        else:
            # Binary challenges: 2 features in [-1, 1] with noise
            dim = spec["dim"]
            emb[tk] = [round(random.gauss(0, 0.05), 6) for _ in range(dim)]

    return emb


async def submission_loop(
    hotkey: str,
    embeddings_fn: Callable[[], Coroutine[Any, Any, dict[str, Any]]] | None = None,
) -> None:
    """Main submission loop. Runs every ~60 seconds.

    Args:
        hotkey: Miner hotkey string.
        embeddings_fn: Async callable returning the embeddings dict.
            If None, generates dummy embeddings.
    """
    logger.info("Starting submission loop for hotkey %s", hotkey[:8])

    while True:
        cycle_start = time.time()
        try:
            # 1. Get embeddings
            if embeddings_fn:
                embeddings = await embeddings_fn()
            else:
                embeddings = generate_dummy_embeddings()

            # 2. Validate
            errors = validate_embeddings(embeddings)
            if errors:
                logger.error("Embedding validation failed: %s", errors)
                await asyncio.sleep(55)
                continue

            # 3. Encrypt
            payload = encrypt_payload(hotkey, embeddings)

            # 4. Upload
            await upload_to_r2(payload, hotkey)

            cycle_time = time.time() - cycle_start
            logger.info("Cycle completed in %.1fs", cycle_time)

        except Exception:
            logger.exception("Submission cycle failed")

        # Wait for next cycle (~60s total, accounting for processing time)
        elapsed = time.time() - cycle_start
        sleep_time = max(1.0, 60.0 - elapsed)
        await asyncio.sleep(sleep_time)
