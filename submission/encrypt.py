"""
Encryption wrapper around the project's generate_v2().
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from generate_and_encrypt import generate_v2
from config import OWNER_HPKE_PUBLIC_KEY_HEX

import miner.config as mcfg


def encrypt_payload(hotkey: str, embeddings: dict[str, Any]) -> dict:
    """Convert embeddings dict to ordered list and produce a V2 encrypted payload.

    The generate_v2 function expects embeddings as a list ordered to match
    config.CHALLENGES. We convert from our dict format here.
    """
    # Convert embeddings dict -> ordered list matching CHALLENGES order
    from config import CHALLENGES
    embeddings_list = [embeddings[c["ticker"]] for c in CHALLENGES]

    payload = generate_v2(
        hotkey=hotkey,
        lock_seconds=mcfg.LOCK_SECONDS,
        owner_pk_hex=OWNER_HPKE_PUBLIC_KEY_HEX,
        payload_text=None,
        embeddings=embeddings_list,
    )
    return payload
