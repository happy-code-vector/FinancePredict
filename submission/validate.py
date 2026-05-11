"""
Pre-submission embedding validation.
Copied from MINER_GUIDE Section 6 and adapted for miner use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    BREAKOUT_ASSETS,
    CHALLENGES,
    FUNDING_ASSETS,
    TRADE_MIX_ASSETS,
)


def validate_embeddings(emb: dict) -> list[str]:
    """Validate an embeddings dict against all challenge specs.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    for spec in CHALLENGES:
        tk = spec["ticker"]
        if tk not in emb:
            errors.append(f"Missing {tk}")
            continue
        val = emb[tk]

        if tk == "MULTIBREAKOUT":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict")
                continue
            for a in BREAKOUT_ASSETS:
                v = val.get(a)
                if not isinstance(v, list) or len(v) != 2:
                    errors.append(f"{tk}.{a}: need [p_cont, p_rev]")
                elif not all(0 < x < 1 for x in v):
                    errors.append(f"{tk}.{a}: values must be in (0,1)")

        elif tk == "TRADEMIX":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict")
                continue
            for a in TRADE_MIX_ASSETS:
                v = val.get(a, None)
                if not isinstance(v, (int, float)) or not (-1 <= v <= 1):
                    errors.append(f"{tk}.{a}: need float in [-1,1]")

        elif tk == "MULTIXSEC":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict")
                continue
            for a in BREAKOUT_ASSETS:
                v = val.get(a, None)
                if not isinstance(v, (int, float)) or not (-1 <= v <= 1):
                    errors.append(f"{tk}.{a}: need float in [-1,1]")

        elif tk == "FUNDINGXSEC":
            if not isinstance(val, dict):
                errors.append(f"{tk}: expected dict")
                continue
            for a in FUNDING_ASSETS:
                v = val.get(a, None)
                if v is not None and (not isinstance(v, (int, float)) or not (-1 <= v <= 1)):
                    errors.append(f"{tk}.{a}: need float in [-1,1]")

        else:
            if not isinstance(val, list) or len(val) != spec["dim"]:
                errors.append(f"{tk}: expected list of length {spec['dim']}")

    return errors
