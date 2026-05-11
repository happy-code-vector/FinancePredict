"""
Miner-specific configuration.
Sensitive values loaded from environment variables or .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINER_ROOT = Path(__file__).resolve().parent

# Docker volumes mount at /app/data and /app/saved_models
# Local runs use miner/ subdirectories
DATA_DIR = Path(os.getenv("MANTIS_DATA_DIR", str(MINER_ROOT / "data_cache")))
MODEL_DIR = Path(os.getenv("MANTIS_MODEL_DIR", str(MINER_ROOT / "saved_models")))
DB_PATH = DATA_DIR / "miner.duckdb"
PARQUET_DIR = DATA_DIR / "parquet"

for d in (MODEL_DIR, DATA_DIR, PARQUET_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bittensor / On-chain
# ---------------------------------------------------------------------------
NETUID = 123
WALLET_NAME = os.getenv("MANTIS_WALLET_NAME", "default")
HOTKEY_NAME = os.getenv("MANTIS_HOTKEY_NAME", "default")
SUBTENSOR_NETWORK = os.getenv("MANTIS_NETWORK", "finney")

# ---------------------------------------------------------------------------
# Cloudflare R2
# ---------------------------------------------------------------------------
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")  # e.g. https://bucket.r2.dev

# ---------------------------------------------------------------------------
# External API Keys
# ---------------------------------------------------------------------------
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY", "")
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "")
LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
SUBMISSION_INTERVAL_SECS = 60
LOCK_SECONDS = int(os.getenv("TLOCK_LOCK_SECONDS", "30"))
RETRAIN_FULL_INTERVAL_SECS = 6 * 3600      # 6 hours
RETRAIN_INCREMENTAL_SECS = 30 * 60          # 30 minutes

# ---------------------------------------------------------------------------
# Feature / Model defaults
# ---------------------------------------------------------------------------
RECENTCY_HALF_LIFE_DAYS = 15
WALK_FORWARD_CHUNK = 12000
TRAINING_LAG = 60
LIGHTGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": 1,
    "seed": 42,
}
