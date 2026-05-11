"""R2 upload for encrypted payloads."""

import json
import logging
from typing import Any

import aiobotocore.session

import miner.config as mcfg

logger = logging.getLogger(__name__)


async def upload_to_r2(payload: dict[str, Any], hotkey: str) -> None:
    """Upload encrypted payload JSON to R2 bucket. Object key = hotkey."""
    session = aiobotocore.session.get_session()
    async with session.create_client(
        "s3",
        endpoint_url=f"https://{mcfg.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=mcfg.R2_ACCESS_KEY_ID,
        aws_secret_access_key=mcfg.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    ) as client:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await client.put_object(
            Bucket=mcfg.R2_BUCKET_NAME,
            Key=hotkey,
            Body=body,
            ContentType="application/json",
        )
        logger.info("Uploaded payload to R2 (key=%s, size=%d bytes)", hotkey[:8], len(body))
