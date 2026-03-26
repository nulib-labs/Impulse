"""
Centralised configuration for the Impulse pipeline.

All tunables come from environment variables so they work identically
in Lambda, ECS Fargate, and local development.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_secret(secret_id: str) -> str:
    """Fetch a secret from AWS Secrets Manager (cached for the process lifetime)."""
    import boto3
    import json

    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_id)
    secret = response["SecretString"]
    # If the secret is a JSON object, return the raw string;
    # callers can parse as needed.
    return secret


def get_mongodb_uri() -> str:
    """Return the MongoDB connection string.

    Resolution order:
      1. ``MONGODB_URI`` env var (for local dev / direct override)
      2. ``MONGODB_SECRET_ID`` env var pointing to a Secrets Manager entry
    """
    uri = os.environ.get("MONGODB_URI")
    if uri:
        return uri

    secret_id = os.environ.get("MONGODB_SECRET_ID")
    if secret_id:
        import json

        payload = _get_secret(secret_id)
        # Support both plain-string secrets and JSON {"uri": "..."} secrets.
        try:
            data = json.loads(payload)
            return data["uri"]
        except (json.JSONDecodeError, KeyError):
            return payload

    raise RuntimeError(
        "MongoDB connection string not configured. "
        "Set MONGODB_URI or MONGODB_SECRET_ID."
    )


# ── S3 ──────────────────────────────────────────────────────────────────────

S3_BUCKET: str = os.environ.get("IMPULSE_BUCKET", "")
S3_INPUT_PREFIX: str = os.environ.get("IMPULSE_INPUT_PREFIX", "uploads")
S3_OUTPUT_PREFIX: str = os.environ.get("IMPULSE_OUTPUT_PREFIX", "results")

# ── MongoDB ─────────────────────────────────────────────────────────────────

MONGODB_DATABASE: str = os.environ.get("MONGODB_DATABASE", "praxis")

# ── Bedrock ─────────────────────────────────────────────────────────────────

BEDROCK_MODEL_ID: str = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
BEDROCK_REGION: str = os.environ.get("BEDROCK_REGION", "us-east-1")

# ── Processing defaults ─────────────────────────────────────────────────────

JP2_QUALITY: int = int(os.environ.get("JP2_QUALITY", "90"))
MARKER_BATCH_SIZE: int = int(os.environ.get("MARKER_BATCH_SIZE", "8"))
