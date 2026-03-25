"""Environmental impact API handlers.

GET /jobs/{jobId}/environmental-impact
GET /collections/{collectionId}/environmental-impact
"""

from __future__ import annotations

import json

from loguru import logger

from impulse.processing.environmental import (
    get_job_impact_summary,
    get_collection_impact_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }


# ── GET /jobs/{jobId}/environmental-impact ──────────────────────────────────


def get_job_environmental_impact(job_id: str, user_id: str) -> dict:
    """Return aggregated environmental impact metrics for a job."""
    logger.info(f"Fetching environmental impact for job {job_id}")

    try:
        result = get_job_impact_summary(job_id)
        return _response(200, result)
    except Exception as e:
        logger.error(f"Failed to get job environmental impact: {e}")
        return _response(500, {"error": str(e)})


# ── GET /collections/{collectionId}/environmental-impact ────────────────────


def get_collection_environmental_impact(collection_id: str, user_id: str) -> dict:
    """Return aggregated environmental impact metrics for a collection."""
    logger.info(f"Fetching environmental impact for collection {collection_id}")

    try:
        result = get_collection_impact_summary(collection_id)
        return _response(200, result)
    except Exception as e:
        logger.error(f"Failed to get collection environmental impact: {e}")
        return _response(500, {"error": str(e)})
