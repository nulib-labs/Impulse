"""GET /jobs -- List all jobs for the authenticated user."""

from __future__ import annotations

import json

from loguru import logger

from impulse.db.client import get_collection


def list_jobs(user_id: str) -> dict:
    """Return all jobs belonging to *user_id*, newest first."""
    jobs_collection = get_collection("jobs")

    cursor = jobs_collection.find(
        {"user_id": user_id},
        {
            "_id": 0,
            "job_id": 1,
            "status": 1,
            "task_type": 1,
            "ocr_engine": 1,
            "custom_id": 1,
            "metadata": 1,
            "total_documents": 1,
            "processed_documents": 1,
            "failed_documents": 1,
            "created_at": 1,
            "updated_at": 1,
        },
    ).sort("created_at", -1)

    jobs = list(cursor)
    logger.info(f"Listed {len(jobs)} jobs for user {user_id}")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"jobs": jobs, "count": len(jobs)}, default=str),
    }
