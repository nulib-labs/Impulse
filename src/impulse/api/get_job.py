"""GET /jobs/{jobId} -- Get a single job's details and progress."""

from __future__ import annotations

import json

from loguru import logger

from impulse.db.client import get_collection


def get_job(job_id: str, user_id: str) -> dict:
    """Return details for a single job, scoped to the user."""
    jobs_collection = get_collection("jobs")

    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0},
    )

    if not job:
        return {
            "statusCode": 404,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "Job not found"}),
        }

    # Compute progress percentage
    total = job.get("total_documents", 0)
    processed = job.get("processed_documents", 0)
    failed = job.get("failed_documents", 0)
    progress = round((processed + failed) / total * 100, 1) if total > 0 else 0
    job["progress_percent"] = progress

    logger.info(f"Retrieved job {job_id}: {job.get('status')}")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"job": job}, default=str),
    }
