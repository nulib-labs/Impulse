"""DELETE /jobs/{jobId} -- Permanently delete a job and its S3 data."""

from __future__ import annotations

import json

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection


def delete_job(job_id: str, user_id: str, claims: dict | None = None) -> dict:
    """Delete a job record from MongoDB and remove all associated S3 objects."""
    from impulse.api.auth import require_admin

    denied = require_admin(claims or {})
    if denied:
        return denied

    jobs_collection = get_collection("jobs")

    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0},
    )

    if not job:
        return _response(404, {"error": "Job not found"})

    if job.get("status") == "PROCESSING":
        return _response(
            409,
            {
                "error": "Cannot delete a job that is currently processing. "
                "Wait for it to complete or fail first.",
            },
        )

    # Delete S3 objects under both input and output prefixes
    s3 = boto3.client("s3")
    deleted_count = 0
    for prefix in [
        job.get("input_s3_prefix", f"uploads/{job_id}"),
        job.get("output_s3_prefix", f"results/{job_id}"),
    ]:
        deleted_count += _delete_s3_prefix(s3, prefix)

    # Delete results from MongoDB
    results_collection = get_collection("results")
    results_deleted = results_collection.delete_many({"job_id": job_id})

    # Delete the job record
    jobs_collection.delete_one({"job_id": job_id, "user_id": user_id})

    logger.info(
        f"Deleted job {job_id}: {deleted_count} S3 objects, "
        f"{results_deleted.deleted_count} result records"
    )

    return _response(
        200,
        {
            "message": "Job deleted permanently.",
            "job_id": job_id,
            "s3_objects_deleted": deleted_count,
            "results_deleted": results_deleted.deleted_count,
        },
    )


def _delete_s3_prefix(s3_client, prefix: str) -> int:
    """Delete all objects under an S3 prefix. Returns count deleted."""
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        s3_client.delete_objects(
            Bucket=S3_BUCKET,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
        )
        deleted += len(objects)
    return deleted


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
