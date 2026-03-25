"""POST /jobs/{jobId}/upload-url -- Generate presigned S3 upload URLs."""

from __future__ import annotations

import json
import os

from loguru import logger

from impulse.db.client import get_collection
from impulse.utils import generate_presigned_upload_url
from impulse.api.create_job import start_step_functions


def generate_upload_urls(job_id: str, body: dict, user_id: str) -> dict:
    """Generate presigned PUT URLs for uploading documents to S3.

    Request body:
        {
            "filenames": ["page001.jp2", "page002.jp2", ...],
            "start_processing": true  // optional, starts the pipeline after URLs are generated
        }

    Returns presigned URLs keyed by filename.
    """
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

    filenames: list[str] = body.get("filenames", [])
    if not filenames:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": "filenames list is required"}),
        }

    input_prefix = job.get("input_s3_prefix", f"uploads/{job_id}")
    urls: dict[str, str] = {}

    for filename in filenames:
        key = f"{input_prefix}/{filename}"
        urls[filename] = generate_presigned_upload_url(key, expires_in=3600)

    # Update the document count
    jobs_collection.update_one(
        {"job_id": job_id},
        {"$set": {"total_documents": len(filenames)}},
    )

    logger.info(f"Generated {len(urls)} presigned URLs for job {job_id}")

    # Optionally start processing immediately
    result: dict = {
        "upload_urls": urls,
        "count": len(urls),
    }

    if body.get("start_processing", False):
        execution_arn = start_step_functions(job_id, job)
        if execution_arn:
            jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "PROCESSING",
                        "step_functions_arn": execution_arn,
                    }
                },
            )
            result["execution_arn"] = execution_arn
            result["status"] = "PROCESSING"

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result, default=str),
    }
