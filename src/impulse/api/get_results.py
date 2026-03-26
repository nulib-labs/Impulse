"""GET /jobs/{jobId}/results -- Paginated extraction results."""

from __future__ import annotations

import json

from loguru import logger

from impulse.db.client import get_collection


def get_results(
    job_id: str,
    user_id: str,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """Return paginated results for a job."""

    # Verify the job belongs to this user
    jobs_collection = get_collection("jobs")
    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0, "job_id": 1},
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

    results_collection = get_collection("results")
    skip = (page - 1) * page_size

    total = results_collection.count_documents({"job_id": job_id})

    cursor = (
        results_collection.find(
            {"job_id": job_id},
            {"_id": 0},
        )
        .sort("page_number", 1)
        .skip(skip)
        .limit(page_size)
    )

    results = list(cursor)
    logger.info(
        f"Retrieved {len(results)} results for job {job_id} "
        f"(page {page}, total {total})"
    )

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(
            {
                "results": results,
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size,
            },
            default=str,
        ),
    }
