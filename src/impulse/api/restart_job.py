"""POST /jobs/{jobId}/restart -- Restart a failed or completed job."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3
from loguru import logger

from impulse.db.client import get_collection
from impulse.api.create_job import start_step_functions


def restart_job(job_id: str, user_id: str) -> dict:
    """Reset a job's counters and re-trigger the Step Functions pipeline."""
    jobs_collection = get_collection("jobs")

    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0},
    )

    if not job:
        return _response(404, {"error": "Job not found"})

    current_status = job.get("status", "")
    if current_status == "PROCESSING":
        return _response(
            409,
            {
                "error": "Job is already processing. Wait for it to finish or fail before restarting.",
            },
        )

    # Reset progress counters
    now = datetime.now(timezone.utc).isoformat()
    jobs_collection.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "PENDING",
                "processed_documents": 0,
                "failed_documents": 0,
                "updated_at": now,
            }
        },
    )

    # Start a new Step Functions execution
    execution_arn = ""
    try:
        execution_arn = start_step_functions(job_id, job)
    except Exception as e:
        # If the execution name already exists, use a suffixed name
        if "ExecutionAlreadyExists" in str(e):
            sfn_client = boto3.client("stepfunctions")
            state_machine_arn = os.environ.get("STATE_MACHINE_ARN", "")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            response = sfn_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=f"job-{job_id}-restart-{ts}",
                input=json.dumps(
                    {
                        "job_id": job_id,
                        "task_type": job.get("task_type", "full_pipeline"),
                        "ocr_engine": job.get("ocr_engine", "textract"),
                        "input_s3_prefix": job.get("input_s3_prefix", ""),
                        "output_s3_prefix": job.get("output_s3_prefix", ""),
                        "impulse_identifier": job.get("custom_id") or job_id,
                    }
                ),
            )
            execution_arn = response["executionArn"]
        else:
            raise

    if execution_arn:
        jobs_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "PROCESSING",
                    "step_functions_arn": execution_arn,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    logger.info(f"Restarted job {job_id}, execution: {execution_arn}")

    return _response(
        200,
        {
            "job_id": job_id,
            "status": "PROCESSING",
            "execution_arn": execution_arn,
            "message": "Job restarted successfully.",
        },
    )


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
