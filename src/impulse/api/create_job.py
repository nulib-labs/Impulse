"""POST /jobs -- Create a new processing job."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.db.models import Job, JobStatus, OcrEngine


def create_job(body: dict, user_id: str) -> dict:
    """Create a job record in MongoDB and start the Step Functions execution."""
    task_type = body.get("task_type", "full_pipeline")
    file_count = body.get("file_count", 0)
    custom_id = body.get("custom_id", "").strip()
    metadata = body.get("metadata", {})
    ocr_engine = body.get("ocr_engine", OcrEngine.TEXTRACT.value)

    # Validate ocr_engine
    valid_engines = {e.value for e in OcrEngine}
    if ocr_engine not in valid_engines:
        ocr_engine = OcrEngine.TEXTRACT.value

    # Validate metadata is a flat dict of strings
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {str(k): str(v) for k, v in metadata.items() if k and v}

    job_id = str(uuid.uuid4())
    input_prefix = f"uploads/{job_id}"
    output_prefix = f"results/{job_id}"

    job = Job(
        job_id=job_id,
        user_id=user_id,
        status=JobStatus.PENDING.value,
        task_type=task_type,
        ocr_engine=ocr_engine,
        custom_id=custom_id,
        metadata=metadata,
        input_s3_prefix=input_prefix,
        output_s3_prefix=output_prefix,
        total_documents=file_count,
    )

    # Insert into MongoDB
    jobs_collection = get_collection("jobs")
    jobs_collection.insert_one(job.to_dict())
    logger.info(f"Created job {job_id} (custom_id={custom_id!r}) for user {user_id}")

    return _response(
        201,
        {
            "job_id": job_id,
            "custom_id": custom_id,
            "status": job.status,
            "input_s3_prefix": input_prefix,
            "output_s3_prefix": output_prefix,
            "message": "Job created. Upload files then call POST /jobs/{jobId}/upload-url to start processing.",
        },
    )


def start_step_functions(job_id: str, job: dict) -> str:
    """Start a Step Functions execution for the given job."""
    sfn_client = boto3.client("stepfunctions")
    state_machine_arn = os.environ.get("STATE_MACHINE_ARN", "")

    if not state_machine_arn:
        logger.warning("STATE_MACHINE_ARN not set, skipping execution start")
        return ""

    response = sfn_client.start_execution(
        stateMachineArn=state_machine_arn,
        name=f"job-{job_id}",
        input=json.dumps(
            {
                "job_id": job_id,
                "task_type": job.get("task_type", "full_pipeline"),
                "ocr_engine": job.get("ocr_engine", "textract"),
                "input_s3_prefix": job.get("input_s3_prefix", ""),
                "output_s3_prefix": job.get("output_s3_prefix", ""),
                "impulse_identifier": job_id,
            }
        ),
    )

    execution_arn = response["executionArn"]
    logger.info(f"Started Step Functions execution: {execution_arn}")
    return execution_arn


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }
