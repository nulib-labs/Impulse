"""ECS Fargate entry point for image processing tasks.

Invoked by Step Functions with the following environment / input:
  - ``DOCUMENT_KEY``: S3 key of the input image
  - ``OUTPUT_KEY``: S3 key for the processed output
  - ``JOB_ID``: Parent job identifier
  - ``IMPULSE_BUCKET``: S3 bucket (set in task definition)
"""

from __future__ import annotations

import os
import sys
import time

from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.processing.environmental import create_and_persist_metrics
from impulse.processing.images import process_image
from impulse.utils import get_s3_content, put_s3_content


def main() -> None:
    document_key = os.environ.get("DOCUMENT_KEY")
    output_key = os.environ.get("OUTPUT_KEY")
    job_id = os.environ.get("JOB_ID")

    if not document_key or not output_key or not job_id:
        logger.critical("Missing required env vars: DOCUMENT_KEY, OUTPUT_KEY, JOB_ID")
        sys.exit(1)

    logger.info(f"Processing image: {document_key}")

    try:
        start_time = time.perf_counter()

        # Download from S3
        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        input_size = len(content)

        # Process: grayscale -> binarize -> denoise -> JPG
        result_bytes = process_image(content, output_format=".jpg")
        output_size = len(result_bytes)

        # Upload result
        put_s3_content(f"s3://{S3_BUCKET}/{output_key}", result_bytes)
        logger.success(f"Saved processed image to {output_key}")

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Update job progress in MongoDB
        jobs = get_collection("jobs")
        jobs.update_one(
            {"job_id": job_id},
            {"$inc": {"processed_documents": 1}},
        )

        # Persist environmental impact metrics
        try:
            create_and_persist_metrics(
                job_id=job_id,
                document_key=document_key,
                task_type="image_transform",
                compute_type="fargate_8gb_2vcpu",
                processing_duration_ms=elapsed_ms,
                input_file_size_bytes=input_size,
                output_file_size_bytes=output_size,
            )
        except Exception as env_err:
            logger.warning(f"Failed to persist environmental metrics: {env_err}")

    except Exception as e:
        logger.error(f"Image processing failed for {document_key}: {e}")

        # Record failure
        jobs = get_collection("jobs")
        jobs.update_one(
            {"job_id": job_id},
            {"$inc": {"failed_documents": 1}},
        )
        raise


if __name__ == "__main__":
    main()
