"""ECS Fargate entry point for document extraction (Marker PDF).

Invoked by Step Functions with the following environment / input:
  - ``DOCUMENT_KEY``: S3 key of the input document
  - ``JOB_ID``: Parent job identifier
  - ``IMPULSE_IDENTIFIER``: Identifier for grouping results
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
from impulse.processing.extraction import extract_documents, save_extraction_results
from impulse.utils import get_s3_content


def main() -> None:
    document_key = os.environ.get("DOCUMENT_KEY")
    job_id = os.environ.get("JOB_ID")
    impulse_identifier = os.environ.get("IMPULSE_IDENTIFIER", "")

    if not document_key or not job_id:
        logger.critical("Missing required env vars: DOCUMENT_KEY, JOB_ID")
        sys.exit(1)

    logger.info(f"Extracting document: {document_key}")

    try:
        start_time = time.perf_counter()

        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        input_size = len(content)
        filename = document_key.split("/")[-1]

        items = [
            {
                "contents": content,
                "filename": filename,
                "impulse_identifier": impulse_identifier or job_id,
            }
        ]

        results = extract_documents(items)

        # Save to MongoDB
        collection = get_collection("results")
        save_extraction_results(results, collection)
        logger.success(f"Extraction complete for {document_key}")

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Update job progress
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
                task_type="document_extraction",
                compute_type="fargate_30gb_4vcpu",
                processing_duration_ms=elapsed_ms,
                input_file_size_bytes=input_size,
            )
        except Exception as env_err:
            logger.warning(f"Failed to persist environmental metrics: {env_err}")

    except Exception as e:
        logger.error(f"Document extraction failed for {document_key}: {e}")

        jobs = get_collection("jobs")
        jobs.update_one(
            {"job_id": job_id},
            {"$inc": {"failed_documents": 1}},
        )
        raise


if __name__ == "__main__":
    main()
