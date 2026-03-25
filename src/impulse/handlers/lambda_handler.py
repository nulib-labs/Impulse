"""AWS Lambda handler for lightweight processing tasks.

Dispatches to the correct processing function based on the ``task_type``
field in the Step Functions input event.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.processing.environmental import create_and_persist_metrics
from impulse.utils import get_s3_content, put_s3_content


def handler(event: dict, context) -> dict:
    """Lambda entry point.  Expected event shape:

    .. code-block:: json

        {
            "task_type": "validate | complete | mets | metadata | summaries | ner | geocode",
            "document_key": "uploads/job-123/doc.pdf",
            "job_id": "job-123",
            "accession_number": "ABC1234",
            "output_key": "results/job-123/doc.yaml"
        }
    """
    task_type = event.get("task_type", "")
    document_key = event.get("document_key", "")
    job_id = event.get("job_id", "")

    logger.info(
        f"Lambda handler: task_type={task_type}, job={job_id}, doc={document_key}"
    )

    try:
        # Time the processing for environmental impact tracking
        start_time = time.perf_counter()
        result = _dispatch(task_type, event)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Only increment progress for per-document processing tasks
        if job_id and task_type not in ("validate", "complete"):
            jobs = get_collection("jobs")
            jobs.update_one(
                {"job_id": job_id},
                {"$inc": {"processed_documents": 1}},
            )

        # Persist environmental metrics for per-document tasks
        if (
            job_id
            and document_key
            and task_type not in ("validate", "complete", "fail")
        ):
            _persist_env_metrics(task_type, event, result, elapsed_ms)

        return {"statusCode": 200, "body": result}

    except Exception as e:
        logger.error(f"Lambda handler failed: {e}")

        if job_id and task_type not in ("validate", "complete"):
            jobs = get_collection("jobs")
            jobs.update_one(
                {"job_id": job_id},
                {"$inc": {"failed_documents": 1}},
            )

        return {"statusCode": 500, "body": str(e)}


# ── Orchestration handlers ──────────────────────────────────────────────


def _validate_job(event: dict) -> dict:
    """List all uploaded documents for a job and build processing descriptors.

    Called by Step Functions as the first step.  Returns a list of document
    descriptors that the DistributedMap fans out over.
    """
    job_id = event.get("job_id", "")
    input_prefix = event.get("input_s3_prefix", "")

    if not input_prefix:
        input_prefix = f"uploads/{job_id}"

    # Look up the job to get the task_type
    jobs = get_collection("jobs")
    job = jobs.find_one({"job_id": job_id}, {"_id": 0})

    if not job:
        raise ValueError(f"Job {job_id} not found in database")

    job_task_type = job.get("task_type", "full_pipeline")
    ocr_engine = job.get("ocr_engine", "textract")
    output_prefix = job.get("output_s3_prefix", f"results/{job_id}")
    custom_id = job.get("custom_id", "")

    # Enumerate S3 objects under the input prefix
    s3 = boto3.client("s3")
    documents: list[dict] = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue

            # Determine the per-document task type
            # For full_pipeline, route based on file extension
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            doc_task_type = job_task_type

            if job_task_type == "full_pipeline":
                if ext in ("jpg", "jpeg", "png", "tif", "tiff", "jp2"):
                    doc_task_type = "image_transform"
                elif ext == "pdf":
                    doc_task_type = "document_extraction"
                elif ext == "xml":
                    doc_task_type = "mets"
                else:
                    doc_task_type = "metadata"

            # For document_extraction, route based on ocr_engine
            # textract and bedrock_claude run on Lambda; marker_pdf on Fargate
            if doc_task_type == "document_extraction":
                if ocr_engine == "marker_pdf":
                    doc_task_type = "document_extraction"  # Fargate
                else:
                    doc_task_type = f"ocr_{ocr_engine}"  # Lambda

            # Build the output key
            # Image transform outputs JPG for browser viewing
            if doc_task_type == "image_transform" and "." in filename:
                base = filename.rsplit(".", 1)[0]
                output_key = f"{output_prefix}/{base}.jpg"
            else:
                output_key = f"{output_prefix}/{filename}"

            documents.append(
                {
                    "task_type": doc_task_type,
                    "ocr_engine": ocr_engine,
                    "document_key": key,
                    "output_key": output_key,
                    "job_id": job_id,
                    "impulse_identifier": custom_id or job_id,
                    "accession_number": custom_id,
                }
            )

            # For full_pipeline with images, also add an OCR task so
            # text extraction runs alongside image transformation
            if (
                job_task_type == "full_pipeline"
                and doc_task_type == "image_transform"
                and ocr_engine != "marker_pdf"
            ):
                documents.append(
                    {
                        "task_type": f"ocr_{ocr_engine}",
                        "ocr_engine": ocr_engine,
                        "document_key": key,
                        "output_key": f"{output_prefix}/{filename}",
                        "job_id": job_id,
                        "impulse_identifier": custom_id or job_id,
                        "accession_number": custom_id,
                    }
                )

    # Update job with accurate document count
    total = len(documents)
    jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "total_documents": total,
                "status": "PROCESSING",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    logger.info(f"Validated job {job_id}: {total} documents found")
    return {"documents": documents, "total": total}


def _complete_job(event: dict) -> dict:
    """Mark a job as COMPLETED after all documents are processed."""
    job_id = event.get("job_id", "")

    jobs = get_collection("jobs")
    jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "COMPLETED",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    logger.info(f"Job {job_id} marked as COMPLETED")
    return {"job_id": job_id, "status": "COMPLETED"}


def _fail_job(event: dict) -> dict:
    """Mark a job as FAILED when the pipeline encounters an error."""
    job_id = event.get("job_id", "")
    error = event.get("error", "Unknown error")
    cause = event.get("cause", "")

    jobs = get_collection("jobs")
    jobs.update_one(
        {"job_id": job_id},
        {
            "$set": {
                "status": "FAILED",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    logger.error(f"Job {job_id} marked as FAILED: {error} — {cause}")
    return {"job_id": job_id, "status": "FAILED"}


# ── Per-document processing dispatch ────────────────────────────────────


def _dispatch(task_type: str, event: dict) -> dict:
    """Route to the correct processing function."""

    # Orchestration tasks (not per-document)
    if task_type == "validate":
        return _validate_job(event)
    elif task_type == "complete":
        return _complete_job(event)
    elif task_type == "fail":
        return _fail_job(event)

    document_key = event.get("document_key", "")
    output_key = event.get("output_key", "")
    job_id = event.get("job_id", "")

    # ── OCR engine tasks (run on Lambda) ────────────────────────────
    if task_type == "ocr_textract":
        from impulse.processing.textract_ocr import ocr_with_textract

        result = ocr_with_textract(document_key, output_key, job_id)
        # Save to MongoDB results collection
        collection = get_collection("results")
        collection.update_one(
            {"job_id": job_id, "document_key": document_key},
            {
                "$set": {
                    "job_id": job_id,
                    "document_key": document_key,
                    "extraction_model": "textract",
                    "extracted_text": result.get("full_text", ""),
                    "page_count": result.get("page_count", 1),
                }
            },
            upsert=True,
        )
        return {"status": "success", "extraction_model": "textract"}

    elif task_type == "ocr_bedrock_claude":
        from impulse.processing.claude_ocr import ocr_with_claude

        result = ocr_with_claude(document_key, output_key, job_id)
        collection = get_collection("results")
        collection.update_one(
            {"job_id": job_id, "document_key": document_key},
            {
                "$set": {
                    "job_id": job_id,
                    "document_key": document_key,
                    "extraction_model": "bedrock_claude",
                    "extracted_text": result.get("full_text", ""),
                    "page_count": result.get("page_count", 1),
                }
            },
            upsert=True,
        )
        return {"status": "success", "extraction_model": "bedrock_claude"}

    # ── Original task types ─────────────────────────────────────────
    if task_type == "mets":
        from impulse.processing.mets import convert_mets_to_yaml

        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        yaml_content = convert_mets_to_yaml(content)
        if output_key:
            put_s3_content(
                f"s3://{S3_BUCKET}/{output_key}",
                yaml_content.encode("utf-8"),
            )
        return {"output_key": output_key, "status": "success"}

    elif task_type == "metadata":
        from impulse.processing.metadata import extract_metadata

        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        accession = event.get("accession_number", "")
        ext = document_key.rsplit(".", 1)[-1].lower()

        kwargs: dict = {"accession_number": accession}
        if ext == "pdf":
            kwargs["pdf_bytes"] = content
        elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
            kwargs["image_bytes"] = content
        else:
            kwargs["document_text"] = content.decode("utf-8", errors="replace")

        metadata = extract_metadata(**kwargs)

        # Save to MongoDB
        collection = get_collection("metadata")
        collection.insert_one(metadata)
        return {"metadata": metadata, "status": "success"}

    elif task_type == "summaries":
        from impulse.processing.summaries import summarise_document

        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        ext = document_key.rsplit(".", 1)[-1].lower()

        if ext == "pdf":
            summary = summarise_document(pdf_bytes=content)
        elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
            summary = summarise_document(image_bytes=content)
        else:
            summary = summarise_document(
                document_text=content.decode("utf-8", errors="replace")
            )

        return {"summary": summary, "status": "success"}

    elif task_type == "ner":
        from impulse.processing.ner import run_ner

        content = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
        text = content.decode("utf-8", errors="replace")
        entities = run_ner(text)
        return {"entities": entities, "status": "success"}

    elif task_type == "geocode":
        from impulse.processing.geocode import geocode

        place = event.get("place", "")
        result = geocode(place)
        return {"geocode_result": result, "status": "success"}

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# ── Environmental metrics persistence ───────────────────────────────────


def _persist_env_metrics(
    task_type: str, event: dict, result: dict, elapsed_ms: int
) -> None:
    """Capture and persist environmental impact metrics for a processing task."""
    job_id = event.get("job_id", "")
    document_key = event.get("document_key", "")

    # Determine input file size from S3 (best-effort)
    input_size = 0
    try:
        s3 = boto3.client("s3")
        head = s3.head_object(Bucket=S3_BUCKET, Key=document_key)
        input_size = head.get("ContentLength", 0)
    except Exception:
        pass

    # Extract AI-service-specific metrics from the processing result
    bedrock_input_tokens = 0
    bedrock_output_tokens = 0
    bedrock_invocations = 0
    textract_api_calls = 0
    page_count = 0

    if isinstance(result, dict):
        bedrock_input_tokens = result.get("bedrock_input_tokens", 0)
        bedrock_output_tokens = result.get("bedrock_output_tokens", 0)
        bedrock_invocations = result.get("bedrock_invocations", 0)
        textract_api_calls = result.get("textract_api_calls", 0)
        page_count = result.get("page_count", 0)

        # summaries.py now returns a dict; handle nested metadata result
        if "metadata" in result and isinstance(result["metadata"], dict):
            meta = result["metadata"]
            bedrock_input_tokens = meta.get(
                "bedrock_input_tokens", bedrock_input_tokens
            )
            bedrock_output_tokens = meta.get(
                "bedrock_output_tokens", bedrock_output_tokens
            )
            bedrock_invocations = meta.get("bedrock_invocations", bedrock_invocations)

        # summaries task returns dict with nested summary info
        if "summary" in result and isinstance(result.get("summary"), dict):
            summary_data = result["summary"]
            bedrock_input_tokens = summary_data.get(
                "bedrock_input_tokens", bedrock_input_tokens
            )
            bedrock_output_tokens = summary_data.get(
                "bedrock_output_tokens", bedrock_output_tokens
            )
            bedrock_invocations = summary_data.get(
                "bedrock_invocations", bedrock_invocations
            )

    try:
        create_and_persist_metrics(
            job_id=job_id,
            document_key=document_key,
            task_type=task_type,
            compute_type="lambda_3008mb",
            processing_duration_ms=elapsed_ms,
            input_file_size_bytes=input_size,
            bedrock_input_tokens=bedrock_input_tokens,
            bedrock_output_tokens=bedrock_output_tokens,
            bedrock_invocations=bedrock_invocations,
            textract_api_calls=textract_api_calls,
            page_count=page_count,
        )
    except Exception as e:
        # Environmental metrics should never break the pipeline
        logger.warning(f"Failed to persist environmental metrics: {e}")
