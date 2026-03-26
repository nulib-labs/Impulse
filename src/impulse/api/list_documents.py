"""GET /jobs/{jobId}/documents -- List input & output files with presigned view URLs."""

from __future__ import annotations

import json
import os

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection
from impulse.utils import generate_presigned_download_url


def list_documents(job_id: str, user_id: str) -> dict:
    """Return input and output S3 objects for a job with presigned GET URLs.

    Groups files by base filename so the frontend can show input/output
    side-by-side for each document.  Also attaches OCR text from the
    results collection when available.
    """
    jobs_collection = get_collection("jobs")
    job = jobs_collection.find_one(
        {"job_id": job_id, "user_id": user_id},
        {"_id": 0, "job_id": 1, "input_s3_prefix": 1, "output_s3_prefix": 1},
    )

    if not job:
        return _response(404, {"error": "Job not found"})

    s3 = boto3.client("s3")
    input_prefix = job.get("input_s3_prefix", f"uploads/{job_id}")
    output_prefix = job.get("output_s3_prefix", f"results/{job_id}")

    input_files = _list_s3_objects(s3, input_prefix)
    output_files = _list_s3_objects(s3, output_prefix)

    # Load OCR results from MongoDB, indexed by document_key
    results_collection = get_collection("results")
    ocr_results: dict[str, dict] = {}
    for r in results_collection.find(
        {"job_id": job_id},
        {
            "_id": 0,
            "document_key": 1,
            "extracted_text": 1,
            "extraction_model": 1,
            "page_count": 1,
        },
    ):
        ocr_results[r.get("document_key", "")] = r

    # Build document list grouped by base filename
    documents: list[dict] = []

    # Index output files by their base name for fast lookup
    output_by_name: dict[str, list[dict]] = {}
    for f in output_files:
        base = (
            f["filename"].rsplit(".", 1)[0] if "." in f["filename"] else f["filename"]
        )
        output_by_name.setdefault(base, []).append(f)

    seen_output_bases: set[str] = set()

    for inp in input_files:
        base = (
            inp["filename"].rsplit(".", 1)[0]
            if "." in inp["filename"]
            else inp["filename"]
        )
        seen_output_bases.add(base)

        # Look up OCR result for this input document
        ocr = ocr_results.get(inp["key"], {})

        doc: dict = {
            "filename": inp["filename"],
            "input": {
                "key": inp["key"],
                "size": inp["size"],
                "url": generate_presigned_download_url(inp["key"], expires_in=3600),
            },
            "outputs": [],
            "ocr_text": ocr.get("extracted_text", ""),
            "ocr_model": ocr.get("extraction_model", ""),
        }
        for out in output_by_name.get(base, []):
            doc["outputs"].append(
                {
                    "key": out["key"],
                    "filename": out["filename"],
                    "size": out["size"],
                    "url": generate_presigned_download_url(out["key"], expires_in=3600),
                }
            )
        documents.append(doc)

    # Include any output files that don't match an input file
    for base, outs in output_by_name.items():
        if base not in seen_output_bases:
            for out in outs:
                documents.append(
                    {
                        "filename": out["filename"],
                        "input": None,
                        "outputs": [
                            {
                                "key": out["key"],
                                "filename": out["filename"],
                                "size": out["size"],
                                "url": generate_presigned_download_url(
                                    out["key"], expires_in=3600
                                ),
                            }
                        ],
                    }
                )

    logger.info(
        f"Listed {len(input_files)} input / {len(output_files)} output files "
        f"for job {job_id}"
    )

    return _response(
        200,
        {
            "documents": documents,
            "input_count": len(input_files),
            "output_count": len(output_files),
        },
    )


def _list_s3_objects(s3_client, prefix: str) -> list[dict]:
    """List all objects under an S3 prefix, returning filename/key/size."""
    objects: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            objects.append(
                {
                    "key": key,
                    "filename": filename,
                    "size": obj["Size"],
                }
            )

    return sorted(objects, key=lambda x: x["filename"])


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
