"""OCR via AWS Textract.

Supports both synchronous (single-page images) and asynchronous (multi-page
PDFs) text detection.  Results are returned as structured text with bounding
boxes and confidence scores.
"""

from __future__ import annotations

import json
import time

import boto3
from loguru import logger

from impulse.config import S3_BUCKET


def ocr_with_textract(
    document_key: str,
    output_key: str,
    job_id: str,
) -> dict:
    """Run Textract on a document stored in S3.

    For images, uses synchronous ``detect_document_text``.
    For PDFs, uses async ``start_document_text_detection`` + polling.

    Returns a dict with extracted text, lines, and word-level details.
    """
    textract = boto3.client("textract")
    s3 = boto3.client("s3")

    ext = document_key.rsplit(".", 1)[-1].lower() if "." in document_key else ""
    is_pdf = ext == "pdf"

    if is_pdf:
        result, api_calls = _textract_async(textract, document_key)
    else:
        result, api_calls = _textract_sync(textract, document_key)

    # Build structured output
    pages: list[dict] = []
    full_text_parts: list[str] = []

    for block in result.get("Blocks", []):
        if block["BlockType"] == "LINE":
            full_text_parts.append(block.get("Text", ""))
        if block["BlockType"] == "PAGE":
            page_num = block.get("Page", 1)
            pages.append({"page_number": page_num, "lines": []})

    # Group lines by page
    current_page = 1
    page_lines: dict[int, list] = {}
    for block in result.get("Blocks", []):
        if block["BlockType"] == "LINE":
            pg = block.get("Page", 1)
            page_lines.setdefault(pg, []).append(
                {
                    "text": block.get("Text", ""),
                    "confidence": block.get("Confidence", 0),
                    "bbox": block.get("Geometry", {}).get("BoundingBox", {}),
                }
            )

    full_text = "\n".join(full_text_parts)

    output = {
        "extraction_model": "textract",
        "document_key": document_key,
        "job_id": job_id,
        "full_text": full_text,
        "page_count": len(page_lines) or 1,
        "pages": [
            {"page_number": pg, "lines": lines}
            for pg, lines in sorted(page_lines.items())
        ],
        "textract_api_calls": api_calls,
    }

    # Save extracted text to S3
    text_key = (
        output_key.rsplit(".", 1)[0] + ".txt"
        if "." in output_key
        else output_key + ".txt"
    )
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=text_key,
        Body=full_text.encode("utf-8"),
        ContentType="text/plain",
    )

    logger.info(
        f"Textract extracted {len(full_text_parts)} lines "
        f"from {document_key} ({len(full_text)} chars)"
    )
    return output


def _textract_sync(textract, document_key: str) -> tuple[dict, int]:
    """Synchronous Textract for single-page images."""
    response = textract.detect_document_text(
        Document={
            "S3Object": {
                "Bucket": S3_BUCKET,
                "Name": document_key,
            }
        }
    )
    return response, 1  # 1 API call


def _textract_async(textract, document_key: str) -> tuple[dict, int]:
    """Asynchronous Textract for multi-page PDFs."""
    api_calls = 1  # start_document_text_detection
    start_response = textract.start_document_text_detection(
        DocumentLocation={
            "S3Object": {
                "Bucket": S3_BUCKET,
                "Name": document_key,
            }
        }
    )
    job_id = start_response["JobId"]
    logger.info(f"Started Textract async job: {job_id}")

    # Poll until complete
    while True:
        time.sleep(2)
        result = textract.get_document_text_detection(JobId=job_id)
        api_calls += 1
        status = result["JobStatus"]
        if status == "SUCCEEDED":
            # Collect all pages of results
            blocks = result.get("Blocks", [])
            next_token = result.get("NextToken")
            while next_token:
                result = textract.get_document_text_detection(
                    JobId=job_id, NextToken=next_token
                )
                api_calls += 1
                blocks.extend(result.get("Blocks", []))
                next_token = result.get("NextToken")
            return {"Blocks": blocks}, api_calls
        elif status == "FAILED":
            raise RuntimeError(
                f"Textract job {job_id} failed: "
                f"{result.get('StatusMessage', 'unknown error')}"
            )
        # IN_PROGRESS -- keep polling
