"""OCR via Amazon Bedrock Claude Vision.

Sends document images to Claude's vision capability for text extraction.
Best for handwritten text, degraded historical documents, and documents
where layout understanding is important.
"""

from __future__ import annotations

import base64
import json

import boto3
from loguru import logger

from impulse.config import BEDROCK_REGION, S3_BUCKET
from impulse.utils import get_s3_content, pdf_to_base64_images

# Use Claude 3 Haiku for cost-effectiveness on OCR tasks.
# Can be overridden via BEDROCK_OCR_MODEL_ID env var.
import os

BEDROCK_OCR_MODEL_ID = os.environ.get(
    "BEDROCK_OCR_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)

_bedrock_client = None


def _get_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock_client


_OCR_PROMPT = """You are a precise OCR engine. Extract ALL text from this document image exactly as it appears.

Rules:
- Preserve the original line breaks and paragraph structure
- Include headers, footers, page numbers, captions, and marginalia
- For tables, use pipe-separated columns (| col1 | col2 |)
- For handwritten text, do your best to transcribe accurately and mark uncertain words with [?]
- Do NOT add commentary, summaries, or descriptions of the image
- Output ONLY the extracted text, nothing else"""


def ocr_with_claude(
    document_key: str,
    output_key: str,
    job_id: str,
) -> dict:
    """Run Claude Vision OCR on a document stored in S3.

    Supports images (jpg, png, etc.) and PDFs (converted to images).
    """
    client = _get_client()
    s3 = boto3.client("s3")

    content_bytes = get_s3_content(f"s3://{S3_BUCKET}/{document_key}")
    ext = document_key.rsplit(".", 1)[-1].lower() if "." in document_key else ""

    # Convert document to image(s) for Claude
    if ext == "pdf":
        image_b64_list = pdf_to_base64_images(content_bytes)
    else:
        image_b64_list = [base64.b64encode(content_bytes).decode("utf-8")]

    # Determine media type
    media_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "jp2": "image/jp2",
        "pdf": "image/png",  # pdf_to_base64_images outputs PNG
    }
    media_type = media_type_map.get(ext, "image/png")

    # Process each page
    all_text_parts: list[str] = []
    pages: list[dict] = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_invocations: int = 0

    for i, img_b64 in enumerate(image_b64_list):
        page_num = i + 1
        content_parts: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            },
            {"type": "text", "text": _OCR_PROMPT},
        ]

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content_parts}],
            }
        )

        response = client.invoke_model(
            modelId=BEDROCK_OCR_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        page_text = result["content"][0]["text"]

        # Extract token usage for environmental impact tracking
        usage = result.get("usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_invocations += 1

        all_text_parts.append(page_text)
        pages.append(
            {
                "page_number": page_num,
                "text": page_text,
            }
        )

    full_text = "\n\n---\n\n".join(all_text_parts)

    output = {
        "extraction_model": "bedrock_claude",
        "document_key": document_key,
        "job_id": job_id,
        "full_text": full_text,
        "page_count": len(pages),
        "pages": pages,
        "bedrock_input_tokens": total_input_tokens,
        "bedrock_output_tokens": total_output_tokens,
        "bedrock_invocations": total_invocations,
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
        f"Claude Vision extracted {len(pages)} page(s) "
        f"from {document_key} ({len(full_text)} chars)"
    )
    return output
