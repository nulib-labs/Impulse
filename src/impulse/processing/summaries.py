"""Document summarisation using Amazon Bedrock."""

from __future__ import annotations

import base64
import json

import boto3
from loguru import logger

from impulse.config import BEDROCK_MODEL_ID, BEDROCK_REGION
from impulse.utils import pdf_to_base64_images

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock_client


SUMMARY_PROMPT = (
    "Provide a short summary of the document. "
    "Do not return anything except a plain text summary. "
    "Use markdown only if necessary."
)


def summarise_document(
    document_text: str | None = None,
    pdf_bytes: bytes | None = None,
    image_bytes: bytes | None = None,
) -> dict:
    """Generate a summary of a document using Amazon Bedrock.

    Accepts text, PDF bytes, or image bytes (mutually preferred in that order
    for multimodal input).

    Returns a dict with ``summary``, ``bedrock_input_tokens``,
    ``bedrock_output_tokens``, and ``bedrock_invocations``.
    """
    client = _get_bedrock_client()

    content_parts: list[dict] = []

    if pdf_bytes:
        for img_b64 in pdf_to_base64_images(pdf_bytes):
            content_parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            )
    elif image_bytes:
        content_parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                },
            }
        )

    prompt = SUMMARY_PROMPT
    if document_text:
        prompt += f"\n\n{document_text}"

    content_parts.append({"type": "text", "text": prompt})

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": content_parts}],
        }
    )

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    summary = result["content"][0]["text"]

    # Extract token usage for environmental impact tracking
    usage = result.get("usage", {})

    logger.info(f"Generated summary ({len(summary)} chars)")
    return {
        "summary": summary,
        "bedrock_input_tokens": usage.get("input_tokens", 0),
        "bedrock_output_tokens": usage.get("output_tokens", 0),
        "bedrock_invocations": 1,
    }
