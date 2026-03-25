"""Metadata extraction: SpaCy NER + Amazon Bedrock LLM."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO

import boto3
from loguru import logger

from impulse.config import BEDROCK_MODEL_ID, BEDROCK_REGION
from impulse.utils import pdf_to_base64_images

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# ── Module-level Bedrock client ─────────────────────────────────────────────

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock_client


# ── SpaCy NER ───────────────────────────────────────────────────────────────


def extract_spacy_entities(text: str) -> tuple[list[str], list[str]]:
    """Run SpaCy NER and return deduplicated (places, people) lists."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    gpes: list[str] = []
    people: list[str] = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            gpes.append(ent.text)
        elif ent.label_ == "PERSON":
            people.append(ent.text)
    return list(dict.fromkeys(gpes)), list(dict.fromkeys(people))


# ── LLM (Bedrock) ──────────────────────────────────────────────────────────


def _build_metadata_prompt(gpes: list[str], people: list[str]) -> str:
    return f"""You are a metadata extraction assistant. SpaCy has already performed
named-entity recognition on the document and surfaced these candidates:

  Candidate places : {gpes if gpes else "none detected"}
  Candidate people : {people if people else "none detected"}

Using these candidates (and the document text below, if provided), identify:

1. The SINGLE most important or most-frequently-mentioned location.
   - Prefer the place most central to the document's subject matter.
   - Resolve ambiguous names to their most specific form
     (e.g. "Washington" -> "Washington, D.C." or "Washington State").
   - If all candidates look spurious, attempt to extract the correct
     place yourself; otherwise return null.

2. Up to 6 KEY PEOPLE mentioned in the document.
   - Prefer people who are subjects of the document over passing references.
   - If all candidates look spurious, attempt to extract them yourself;
     otherwise return an empty list.

Return ONLY valid JSON -- no prose, no markdown fences -- in exactly this shape:
{{
  "main_place": "<place or null>",
  "key_people": ["<name>", "..."]
}}"""


def ask_bedrock(
    gpes: list[str],
    people: list[str],
    document_text: str | None = None,
    pdf_bytes: bytes | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """Call Amazon Bedrock to extract metadata from document content."""
    client = _get_bedrock_client()
    prompt = _build_metadata_prompt(gpes, people)

    if document_text:
        prompt += f"\n\nDocument text:\n{document_text}"

    # Build the messages payload for Claude on Bedrock
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

    content_parts.append({"type": "text", "text": prompt})

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
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

    # Extract token usage for environmental impact tracking
    usage = result.get("usage", {})

    return {
        "text": result["content"][0]["text"],
        "bedrock_input_tokens": usage.get("input_tokens", 0),
        "bedrock_output_tokens": usage.get("output_tokens", 0),
        "bedrock_invocations": 1,
    }


# ── JSON parsing ────────────────────────────────────────────────────────────


def extract_valid_json(content: str) -> dict:
    """Strip non-JSON content and parse the first valid JSON object found."""
    content = re.sub(r"```(?:json)?\s*", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"main_place": None, "key_people": []}


# ── High-level entry point ──────────────────────────────────────────────────


def extract_metadata(
    document_text: str | None = None,
    pdf_bytes: bytes | None = None,
    image_bytes: bytes | None = None,
    accession_number: str = "",
) -> dict:
    """Extract metadata from a document and return a dict ready for MongoDB.

    Runs SpaCy NER on any available text, then calls Bedrock for refinement.
    """
    gpes: list[str] = []
    people: list[str] = []

    if document_text:
        gpes, people = extract_spacy_entities(document_text)

    raw_response = ask_bedrock(
        gpes=gpes,
        people=people,
        document_text=document_text,
        pdf_bytes=pdf_bytes,
        image_bytes=image_bytes,
    )

    result = extract_valid_json(raw_response["text"])
    result["accession_number"] = accession_number

    # Propagate Bedrock token usage for environmental impact tracking
    result["bedrock_input_tokens"] = raw_response.get("bedrock_input_tokens", 0)
    result["bedrock_output_tokens"] = raw_response.get("bedrock_output_tokens", 0)
    result["bedrock_invocations"] = raw_response.get("bedrock_invocations", 0)

    return result
