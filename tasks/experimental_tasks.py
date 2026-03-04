import io
import json
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import override

import boto3
import certifi
import spacy
from fireworks.core.firework import FWAction, FireTaskBase
from fireworks.utilities.filepad import FilePad
from loguru import logger
import ollama
from pymongo import MongoClient

mongo_client = MongoClient(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING_IMPULSE"))
db = mongo_client["praxis"]

fp = FilePad(
    host=str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")),
    port=27017,
    name="fireworks",
    uri_mode=True,
    mongoclient_kwargs={"tls": True, "tlsCAFile": certifi.where()},
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LLM_MODEL = "google/gemma-3-27b-it"
IIIF_IMAGE_WIDTH = 1500  # px wide to request from IIIF Image API
NOMINATIM_UA = "impulse-eis-pipeline/1.0 (Northwestern University Libraries)"

VALID_THEMES = [
    "Transportation Infrastructure",
    "Energy Systems",
    "Wildlife and Natural Areas",
    "Water Systems",
    "Urban Development",
    "Industrial Production and Materials",
    "Climate and Weather Modification",
    "Governance and Institutional Control",
    "Place Based Development Conflicts",
    "Indigenous Narratives and Sovereignty",
]


# ── Shared S3 helpers (mirrors pattern in existing tasks.py) ─────────────────


def s3_read_json(s3_path: str) -> dict:
    match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
    bucket, key = match.group(1), match.group(2)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_write_json(s3_path: str, data) -> None:
    match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
    bucket, key = match.group(1), match.group(2)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
    )


def s3_read_bytes(s3_path: str) -> bytes:
    match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
    bucket, key = match.group(1), match.group(2)
    buf = BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.read()


# ── MongoDB pipeline store (replaces S3 for intermediate files) ───────────────

_pipeline_store = mongo_client["praxis"]["pipeline_store"]


def mongo_read_json(key: str) -> dict:
    doc = _pipeline_store.find_one({"_id": key})
    if not doc:
        raise ValueError(f"Pipeline store missing key: {key!r}")
    return doc["data"]


def mongo_write_json(key: str, data) -> None:
    _pipeline_store.replace_one(
        {"_id": key},
        {"_id": key, "data": data},
        upsert=True,
    )


def load_json(path: str) -> dict:
    """Read JSON from S3 (s3://), local file (starts with /), or MongoDB pipeline store."""
    if re.match(r"^s3a?://", path):
        return s3_read_json(path)
    if path.startswith("/"):
        with open(path) as f:
            return json.load(f)
    return mongo_read_json(path)


def save_json(path: str, data) -> None:
    """Write JSON to S3 (s3://), local file (starts with /), or MongoDB pipeline store."""
    if re.match(r"^s3a?://", path):
        s3_write_json(path, data)
    elif path.startswith("/"):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        mongo_write_json(path, data)


# ── Shared LLM helper (mirrors Summaries.call_llm) ───────────────────────────


def call_llm(
    prompt: str, model: str = DEFAULT_LLM_MODEL, max_tokens: int = 2048
) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_predict": max_tokens,
        },
    )
    return response["message"]["content"].strip()


def parse_json_response(text: str):
    """Strip markdown fences and parse JSON robustly."""
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def llm_host_from_spec(fw_spec: dict) -> str:
    """
    Get vLLM host. If llm_host is explicitly set in fw_spec, use that.
    Otherwise auto-detect using the current machine's hostname — same
    approach as Summaries/Themes/Context in tasks.py.
    """
    import socket

    return fw_spec.get("llm_host", f"http://{socket.gethostname()}:8000/v1")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Fetch IIIF Collection
# ─────────────────────────────────────────────────────────────────────────────


class FetchIIIFCollectionTask(FireTaskBase):
    """
    Fetch the Northwestern IIIF v3 collection manifest, following pagination,
    and save it to S3 as a single merged JSON.

    fw_spec keys:
      collection_url  — full IIIF collection URL
      output_path     — S3 path to write merged collection JSON
    """

    _fw_name = "Fetch IIIF Collection Task"

    @staticmethod
    def _fetch_json(url: str, retries: int = 3) -> dict:
        import requests

        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{retries} for {url}: {e}")
                time.sleep(2**attempt)

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        collection_url = fw_spec["collection_url"]
        output_path = fw_spec["output_path"]

        logger.info(f"Fetching IIIF collection: {collection_url}")

        all_items = []
        current_url = collection_url

        while current_url:
            page = self._fetch_json(current_url)
            for item in page.get("items", []):
                if item.get("type") == "Manifest":
                    all_items.append(item)
            # Follow pagination
            current_url = next(
                (
                    i["id"]
                    for i in page.get("items", [])
                    if i.get("type") == "Collection" and "page=" in i.get("id", "")
                ),
                None,
            )

        merged = {**page, "items": all_items}
        logger.info(f"Fetched {len(all_items)} works")

        save_json(output_path, merged)
        return FWAction(
            update_spec={
                "collection_path": output_path,
                "work_count": len(all_items),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: OCR (uses existing DocumentExtractionTask from tasks.py)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Task 2b: Aggregate OCR blocks → {work_uuid: text} in pipeline_store
# ─────────────────────────────────────────────────────────────────────────────


class AggregateOCRTask(FireTaskBase):
    """
    Read per-block OCR output from praxis.pages (written by DocumentExtractionTask),
    resolve each accession_number to its IIIF work UUID by loading the collection
    manifest (saved by FetchIIIFCollectionTask) and calling the NUL non-IIIF work
    API for each UUID to get its accession_number, then write {uuid: full_text}
    to pipeline_store.

    Strips HTML tags from the html field and concatenates blocks in page order.

    fw_spec keys:
      collection_path — pipeline_store key for the IIIF collection (set by FW1)
      output_path     — pipeline_store key to write (default: "eis-enrichment/ocr_output")
    """

    _fw_name = "Aggregate OCR Task"

    _NUL_WORK_API = "https://api.dc.library.northwestern.edu/api/v2/works"

    @staticmethod
    def _strip_html(html_text: str) -> str:
        return re.sub(r"<[^>]+>", " ", html_text or "").strip()

    @staticmethod
    def _uuid_from_iiif_url(iiif_id: str) -> str | None:
        """Extract UUID from a NUL IIIF manifest URL like .../works/{uuid}?as=iiif."""
        match = re.search(r"/works/([0-9a-f-]{36})", iiif_id)
        return match.group(1) if match else None

    def _build_accession_to_uuid(self, collection_path: str) -> dict[str, str]:
        """
        Load the IIIF collection, then call the non-IIIF work API for each UUID
        to get its accession_number.  Returns {accession_number: uuid}.
        """
        import requests

        collection = load_json(collection_path)
        uuids = [
            self._uuid_from_iiif_url(item.get("id", ""))
            for item in collection.get("items", [])
            if item.get("type") == "Manifest"
        ]
        uuids = [u for u in uuids if u]
        logger.info(f"Building accession→UUID map for {len(uuids)} works")

        mapping: dict[str, str] = {}
        for uuid in uuids:
            try:
                resp = requests.get(
                    f"{self._NUL_WORK_API}/{uuid}",
                    timeout=15,
                )
                resp.raise_for_status()
                acc = resp.json().get("data", {}).get("accession_number")
                if acc:
                    mapping[acc] = uuid
                    logger.debug(f"  {acc} → {uuid}")
                else:
                    logger.warning(f"  No accession_number for UUID {uuid}")
            except Exception as e:
                logger.warning(f"  Work API call failed for {uuid}: {e}")
            time.sleep(0.1)  # be polite to the API

        logger.info(f"Resolved {len(mapping)} accession numbers to UUIDs")
        return mapping

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        collection_path = fw_spec.get("collection_path", "eis-enrichment/00_collection")
        output_path = fw_spec.get("output_path", "eis-enrichment/ocr_output")

        pages_col = mongo_client["praxis"]["pages"]

        # Group blocks by accession_number, sorted by page_number
        work_blocks: dict[str, list] = {}
        for block in pages_col.find(
            {}, {"accession_number": 1, "html": 1, "page_number": 1}
        ):
            acc = block.get("accession_number")
            if not acc:
                continue
            work_blocks.setdefault(acc, []).append(
                (block.get("page_number", 0), block.get("html", ""))
            )

        # Build accession_number → UUID from collection + non-IIIF work API
        accession_to_uuid = self._build_accession_to_uuid(collection_path)

        # Build {uuid: full_text}
        ocr_output: dict[str, str] = {}
        for acc, blocks in work_blocks.items():
            uuid = accession_to_uuid.get(acc)
            if not uuid:
                logger.warning(f"Skipping {acc} — could not resolve to UUID")
                continue
            blocks.sort(key=lambda x: x[0])
            text = "\n".join(self._strip_html(html) for _, html in blocks)
            ocr_output[uuid] = text
            logger.info(
                f"Aggregated {len(blocks)} blocks for {uuid} ({len(text)} chars)"
            )

        save_json(output_path, ocr_output)
        logger.info(
            f"Wrote {len(ocr_output)} works to pipeline_store key {output_path!r}"
        )

        return FWAction(update_spec={"docs_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Get Metadata (place + people)
# ─────────────────────────────────────────────────────────────────────────────


class GetMetadataTask(FireTaskBase):
    """
    Extract main location and key people from each document using
    spaCy NER followed by a Gemma LLM call for disambiguation.
    Mirrors and extends the existing GetMetadata task.

    fw_spec keys:
      docs_path    — S3 path to OCR output JSON
      output_path  — S3 path to write metadata results JSON
      llm_host     — vLLM base URL (e.g. http://localhost:8000/v1)
      debug        — if True, read/write local files instead of S3
    """

    _fw_name = "Get Metadata Task"

    def _ask_llm(self, gpes: list, people: list, text: str, host: str) -> dict:
        prompt = f"""Document text (first 2000 words):
{" ".join(text.split()[:2000])}

Places found by NER: {gpes}
People found by NER: {people}

Task: Identify:
1) The single most important location in this document. Be specific
   (e.g. "Rogers Park, Chicago, Illinois" not just "Chicago").
   Augment vague place names where confident (e.g. "washington" -> "Washington, D.C.").
2) 5-7 key people mentioned by name.

If places or names are unclear or nonsensical (OCR noise), return null.
Return ONLY this exact JSON, no other text:
{{
  "main_place": "...",
  "key_people": ["Person One", "Person Two"]
}}"""
        raw = call_llm(prompt, host, max_tokens=200)
        try:
            return parse_json_response(raw)
        except Exception:
            return {"main_place": None, "key_people": []}

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        docs_path = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        llm_host = llm_host_from_spec(fw_spec)

        nlp = spacy.load("en_core_web_sm")

        docs = load_json(docs_path)

        results = {}
        for doc_id, text in docs.items():
            logger.info(f"GetMetadata: {doc_id}")
            doc = nlp(text)
            gpes = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "GPE"))
            people = list(
                dict.fromkeys(e.text for e in doc.ents if e.label_ == "PERSON")
            )

            ai = self._ask_llm(gpes, people, text, llm_host)
            results[doc_id] = {
                "doc_id": doc_id,
                "main_place": ai.get("main_place"),
                "key_people": ai.get("key_people", []),
            }
            logger.info(
                f"  place={results[doc_id]['main_place']}, "
                f"people={len(results[doc_id]['key_people'])}"
            )

        save_json(output_path, results)

        return FWAction(update_spec={"metadata_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Geocode
# ─────────────────────────────────────────────────────────────────────────────


class GeocodeTask(FireTaskBase):
    """
    Geocode each document's main_place using Nominatim (OpenStreetMap).
    Adds lat/lon coordinates to the metadata results.
    Rate-limited to 1 request/second per Nominatim's usage policy.

    fw_spec keys:
      metadata_path — S3 path to GetMetadataTask output
      output_path   — S3 path to write geocoded results JSON
      debug         — if True, read/write local files
    """

    _fw_name = "Geocode Task"

    @staticmethod
    def _geocode(place: str) -> dict | None:
        """Call Nominatim and return {lat, lon, display_name} or None."""
        import requests

        if not place:
            return None

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        headers = {"User-Agent": NOMINATIM_UA}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            results = resp.json()
            if results:
                r = results[0]
                return {
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "display_name": r.get("display_name", place),
                }
        except Exception as e:
            logger.warning(f"Geocode failed for '{place}': {e}")
        return None

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        metadata_path = fw_spec["metadata_path"]
        output_path = fw_spec["output_path"]

        metadata = load_json(metadata_path)

        results = {}
        for doc_id, meta in metadata.items():
            place = meta.get("main_place")
            logger.info(f"Geocoding '{place}' for {doc_id}")

            coords = self._geocode(place)
            results[doc_id] = {
                **meta,
                "coordinates": coords,  # {lat, lon, display_name} or None
            }

            if coords:
                logger.info(f"  → {coords['lat']}, {coords['lon']}")
            else:
                logger.warning(f"  → no result")

            # Nominatim requires max 1 request/second
            time.sleep(1.1)

        save_json(output_path, results)

        return FWAction(update_spec={"geocoded_metadata_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Summaries
# ─────────────────────────────────────────────────────────────────────────────


class SummaryTask(FireTaskBase):
    """
    Generate a ~125-word factual summary for each document.
    Mirrors the existing Summaries task with improved prompting.

    fw_spec keys:
      docs_path    — S3 path to OCR output JSON
      output_path  — S3 path to write summaries JSON
      llm_host     — vLLM base URL
      debug        — if True, read/write local files
    """

    _fw_name = "Summary Task"

    def _ask_llm(self, text: str, host: str) -> dict:
        # Truncate to ~6k tokens — safe for gemma3:27b context
        max_chars = 24000
        if len(text) > max_chars:
            chunk = (
                text[: max_chars // 2]
                + "\n...[middle truncated]...\n"
                + text[-(max_chars // 2) :]
            )
        else:
            chunk = text

        prompt = f"""Document text:
{" ".join(chunk.split())}

Task: Write a factual summary of this document in approximately 125 words (minimum 70, maximum 150).
Cover: what project or proposal it describes, where it is located, which agencies are involved, and the stated goal.
Only include information from the document — do not add outside knowledge.

Return ONLY this JSON, no other text:
{{"summary": "your summary here"}}
If the document is too vague, return: {{"summary": null}}"""

        raw = call_llm(prompt, host, max_tokens=256)
        try:
            return parse_json_response(raw)
        except Exception:
            return {"summary": None}

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        docs_path = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        llm_host = llm_host_from_spec(fw_spec)

        docs = load_json(docs_path)

        results = {}
        for doc_id, text in docs.items():
            logger.info(f"Summary: {doc_id}")
            result = self._ask_llm(text, llm_host)
            results[doc_id] = {
                "doc_id": doc_id,
                "summary": result.get("summary"),
            }
            wc = len((results[doc_id]["summary"] or "").split())
            logger.info(f"  {wc} words")

        save_json(output_path, results)

        return FWAction(update_spec={"summaries_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Themes
# ─────────────────────────────────────────────────────────────────────────────


class ThemesEnrichmentTask(FireTaskBase):
    """
    Assign 1-3 themes from a fixed controlled vocabulary.
    Mirrors and replaces the existing Themes task.

    fw_spec keys:
      docs_path    — S3 path to OCR output JSON
      output_path  — S3 path to write themes JSON
      llm_host     — vLLM base URL
      debug        — if True, read/write local files
    """

    _fw_name = "Themes Enrichment Task"

    def _ask_llm(self, text: str, host: str) -> dict:
        max_chars = 24000
        chunk = text[:max_chars] if len(text) > max_chars else text
        themes_list = "\n".join(f"- {t}" for t in VALID_THEMES)

        prompt = f"""Document text:
{" ".join(chunk.split())}

Task: Assign 1-3 themes to this document from the list below.
Use ONLY themes from this list, spelled exactly as written:
{themes_list}

Return ONLY valid JSON, no other text:
{{"themes": ["Theme One", "Theme Two"]}}
If no themes apply: {{"themes": []}}"""

        raw = call_llm(prompt, host, max_tokens=128)
        try:
            parsed = parse_json_response(raw)
            # Filter to valid themes only — model may hallucinate variants
            themes = [t for t in parsed.get("themes", []) if t in VALID_THEMES]
            return {"themes": themes}
        except Exception:
            return {"themes": []}

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        docs_path = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        llm_host = llm_host_from_spec(fw_spec)

        docs = load_json(docs_path)

        results = {}
        for doc_id, text in docs.items():
            logger.info(f"Themes: {doc_id}")
            result = self._ask_llm(text, llm_host)
            results[doc_id] = {"doc_id": doc_id, "themes": result["themes"]}
            logger.info(f"  {result['themes']}")

        save_json(output_path, results)

        return FWAction(update_spec={"themes_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: Quotes
# ─────────────────────────────────────────────────────────────────────────────


class QuotesEnrichmentTask(FireTaskBase):
    """
    Extract verbatim public comment excerpts.
    Mirrors and replaces the existing Quotes task.

    fw_spec keys:
      docs_path    — S3 path to OCR output JSON
      output_path  — S3 path to write quotes JSON
      llm_host     — vLLM base URL
      debug        — if True, read/write local files
    """

    _fw_name = "Quotes Enrichment Task"

    @staticmethod
    def _extract_comment_section(text: str) -> str:
        """Find the public comment section, fall back to last 12k chars."""
        markers = [
            "public comment",
            "public hearing",
            "community comment",
            "citizen comment",
            "oral comment",
            "written comment",
            "comment period",
            "testimony",
            "public testimony",
        ]
        lower = text.lower()
        best_pos = max((lower.rfind(m) for m in markers), default=-1)
        return text[best_pos:] if best_pos != -1 else text[-12000:]

    def _ask_llm(self, text: str, host: str) -> dict:
        section = self._extract_comment_section(text)
        prompt = f"""Document section:
{" ".join(section.split())}

Task: QUOTES
Find 2-4 verbatim excerpts from named public commenters or organizations.
Rules:
- Copy text EXACTLY as written — do not paraphrase or change wording
- Include commenter name or organization if present
- Format: "'exact quote' - Name or Organization"
- If no clearly attributed public comments exist, return an empty list

Return ONLY this JSON, no other text:
{{"PUBLIC_COMMENT": ["'quote one' - Name", "'quote two' - Organization"]}}"""

        raw = call_llm(prompt, host, max_tokens=512)
        try:
            parsed = parse_json_response(raw)
            comments = parsed.get("PUBLIC_COMMENT", [])
            return {"PUBLIC_COMMENT": comments if isinstance(comments, list) else []}
        except Exception:
            return {"PUBLIC_COMMENT": []}

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        docs_path = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        llm_host = llm_host_from_spec(fw_spec)

        docs = load_json(docs_path)

        results = {}
        for doc_id, text in docs.items():
            logger.info(f"Quotes: {doc_id}")
            result = self._ask_llm(text, llm_host)
            results[doc_id] = {
                "doc_id": doc_id,
                "PUBLIC_COMMENT": result["PUBLIC_COMMENT"],
            }
            logger.info(f"  {len(result['PUBLIC_COMMENT'])} quotes found")

        save_json(output_path, results)

        return FWAction(update_spec={"quotes_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 8: Context + Completed
# ─────────────────────────────────────────────────────────────────────────────


class ContextEnrichmentTask(FireTaskBase):
    """
    Generate historical context and determine whether the project was
    ultimately completed.

    Adds two fields:
      context   — 2-4 sentence explanation of what happened
      completed — True (built/executed), False (cancelled/rejected),
                  or None (outcome unclear)

    fw_spec keys:
      docs_path    — S3 path to OCR output JSON
      output_path  — S3 path to write context JSON
      llm_host     — vLLM base URL
      debug        — if True, read/write local files
    """

    _fw_name = "Context Enrichment Task"

    def _ask_llm(self, text: str, host: str) -> dict:
        max_chars = 24000
        if len(text) > max_chars:
            chunk = (
                text[: max_chars // 2]
                + "\n...[middle truncated]...\n"
                + text[-(max_chars // 2) :]
            )
        else:
            chunk = text

        prompt = f"""Document text:
{" ".join(chunk.split())}

Task: Context
Provide a 2-4 sentence explanation of the historical context and outcome of
this project or proposal. You may use outside knowledge if you are confident
and could cite a source.

Also determine whether the project was ultimately completed:
  true  — the project was executed, built, or implemented
  false — the project was cancelled, rejected, or stopped
  null  — the outcome is unclear from the document and your knowledge

Return ONLY one of these exact JSON formats, no other text:
{{"context": "explanation here", "completed": true}}
{{"context": "explanation here", "completed": false}}
{{"context": "explanation here", "completed": null}}"""

        raw = call_llm(prompt, host, max_tokens=512)
        try:
            parsed = parse_json_response(raw)
            return {
                "context": parsed.get("context", "").strip(),
                "completed": parsed.get("completed"),  # True, False, or None
            }
        except Exception:
            return {"context": "", "completed": None}

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        docs_path = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        llm_host = llm_host_from_spec(fw_spec)

        docs = load_json(docs_path)

        results = {}
        for doc_id, text in docs.items():
            logger.info(f"Context: {doc_id}")
            result = self._ask_llm(text, llm_host)
            results[doc_id] = {
                "doc_id": doc_id,
                "context": result["context"],
                "completed": result["completed"],
            }
            label = {
                True: "COMPLETED ✓",
                False: "NOT COMPLETED ✗",
                None: "UNCLEAR ?",
            }.get(result["completed"], "?")
            logger.info(f"  {label}")

        save_json(output_path, results)

        return FWAction(update_spec={"context_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 9: Build Enriched IIIF Manifest
# ─────────────────────────────────────────────────────────────────────────────


class BuildEnrichedManifestTask(FireTaskBase):
    """
    Merge all enrichment outputs into a single IIIF v3 Collection manifest.

    Standard IIIF metadata[] entries are readable by any viewer including Canopy.
    The x-impulse:enrichment block provides structured data for custom plugins:
      - geo.coordinates  → map view
      - themes           → faceted filtering
      - public_comments  → quotes panel
      - completed        → outcome filter

    fw_spec keys:
      collection_path       — S3 path to base IIIF collection JSON
      geocoded_metadata_path — S3 path to geocoded metadata JSON
      summaries_path        — S3 path to summaries JSON
      themes_path           — S3 path to themes JSON
      quotes_path           — S3 path to quotes JSON
      context_path          — S3 path to context JSON
      output_path           — S3 path to write enriched manifest JSON
      llm_model             — model name used (for provenance)
      debug                 — if True, read/write local files
    """

    _fw_name = "Build Enriched Manifest Task"

    @staticmethod
    def _get_label(obj) -> str:
        """Unwrap a IIIF v3 label object to a plain string."""
        if not obj or not isinstance(obj, dict):
            return str(obj) if obj else ""
        for values in obj.values():
            if values:
                return values[0]
        return ""

    @staticmethod
    def _meta_entry(label: str, value: str) -> dict:
        """Build a IIIF v3 metadata entry."""
        return {
            "label": {"none": [label]},
            "value": {"none": [value]},
        }

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        collection = load_json(fw_spec["collection_path"])
        geo_meta = load_json(fw_spec["geocoded_metadata_path"])
        summaries = load_json(fw_spec["summaries_path"])
        themes = load_json(fw_spec["themes_path"])
        quotes = load_json(fw_spec["quotes_path"])
        context = load_json(fw_spec["context_path"])
        llm_model = fw_spec.get("llm_model", DEFAULT_LLM_MODEL)
        output_path = fw_spec["output_path"]

        completed_labels = {True: "Completed", False: "Not Completed", None: "Unknown"}
        enriched_items = []

        for stub in collection.get("items", []):
            doc_id = stub["id"].split("/works/")[1].split("?")[0]

            meta = geo_meta.get(doc_id, {})
            summ = summaries.get(doc_id, {})
            thm = themes.get(doc_id, {})
            quot = quotes.get(doc_id, {})
            ctx = context.get(doc_id, {})

            # ── Standard IIIF v3 metadata (visible in any viewer) ────────────
            enriched_metadata = list(stub.get("metadata", []))

            if summ.get("summary"):
                enriched_metadata.append(self._meta_entry("Summary", summ["summary"]))
            if thm.get("themes"):
                enriched_metadata.append(
                    self._meta_entry("Themes", ", ".join(thm["themes"]))
                )
            if meta.get("main_place"):
                enriched_metadata.append(
                    self._meta_entry("Main Location", meta["main_place"])
                )
            if meta.get("coordinates"):
                coords = meta["coordinates"]
                enriched_metadata.append(
                    self._meta_entry("Coordinates", f"{coords['lat']}, {coords['lon']}")
                )
            if meta.get("key_people"):
                enriched_metadata.append(
                    self._meta_entry("Key People", "; ".join(meta["key_people"]))
                )
            if ctx.get("context"):
                enriched_metadata.append(
                    self._meta_entry("Historical Context", ctx["context"])
                )

            enriched_metadata.append(
                self._meta_entry(
                    "Completed", completed_labels.get(ctx.get("completed"), "Unknown")
                )
            )

            # ── x-impulse:enrichment (structured block for Canopy plugins) ──
            enriched_items.append(
                {
                    **stub,
                    "metadata": enriched_metadata,
                    "x-impulse:enrichment": {
                        "summary": summ.get("summary"),
                        "themes": thm.get("themes", []),
                        "geo": {
                            "main_place": meta.get("main_place"),
                            "label": self._get_label(stub.get("label", {})),
                            "coordinates": meta.get(
                                "coordinates"
                            ),  # {lat, lon, display_name}
                        },
                        "key_people": meta.get("key_people", []),
                        "public_comments": quot.get("PUBLIC_COMMENT", []),
                        "context": ctx.get("context"),
                        "completed": ctx.get("completed"),  # True / False / None
                        "pipeline_version": "2.0.0",
                        "pipeline_model": llm_model,
                        "pipeline_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

        enriched_collection = {
            **collection,
            "label": {
                "none": [self._get_label(collection.get("label", {})) + " [Enriched]"]
            },
            "x-impulse:pipeline": {
                "version": "2.0.0",
                "model": llm_model,
                "steps": [
                    "iiif-fetch",
                    "ocr-marker",
                    "metadata",
                    "geocode",
                    "summaries",
                    "themes",
                    "quotes",
                    "context",
                ],
                "document_count": len(enriched_items),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "items": enriched_items,
        }

        logger.info(f"Built enriched manifest with {len(enriched_items)} works")

        save_json(output_path, enriched_collection)

        return FWAction(update_spec={"enriched_manifest_path": output_path})


# ─────────────────────────────────────────────────────────────────────────────
# Task 10: Save Manifest to MongoDB
# ─────────────────────────────────────────────────────────────────────────────


class SaveManifestToMongoTask(FireTaskBase):
    """
    Write the enriched IIIF manifest to MongoDB.

    Stores the full collection document in praxis.enriched_manifests,
    and also upserts each individual work's enrichment data into
    praxis.work_enrichments for easy per-document querying.

    fw_spec keys:
      enriched_manifest_path — S3 path to enriched manifest JSON
      collection_id          — Northwestern collection UUID (for the _id field)
      debug                  — if True, read from local file
    """

    _fw_name = "Save Manifest To Mongo Task"

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        manifest_path = fw_spec["enriched_manifest_path"]
        collection_id = fw_spec.get("collection_id", "unknown")

        manifest = load_json(manifest_path)

        # ── Save full collection manifest ────────────────────────────────────
        manifests_col = mongo_client["praxis"]["enriched_manifests"]
        manifests_col.replace_one(
            {"_id": collection_id},
            {"_id": collection_id, **manifest},
            upsert=True,
        )
        logger.info(f"Saved collection manifest to enriched_manifests/{collection_id}")

        # ── Upsert per-work enrichment records ───────────────────────────────
        works_col = mongo_client["praxis"]["work_enrichments"]
        count = 0
        for item in manifest.get("items", []):
            doc_id = item["id"].split("/works/")[1].split("?")[0]
            enrichment = item.get("x-impulse:enrichment", {})
            label = ""
            for values in (item.get("label") or {}).values():
                if values:
                    label = values[0]
                    break

            works_col.replace_one(
                {"_id": doc_id},
                {
                    "_id": doc_id,
                    "collection_id": collection_id,
                    "iiif_id": item["id"],
                    "label": label,
                    "summary": enrichment.get("summary"),
                    "themes": enrichment.get("themes", []),
                    "geo": enrichment.get("geo", {}),
                    "key_people": enrichment.get("key_people", []),
                    "public_comments": enrichment.get("public_comments", []),
                    "context": enrichment.get("context"),
                    "completed": enrichment.get("completed"),
                    "pipeline_model": enrichment.get("pipeline_model"),
                    "pipeline_timestamp": enrichment.get("pipeline_timestamp"),
                },
                upsert=True,
            )
            count += 1

        logger.info(f"Upserted {count} work enrichment records")

        return FWAction(
            update_spec={
                "mongo_collection_id": collection_id,
                "mongo_work_count": count,
            }
        )
