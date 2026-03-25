"""GET /search -- Full-text search across jobs, documents, collections, and analyses."""

from __future__ import annotations

import json
import re

from loguru import logger

from impulse.db.client import get_collection
from impulse.db.indexes import ensure_text_indexes


# Maximum snippet length for document text results
_SNIPPET_LEN = 200


def _snippet(text: str, query: str, length: int = _SNIPPET_LEN) -> str:
    """Extract a snippet from *text* centred around the first occurrence of a query word.

    Falls back to the first *length* characters if no match is found.
    """
    if not text:
        return ""

    # Try to find the first query word in the text (case-insensitive)
    words = query.strip().split()
    best_pos = len(text)  # fallback: start
    for word in words:
        match = re.search(re.escape(word), text, re.IGNORECASE)
        if match and match.start() < best_pos:
            best_pos = match.start()

    if best_pos == len(text):
        # No match found, return from start
        return text[:length] + ("..." if len(text) > length else "")

    # Centre the snippet around the match
    half = length // 2
    start = max(0, best_pos - half)
    end = min(len(text), start + length)
    if start > 0 and end < len(text):
        return "..." + text[start:end] + "..."
    elif start > 0:
        return "..." + text[start:end]
    elif end < len(text):
        return text[start:end] + "..."
    return text[start:end]


def search(user_id: str, query_params: dict) -> dict:
    """Run full-text search across multiple MongoDB collections.

    Query parameters:
        q          -- the search query (required)
        types      -- comma-separated list: jobs,documents,collections,analyses (default: all)
        page       -- page number, 1-based (default: 1)
        page_size  -- results per type (default: 20)
    """
    q = (query_params.get("q") or "").strip()
    if not q:
        return _response(400, {"error": "Missing required query parameter 'q'"})

    types_raw = query_params.get("types", "jobs,documents,collections,analyses")
    requested_types = {t.strip() for t in types_raw.split(",") if t.strip()}
    page = max(1, int(query_params.get("page", "1")))
    page_size = min(50, max(1, int(query_params.get("page_size", "20"))))
    skip = (page - 1) * page_size

    # Ensure text indexes exist (idempotent, cached after first call)
    try:
        ensure_text_indexes()
    except Exception as exc:
        logger.warning(f"Text index creation failed (may already exist): {exc}")

    results: dict = {}
    counts: dict = {}

    text_filter = {"$text": {"$search": q}}
    score_proj = {"score": {"$meta": "textScore"}}
    score_sort = [("score", {"$meta": "textScore"})]

    # ── Jobs ─────────────────────────────────────────────────────────────
    if "jobs" in requested_types:
        try:
            coll = get_collection("jobs")
            job_filter = {"user_id": user_id, **text_filter}
            total = coll.count_documents(job_filter)
            cursor = (
                coll.find(
                    job_filter,
                    {
                        "_id": 0,
                        "job_id": 1,
                        "custom_id": 1,
                        "status": 1,
                        "task_type": 1,
                        "ocr_engine": 1,
                        "metadata": 1,
                        "total_documents": 1,
                        "processed_documents": 1,
                        "failed_documents": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        **score_proj,
                    },
                )
                .sort(score_sort)
                .skip(skip)
                .limit(page_size)
            )
            results["jobs"] = list(cursor)
            counts["jobs"] = total
        except Exception as exc:
            logger.error(f"Job search failed: {exc}")
            results["jobs"] = []
            counts["jobs"] = 0

    # ── Documents (results collection) ───────────────────────────────────
    if "documents" in requested_types:
        try:
            coll = get_collection("results")
            # Results don't have user_id directly -- we need to join through
            # jobs.  For efficiency, first get the user's job_ids, then search
            # results within those jobs.
            jobs_coll = get_collection("jobs")
            user_job_ids = jobs_coll.distinct("job_id", {"user_id": user_id})

            doc_filter = {"job_id": {"$in": user_job_ids}, **text_filter}
            total = coll.count_documents(doc_filter)
            cursor = (
                coll.find(
                    doc_filter,
                    {
                        "_id": 0,
                        "result_id": 1,
                        "job_id": 1,
                        "document_key": 1,
                        "page_number": 1,
                        "extraction_model": 1,
                        "extracted_text": 1,
                        "summary": 1,
                        "created_at": 1,
                        **score_proj,
                    },
                )
                .sort(score_sort)
                .skip(skip)
                .limit(page_size)
            )
            docs = []
            for doc in cursor:
                # Replace full text with a snippet for transfer efficiency
                if doc.get("extracted_text"):
                    doc["extracted_text"] = _snippet(doc["extracted_text"], q)
                if doc.get("summary") and len(doc["summary"]) > 300:
                    doc["summary"] = doc["summary"][:300] + "..."
                # Extract a readable filename from the document_key
                key = doc.get("document_key", "")
                doc["filename"] = key.rsplit("/", 1)[-1] if "/" in key else key
                docs.append(doc)
            results["documents"] = docs
            counts["documents"] = total
        except Exception as exc:
            logger.error(f"Document search failed: {exc}")
            results["documents"] = []
            counts["documents"] = 0

    # ── Collections ──────────────────────────────────────────────────────
    if "collections" in requested_types:
        try:
            coll = get_collection("collections")
            col_filter = {"user_id": user_id, **text_filter}
            total = coll.count_documents(col_filter)
            cursor = (
                coll.find(
                    col_filter,
                    {
                        "_id": 0,
                        "collection_id": 1,
                        "name": 1,
                        "description": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        **score_proj,
                    },
                )
                .sort(score_sort)
                .skip(skip)
                .limit(page_size)
            )
            col_results = list(cursor)
            # Add document_count by counting the documents array length
            # (not returned from text search projection for efficiency)
            for c in col_results:
                detail = coll.find_one(
                    {"collection_id": c["collection_id"]},
                    {"documents": 1},
                )
                c["document_count"] = len(detail.get("documents", [])) if detail else 0
            results["collections"] = col_results
            counts["collections"] = total
        except Exception as exc:
            logger.error(f"Collection search failed: {exc}")
            results["collections"] = []
            counts["collections"] = 0

    # ── Analyses ─────────────────────────────────────────────────────────
    if "analyses" in requested_types:
        try:
            coll = get_collection("analyses")
            ana_filter = {"user_id": user_id, **text_filter}
            total = coll.count_documents(ana_filter)
            cursor = (
                coll.find(
                    ana_filter,
                    {
                        "_id": 0,
                        "analysis_id": 1,
                        "name": 1,
                        "description": 1,
                        "status": 1,
                        "sources": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        **score_proj,
                    },
                )
                .sort(score_sort)
                .skip(skip)
                .limit(page_size)
            )
            results["analyses"] = list(cursor)
            counts["analyses"] = total
        except Exception as exc:
            logger.error(f"Analysis search failed: {exc}")
            results["analyses"] = []
            counts["analyses"] = 0

    total_count = sum(counts.values())
    logger.info(
        f"Search for '{q}' by user {user_id}: {total_count} total results "
        f"(jobs={counts.get('jobs', 0)}, documents={counts.get('documents', 0)}, "
        f"collections={counts.get('collections', 0)}, analyses={counts.get('analyses', 0)})"
    )

    return _response(
        200,
        {
            "query": q,
            "results": results,
            "counts": counts,
            "total": total_count,
            "page": page,
            "page_size": page_size,
        },
    )


def _response(status_code: int, body: dict) -> dict:
    """Build an API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }
