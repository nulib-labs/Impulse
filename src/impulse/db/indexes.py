"""
MongoDB text index management for full-text search.

Creates compound text indexes on each collection to enable $text queries.
Index creation is idempotent -- MongoDB will skip if the index already exists.
"""

from __future__ import annotations

from pymongo import TEXT
from loguru import logger

from impulse.db.client import get_db

_indexes_ensured = False


def ensure_text_indexes() -> None:
    """Create text indexes on all searchable collections (idempotent).

    Called once per process; subsequent calls are no-ops.
    """
    global _indexes_ensured
    if _indexes_ensured:
        return

    db = get_db()

    # Jobs: search by custom_id, task_type, status, and metadata values
    db["jobs"].create_index(
        [
            ("custom_id", TEXT),
            ("task_type", TEXT),
            ("status", TEXT),
        ],
        weights={"custom_id": 10, "task_type": 3, "status": 1},
        name="jobs_text_search",
        default_language="english",
        background=True,
    )
    logger.info("Ensured text index on 'jobs' collection")

    # Results: search across extracted OCR text, summaries, and document keys
    db["results"].create_index(
        [
            ("extracted_text", TEXT),
            ("summary", TEXT),
            ("document_key", TEXT),
        ],
        weights={"extracted_text": 5, "summary": 3, "document_key": 1},
        name="results_text_search",
        default_language="english",
        background=True,
    )
    logger.info("Ensured text index on 'results' collection")

    # Collections: search by name and description
    db["collections"].create_index(
        [
            ("name", TEXT),
            ("description", TEXT),
        ],
        weights={"name": 10, "description": 5},
        name="collections_text_search",
        default_language="english",
        background=True,
    )
    logger.info("Ensured text index on 'collections' collection")

    # Analyses: search by name and description
    db["analyses"].create_index(
        [
            ("name", TEXT),
            ("description", TEXT),
        ],
        weights={"name": 10, "description": 5},
        name="analyses_text_search",
        default_language="english",
        background=True,
    )
    logger.info("Ensured text index on 'analyses' collection")

    _indexes_ensured = True
    logger.info("All text indexes ensured")
