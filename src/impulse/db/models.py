"""
Data models for the Impulse pipeline.

These are plain dataclasses used for validation and serialisation.
No binary fields -- all images/documents live in S3.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskType(str, Enum):
    IMAGE_TRANSFORM = "image_transform"
    DOCUMENT_EXTRACTION = "document_extraction"
    METS_CONVERSION = "mets_conversion"
    METADATA_EXTRACTION = "metadata_extraction"
    SUMMARISATION = "summarisation"
    NER = "ner"
    GEOCODE = "geocode"
    FULL_PIPELINE = "full_pipeline"


class OcrEngine(str, Enum):
    TEXTRACT = "textract"
    BEDROCK_CLAUDE = "bedrock_claude"
    MARKER_PDF = "marker_pdf"


# ── Job ──────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    """Top-level processing job submitted by a user."""

    job_id: str
    user_id: str
    status: str = JobStatus.PENDING.value
    task_type: str = TaskType.FULL_PIPELINE.value
    ocr_engine: str = OcrEngine.TEXTRACT.value
    custom_id: str = ""
    metadata: dict = field(default_factory=dict)
    input_s3_prefix: str = ""
    output_s3_prefix: str = ""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    step_functions_arn: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass
class Result:
    """Processing result for a single document/page."""

    result_id: str
    job_id: str
    document_key: str  # S3 key of the input document
    page_number: int = 0
    extraction_model: str = ""
    extracted_text: str = ""
    structured_output: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    summary: str = ""
    ner_entities: list = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Collection ───────────────────────────────────────────────────────────────


@dataclass
class Collection:
    """A user-curated group of documents drawn from across jobs."""

    collection_id: str
    user_id: str
    name: str
    description: str = ""
    documents: list = field(default_factory=list)
    # Each document: {s3_key, filename, job_id, source_type, size}
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Analysis ─────────────────────────────────────────────────────────────────


@dataclass
class Analysis:
    """A user-created analytical workspace over jobs/collections/documents."""

    analysis_id: str
    user_id: str
    name: str
    description: str = ""
    status: str = "IDLE"  # IDLE | RUNNING | COMPLETED | FAILED
    sources: list = field(default_factory=list)
    # [{type: "job"|"collection"|"document", id: str, name: str}]

    # Computed results (populated by run_analysis)
    entities: list = field(default_factory=list)
    # [{text, type, count, documents: [doc_key]}]
    entity_edges: list = field(default_factory=list)
    # [{source, target, weight, documents: [doc_key]}]
    locations: list = field(default_factory=list)
    # [{name, lat, lon, count, documents: [doc_key]}]
    word_frequencies: list = field(default_factory=list)
    # [{word, count}]
    doc_coordinates: list = field(default_factory=list)
    # [{doc_key, filename, x, y}]
    summary_stats: dict = field(default_factory=dict)
    # {total_docs, total_chars, entity_counts, ocr_engines, ...}
    timeline_events: list = field(default_factory=list)
    # [{job_id, custom_id, status, created_at, updated_at, total, processed}]

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Environmental Metrics ────────────────────────────────────────────────────


@dataclass
class EnvironmentalMetrics:
    """Per-document environmental impact metrics captured during processing."""

    metrics_id: str
    job_id: str
    document_key: str  # S3 key of the input document
    task_type: str  # e.g. "ocr_textract", "image_transform", "document_extraction"

    # Timing
    processing_duration_ms: int = 0

    # Resource consumption (measured)
    input_file_size_bytes: int = 0
    output_file_size_bytes: int = 0
    bedrock_input_tokens: int = 0
    bedrock_output_tokens: int = 0
    bedrock_invocations: int = 0
    textract_api_calls: int = 0
    page_count: int = 0

    # Compute profile (from task routing)
    compute_type: str = ""  # "lambda_3008mb", "fargate_8gb_2vcpu", "fargate_30gb_4vcpu"

    # Calculated impact
    energy_kwh: float = 0.0
    carbon_g_co2e: float = 0.0
    water_ml: float = 0.0

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Legacy models (kept for backward-compat with existing data) ──────────────


@dataclass
class WorkMetadata:
    """Metadata about a work."""

    creators: list | None = None
    title: str | None = None
    medium: str | None = None
    language: str | None = None


@dataclass
class PageProcessingMeta:
    deskew_angle: float | None = None
    noise_level: str = ""


@dataclass
class Page:
    _id: str = ""
    work_id: str = ""
    page_number: int = 0
    status: str = ""
    representations: dict = field(default_factory=dict)
    processed: dict = field(default_factory=dict)


@dataclass
class Work:
    """Tracks a complete work/document."""

    _id: str = ""
    page_count: int = 0
    metadata: dict = field(default_factory=dict)
    status: str = JobStatus.PENDING.value
