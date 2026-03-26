"""Environmental impact calculation for document processing.

Estimates energy consumption, carbon emissions, and water usage for each
document processed by the Impulse pipeline, based on measured resource
consumption (processing time, tokens, API calls, file sizes) and published
emission factors.

Sources:
  - Cloud Carbon Footprint (CCF) methodology: https://www.cloudcarbonfootprint.org/
  - IEA grid emission factors (2024)
  - AWS sustainability reports for PUE and WUE
  - Luccioni et al. (2023) for LLM inference energy estimates
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from loguru import logger

from impulse.db.client import get_collection
from impulse.db.models import EnvironmentalMetrics


# ── Configurable region-specific factors ────────────────────────────────────

# Grid carbon intensity in kg CO2e per kWh.
# Default is US average (~0.379).  Override for your AWS region:
#   us-west-2 (Oregon):   ~0.08  (very clean — hydro)
#   eu-west-1 (Ireland):  ~0.28
#   us-east-1 (Virginia): ~0.38
#   ap-southeast-1:       ~0.49
CARBON_INTENSITY_KG_PER_KWH = float(
    os.environ.get("CARBON_INTENSITY_KG_PER_KWH", "0.379")
)

# AWS Power Usage Effectiveness (PUE): ratio of total facility energy
# to IT equipment energy.  AWS average is ~1.2.
AWS_PUE = float(os.environ.get("AWS_PUE", "1.2"))

# AWS Water Usage Effectiveness (WUE): litres of water per kWh of
# IT equipment energy.  AWS average is ~1.8 L/kWh.
AWS_WUE_L_PER_KWH = float(os.environ.get("AWS_WUE_L_PER_KWH", "1.8"))


# ── Compute energy profiles ────────────────────────────────────────────────
# Estimated average power draw (watts) for each compute type.
# These account for the CPU/memory share but not the PUE overhead
# (that's applied separately).

COMPUTE_POWER_W: dict[str, float] = {
    # Lambda 3008 MB: ~1.5 vCPU equivalent, estimated ~5W average draw
    # (conservative; Lambda shares physical hardware across many invocations)
    "lambda_3008mb": 5.0,
    # Fargate 2 vCPU / 8 GB: dedicated container, ~14W average
    "fargate_8gb_2vcpu": 14.0,
    # Fargate 4 vCPU / 30 GB: heavy compute container, ~36W average
    "fargate_30gb_4vcpu": 36.0,
}


# ── AI service energy estimates ─────────────────────────────────────────────
# Based on Luccioni et al. (2023) and inference benchmarks for
# GPU-accelerated models.

# Bedrock (Claude Haiku): energy per 1000 tokens.
# Haiku is a small model; inference energy is modest.
BEDROCK_KWH_PER_1K_INPUT_TOKENS = float(
    os.environ.get("BEDROCK_KWH_PER_1K_INPUT_TOKENS", "0.0003")
)
BEDROCK_KWH_PER_1K_OUTPUT_TOKENS = float(
    os.environ.get("BEDROCK_KWH_PER_1K_OUTPUT_TOKENS", "0.0006")
)

# AWS Textract: energy per API call.
# Textract uses GPU-backed ML models; estimated ~0.0004 kWh per
# synchronous detect call (based on typical latency of 1-3s on
# inference hardware drawing ~150W).
TEXTRACT_KWH_PER_CALL = float(os.environ.get("TEXTRACT_KWH_PER_CALL", "0.0004"))

# S3 transfer: energy per MB transferred.
# Network + disk I/O; very small compared to compute.
S3_KWH_PER_MB = float(os.environ.get("S3_KWH_PER_MB", "0.0000002"))


# ── Comparison factors (for human-readable equivalences) ────────────────────
# All values per gram of CO2e.

# Average passenger car: ~120 g CO2e per km (EU average)
G_CO2E_PER_KM_DRIVEN = 120.0

# LED bulb (10W) running for 1 hour at US grid average:
# 0.01 kWh * 379 g/kWh = 3.79 g CO2e
G_CO2E_PER_HOUR_LED = 3.79

# Smartphone charge (~0.012 kWh): 0.012 * 379 = 4.55 g CO2e
G_CO2E_PER_SMARTPHONE_CHARGE = 4.55

# Google search: ~0.2 g CO2e (Google's published figure)
G_CO2E_PER_GOOGLE_SEARCH = 0.2


# ── Core calculation ────────────────────────────────────────────────────────


def calculate_impact(metrics: EnvironmentalMetrics) -> EnvironmentalMetrics:
    """Calculate energy, carbon, and water impact from raw measurements.

    Mutates and returns the same ``EnvironmentalMetrics`` instance with the
    ``energy_kwh``, ``carbon_g_co2e``, and ``water_ml`` fields populated.
    """
    energy = 0.0

    # 1. Compute energy (from processing duration and compute profile)
    power_w = COMPUTE_POWER_W.get(metrics.compute_type, 5.0)
    duration_s = metrics.processing_duration_ms / 1000.0
    compute_energy = (power_w * duration_s) / 3_600_000  # W·s -> kWh
    energy += compute_energy

    # 2. AI inference energy
    bedrock_energy = (
        metrics.bedrock_input_tokens / 1000.0
    ) * BEDROCK_KWH_PER_1K_INPUT_TOKENS + (
        metrics.bedrock_output_tokens / 1000.0
    ) * BEDROCK_KWH_PER_1K_OUTPUT_TOKENS
    energy += bedrock_energy

    textract_energy = metrics.textract_api_calls * TEXTRACT_KWH_PER_CALL
    energy += textract_energy

    # 3. Data transfer energy
    total_transfer_mb = (
        metrics.input_file_size_bytes + metrics.output_file_size_bytes
    ) / (1024 * 1024)
    transfer_energy = total_transfer_mb * S3_KWH_PER_MB
    energy += transfer_energy

    # Apply PUE (facility overhead)
    energy *= AWS_PUE

    # Populate calculated fields
    metrics.energy_kwh = round(energy, 10)
    metrics.carbon_g_co2e = round(
        energy * CARBON_INTENSITY_KG_PER_KWH * 1000, 6
    )  # kg -> g
    metrics.water_ml = round(energy * AWS_WUE_L_PER_KWH * 1000, 4)  # L -> mL

    return metrics


# ── Persistence ─────────────────────────────────────────────────────────────


def persist_metrics(metrics: EnvironmentalMetrics) -> None:
    """Save environmental metrics to MongoDB."""
    collection = get_collection("environmental_metrics")
    doc = metrics.to_dict()
    collection.update_one(
        {
            "job_id": metrics.job_id,
            "document_key": metrics.document_key,
            "task_type": metrics.task_type,
        },
        {"$set": doc},
        upsert=True,
    )
    logger.debug(
        f"Saved environmental metrics for {metrics.document_key} "
        f"({metrics.task_type}): {metrics.energy_kwh:.8f} kWh, "
        f"{metrics.carbon_g_co2e:.4f} g CO2e"
    )


def create_and_persist_metrics(
    *,
    job_id: str,
    document_key: str,
    task_type: str,
    compute_type: str,
    processing_duration_ms: int,
    input_file_size_bytes: int = 0,
    output_file_size_bytes: int = 0,
    bedrock_input_tokens: int = 0,
    bedrock_output_tokens: int = 0,
    bedrock_invocations: int = 0,
    textract_api_calls: int = 0,
    page_count: int = 0,
) -> EnvironmentalMetrics:
    """Create, calculate, and persist environmental metrics in one call."""
    metrics = EnvironmentalMetrics(
        metrics_id=str(uuid.uuid4()),
        job_id=job_id,
        document_key=document_key,
        task_type=task_type,
        compute_type=compute_type,
        processing_duration_ms=processing_duration_ms,
        input_file_size_bytes=input_file_size_bytes,
        output_file_size_bytes=output_file_size_bytes,
        bedrock_input_tokens=bedrock_input_tokens,
        bedrock_output_tokens=bedrock_output_tokens,
        bedrock_invocations=bedrock_invocations,
        textract_api_calls=textract_api_calls,
        page_count=page_count,
    )
    calculate_impact(metrics)
    persist_metrics(metrics)
    return metrics


# ── Aggregation ─────────────────────────────────────────────────────────────


def get_job_impact_summary(job_id: str) -> dict:
    """Aggregate environmental metrics for all documents in a job."""
    collection = get_collection("environmental_metrics")
    docs = list(collection.find({"job_id": job_id}, {"_id": 0}))

    if not docs:
        return {
            "job_id": job_id,
            "summary": None,
            "per_document": [],
            "comparisons": None,
        }

    summary = _aggregate_docs(docs)
    per_document = _build_per_document_list(docs)
    comparisons = _build_comparisons(summary["total_carbon_g_co2e"])

    return {
        "job_id": job_id,
        "summary": summary,
        "per_document": per_document,
        "comparisons": comparisons,
    }


def get_collection_impact_summary(collection_id: str) -> dict:
    """Aggregate environmental metrics for all documents in a collection."""
    coll = get_collection("collections")
    collection_doc = coll.find_one({"collection_id": collection_id}, {"_id": 0})

    if not collection_doc:
        return {
            "collection_id": collection_id,
            "summary": None,
            "per_document": [],
            "comparisons": None,
        }

    documents = collection_doc.get("documents", [])
    if not documents:
        return {
            "collection_id": collection_id,
            "summary": None,
            "per_document": [],
            "comparisons": None,
        }

    # Look up environmental metrics for each document by its S3 key
    env_collection = get_collection("environmental_metrics")
    all_metrics: list[dict] = []

    for doc in documents:
        s3_key = doc.get("s3_key", "")
        job_id = doc.get("job_id", "")
        if not s3_key:
            continue

        # A document may have multiple metrics entries (e.g., image_transform + OCR)
        query: dict = {"document_key": s3_key}
        if job_id:
            query["job_id"] = job_id
        metrics_docs = list(env_collection.find(query, {"_id": 0}))
        all_metrics.extend(metrics_docs)

    if not all_metrics:
        return {
            "collection_id": collection_id,
            "summary": None,
            "per_document": [],
            "comparisons": None,
        }

    summary = _aggregate_docs(all_metrics)
    per_document = _build_per_document_list(all_metrics)
    comparisons = _build_comparisons(summary["total_carbon_g_co2e"])

    return {
        "collection_id": collection_id,
        "summary": summary,
        "per_document": per_document,
        "comparisons": comparisons,
    }


# ── Internal helpers ────────────────────────────────────────────────────────


def _aggregate_docs(docs: list[dict]) -> dict:
    """Sum up metrics across a list of metric documents."""
    return {
        "total_energy_kwh": round(sum(d.get("energy_kwh", 0) for d in docs), 8),
        "total_carbon_g_co2e": round(sum(d.get("carbon_g_co2e", 0) for d in docs), 4),
        "total_water_ml": round(sum(d.get("water_ml", 0) for d in docs), 4),
        "total_bedrock_input_tokens": sum(
            d.get("bedrock_input_tokens", 0) for d in docs
        ),
        "total_bedrock_output_tokens": sum(
            d.get("bedrock_output_tokens", 0) for d in docs
        ),
        "total_bedrock_invocations": sum(d.get("bedrock_invocations", 0) for d in docs),
        "total_textract_api_calls": sum(d.get("textract_api_calls", 0) for d in docs),
        "total_processing_duration_ms": sum(
            d.get("processing_duration_ms", 0) for d in docs
        ),
        "total_input_bytes": sum(d.get("input_file_size_bytes", 0) for d in docs),
        "total_output_bytes": sum(d.get("output_file_size_bytes", 0) for d in docs),
        "document_count": len(set(d.get("document_key", "") for d in docs)),
        "metrics_count": len(docs),
    }


def _build_per_document_list(docs: list[dict]) -> list[dict]:
    """Build a per-document summary from raw metrics documents."""
    return [
        {
            "document_key": d.get("document_key", ""),
            "task_type": d.get("task_type", ""),
            "energy_kwh": d.get("energy_kwh", 0),
            "carbon_g_co2e": d.get("carbon_g_co2e", 0),
            "water_ml": d.get("water_ml", 0),
            "processing_duration_ms": d.get("processing_duration_ms", 0),
            "bedrock_input_tokens": d.get("bedrock_input_tokens", 0),
            "bedrock_output_tokens": d.get("bedrock_output_tokens", 0),
            "textract_api_calls": d.get("textract_api_calls", 0),
            "compute_type": d.get("compute_type", ""),
        }
        for d in docs
    ]


def _build_comparisons(total_carbon_g: float) -> dict:
    """Convert total carbon emissions to human-readable equivalences."""
    if total_carbon_g <= 0:
        return {
            "km_driven": 0,
            "hours_led_bulb": 0,
            "smartphone_charges": 0,
            "google_searches": 0,
        }

    return {
        "km_driven": round(total_carbon_g / G_CO2E_PER_KM_DRIVEN, 4),
        "hours_led_bulb": round(total_carbon_g / G_CO2E_PER_HOUR_LED, 2),
        "smartphone_charges": round(total_carbon_g / G_CO2E_PER_SMARTPHONE_CHARGE, 2),
        "google_searches": round(total_carbon_g / G_CO2E_PER_GOOGLE_SEARCH, 1),
    }
