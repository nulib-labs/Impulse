"""Document extraction using Marker PDF."""

from __future__ import annotations

import io
import json
from itertools import batched

from loguru import logger

from impulse.config import MARKER_BATCH_SIZE


def extract_documents(items: list[dict]) -> list[dict]:
    """Run Marker PDF extraction on a list of document items.

    Each *item* must have:
      - ``contents`` (bytes): raw file bytes
      - ``filename`` (str)
      - ``impulse_identifier`` (str)

    Returns a list of dicts with Marker output plus filename/identifier.
    """
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config = {"output_format": "json"}
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    results: list[dict] = []

    for batch in batched(items, MARKER_BATCH_SIZE):
        for item in batch:
            file_input = io.BytesIO(item["contents"])
            rendered = converter(file_input)

            # Serialise via Pydantic to avoid unhashable-type bugs
            rendered_dict = json.loads(rendered.model_dump_json())
            rendered_dict["filename"] = item["filename"]
            rendered_dict["impulse_identifier"] = item["impulse_identifier"]
            results.append(rendered_dict)

    logger.info(f"Extracted {len(results)} document(s) via Marker PDF")
    return results


def save_extraction_results(results: list[dict], collection) -> None:
    """Upsert extraction results into MongoDB."""
    for i, page in enumerate(results):
        page["document_extraction_model"] = "marker"
        collection.update_one(
            {
                "page_number": i + 1,
                "impulse_identifier": page["impulse_identifier"],
            },
            {"$set": page},
            upsert=True,
        )
    logger.success(f"Saved {len(results)} extraction results to MongoDB")
