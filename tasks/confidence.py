"""
Custom marker subclasses that capture OCR confidence scores from surya
before marker discards them during span/char conversion.

Surya's RecognitionPredictor produces per-character and per-line confidence
scores (softmax max probabilities), but marker's OcrBuilder.spans_from_html_chars()
drops them when converting surya TextChar -> marker Char objects.

This module provides:
  - ConfidenceCapturingOcrBuilder: captures confidences from the main OCR pass
  - ConfidenceCapturingTableProcessor: captures confidences from table cell OCR
  - ConfidencePdfConverter: wires everything together and exposes confidence data
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ftfy import fix_text
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, OCRResult, TextLine as SuryaTextLine
from surya.table_rec import TableRecPredictor

from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters.pdf import PdfConverter
from marker.processors.table import TableProcessor
from marker.providers.registry import provider_from_filepath
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.text.line import Line


def _extract_line_confidence(text_line: SuryaTextLine) -> Dict[str, Any]:
    """Extract confidence data from a surya TextLine into a plain dict."""
    return {
        "text": text_line.text,
        "confidence": float(text_line.confidence) if text_line.confidence else 0.0,
        "chars": [
            {
                "text": c.text,
                "confidence": float(c.confidence) if c.confidence else 0.0,
            }
            for c in text_line.chars
        ],
    }


class ConfidenceCapturingOcrBuilder(OcrBuilder):
    """OcrBuilder subclass that intercepts surya recognition results to
    capture per-line and per-character confidence scores.

    The recognition model is called exactly once (no double-inference).
    After capturing, the rest of the span-creation logic is identical
    to the parent class.
    """

    def __init__(self, recognition_model: RecognitionPredictor, config=None):
        super().__init__(recognition_model, config)
        self.captured_confidences: Dict[int, List[Dict[str, Any]]] = {}

    def ocr_extraction(
        self,
        document: Document,
        pages,
        images,
        block_polygons,
        block_ids,
        block_original_texts,
    ):
        if sum(len(b) for b in block_polygons) == 0:
            return

        # --- Run recognition model (single call) ---
        self.recognition_model.disable_tqdm = self.disable_tqdm
        recognition_results: List[OCRResult] = self.recognition_model(
            images=images,
            task_names=[self.ocr_task_name] * len(images),
            polygons=block_polygons,
            input_text=block_original_texts,
            recognition_batch_size=int(self.get_recognition_batch_size()),
            sort_lines=False,
            math_mode=not self.disable_ocr_math,
            drop_repeated_text=self.drop_repeated_text,
            max_sliding_window=2148,
            max_tokens=2048,
        )

        # --- Capture confidence data before marker discards it ---
        for page, page_result in zip(pages, recognition_results):
            page_lines = []
            for text_line in page_result.text_lines:
                page_lines.append(_extract_line_confidence(text_line))
            self.captured_confidences[page.page_id] = page_lines

        # --- Span creation (replicated from parent to avoid double-inference) ---
        assert (
            len(recognition_results) == len(images) == len(pages) == len(block_ids)
        ), (
            f"Mismatch in OCR lengths: {len(recognition_results)}, {len(images)}, {len(pages)}, {len(block_ids)}"
        )
        for document_page, page_recognition_result, page_block_ids, image in zip(
            pages, recognition_results, block_ids, images
        ):
            for block_id, block_ocr_result in zip(
                page_block_ids, page_recognition_result.text_lines
            ):
                if block_ocr_result.original_text_good:
                    continue
                if not fix_text(block_ocr_result.text):
                    continue

                block = document_page.get_block(block_id)
                all_line_spans = self.spans_from_html_chars(
                    block_ocr_result.chars, document_page, image
                )
                if block.block_type == BlockTypes.Line:
                    flat_spans = [
                        s for line_spans in all_line_spans for s in line_spans
                    ]
                    self.replace_line_spans(document, document_page, block, flat_spans)
                else:
                    for line in block.contained_blocks(
                        document_page, block_types=[BlockTypes.Line]
                    ):
                        line.removed = True
                    block.structure = []

                    for line_spans in all_line_spans:
                        new_line = Line(
                            polygon=block.polygon,
                            page_id=block.page_id,
                            text_extraction_method="surya",
                        )
                        document_page.add_full_block(new_line)
                        block.add_structure(new_line)
                        self.replace_line_spans(
                            document, document_page, new_line, line_spans
                        )


class ConfidenceCapturingTableProcessor(TableProcessor):
    """TableProcessor subclass that captures OCR confidence from table cell recognition.

    Table cells that need OCR go through a separate recognition pass in
    TableProcessor.get_ocr_results(). This subclass intercepts those results
    to capture per-cell line/char confidence.
    """

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        table_rec_model: TableRecPredictor,
        detection_model: DetectionPredictor,
        config=None,
    ):
        super().__init__(recognition_model, table_rec_model, detection_model, config)
        self.captured_confidences: Dict[int, List[Dict[str, Any]]] = {}

    def get_ocr_results(self, table_images, ocr_polys):
        """Override to capture confidence from table cell OCR before parent discards it."""
        # Let the parent do all the work (filtering, recognition, re-alignment)
        ocr_results = super().get_ocr_results(table_images, ocr_polys)

        # Capture confidence from the (possibly re-aligned) results
        for table_idx, table_ocr_result in enumerate(ocr_results):
            table_lines = []
            for text_line in table_ocr_result.text_lines:
                table_lines.append(_extract_line_confidence(text_line))
            # Use negative indices to distinguish table confidences from page confidences
            # Keyed by table_idx; the converter will merge these into the page data
            self.captured_confidences[table_idx] = table_lines

        return ocr_results


class ConfidencePdfConverter(PdfConverter):
    """PdfConverter that captures OCR confidence scores from both the main
    OCR pass and the table processor OCR pass.

    After calling the converter, access confidence data via:
      - converter.ocr_confidences: per-page main OCR confidence
      - converter.table_confidences: per-table cell OCR confidence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ocr_builder: Optional[ConfidenceCapturingOcrBuilder] = None
        self._table_processor: Optional[ConfidenceCapturingTableProcessor] = None

    def build_document(self, filepath: str) -> Document:
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(LayoutBuilder)
        line_builder = self.resolve_dependencies(LineBuilder)

        # Inject our confidence-capturing OCR builder (fresh each call)
        ocr_builder = self.resolve_dependencies(ConfidenceCapturingOcrBuilder)
        self._ocr_builder = ocr_builder

        provider = provider_cls(filepath, self.config)
        document = DocumentBuilder(self.config)(
            provider, layout_builder, line_builder, ocr_builder
        )
        structure_builder = self.resolve_dependencies(StructureBuilder)
        structure_builder(document)

        # Replace the TableProcessor in the processor list with our subclass
        # (only on first call; subsequent calls reuse the same instance)
        for i, processor in enumerate(self.processor_list):
            if isinstance(processor, TableProcessor) and not isinstance(
                processor, ConfidenceCapturingTableProcessor
            ):
                self._table_processor = self.resolve_dependencies(
                    ConfidenceCapturingTableProcessor
                )
                self.processor_list[i] = self._table_processor
                break

        # Clear table confidences from any previous run
        if self._table_processor:
            self._table_processor.captured_confidences = {}

        for processor in self.processor_list:
            processor(document)

        return document

    @property
    def ocr_confidences(self) -> Dict[int, List[Dict[str, Any]]]:
        """Per-page OCR confidence from the main OCR pass.
        Keys are page_ids (int), values are lists of line confidence dicts."""
        if self._ocr_builder:
            return self._ocr_builder.captured_confidences
        return {}

    @property
    def table_confidences(self) -> Dict[int, List[Dict[str, Any]]]:
        """Per-table OCR confidence from the table processor OCR pass.
        Keys are table indices (int), values are lists of line confidence dicts."""
        if self._table_processor:
            return self._table_processor.captured_confidences
        return {}
