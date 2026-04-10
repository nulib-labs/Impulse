import base64
from io import BytesIO
import os
import re

from fireworks.core.firework import FWAction, FireTaskBase
import fitz
from loguru import logger

from tasks.helpers import _get_db, get_s3_content, parse_s3_path

class ExtractMetadata(FireTaskBase):
    _fw_name = "Extract Metadata"

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    @staticmethod
    def pdf_to_images(pdf_bytes: bytes) -> list[str]:
        """Convert each PDF page to a base64-encoded PNG string."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            images.append(img_b64)
        return images

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        """
        Parse S3 path into bucket and key.

        Args:
            s3_path: S3 URI in format s3://bucket/key or s3a://bucket/key

        Returns:
            Tuple of (bucket, key)
        """
        # Remove s3:// or s3a:// prefix
        path = re.sub(r"^s3a?://", "", s3_path)
        # Split into bucket and key
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    @staticmethod
    def extract_valid_json(content: str) -> dict:
        import json

        """Strip non-JSON content and parse the first valid JSON object found."""
        # Remove markdown fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip()

        # Try parsing directly first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Find the first { ... } block
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Return safe default if nothing works
        return {"main_place": None, "key_people": []}

    @staticmethod
    def extract_spacy_entities(text: str) -> tuple[list[str], list[str]]:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        gpes, people = [], []
        for ent in doc.ents:
            if ent.label_ == "GPE":
                gpes.append(ent.text)
            elif ent.label_ == "PERSON":
                people.append(ent.text)
        return list(dict.fromkeys(gpes)), list(dict.fromkeys(people))

    @staticmethod
    def ask_ai(
        gpes: list[str],
        people: list[str],
        document_text=None,
        pdf_bytes=None,
        image_bytes=None,
    ):
        from ollama import chat

        prompt = f"""
You are a metadata extraction assistant. SpaCy has already performed
named-entity recognition on the document and surfaced these candidates:

  Candidate places : {gpes if gpes else "none detected"}
  Candidate people : {people if people else "none detected"}

Using these candidates (and the document text below, if provided), identify:

1. The SINGLE most important or most-frequently-mentioned location.
   - Prefer the place most central to the document's subject matter.
   - Resolve ambiguous names to their most specific form
     (e.g. "Washington" → "Washington, D.C." or "Washington State").
   - If all candidates look spurious, attempt to extract the correct
     place yourself; otherwise return null.

2. Up to 6 KEY PEOPLE mentioned in the document.
   - Prefer people who are subjects of the document over passing references.
   - If all candidates look spurious, attempt to extract them yourself;
     otherwise return an empty list.

Return ONLY valid JSON — no prose, no markdown fences — in exactly this shape:
{{
  "main_place": "<place or null>",
  "key_people": ["<name>", "..."]
}}
"""
        message = {"role": "user", "content": prompt}

        if pdf_bytes:
            message["images"] = ExtractMetadata.pdf_to_images(pdf_bytes)
        elif image_bytes:
            message["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

        if document_text:
            message["content"] += f"\n\nDocument text:\n{document_text}"
        response = chat(model="gemma3:27b", messages=[message], stream=False)
        print(response["message"]["content"])
        # or access fields directly from the response object
        return response.message.content

    def _load_bytes(self, document: str) -> bytes:
        if document.startswith(("s3://", "s3a://")):
            logger.info("Document starts with s3, pulling from bucket")
            return get_s3_content(document)
        with open(document, "rb") as f:
            return f.read()

    def save_to_mongo(self, document, collection):
        collection.insert_one(document)
        return True

    def run_task(self, fw_spec: dict):
        document = fw_spec["document"]
        accession_number: str | None = fw_spec.get("accession_number", None)
        if not accession_number:
            raise KeyError(f"Accession number not in spec!")
        # Case 1: PDF
        if isinstance(document, str) and document.lower().endswith(".pdf"):
            pdf_bytes = self._load_bytes(document)
            try:
                import pdfminer.high_level as pdfminer

                plain_text = pdfminer.extract_text(BytesIO(pdf_bytes))
            except Exception:
                plain_text = ""
            gpes, people = (
                self.extract_spacy_entities(plain_text) if plain_text else ([], [])
            )
            metadata = self.ask_ai(gpes=gpes, people=people, pdf_bytes=pdf_bytes)

        # Case 2: Image file
        elif (
            isinstance(document, str)
            and os.path.splitext(document)[1].lower() in self.IMAGE_EXTENSIONS
        ):
            image_bytes = self._load_bytes(document)
            metadata = self.ask_ai(gpes=[], people=[], image_bytes=image_bytes)

        # Case 3: List of S3/file paths (text files)
        elif isinstance(document, list):
            parts = [self.get_s3_content(p).decode("utf-8") for p in document]
            combined = "\n".join(parts)
            gpes, people = self.extract_spacy_entities(combined)
            metadata = self.ask_ai(gpes=gpes, people=people, document_text=combined)
        # Case 4: Raw string text
        elif isinstance(document, str):
            gpes, people = self.extract_spacy_entities(document)
            metadata = self.ask_ai(gpes=gpes, people=people, document_text=document)
        else:
            raise ValueError(f"Unsupported document type: {type(document)}")
        document = self.extract_valid_json(metadata if metadata else "")
        document["accession_number"] = accession_number
        self.save_to_mongo(document=document, collection=_get_db()["metadata"])
        return FWAction(update_spec={"document_metadata": document})


class GetMetadata(FireTaskBase):
    _fw_name = "Get Metadata Task"

class EnrichMetadata(FireTaskBase):
    _fw_name = "Build Enriched Manifest Task"
class SaveManifest(FireTaskBase):
    _fw_name = "Save Manifest To Mongo Task"
class SummaryManifest(FireTaskBase):
    _fw_name = "Summary Task"
class SummaryManifest(FireTaskBase):
    _fw_name = "Context Enrichment Task"
class Task187(FireTaskBase):
    _fw_name = "Aggregate OCR Task"
class Task124(FireTaskBase):
    _fw_name = "Fetch IIIF Collection Task"
class IIIF(FireTaskBase):
    _fw_name = "IIIF OCR Task"
class IIIF(FireTaskBase):
    _fw_name = "Benchmark Networking"
class IIIF(FireTaskBase):
    _fw_name = "Experimental Metadata Task"
