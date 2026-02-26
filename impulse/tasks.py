import io
from typing import NoReturn, override
import re
import certifi
import boto3
from io import BytesIO
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
from fireworks.utilities.filepad import FilePad
import os
from uuid import uuid4
from pymongo import MongoClient
import json
from pathlib import Path
from impulse.auxiliary import convert_mets_to_yml
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
import boto3, json
from openai import OpenAI


client = MongoClient("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
db = client["praxis"]
collection = db["pages"]

fp = FilePad(
    host=str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")),
    port=27017,
    name="fireworks",
    uri_mode=True,
    mongoclient_kwargs={"tls": True, "tlsCAFile": certifi.where()},
)


class BinarizationTask(FireTaskBase):
    _fw_name = "Binarization Task"
    output_path: str | None

    @staticmethod
    def _save_content(output_path, content):
        import cv2

        _ = cv2.imwrite(output_path, content)
        pass

    @staticmethod
    def _binarize(content: bytes) -> bytes:
        import cv2
        import numpy as np

        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        success, encoded = cv2.imencode(".png", binary)
        if not success:
            raise ValueError("Failed to encode binarized image.")

        return encoded.tobytes()

    @staticmethod
    def is_s3_path(path: str) -> bool:
        """
        Check if the path is an S3 URI.
        Supports both s3:// and s3a:// formats.
        """
        return bool(re.match(r"^s3a?://", path))

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

    def get_s3_content(self, s3_path: str) -> bytes:
        """
        Retrieve content from S3.

        Args:
            s3_path: S3 URI

        Returns:
            File content as bytes
        """
        bucket, key = self.parse_s3_path(s3_path)

        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Download file content
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        return buffer.read()

    @override
    def run_task(self, fw_spec: dict[str, str]) -> FWAction:
        """
        This method runs the binarization task.
        This task takes one spec entry: `path` of type `str`
        """
        llm_host = fw_spec["llm_host"]  # required, no default

        path_array_key = fw_spec["find_path_array_in"]
        path_array: str = fw_spec[path_array_key]
        binarized_objects: list[tuple[str, str]]
        binarized_objects = []
        for path in path_array:
            logger.info(f"`path` is {path}")
            if self.is_s3_path(path):
                logger.info("Now loading content from S3")
                content = self.get_s3_content(path)
                binarized = self._binarize(content)
            else:
                with open(path, "rb") as f:
                    content = f.read()
                binarized = self._binarize(content)
            id, identifier = fp.add_contents(
                binarized, identifier=f"impulse:binarized:{path}:{uuid4()}"
            )
            binarized_objects.append((id, identifier))

        # Returns a FW Action that is appending all objects to the spec key `binarized_objects`
        return FWAction(update_spec={"binarized_objects": binarized_objects})


class DocumentExtractionTask(FireTaskBase):
    _fw_name = "Document Extraction Task"

    def filetype(self, contents: bytes) -> str | None:
        """
        Determine file type from raw bytes using magic numbers.

        Args:
            contents: File contents as bytes

        Returns:
            File extension string (e.g. 'png', 'pdf') or None if unknown
        """
        if not contents or len(contents) < 4:
            return None

        # PNG
        if contents.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"

        # JPEG
        if contents.startswith(b"\xff\xd8\xff"):
            return "jpg"

        # GIF
        if contents.startswith((b"GIF87a", b"GIF89a")):
            return "gif"

        # PDF
        if contents.startswith(b"%PDF"):
            return "pdf"

        # ZIP (also used by docx, xlsx, pptx, etc.)
        if contents.startswith(b"PK\x03\x04"):
            return "zip"

        # GZIP
        if contents.startswith(b"\x1f\x8b"):
            return "gz"

        # MP3 (ID3 tag)
        if contents.startswith(b"ID3"):
            return "mp3"

        # MP4
        if len(contents) > 8 and contents[4:8] == b"ftyp":
            return "mp4"

        # JP2 (JPEG 2000)
        if contents.startswith(b"\x00\x00\x00\x0cjP  \r\n\x87\n"):
            return "jp2"

        # Plain text (heuristic)
        try:
            contents.decode("utf-8")
            return "txt"
        except UnicodeDecodeError:
            pass

        return None

    def _predict(self, contents):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        config = {"output_format": "json"}
        config_parser = ConfigParser(config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        if self.filetype(contents) != "pdf":
            contents = self.load_jp2(contents)
        else:
            contents: BytesIO = io.BytesIO(contents)
        rendered = converter(contents)
        return rendered

    @staticmethod
    def get_filepad_contents(gfs_id):
        contents, doc = fp.get_file(gfs_id)
        logger.info(f"Type of contents: {type(contents)}")
        return contents

    @staticmethod
    def is_s3_path(path: str) -> bool:
        """
        Check if the path is an S3 URI.
        Supports both s3:// and s3a:// formats.
        """
        return bool(re.match(r"^s3a?://", path))

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

    def get_s3_content(self, s3_path: str) -> bytes:
        """
        Retrieve content from S3.

        Args:
            s3_path: S3 URI

        Returns:
            File content as bytes
        """
        bucket, key = self.parse_s3_path(s3_path)

        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Download file content
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        return buffer.read()

    @staticmethod
    def is_impulse_identifier(value: str) -> bool:
        """
        Checks if value is impulse identifier.
        """

        if "impulse:" in value:
            return True
        else:
            return False

    @staticmethod
    def get_file_from_filepad(identifier):
        return fp

    @staticmethod
    def load_jp2(contents: bytes):
        import numpy as np
        import cv2
        from PIL import Image
        from io import BytesIO

        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR_RGB)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        return img_byte_arr

    def save_to_mongo(self, model, collection):
        """Save any Pydantic model to MongoDB."""

        for page in model:
            collection.insert_one(page.dict())
        return True

    @override
    def run_task(self, fw_spec: dict[str, list[str]]) -> FWAction:
        """
        This method runs the OCR task.
        This method looks for `path_array`.
        """
        find_path_array_in: list[str] = fw_spec["find_path_array_in"]
        path_array: list[tuple(str, str)] = fw_spec[find_path_array_in]
        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{path_array}")
        for path in path_array:
            if self.is_s3_path(path):
                # Get content from S3
                logger.info("Now loading content from S3")
                content = self.get_s3_content(path)
                predictions = self._predict(content)
                self.save_to_mongo(predictions)
                logger.info(f"Predictions:\n{predictions}")
            elif self.is_impulse_identifier(path[1]):
                logger.info("Detected Impulse identifier")
                content = self.get_filepad_contents(path[1])
                predictions = self._predict(content)
                self.save_to_mongo(model=predictions, collection=collection)
                logger.info(f"Type of predictions:\n{type(predictions)}")
            else:
                # Handle local file path
                with open(path, "rb") as f:
                    content = f.read()
                predictions = self._predict(content)
                logger.info(f"Predictions:\n{predictions}")

        return FWAction()


class TextExtractionTask(FireTaskBase):
    _fw_name = "Text Extraction Task"
    path_array: list[str]

    @staticmethod
    def _extract(content):
        pass

    @staticmethod
    def is_s3_path(path: str) -> bool:
        """
        Check if the path is an S3 URI.
        Supports both s3:// and s3a:// formats.
        """
        return bool(re.match(r"^s3a?://", path))

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

    def get_s3_content(self, s3_path: str) -> bytes:
        """
        Retrieve content from S3.

        Args:
            s3_path: S3 URI

        Returns:
            File content as bytes
        """
        bucket, key = self.parse_s3_path(s3_path)

        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Download file content
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        return buffer.read()

    @override
    def run_task(self, fw_spec: dict[str, str]) -> FWAction:
        return FWAction()


class METSXMLToHathiTrustManifestTask(FireTaskBase):
    _fw_name = "MetsXML To HathiTrust Manifest Task"
    path: str

    @staticmethod
    def is_s3_path(path: str) -> bool:
        """
        Check if the path is an S3 URI.
        Supports both s3:// and s3a:// formats.
        """
        return bool(re.match(r"^s3a?://", path))

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

    def get_s3_content(self, s3_path: str) -> bytes:
        """
        Retrieve content from S3.

        Args:
            s3_path: S3 URI

        Returns:
            File content as bytes
        """
        bucket, key = self.parse_s3_path(s3_path)

        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Download file content
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        return buffer.read()

    def save_to_s3(self, s3_path: str, content: str) -> str:
        """
        Save string content to S3.

        Args:
            s3_path: S3 URI (e.g. s3://bucket/key)
            content: File content as a string
        """
        bucket, key = self.parse_s3_path(s3_path)

        s3_client = boto3.client("s3")

        # Convert string to bytes
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode("utf-8"),  # Encode string as bytes
        )

        return s3_path

    def _convert_mets_to_yml(*args):
        """
        FireWorks task function: Convert a METS XML file from S3 into a YAML file
        following HathiTrust ingest specs, and save the YAML back to S3.
        args[0] = s3_key of the METS XML file in S3
        args[1] = filename
        args[2] = accession_number
        """

        s3_key = args[0]
        filename = args[1]
        accession_number = args[2]

        s3_impulse = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
        )

        output_filename = filename.replace(".xml", ".yaml")
        logger.info(f"Value of output filename: {output_filename}")
        response = s3_impulse.get_object(Bucket="nu-impulse-production", Key=s3_key)
        data = response["Body"].read()

        # === Step 2: Parse the XML directly from memory ===
        try:
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.ElementTree(ET.fromstring(data, parser))
            root = tree.getroot()
        except ET.XMLSyntaxError as e:
            print(f"Error: Invalid XML file: {e}")
            raise RuntimeError("Invalid METS XML file") from e

        ns = {
            "xmlns": "http://www.loc.gov/METS/",
            "xlink": "http://www.w3.org/1999/xlink",
        }

        def find_filename_by_file_id(file_id):
            node = root.xpath(
                f"//xmlns:file[@ID='{file_id}']/xmlns:FLocat", namespaces=ns
            )
            if not node:
                return None
            href = node[0].get("{http://www.w3.org/1999/xlink}href", "")
            return href[7:] if href.startswith("file://") else href

        # === Step 3: Extract header information ===
        mets_hdr = root.xpath("//xmlns:metsHdr", namespaces=ns)
        capture_date = mets_hdr[0].get("CREATEDATE") + "-06:00" if mets_hdr else None

        suprascan = False
        scanning_order_rtl = False
        reading_order_rtl = False
        resolution = 400

        yaml_lines = []
        yaml_lines.append(f"capture_date: {capture_date}")
        if suprascan:
            yaml_lines.append("scanner_make: SupraScan")
            yaml_lines.append("scanner_model: Quartz A1")
        else:
            yaml_lines.append("scanner_make: Kirtas")
            yaml_lines.append("scanner_model: APT 1200")
        yaml_lines.append(
            'scanner_user: "Northwestern University Library: Repository & Digital Curation"'
        )
        yaml_lines.append(f"contone_resolution_dpi: {resolution}")
        yaml_lines.append(f"image_compression_date: {capture_date}")
        yaml_lines.append("image_compression_agent: northwestern")
        yaml_lines.append('image_compression_tool: ["LIMB v4.5.0.0"]')
        yaml_lines.append(
            f"scanning_order: {'right-to-left' if scanning_order_rtl else 'left-to-right'}"
        )
        yaml_lines.append(
            f"reading_order: {'right-to-left' if reading_order_rtl else 'left-to-right'}"
        )
        yaml_lines.append("pagedata:")

        # === Step 4: Logical structMap page iteration ===
        logical_pages = root.xpath(
            '//xmlns:structMap[@TYPE="logical"]//xmlns:div[@TYPE="page"]', namespaces=ns
        )

        for element in logical_pages:
            fileptr = element.xpath(
                "./xmlns:fptr[starts-with(@FILEID, 'JP2')]", namespaces=ns
            )
            if not fileptr:
                continue
            file_id = fileptr[0].get("FILEID")
            page_filename = find_filename_by_file_id(file_id)
            if not page_filename:
                continue

            parent = element.getparent()
            parent_label = parent.get("LABEL", "")
            parent_type = parent.get("TYPE", "")
            orderlabel = element.get("ORDERLABEL", "")
            line = None

            # Label logic
            if element == parent[0]:
                if (
                    parent_label == "Cover"
                    and parent_type == "cover"
                    and parent == logical_pages[0].getparent()
                ):
                    label = "FRONT_COVER"
                    line = f'{page_filename}: {{ label: "{label}" }}'
                elif parent_label == "Front Matter" and orderlabel:
                    line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'
                elif parent_label == "Title":
                    label = "TITLE"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Contents":
                    label = "TABLE_OF_CONTENTS"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Preface":
                    label = "PREFACE"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label.startswith("Chapter") or parent_label == "Appendix":
                    label = "CHAPTER_START"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label in ("Notes", "Bibliography"):
                    label = "REFERENCES"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Index":
                    label = "INDEX"
                    line = (
                        f'{page_filename}: {{ orderlabel: "{orderlabel}", label: "{
                            label
                        }" }}'
                        if orderlabel
                        else f'{page_filename}: {{ label: "{label}" }}'
                    )
                elif parent_label == "Cover" and parent_type == "cover":
                    label = "BACK_COVER"
                    line = f'{page_filename}: {{ label: "{label}" }}'
            else:
                if orderlabel:
                    line = f'{page_filename}: {{ orderlabel: "{orderlabel}" }}'

            if line:
                yaml_lines.append("    " + line)

        # === Step 5: Write YAML to S3 ===
        yaml_content: str = "\n".join(yaml_lines)
        return yaml_content

    @override
    def run_task(self, fw_spec: dict[str, str]) -> FWAction:
        logger.info("Now loading content from S3")
        input_path = fw_spec["input_path"]
        output_path = fw_spec["output_path"]
        content = self.get_s3_content(input_path)

        yaml_content: str = self.convert_mets_to_yml(content)
        s3_path = self.save_to_s3(output_path, yaml_content)

        return FWAction(update_spec={"hathitrust_yaml_path": s3_path})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_URL = "https://nu-impulse-production.s3.us-east-1.amazonaws.com/"
MAX_WORDS = 980000




# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class GetMetadata(FireTaskBase):
    _fw_name = "Get Metadata"

    @staticmethod
    def s3_read_json(s3_path: str) -> dict:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    @staticmethod
    def s3_write_json(s3_path: str, data: dict) -> None:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def ask_ai_func(self, gpes, people, text=""):

        prompt = f"""
    Document text (first 2000 words):
    {" ".join(text.split()[:2000])}

    Places mentioned by NER: {gpes}
    People mentioned by NER: {people}

Task: Identify:
1) The main location of the document (the most important place, most frequently
   mentioned, or the location where the main project occurs)
2) 5-7 key people mentioned

Please ONLY return one location as the main place, and a list of up to 6 key
people. If the most important place is not very specific, please augment it to
make it more specific (e.g. "washington" to "Washington DC" or
"Washington State"). If the listed places and names are bogus, try to extract
them on your own, or put None. Do NOT include any other information or
explanations. Return ONLY this exact format:
{{
  "main_place": "...",
  "key_people": ["...", "..."]
}}
        """

        from ollama import chat

        response = chat(
            model="gemma3:270m",
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.message.content
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"main_place": None, "key_people": None}

    def run_task(self, fw_spec):
        import spacy

        debug = fw_spec.get("debug", False)

        nlp = spacy.load("en_core_web_sm")

        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]

        # load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = self.s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            doc = nlp(text)

            gpes = []
            people = []

            for ent in doc.ents:
                if ent.label_ == "GPE":
                    gpes.append(ent.text)
                elif ent.label_ == "PERSON":
                    people.append(ent.text)

            gpes = list(dict.fromkeys(gpes))
            people = list(dict.fromkeys(people))

            # call ai
            ai_result = self.ask_ai_func(gpes=gpes, people=people, text=text)

            results[doc_id] = {
                "doc_id":     doc_id,
                "main_place": ai_result.get("main_place"),
                "key_people": ai_result.get("key_people"),
            }

        # Save
        if debug:
            local_path = Path.cwd() / "overall_metadata.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved metadata to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            self.s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})



# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
class Summaries(FireTaskBase):
    _fw_name = "Summaries"
    path: str

    @staticmethod
    def s3_read_json(s3_path: str) -> dict:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    @staticmethod
    def s3_write_json(s3_path: str, data: dict) -> None:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def ask_ai_func_summaries(self, raw_text="", host=None):
        max_chars = 24000  # 6k tokens (ok for gemma 27b, maybe don't use a smaller model)
        was_truncated = False

        if len(raw_text) > max_chars:
            # Trying to take beginning and end, probably ok for prototype, maybe use chunked map reduce later
            chunk = raw_text[:max_chars // 2] + "\n...[middle truncated]...\n" + raw_text[-(max_chars // 2):]
            was_truncated = True
        else:
            chunk = raw_text

        prompt = f"""
Document text:
{" ".join(chunk.split())}

Task: SUMMARY
Write a detailed summary of the document in approximately 125 words (minimum 70 words, maximum 150 words). The summary should explain:
 - What project, proposal, or decision the document addresses, where is it, which agencies are involved, what is the goal.
 - Only include information supported by the document text. - Do not introduce outside knowledge
 - return only a dictionary in this exact format: 
    "summary": "your text here" and don't ask any follow up questions. If the document is too vague to summarize, return "summary": null.
"""

        response = self.call_llm(prompt, host=host)

        # Strip accidental markdown fences before parsing
        response = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": None}

    def call_llm(self, prompt: str, host: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url=host)

        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )

        return response.choices[0].message.content.strip()

    def run_task(self, fw_spec):

        debug = fw_spec.get("debug", False)

        import socket
        llm_host = f"http://{socket.gethostname()}:8000/v1"

        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]

        # 1. Load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = self.s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            summary = self.ask_ai_func_summaries(raw_text=text, host=llm_host)

            results[doc_id] = {
                "doc_id":  doc_id,
                "summary": summary.get("summary"),  #THIS MEANS RESPONSE MUST HAVE "summary"
            }

        # 4. Save
        if debug:
            local_path = Path.cwd() / "overall_summaries.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved metadata to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            self.s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})




# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

class Themes(FireTaskBase):
    _fw_name = "Themes"
    path: str

    @staticmethod
    def s3_read_json(s3_path: str) -> dict:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    @staticmethod
    def s3_write_json(s3_path: str, data: dict) -> None:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def ask_ai_func_themes(self, raw_text="", host=None):
        max_chars = 24000  # 6k tokens (ok for gemma 27b, maybe don't use a smaller model)
        was_truncated = False

        if len(raw_text) > max_chars:
            # Trying to take beginning and end, probably ok for prototype, maybe use chunked map reduce later
            chunk = raw_text[:max_chars // 2] + "\n...[middle truncated]...\n" + raw_text[-(max_chars // 2):]
            was_truncated = True
        else:
            chunk = raw_text

        prompt = f"""
Document text:
{" ".join(chunk.split())}

Task: Assign 1-3 themes to this document from the list below. Only use themes from this list, spelled exactly as written.

Themes:
- Transportation Infrastructure
- Energy Systems
- Wildlife and Natural Areas
- Water Systems
- Urban Development
- Industrial Production and Materials
- Climate and Weather Modification
- Governance and Institutional Control
- Place Based Development Conflicts
- Indigenous Narratives and Sovereignty

Respond ONLY with valid JSON, no other text:
{{"themes": ["theme1", "theme2"]}}

If no themes apply or the document is too vague, return {{"themes": []}}
"""
        response = self.call_llm(prompt, host=host)

        # Strip accidental markdown fences before parsing
        response = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"themes": []}

    def call_llm(self, prompt: str, host: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url=host)

        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )

        return response.choices[0].message.content.strip()

    def run_task(self, fw_spec):

        debug = fw_spec.get("debug", False)

        import socket
        llm_host = f"http://{socket.gethostname()}:8000/v1"

        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]

        # 1. Load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = self.s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            summary = self.ask_ai_func_themes(raw_text=text, host=llm_host)

            results[doc_id] = {
                "doc_id":  doc_id,
                "themes": summary.get("themes", []),  #THIS MEANS RESPONSE MUST HAVE "themes"
            }

        # 4. Save
        if debug:
            local_path = Path.cwd() / "overall_themes.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved metadata to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            self.s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})




# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

class Quotes(FireTaskBase):
    _fw_name = "Quotes"
    path: str

    @staticmethod
    def extract_comment_section(raw_text: str) -> tuple[str, bool]:
        """Try to find where public comments start, fall back to last N chars."""
        markers = [
            "public comment", "public hearing", "community comment",
            "citizen comment", "oral comment", "written comment",
            "comment period", "testimony", "public testimony"
        ]
        lower = raw_text.lower()
        best_pos = -1
        for marker in markers:
            pos = lower.rfind(marker)  # rfind gets the LAST occurrence
            if pos != -1 and pos > best_pos:
                best_pos = pos
        if best_pos != -1:
            return raw_text[best_pos:], True
        else:
            return raw_text[-12000:], False

    @staticmethod
    def s3_read_json(s3_path: str) -> dict:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    @staticmethod
    def s3_write_json(s3_path: str, data: dict) -> None:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def ask_ai_func_quotes(self, raw_text="", host=None):
        section, found = self.extract_comment_section(raw_text)

        prompt = f"""
Document text:
{" ".join(section.split())}

Task: QUOTES
Locate a section explicitly labeled “Public Comment,” “Comments Received,” “Response to Comments,” or similar. If there is no clear section, please look for messages from individuals or organizations expressing support, opposition, or concerns about the project, proposal, or decision in the document, and use that as the section for this task.
- Identify text explicitly attributed to commenters. 
- Select 2–4 verbatim excerpts that reflect commonly repeated concerns or positions. Please try to select representative but interesting/evocative excerpts. Strict requirements: 
- Copy text exactly as written.
- Do not paraphrase. 
- Do not summarize. 
- Do not explain. 
- Output only the excerpts AND (if possible) the name of the commenter or organization, in this exact format "'quote text' - commenter/organization". If the commenter or organization is not named, just return the quote text in single quotes.
- You can clean the text, removing things like formatting issues, but do not change the wording.
- If the document contains no public comments or similar sections, or if the comments are too vague to extract meaningful excerpts, return an empty list.
---------------------------------------- 
FORMAT YOUR RESPONSE EXACTLY AS: 
PUBLIC_COMMENT: ["'quote 1' - commenter1", "'quote 2' - commenter2"]
"""
        response = self.call_llm(prompt, host=host)

        # Strip accidental markdown fences before parsing
        response = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"PUBLIC_COMMENT": []}

    def call_llm(self, prompt: str, host: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url=host)

        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()

    def run_task(self, fw_spec):

        debug = fw_spec.get("debug", False)

        import socket
        llm_host = f"http://{socket.gethostname()}:8000/v1"

        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]

        # Load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = self.s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            result = self.ask_ai_func_quotes(raw_text=text, host=llm_host)

            results[doc_id] = {
                "doc_id": doc_id,
                "PUBLIC_COMMENT": result.get("PUBLIC_COMMENT", []),  # RESPONSE MUST HAVE "PUBLIC_COMMENT"
            }

        # Save
        if debug:
            local_path = Path.cwd() / "overall_quotes.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved quotes to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            self.s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})





# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class Context(FireTaskBase):
    _fw_name = "Context"
    path: str

    @staticmethod
    def s3_read_json(s3_path: str) -> dict:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    @staticmethod
    def s3_write_json(s3_path: str, data: dict) -> None:
        match = re.match(r"^s3a?://([^/]+)/(.+)$", s3_path)
        bucket, key = match.group(1), match.group(2)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))

    def ask_ai_func_context(self, raw_text="", host=None):
        max_chars = 24000  # 6k tokens (ok for gemma 27b, maybe don't use a smaller model)
        was_truncated = False

        if len(raw_text) > max_chars:
            # Trying to take beginning and end, probably ok for prototype, maybe use chunked map reduce later
            chunk = raw_text[:max_chars // 2] + "\n...[middle truncated]...\n" + raw_text[-(max_chars // 2):]
            was_truncated = True
        else:
            chunk = raw_text

        prompt = f"""
Document text:
{" ".join(chunk.split())}

Task: Context
Please give a short explanation of context and outcome of this event. You can use outside knowledge for this question but ONLY IF YOU ARE CONFIDENT and could give your source when asked.
Explain if the project was excecuted, if it was cancelled, if it is still in progress, or if the outcome is unclear, and why.
If the document is too vague to determine an outcome, return null.

Please also include a 1 or 0 if the project happened or didn't happen. If you're unsure, return null.

Only return a dictionary in one of these exact formats:
{"context": "your summary here", "outcome": 1}
{"context": "your summary here", "outcome": 0}
{"context": "your summary here", "outcome": null}

"""
        response = self.call_llm(prompt, host=host)

        # Strip accidental markdown fences before parsing
        response = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"context": "", "outcome": None}

    def call_llm(self, prompt: str, host: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url=host)

        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()

    def run_task(self, fw_spec):

        debug = fw_spec.get("debug", False)

        import socket
        llm_host = f"http://{socket.gethostname()}:8000/v1"

        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]

        # Load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = self.s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            result = self.ask_ai_func_context(raw_text=text, host=llm_host)

            results[doc_id] = {
                "doc_id": doc_id,
                "context": result.get("context", ""),
                "outcome": result.get("outcome", None),
            }

        # Save
        if debug:
            local_path = Path.cwd() / "overall_context.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved context to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            self.s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})






# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------




class ThemesTask(FireTaskBase):
    _fw_name = "Themes Task"

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        path = re.sub(r"^s3a?://", "", s3_path)
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def get_s3_content(self, s3_path: str) -> bytes:
        bucket, key = self.parse_s3_path(s3_path)
        s3_client = boto3.client("s3")
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        return buffer.read()

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
    def image_to_b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def ask_ai(document_text=None, pdf_bytes=None, image_bytes=None):
        from ollama import chat
        prompt = """
        Task: Assign 1-3 themes to this document from the list below. Only use themes from this list, spelled exactly as written.
            Themes:
            - Transportation Infrastructure
            - Energy Systems
            - Wildlife and Natural Areas
            - Water Systems
            - Urban Development
            - Industrial Production and Materials
            - Climate and Weather Modification
            - Governance and Institutional Control
            - Place Based Development Conflicts
            - Indigenous Narratives and Sovereignty

            Respond ONLY with a list no other text:
            ["theme1", "theme2"]

            If no themes apply or the document is too vague, return []
        """
        message = {
            "role": "user",
            "content": prompt,
        }

        if pdf_bytes:
            # Convert PDF pages to images — gemma3 can't read raw PDF bytes
            message["images"] = ThemesTask.pdf_to_images(pdf_bytes)

        elif image_bytes:
            message["images"] = [ThemesTask.image_to_b64(image_bytes)]

        if document_text:
            message["content"] += f"\n\n{document_text}"

        response = chat(
            model="gemma3:4b",
            messages=[message],
        )
        return response.message.content

    def run_task(self, fw_spec):
        document = fw_spec["document"]

        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

        # Case 1: S3 or local file path
        if isinstance(document, str) and (
            document.startswith("s3://") or document.startswith("s3a://") or "/" in document
        ):
            ext = os.path.splitext(document)[1].lower()

            if document.startswith(("s3://", "s3a://")):
                print("Document starts with s3, pulling from bucket")
                content_bytes = self.get_s3_content(document)
            else:
                with open(document, "rb") as f:
                    content_bytes = f.read()

            if ext == ".pdf":
                themes = self.ask_ai(pdf_bytes=content_bytes)
            elif ext in IMAGE_EXTENSIONS:
                themes = self.ask_ai(image_bytes=content_bytes)
            else:
                # Treat as text file
                themes = self.ask_ai(document_text=content_bytes.decode("utf-8"))

        # Case 2: List of S3/file paths
        elif isinstance(document, list):
            document_text = []
            for path in document:
                content = self.get_s3_content(path)
                document_text.append(content.decode("utf-8"))
            themes = self.ask_ai(document_text="\n".join(document_text))

        # Case 3: Raw string text
        elif isinstance(document, str):
            themes = self.ask_ai(document_text=document)

        else:
            raise ValueError(f"Unsupported document type: {type(document)}")

        print(themes)
        try:
            themes = json.loads(themes)
            if not isinstance(themes, list):
                themes = []
        except (json.JSONDecodeError, TypeError):
            themes = []
        return FWAction(update_spec={"document_themes": themes})




# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

    

class QuotesTask(FireTaskBase):
    _fw_name = "Quotes Task"

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        path = re.sub(r"^s3a?://", "", s3_path)
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def get_s3_content(self, s3_path: str) -> bytes:
        bucket, key = self.parse_s3_path(s3_path)
        s3_client = boto3.client("s3")
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        return buffer.read()

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
    def image_to_b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def ask_ai(document_text=None, pdf_bytes=None, image_bytes=None):
        from ollama import chat
        prompt = """
    
    Task: QUOTES
    Locate a section explicitly labeled “Public Comment,” “Comments Received,” “Response to Comments,” or similar. If there is no clear section, please look for messages from individuals or organizations expressing support, opposition, or concerns about the project, proposal, or decision in the document, and use that as the section for this task.
    - Identify text explicitly attributed to commenters. 
    - Select 2–4 verbatim excerpts that reflect commonly repeated concerns or positions. Please try to select representative but interesting/evocative excerpts. Requirements: 
    - Copy text exactly as written.
    - Do not paraphrase. 
    - Do not summarize. 
    - Do not explain. 
    - Output only the excerpts AND (if possible) the name of the commenter or organization, in this exact format "'quote text' - commenter/organization". If the commenter or organization is not named, just return the quote text in single quotes.
    - You can clean the text, removing things like formatting issues, but do not change the wording.
    - If the document contains no public comments or similar sections, or if the comments are too vague to extract meaningful excerpts, return an empty list.
    ---------------------------------------- 
    FORMAT YOUR RESPONSE EXACTLY AS: 
    PUBLIC_COMMENT: ["'quote 1' - commenter1", "'quote 2' - commenter2"]

        """
        message = {
            "role": "user",
            "content": prompt,
        }

        if pdf_bytes:
            # Convert PDF pages to images — gemma3 can't read raw PDF bytes
            message["images"] = QuotesTask.pdf_to_images(pdf_bytes)

        elif image_bytes:
            message["images"] = [QuotesTask.image_to_b64(image_bytes)]

        if document_text:
            message["content"] += f"\n\n{document_text}"

        response = chat(
            model="gemma3:4b",
            messages=[message],
        )
        return response.message.content

    def run_task(self, fw_spec):
        document = fw_spec["document"]

        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

        # Case 1: S3 or local file path
        if isinstance(document, str) and (
            document.startswith("s3://") or document.startswith("s3a://") or "/" in document
        ):
            ext = os.path.splitext(document)[1].lower()

            if document.startswith(("s3://", "s3a://")):
                print("Document starts with s3, pulling from bucket")
                content_bytes = self.get_s3_content(document)
            else:
                with open(document, "rb") as f:
                    content_bytes = f.read()

            if ext == ".pdf":
                quotes = self.ask_ai(pdf_bytes=content_bytes)
            elif ext in IMAGE_EXTENSIONS:
                quotes = self.ask_ai(image_bytes=content_bytes)
            else:
                # Treat as text file
                quotes = self.ask_ai(document_text=content_bytes.decode("utf-8"))

        # Case 2: List of S3/file paths
        elif isinstance(document, list):
            document_text = []
            for path in document:
                content = self.get_s3_content(path)
                document_text.append(content.decode("utf-8"))
            quotes = self.ask_ai(document_text="\n".join(document_text))

        # Case 3: Raw string text
        elif isinstance(document, str):
            quotes = self.ask_ai(document_text=document)

        else:
            raise ValueError(f"Unsupported document type: {type(document)}")

        print(quotes)
        return FWAction(update_spec={"document_quotes": quotes})



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
    

class ContextTask(FireTaskBase):
    _fw_name = "Context Task"

    @staticmethod
    def parse_s3_path(s3_path: str) -> tuple[str, str]:
        path = re.sub(r"^s3a?://", "", s3_path)
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def get_s3_content(self, s3_path: str) -> bytes:
        bucket, key = self.parse_s3_path(s3_path)
        s3_client = boto3.client("s3")
        buffer = BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        return buffer.read()

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
    def image_to_b64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def ask_ai(document_text=None, pdf_bytes=None, image_bytes=None):
        from ollama import chat
        prompt = """
        
Task: Context
Please give a short explanation of context and outcome of this event. You can use outside knowledge for this question but ONLY IF YOU ARE CONFIDENT and could give your source when asked.
Explain if the project was excecuted, if it was cancelled, if it is still in progress, or if the outcome is unclear, and why.
If the document is too vague to determine an outcome, return null.

Please also include a 1 or 0 if the project happened or didn't happen. If you're unsure, return null.

Only return a dictionary in one of these exact formats:
{"context": "your summary here", "outcome": 1}
{"context": "your summary here", "outcome": 0}
{"context": "your summary here", "outcome": null}

        """
        message = {
            "role": "user",
            "content": prompt,
        }

        if pdf_bytes:
            # Convert PDF pages to images — gemma3 can't read raw PDF bytes
            message["images"] = ContextTask.pdf_to_images(pdf_bytes)

        elif image_bytes:
            message["images"] = [ContextTask.image_to_b64(image_bytes)]

        if document_text:
            message["content"] += f"\n\n{document_text}"

        response = chat(
            model="gemma3:4b",
            messages=[message],
        )
        return response.message.content

    def run_task(self, fw_spec):
        document = fw_spec["document"]

        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

        # Case 1: S3 or local file path
        if isinstance(document, str) and (
            document.startswith("s3://") or document.startswith("s3a://") or "/" in document
        ):
            ext = os.path.splitext(document)[1].lower()

            if document.startswith(("s3://", "s3a://")):
                print("Document starts with s3, pulling from bucket")
                content_bytes = self.get_s3_content(document)
            else:
                with open(document, "rb") as f:
                    content_bytes = f.read()

            if ext == ".pdf":
                context = self.ask_ai(pdf_bytes=content_bytes)
            elif ext in IMAGE_EXTENSIONS:
                context = self.ask_ai(image_bytes=content_bytes)
            else:
                # Treat as text file
                context = self.ask_ai(document_text=content_bytes.decode("utf-8"))

        # Case 2: List of S3/file paths
        elif isinstance(document, list):
            document_text = []
            for path in document:
                content = self.get_s3_content(path)
                document_text.append(content.decode("utf-8"))
            context = self.ask_ai(document_text="\n".join(document_text))

        # Case 3: Raw string text
        elif isinstance(document, str):
            context = self.ask_ai(document_text=document)

        else:
            raise ValueError(f"Unsupported document type: {type(document)}")

        print(context)
        return FWAction(update_spec={"document_context": context})