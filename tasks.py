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
# Shared helpers
# ---------------------------------------------------------------------------

skipped_txt = 0  # simple global counter; fine for single-process use

def s3_write_json(s3_path: str, data: dict):
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    boto3.client("s3").put_object(
        Bucket=bucket, Key=key, Body=json.dumps(data, ensure_ascii=False)
    )

def s3_read_json(s3_path: str) -> dict:
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def get_doc_id(url: str) -> str | None:
    """Extract document ID of the form P####_############## from a URL."""
    match = re.search(r"(P\d{4}_\d{14})", url, re.IGNORECASE)
    return match.group(1) if match else None


def extract_page_number(url: str) -> int:
    """Extract the 8-digit page number from a TXT URL for correct page ordering."""
    match = re.search(r"_(\d{8})\.txt", url)
    return int(match.group(1)) if match else 0


def get_txt_urls() -> list[str]:
    """Scrape the Impulse S3 index page and return all .txt file URLs."""
    html = requests.get(INDEX_URL).text
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or not href.endswith(".txt"):
            continue

        # Strip the bucket prefix if it was embedded in the href
        if href.startswith(INDEX_URL):
            href = href[len(INDEX_URL):]

        if href.startswith("http"):
            urls.append(href)
        else:
            urls.append(INDEX_URL + href)

    return urls


def download_txt(url: str) -> str:
    """Download a single .txt page; returns empty string on failure."""
    global skipped_txt
    try:
        r = requests.get(url, timeout=3)
        if "AccessDenied" in r.text:
            skipped_txt += 1
            return ""
        r.raise_for_status()
        return r.text
    except Exception:
        skipped_txt += 1
        return ""


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[•·●■□◆◇]+", " ", text)
    text = re.sub(r"(?m)^[\d\W_]{1,5}$", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b(bbox|polygon|confidence|bbox_valid|text)\b", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return text.strip()

def group_urls_by_doc(urls: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        doc_id = get_doc_id(url)
        if doc_id:
            groups[doc_id].append(url)
    return groups


def load_grouped_docs(grouped_urls: dict, limit: int | None = None) -> dict[str, str]:
    """Download and concatenate all pages for each doc, in page order."""
    docs: dict[str, str] = {}
    items = list(grouped_urls.items())
    if limit:
        items = items[:limit]

    for doc_id, url_list in items:
        url_list_sorted = sorted(url_list, key=extract_page_number)

        with ThreadPoolExecutor(max_workers=32) as ex:
            future_to_url = {ex.submit(download_txt, url): url for url in url_list_sorted}

            page_results: dict[str, str] = {}
            for fut in tqdm(
                as_completed(future_to_url),
                total=len(future_to_url),
                desc=f"Doc {doc_id}",
            ):
                url = future_to_url[fut]
                page_results[url] = clean_text(fut.result())

        # Reassemble in sorted order after all futures complete
        docs[doc_id] = "\n".join(page_results[url] for url in url_list_sorted)

    return docs


# ---------------------------------------------------------------------------
# Task 1 — Fetch docs from Impulse S3 and persist as a single JSON
# ---------------------------------------------------------------------------

class FetchDocs(FireTaskBase):
    """
    Scrapes the Impulse S3 index, downloads every .txt page, cleans and
    concatenates them in page order, then writes a single JSON file mapping
    doc_id -> full cleaned text.

    fw_spec keys consumed
    ---------------------
    docs_path : str       -- where to write the combined-text JSON
    doc_limit : int|None  -- optional cap on number of docs (useful for testing)
    """

    _fw_name = "Fetch Docs"

    def run_task(self, fw_spec):
        debug = fw_spec.get("debug", False)

        docs_path = fw_spec["docs_path"]
        limit     = fw_spec.get("doc_limit", None)

        # 1. Discover all TXT URLs from Impulse
        print("Scraping Impulse index for .txt URLs ...")
        all_urls = get_txt_urls()
        print(f"  Found {len(all_urls)} .txt URLs")

        grouped = group_urls_by_doc(all_urls)
        print(f"  Grouped into {len(grouped)} documents")

        # 2. Download, clean, and concatenate pages
        docs_dict = load_grouped_docs(grouped, limit=limit)
        print(f"  Downloaded {len(docs_dict)} documents  (skipped pages: {skipped_txt})")

    # 3. Persist
        if debug:
            local_path = Path.cwd() / "overall_concat_docs_dict.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(docs_dict, f, ensure_ascii=False, indent=2)
            print(f"  [DEBUG] Saved docs to {local_path}")
            return FWAction(update_spec={"docs_path": str(local_path)})
        else:
            s3_write_json(docs_path, docs_dict)
            print(f"  Saved docs to {docs_path}")

            return FWAction(update_spec={"docs_path": docs_path})

# ---------------------------------------------------------------------------
# Task 2 -- Run NER on fetched docs, call LLM, produce final metadata JSON
# ---------------------------------------------------------------------------

class ExtractMetadata(FireTaskBase):
    """
    Loads the docs JSON written by FetchDocs, runs spaCy NER on each document
    in-memory (no intermediate NER files on disk), calls a local LLM to
    resolve the main location and key people, then writes a final metadata JSON:

        {
          "<doc_id>": {
            "doc_id":     "<doc_id>",
            "main_place": "...",
            "key_people": ["...", ...]
          },
          ...
        }

    fw_spec keys consumed
    ---------------------
    docs_path   : str  -- path to the combined-text JSON (written by FetchDocs)
    output_path : str  -- where to write the final metadata JSON
    spacy_model : str  -- spaCy model name (default: "en_core_web_sm")
    """

    _fw_name = "Extract Metadata"

    @staticmethod
    def truncate_text(text: str, max_words: int = MAX_WORDS) -> str:
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text
        
    def run_task(self, fw_spec):

        debug = fw_spec.get("debug", False)

        import socket
        llm_host = f"http://{socket.gethostname()}:8000/v1"


        docs_path   = fw_spec["docs_path"]
        output_path = fw_spec["output_path"]
        model       = fw_spec.get("spacy_model", "en_core_web_sm")

        # 1. Load docs
        if debug:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_dict = json.load(f)
        else:
            docs_dict = s3_read_json(docs_path)


        # 2. Load spaCy model once for the whole run
        nlp = spacy.load(model)

        results = {}

        # 3. NER -> LLM loop
        for doc_id, text in tqdm(docs_dict.items(), desc="NER + LLM", total=len(docs_dict)):

            # Run NER
            spacy_doc = nlp(self.truncate_text(text))

            # Deduplicate while preserving first-seen order
            gpes = list(dict.fromkeys(
                ent.text for ent in spacy_doc.ents if ent.label_ == "GPE"
            ))
            people = list(dict.fromkeys(
                ent.text for ent in spacy_doc.ents if ent.label_ == "PERSON"
            ))

            # Call LLM
            places_n_people = self.ask_ai_func(gpes, people, raw_text=text, host=llm_host)



            # Collect result
            results[doc_id] = {
                "doc_id":     doc_id,
                "main_place": places_n_people.get("main_place"),
                "key_people": places_n_people.get("key_people", [])
            }

        # 4. Save
        if debug:
            local_path = Path.cwd() / "dates_people_metadata.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved metadata to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})


    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def ask_ai_func(self, gpes, people, raw_text="", host=None):

        prompt = f"""
    Document text (first 2000 words):
    {" ".join(raw_text.split()[:2000])}

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
        response = self.call_llm(prompt, host=host)


        # Strip accidental markdown fences before parsing
        response = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"main_place": None, "key_people": []}

    def call_llm(self, prompt: str, host: str) -> str:
        client = OpenAI(api_key="EMPTY", base_url=host)

        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )

        return response.choices[0].message.content.strip()

#-------------------------------------------------------------------------

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

Task: summarize...
    summary summary summary summary  #FIX FIX FIX AAHHFOINFODN
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

#helper for following taskkkk


class Summaries(FireTaskBase):
    _fw_name = "Summaries"
    path: str

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
            docs_dict = s3_read_json(docs_path)

        results = {}
        for doc_id, text in docs_dict.items():
            summary = self.ask_ai_func_summaries(raw_text=text, host=llm_host)
            
            results[doc_id] = {
                "doc_id":     doc_id,
                "summary": summary.get("summary"), #THIS MEANS RESPONSE MUST HAVE "summary"
            }

        # 4. Save
        if debug:
            local_path = Path.cwd() / "overall_summaries.json"
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [DEBUG] Saved metadata to {local_path}")
            return FWAction(update_spec={"metadata_path": str(local_path)})
        else:
            s3_write_json(output_path, results)
            return FWAction(update_spec={"metadata_path": output_path})