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

from impulse.auxiliary import convert_mets_to_yml

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
        logger.info(f"Rendered length: {len(rendered)}")
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

    def save_to_mongo(model, collection):
        """Save any Pydantic model to MongoDB."""
        return collection.insert_one(model.dict())

    @override
    def run_task(self, fw_spec: dict[str, list[str]]) -> FWAction:
        """
        This method runs the OCR task.
        This method looks for `path_array`.
        """
        find_path_array_in = fw_spec["find_path_array_in"]
        path_array: list[tuple(str, str)] = fw_spec[find_path_array_in]
        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{path_array}")
        for path in path_array:
            if self.is_s3_path(path):
                # Get content from S3
                logger.info("Now loading content from S3")
                content = self.get_s3_content(path)
                predictions = self._predict(content)
                logger.info(f"Predictions:\n{predictions}")
            elif self.is_impulse_identifier(path[1]):
                logger.info("Detected Impulse identifier")
                content = self.get_filepad_contents(path[1])
                predictions = self._predict(content)
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
