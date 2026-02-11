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
        contents = self.load_jp2(contents)
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
            if self.is_s3_path(path[0]):
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
