from typing import NoReturn, override
import re
import boto3
from io import BytesIO
from fireworks.core.firework import FWAction, FireTaskBase
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from typing_extensions import override
from loguru import logger
from fireworks.utilities.filepad import FilePad
import os
from uuid import uuid4

fp = FilePad(
    host=str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")),
    uri_mode=True,
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
    def _binarize(content):
        import cv2
        import numpy as np

        # Convert to grayscale if RGB
        try:
            content_array = np.frombuffer(content, np.uint8)
        except Exception:
            logger.critical("Unable to convert bytes to numpy array for binarization.")
            exit()

        if len(content_array.shape) == 3 and content_array.shape[2] == 3:
            grayscale_content_array = cv2.cvtColor(content_array, cv2.COLOR_RGB2GRAY)
        else:
            grayscale_content_array = content_array

        # Otsu’s thresholding (black text on white)
        otsu_thresh, _ = cv2.threshold(
            grayscale_content_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adjusted_thresh = otsu_thresh * 1.2
        _, binary_content_array = cv2.threshold(
            grayscale_content_array, adjusted_thresh, 255, cv2.THRESH_BINARY
        )

        return binary_content_array

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
        return FWAction(
            mod_spec=[
                {"_push": {"binarized_objects": binarized_object}}
                for binarized_object in binarized_objects
            ]
        )


class OCRTask(FireTaskBase):
    _fw_name = "OCR Task"

    @staticmethod
    def _predict(contents):
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from loguru import logger
        from PIL import Image

        logger.info("Making Predictions")
        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()
        predictions = recognition_predictor(
            [Image.open(BytesIO(contents))], det_predictor=detection_predictor
        )
        return predictions

    @staticmethod
    def get_filepad_contents(gfs_id):
        contents, doc = fp.get_file_by_id(gfs_id)

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
            elif self.is_impulse_identifier(path[-1]):
                logger.info("Detected Impulse identifier")
                content = self.get_s3_content(path)
                predictions = self._predict(content)
                logger.info(f"Predictions:\n{predictions}")
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
