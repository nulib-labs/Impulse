from typing import NoReturn, override
from fireworks import FiretaskBase, explicit_serialize
import re
import boto3
from io import BytesIO
from fireworks.core.firework import FWAction, FireTaskBase
from typing_extensions import override
from loguru import logger


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
        path: str = fw_spec["path"]
        output_path: str = fw_spec.get("output_path", None)
        logger.info(f"`path` is {path}")
        logger.info(f"`output_path` is {output_path}")
        if self.is_s3_path(path):
            # Get content from S3
            logger.info("Now loading content from S3")
            content = self.get_s3_content(path)
            binarized = self._binarize(content)
            try:
                _ = self._save_content(output_path, binarized)
                logger.info("Successfully saved image")
            except Exception as e:
                logger.critical(f"Failed to save {e}")
        else:
            # Handle local file path
            with open(path, "rb") as f:
                content = f.read()
            binarized = self._binarize(content)
            try:
                _ = self._save_content(output_path, binarized)
                logger.info("Successfully saved image")
            except Exception as e:
                logger.critical("Failed to save")

        return FWAction()
