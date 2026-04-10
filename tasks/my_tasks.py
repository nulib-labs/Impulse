import re
from typing import override
from uuid import uuid4
import boto3
from chandra.model.schema import BatchInputItem
import cv2
from cv2.typing import MatLike
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
import numpy as np
from chandra.model import InferenceManager
from tasks.helpers import _get_db, get_s3_content
from dataclasses import asdict
from pymongo import UpdateOne
import io
import base64

class ImageProcessingTask(FireTaskBase):
    _fw_name = "Image Processing Task"
    @staticmethod
    def _save_content(output_path, content):
        import cv2

        _ = cv2.imwrite(output_path, content)
        pass
    
    @staticmethod
    def _to_array(content: bytes) -> np.ndarray:
        import cv2
        arr = np.frombuffer(content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # actually decode it

    @staticmethod
    def _binarize(arr: MatLike) -> MatLike:

        if len(arr.shape) == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # arr is now guaranteed to be single-channel
        _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized
        
        return binary

    @staticmethod
    def _denoise(array: MatLike) -> MatLike:
        import cv2


        return cv2.fastNlMeansDenoising(array, None, 10, 7, 21)
    
    @staticmethod
    def _process(content: bytes) -> None:
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

    def save_to_s3(self, s3_path: str, content: bytes) -> bool:
        """
        Save string content to S3.

        Args:
            s3_path: S3 URI (e.g. s3://bucket/key)
            content: File content as a string
        """
        logger.debug(f's3_path: {s3_path}')
        bucket, key = self.parse_s3_path(s3_path)

        session = boto3.Session(profile_name="impulse")
        s3_client = session.client("s3")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,  # Encode string as bytes
        )
        logger.success(f"Successfully saved file to s3: {key}")
        return True
    
    @staticmethod
    def _to_grayscale(arr: MatLike) -> MatLike:
        import cv2
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _is_RGB(arr: MatLike) -> bool:
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            return True
        else:
            return False
    
    @staticmethod
    def _encode_to_image(arr: MatLike, filetype: str) -> tuple[bytes, str]:
        import cv2
        from PIL import Image
        import io

        if filetype == ".jp2":
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB) if len(arr.shape) == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG2000")
            return buf.getvalue(), filetype

        success, buffer = cv2.imencode(filetype, arr)
        if not success:
            raise RuntimeError(f"cv2.imencode failed for {filetype}")
        return buffer.tobytes(), filetype

        
    @override
    def run_task(self, fw_spec: dict[str, str]) -> FWAction:
        """
        This method runs the image processing task.
        """
        path_array_key = fw_spec.get("find_path_array_in", None)
        if not path_array_key:
            logger.critical("Critical spec keys missing. Abandoning.")
            raise KeyError(f"Find path array not in spec!")

        path_array: str | None = fw_spec.get(path_array_key, None)

        if not path_array:
            logger.critical("Critical spec keys missing. Abandoning.")
            raise KeyError(f"{path_array_key} not in spec!")

        impulse_identifier = fw_spec.get(
            "impulse_identifier", None
        )  # The impulse identifier can be anything that would be a valid directory name in S3
        impulse_identifier = uuid4() if not impulse_identifier else impulse_identifier
        output_paths: list[tuple[str, str]] = []
        import cv2
        from pathlib import Path
        for path in path_array:
            logger.info(f"`path` is {path}")
            if self.is_s3_path(path):
                filestem = Path(path.split("/")[-1])

                content = get_s3_content(path)
                
                # Fix 1: actually decode the image
                raw_arr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
                if raw_arr is None:
                    logger.error(f"Failed to decode image at {path}, skipping.")
                    continue

                if self._is_RGB(raw_arr):
                    raw_arr = self._to_grayscale(raw_arr)

                bin_arr: MatLike = self._binarize(raw_arr)
                dst_arr: MatLike = self._denoise(bin_arr)
                buffer, filetype = self._encode_to_image(dst_arr, ".jp2")

                output_s3_path = "/".join([
                    "nu-impulse-production",
                    "DATA",
                    str(impulse_identifier).upper(),
                    str(filestem.with_suffix(filetype)),
                ])

                # Fix 2: always save, buffer is already bytes
                self.save_to_s3("".join(["s3://", output_s3_path]), buffer)

        return FWAction()


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

    def save_to_s3(self, s3_path: str, content: bytes) -> bool:
        """
        Save string content to S3.

        Args:
            s3_path: S3 URI (e.g. s3://bucket/key)
            content: File content as a string
        """
        logger.debug(f's3_path: {s3_path}')
        bucket, key = self.parse_s3_path(s3_path)

        session = boto3.Session(profile_name="impulse")
        s3_client = session.client("s3")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,  # Encode string as bytes
        )
        logger.success(f"Successfully saved file to s3: {key}")
        return True
    
    def _predict(self, contents: list[dict]):

        manager = InferenceManager(method="vllm")
        output: list[dict] = []
        batch: list[BatchInputItem] = [BatchInputItem(image=i["contents"], prompt_type="ocr_layout") for i in contents] # Define a batch as a list of InputItems for Chandra
        results = manager.generate(batch) # Generate the results
        for i, item in enumerate(results): # Generate a rendered dictionary
            rendered_dict = asdict(item)
            rendered_dict["filename"] = contents[i]["filename"]
            rendered_dict["page_number"] = contents[i]["page_number"]
            rendered_dict["impulse_identifier"] = contents[i]["impulse_identifier"]
            output.append(rendered_dict)

        return output

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
    def load_image(contents: bytes):
        import numpy as np
        import cv2
        from PIL import Image

        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR_RGB)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img


    def save_to_mongo(self, model, collection, s3_base_path: str):
        """Save any Pydantic model to MongoDB, with images stored in S3.
        
        Args:
            s3_base_path: S3 URI prefix e.g. s3://your-bucket/images
        """
        operations = []
        for i, page in enumerate(model):
            page = page.copy()
            impulse_id = page["impulse_identifier"]
            page_number = page["page_number"]

            image_keys = []
            for filename, pil_image in page.pop("images", {}).items():
                buffer = io.BytesIO()
                pil_image.save(buffer, format="WEBP")
                
                s3_path = f"{s3_base_path}/{impulse_id}/{page_number:010d}/{filename}.webp"
                self.save_to_s3(s3_path, buffer.getvalue())
                image_keys.append(s3_path)

            page["images"] = image_keys
            page["document_extraction_model"] = "chandra"

            operations.append(
                UpdateOne(
                    {
                        "page_number": page_number,
                        "impulse_identifier": impulse_id,
                    },
                    {"$set": page},
                    upsert=True,
                )
            )

        if operations:
            collection.bulk_write(operations)
        logger.success("Successfully uploaded all documents!")
        return True

    @override
    def run_task(self, fw_spec: dict[str, list[str]]) -> FWAction:
        """
        This method runs the OCR task.
        This method looks for `path_array`.
        """
        find_path_array_in: list[str] = fw_spec["find_path_array_in"] # What key to get the array of S3 keys from
        path_array: list[str] = fw_spec[find_path_array_in] # Get the list of S3 keys
        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{path_array}")
        from itertools import batched
        i = 0
        for batch in batched(path_array, n=32): 
            contents: list[dict] = []
            for path in batch:
                i += 1
                logger.info(f"`path`: {path}")
                filename = path.split("/")[-1]
                logger.info(f"Filename: {filename}")
                # Get content from S3
                logger.info("Now loading content from S3")
                contents.append({
                "filename": filename,
                "page_number": i,
                "contents": get_s3_content(path),
                "impulse_identifier": fw_spec["impulse_identifier"],
                "source_image": path
                })

            results = self._predict(contents)
            print(results[0])
            self.save_to_mongo(results, collection=_get_db()["colt"], s3_base_path="nu-impulse-production")

        return FWAction()

