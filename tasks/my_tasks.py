from typing import NoReturn, override
from cv2.typing import MatLike
import numpy as np
import dataclasses
import re
import boto3
from io import BytesIO
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
import os
from uuid import uuid4
import base64
import fitz
from tasks.helpers import parse_s3_path, get_s3_content, _get_db
import cv2
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

        return arr

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
        bucket, key = self.parse_s3_path(s3_path)

        session = boto3.Session(profile_name="impulse")
        s3_client = session.client("s3")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,  # Encode string as bytes
        )

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
                raw_arr = self._to_array(content)
                
                if self._is_RGB(raw_arr):
                    raw_arr = self._to_grayscale(raw_arr)

                bin_arr: MatLike = self._binarize(raw_arr)
                dst_arr: MatLike = self._denoise(bin_arr)
                buffer, filetype = self._encode_to_image(bin_arr, ".jp2")
                output_s3_path = "/".join(
                    [
                        "nu-impulse-production",
                        "DATA",
                        str(impulse_identifier).upper(),
                        str(filestem.with_suffix(filetype)),
                    ]
                )
                if type(buffer) == np.ndarray:
                    buffer = buffer.to_bytes
                else:
                    buffer = buffer
                    self.save_to_s3("".join(["s3://", output_s3_path]), buffer)
            else:
                with open(path, "rb") as f:
                    content = f.read()
                    exit()

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

    def _predict(self, contents: list[dict]):
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem, BatchOutputItem
        from tqdm import tqdm
        from PIL import Image
        import io
        from itertools import batched

        results: list[BatchOutputItem] = []
        manager = InferenceManager(method="vllm")
        logger.info(f"Now predicting data")
        for batch in tqdm(batched(contents, 8), desc="Predicting"):
            for b in batch:
                img = Image.open(io.BytesIO(b["contents"])).convert("RGB")
                
                batch_input_items: list[BatchInputItem] = [
                    BatchInputItem(
                        image=img,
                        prompt="Extract the text from this document",
                    )
                ]
                results.append(manager.generate(batch_input_items))
        logger.success("Predictions complete!")
        for i, result in enumerate(results):
            contents[i]["predictions"] = result
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
    def load_jp2(contents: bytes):
        import numpy as np
        import cv2
        from PIL import Image
        from io import BytesIO

        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR_RGB)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def save_to_mongo(self, results, collection):
        """Save any Pydantic model to MongoDB."""
        from tqdm import tqdm
        for i, page in tqdm(enumerate(results), desc="Saving results to database"):
            page_dict = dataclasses.asdict(page["result"])
            page_dict["filename"] = results[i]["filename"]
            page_dict["impulse_identifier"] = results[i]["impulse_identifier"]
            page_dict["page_number"] = results[i]["page_number"]
            collection.update_one(
                {
                    "filename": page_dict["filename"],
                    "impulse_identifier": page_dict["impulse_identifier"],
                    "page_number": page_dict,  # add this
                },
                {"$set": page_dict},
                upsert=True,
            )
        logger.success("Successfully uploaded all documents!")
        return True

    @override
    def run_task(self, fw_spec: dict[str, list[str]]) -> FWAction:
        """
        This method runs the OCR task.
        This method looks for `path_array`.
        """
        find_path_array_in: list[str] = fw_spec["find_path_array_in"]
        path_array: list[str] = fw_spec[find_path_array_in]
        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{path_array}")
        contents: list[dict] = []
        i = 1
        for path in path_array:
            filename = path.split("/")[-1]
            logger.info(f"Filename: {filename}")
            if self.is_s3_path(path):
                # Get content from S3
                logger.info("Now loading content from S3")
                contents.append({
                "filename": filename,
                "page_number": i,
                "contents": get_s3_content(path),
                "impulse_identifier": fw_spec["impulse_identifier"]
                })
        results = self._predict(contents)
        self.save_to_mongo(results, collection=_get_db()["colt"])

        return FWAction()


