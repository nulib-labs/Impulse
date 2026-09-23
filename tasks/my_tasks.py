import io
import json
from math import floor
import queue
import re
import threading
from typing import Generator, override
from uuid import uuid4

from PIL import Image
import PIL
from natsort import natsorted
import boto3
from bs4 import BeautifulSoup
import certifi
import cv2
from cv2.typing import MatLike
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
import numpy as np
from pymongo import ReplaceOne, UpdateOne
from pymongo import MongoClient
from tasks import common, config
import tasks
from tasks.common.s3 import upload_pil_image_to_s3, s3_key_exists
from tasks.helpers import _get_db, funcs, get_s3_content
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://localhost:27017",
)

MONGO_DB = os.environ.get(
    "MONGO_DB",
    "fireworks",
)

S3_BUCKET = os.environ["S3_BUCKET"]
AWS_PROFILE = os.environ["AWS_PROFILE"]
AWS_REGION = os.environ["AWS_REGION"]

SENTENCE_SPLIT = re.compile(r"(?<=[a-z0-9]{2}[.!?])\s+(?=[A-Z])")


SENTENCE_FILTER = re.compile(r"[a-zA-Z]{4,}")

@dataclass
class ImpulseItem:
    """ Class for passing image meta/data to impulse."""
    impulse_identifier: str
    page_number: int

@dataclass
class ImpulseInputItem(ImpulseItem):
    image_data: Image.Image
    source_path: str

@dataclass
class ImpulseOutputItem(ImpulseInputItem):
    layout_data: dict
    ocr_data: dict
    data_save_path: str
    extraction_model: str = "surya-2"

class IOTask(FireTaskBase):
    _fw_name = "I/O Task"
    def run_task(self, fw_spec: dict[str, list[str]]):
        from itertools import batched
        find_path_array_in: str = fw_spec["find_path_array_in"]
        path_array: list[str] = fw_spec[find_path_array_in]
        path_array = natsorted(path_array)

        impulse_identifier: str = fw_spec["impulse_identifier"]
        impulse_identifier = (
            impulse_identifier.replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .lower()
        )
        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{type(path_array)}")

        bucket = "nu-impulse-data"
        impulse_input_items: list[ImpulseInputItem] = []
        new_keys: list[str] = []
        keys_lock = threading.Lock()
        items_lock = threading.Lock()

        def normalize_paths(i, image_path):
            if image_path.startswith('s3://'):
                from tasks.common.s3 import download_s3_file
                project_number = impulse_identifier.split("_")[0].lower()
                accession_number = impulse_identifier.split("_")[1].lower()
                filename = "_".join([project_number, accession_number, f"{i+1:010d}.jpg"])
                key = "/".join([project_number, accession_number, "raw_images", filename])

                item = ImpulseInputItem(impulse_identifier, i + 1, download_s3_file(image_path))
                with items_lock:
                    impulse_input_items.append(item)

                if not s3_key_exists(bucket, key):
                    print(f"Uploading to key: {key}")
                    upload_pil_image_to_s3(item.image_data, bucket, key)
                else:
                    print(f"Skipping existing key: {key}")

                with keys_lock:
                    new_keys.append(f"s3://{bucket}/{key}")

        for batch in batched(enumerate(path_array), 4):
            threads = []
            for i, image_path in batch:
                t = threading.Thread(target=normalize_paths, args=(i, image_path))
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        new_keys = natsorted(new_keys)

        return FWAction(update_spec={"keys": new_keys})

class EmbeddingTask(FireTaskBase):
    _fw_name = "Embedding Task"

    # ----------------------------
    # HTML CLEANING
    # ----------------------------
    #
    def find_text_values(self, d, results=None):
        """
        Recursively extract HTML strings from extracted_data tree.
        """
        if results is None:
            results = []

        if isinstance(d, dict):
            for k, v in d.items():
                if k == "html" and isinstance(v, str) and v.strip():
                    results.append(v)
                else:
                    self.find_text_values(v, results)

        elif isinstance(d, list):
            for item in d:
                self.find_text_values(item, results)

        return results

    def clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["math", "script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)

    # ----------------------------
    # EXTRACT ORDERED STREAM
    # ----------------------------
    def extract_stream(self, impulse_identifier: str, coll):
        cursor = coll.find({"impulse_identifier": impulse_identifier}).sort(
            "page_number", 1
        )

        stream = []

        for doc in cursor:
            page_number = doc["page_number"]

            html_blocks = self.find_text_values(doc.get("extracted_data", {}))

            for html in html_blocks:
                if not html:
                    continue

                cleaned = self.clean_html(html)

                if not cleaned:
                    continue

                if not SENTENCE_FILTER.search(cleaned):
                    continue

                stream.append((page_number, cleaned))

        return stream

    # ----------------------------
    # BUILD FULL DOCUMENT
    # ----------------------------
    def build_document(self, stream):
        full_text_parts = []
        char_map = []

        cursor = 0

        for page_number, text in stream:
            start = cursor
            full_text_parts.append(text)

            cursor += len(text) + 1

            char_map.append((start, cursor, page_number))

        full_text = " ".join(full_text_parts)
        full_text = re.sub(r"\s+", " ", full_text)

        return full_text, char_map

    # ----------------------------
    # SENTENCE SPLITTING (BLINGFIRE)
    # ----------------------------
    def split_sentences(self, text: str):
        raw = text_to_sentences(text).split("\n")

        sentences = [
            s.strip() for s in raw if len(s.strip()) > 20 and SENTENCE_FILTER.search(s)
        ]

        return sentences

    # ----------------------------
    # FIX BROKEN SENTENCES (PAGE SPLITS)
    # ----------------------------
    def merge_broken_sentences(self, sentences):
        merged = []

        for s in sentences:
            if not merged:
                merged.append(s)
                continue

            prev = merged[-1]

            if not re.search(r'[.!?]["\']?$', prev):
                merged[-1] = prev + " " + s
            elif s and s[0].islower():
                merged[-1] = prev + " " + s
            else:
                merged.append(s)

        return merged

    # ----------------------------
    # MAP SENTENCES TO PAGES
    # ----------------------------
    def map_pages(self, sentences, full_text, char_map):
        results = []
        cursor = 0

        for sentence in sentences:
            idx = full_text.find(sentence, cursor)
            if idx == -1:
                continue

            end = idx + len(sentence)
            cursor = end

            page = None
            for s, e, p in char_map:
                if s <= idx < e:
                    page = p
                    break

            results.append(
                {
                    "sentence": sentence,
                    "page_number": page,
                }
            )

        return results

    def embed(
        self, items: list[dict], model: SentenceTransformer, batch_size: int = 4, k=4
    ):
        from collections import deque
        from itertools import islice, batched

        def sliding_window(iterable, k):
            iterator = iter(iterable)
            window = deque(islice(iterator, k - 1), maxlen=k)
            for x in iterator:
                window.append(x)
                yield tuple(window)

        sentences = [x["sentence"] for x in items]
        chunks = [" ".join(c) for c in sliding_window(sentences, k)]
        print(f"Length of chunks: {len(chunks)}")

        to_store = []
        for batch_chunk in batched(chunks, 256):
            all_embeddings = model.encode(
                list(batch_chunk),
                batch_size=8,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            for chunk, emb in zip(chunks, all_embeddings):
                to_store.append({"chunk": chunk, "embedding": emb.tolist()})

        return to_store

    def store(self, items, coll, impulse_identifier):
        ops = []
        for item in items:
            ops.append(
                UpdateOne(
                    {
                        "impulse_identifier": item.get("impulse_identifier"),
                        "chunk": item["chunk"],
                        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                        "impulse_identifier": impulse_identifier,
                    },
                    {"$set": item},
                    upsert=True,
                )
            )

        if ops:
            coll.bulk_write(ops)

        logger.success(f"Stored {len(ops)} embeddings")

    def _run_pipeline(
        self,
        model: SentenceTransformer,
        impulse_identifier: str,
        db,
        batch_size: int = 4,
    ) -> int:
        """Stream extraction into embedding using a threaded producer/consumer.

        A background thread runs the full extraction pipeline
        (``get_documents_batched``) and pushes sentence batches onto a
        bounded queue.  The main thread pulls batches off the queue,
        encodes them on GPU, and stores the results — so the GPU never
        idles waiting for extraction to finish.

        Args:
            impulse_identifier: The document identifier to process.
            db: A pymongo ``Database`` instance (e.g. ``client["praxis"]``).
            batch_size: Number of sentences per batch.

        Returns:
            Total number of embedded sentences.
        """

        def get_documents(impulse_identifier: str, coll) -> list[dict]:
            stream = self.extract_stream(impulse_identifier, coll)

            full_text, char_map = self.build_document(stream)

            sentences = self.split_sentences(full_text)
            sentences = self.merge_broken_sentences(sentences)

            mapped = self.map_pages(sentences, full_text, char_map)

            logger.info(f"Extracted {len(mapped)} sentences")
            return mapped

        documents: list[dict] = get_documents(impulse_identifier, db["colt"])
        embedded_documents = self.embed(
            documents, model=model, batch_size=batch_size, k=4
        )
        self.store(
            embedded_documents,
            coll=db["embeddings"],
            impulse_identifier=impulse_identifier,
        )
        return True

    def run_task(self, fw_spec: dict) -> FWAction:
        print("Now running embedding task")
        client = MongoClient(config.MONGO_URI, tlsCAFile=certifi.where())
        db = client["praxis"]

        import torch

        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device="cuda",
            model_kwargs={"torch_dtype": torch.float16},
        )
        impulse_identifier = fw_spec.get("impulse_identifier")
        if not impulse_identifier:
            raise ValueError("Missing impulse_identifier")

        total = self._run_pipeline(model, impulse_identifier, db, batch_size=4)

        return FWAction(stored_data={"num_embeddings": total})


class ImageProcessingTask(FireTaskBase):
    _fw_name = "Image Processing Task"

    @staticmethod
    def _decode(content: bytes) -> MatLike | None:
        """Decode raw image bytes into an OpenCV array.

        Returns None if the bytes could not be decoded.
        """
        import cv2

        arr = np.frombuffer(content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _denoise_gray(arr: MatLike) -> MatLike:
        """Denoise a single-channel grayscale image.

        Uses a small median filter, which is cheap and kills salt-and-
        pepper scan speckle without blurring stroke edges. We deliberately
        avoid ``fastNlMeansDenoising`` here: it is dramatically more
        expensive (seconds per multi-megapixel page) and offers little
        additional benefit once illumination normalization + Sauvola do
        the heavy lifting. On a multi-worker pipeline the NLM cost
        (both CPU and RAM) is the main cause of OOM crashes.
        """
        import cv2

        return cv2.medianBlur(arr, 3)

    @staticmethod
    def _normalize_illumination(gray: MatLike) -> MatLike:
        """Flatten uneven scan lighting / shadows / page-curl gradients.

        Estimates the background via a large morphological close, then
        divides the input by that background. The result has a near-
        uniform bright background so downstream local thresholding
        performs much more consistently.

        The structuring-element size scales with the shorter image
        dimension so this works across DPIs.
        """
        import cv2

        h, w = gray.shape[:2]
        k = max(15, (min(h, w) // 30) | 1)  # odd, ~3% of short side
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # cv2.divide handles zero-background pixels safely and clips to uint8.
        return cv2.divide(gray, background, scale=255)

    @staticmethod
    def _binarize_adaptive(
        gray: MatLike, k: float = 0.34, R: float = 128.0
    ) -> MatLike:
        """Sauvola adaptive binarization.

        Threshold per pixel is::

            T(x, y) = mean(x, y) * (1 + k * (std(x, y) / R - 1))

        with ``k`` a sensitivity parameter (Sauvola & Pietikäinen 2000
        recommend ~0.2–0.5; 0.34 is a good default for scanned text)
        and ``R`` the dynamic range of standard deviation (128 for
        8-bit imagery).

        Implementation uses OpenCV integral images so per-pixel local
        mean/std are computed in O(N) regardless of window size.

        The window size scales with the shorter image dimension so the
        same code works at arbitrary DPI.
        """
        import cv2

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        win = max(15, (min(h, w) // 40) | 1)  # odd, ~2.5% of short side
        r = win // 2

        g = gray.astype(np.float32)
        # Integral images: shape (h+1, w+1).
        # S can safely be float32 (max value 255*W*H fits in 24-bit mantissa
        # for realistic page sizes). S2 must stay float64 — squared pixel
        # sums grow ~65k× faster and would lose precision in float32 on
        # high-res scans.
        S = cv2.integral(g, sdepth=cv2.CV_32F)
        g2 = g * g
        S2 = cv2.integral(g2, sdepth=cv2.CV_64F)
        # Drop the per-pixel intermediates as soon as their integrals exist.
        del g, g2

        # Pad-index trick: for each pixel (y, x) we want the box
        # [y-r .. y+r] x [x-r .. x+r], clipped to image bounds.
        ys = np.arange(h)
        xs = np.arange(w)
        y0 = np.clip(ys - r, 0, h)
        y1 = np.clip(ys + r + 1, 0, h)
        x0 = np.clip(xs - r, 0, w)
        x1 = np.clip(xs + r + 1, 0, w)

        # Broadcast to (h, w) index arrays.
        Y0 = y0[:, None]
        Y1 = y1[:, None]
        X0 = x0[None, :]
        X1 = x1[None, :]

        area = (Y1 - Y0) * (X1 - X0)
        # Guard against zero-area boxes (shouldn't happen, but be safe).
        area = np.where(area == 0, 1, area).astype(np.float64)

        sum_ = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]
        sum_sq = S2[Y1, X1] - S2[Y0, X1] - S2[Y1, X0] + S2[Y0, X0]

        mean = sum_ / area
        # Variance clamped to zero to avoid tiny negatives from FP noise.
        var = np.maximum(sum_sq / area - mean * mean, 0.0)
        std = np.sqrt(var)

        threshold = mean * (1.0 + k * ((std / R) - 1.0))
        binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
        return binary

    @staticmethod
    def _deskew(binary: MatLike) -> MatLike:
        """Estimate page skew from the binary image and rotate to correct.

        The ``deskew`` package expects ink-as-high, so we pass an
        inverted copy for angle detection only. Angles outside
        ``[0.1°, 15°]`` (absolute) are ignored — smaller is noise,
        larger is almost certainly a bad estimate on a sparse page.

        Rotation is done on the binary itself with cubic interpolation
        and white border fill so exposed corners match the page
        background and don't confuse downstream OCR.
        """
        import cv2
        from deskew import determine_skew

        angle = determine_skew(cv2.bitwise_not(binary))
        if angle is None:
            return binary
        if abs(angle) < 0.1 or abs(angle) > 15.0:
            return binary

        h, w = binary.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            binary,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

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
        path = re.sub(r"^s3a?://", "", s3_path)
        parts = path.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key

    def save_to_s3(self, s3_path: str, content: bytes) -> bool:
        """
        Save binary content to S3.

        Args:
            s3_path: S3 URI (e.g. s3://bucket/key)
            content: File content as bytes
        """
        logger.debug(f"s3_path: {s3_path}")
        bucket, key = self.parse_s3_path(s3_path)

        session = boto3.Session(profile_name="impulse")
        s3_client = session.client("s3")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
        )
        logger.success(f"Successfully saved file to s3: {key}")
        return True

    @staticmethod
    def _to_grayscale(arr: MatLike) -> MatLike:
        import cv2

        # cv2.imdecode / imread produce BGR, not RGB — BGR2GRAY is correct here
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _is_RGB(arr: MatLike) -> bool:
        return len(arr.shape) == 3 and arr.shape[2] == 3

    @staticmethod
    def _encode_to_image(arr: MatLike, filetype: str) -> tuple[bytes, str]:
        import cv2
        from PIL import Image
        import io

        if filetype == ".jp2":
            rgb = (
                cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                if len(arr.shape) == 2
                else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            )
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
        import cv2
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Cap OpenCV's internal thread pool. When we run several page
        # pipelines concurrently, each calling into OpenCV, the library's
        # own OpenMP/pthread pool multiplies with our worker count and
        # causes oversubscription — the process either thrashes or gets
        # OOM-killed, which surfaces as a hard "crash" under tmux.
        try:
            cv2.setNumThreads(2)
        except Exception:
            pass

        path_array_key = fw_spec.get("find_path_array_in", None)
        if not path_array_key:
            logger.critical("Critical spec keys missing. Abandoning.")
            raise KeyError("Find path array not in spec!")

        path_array: list[str] | None = fw_spec.get(path_array_key, None)
        if not path_array:
            logger.critical("Critical spec keys missing. Abandoning.")
            raise KeyError(f"{path_array_key} not in spec!")

        path_array = natsorted(path_array)

        impulse_identifier = fw_spec.get("impulse_identifier", None)
        impulse_identifier = str(uuid4()) if not impulse_identifier else impulse_identifier
        impulse_identifier = (
            impulse_identifier.replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .lower()
        )

        project_number = impulse_identifier.split("_")[0].lower()
        accession_number = impulse_identifier.split("_")[1].lower()

        new_keys: list[str] = []
        keys_lock = threading.Lock()

        def sibling_key(path: str, new_extension: str) -> str:
            """
            Return the S3 key for a sibling file with a different extension.

            Example:
                s3://bucket/foo/bar/page.png
                -> foo/bar/page.jp2
            """
            bucket, key = self.parse_s3_path(path)
            if not key:
                raise ValueError(f"Invalid S3 path: {path}")

            base, _ = os.path.splitext(key)
            return f"{base}{new_extension}"

        def process_one(i: int, path: str) -> None:
            if not self.is_s3_path(path):
                logger.warning(f"Skipping non-S3 path: {path}")
                return

            try:
                bucket, source_key = self.parse_s3_path(path)

                # The transformed image is a sibling of the source image.
                output_key = sibling_key(path, ".jp2")

                if s3_key_exists(bucket, output_key):
                    logger.info(
                        f"Skipping existing transformed image: "
                        f"s3://{bucket}/{output_key}"
                    )
                    with keys_lock:
                        new_keys.append(output_key)
                    return

                content = get_s3_content(path)

                raw_arr = self._decode(content)
                if raw_arr is None:
                    logger.error(
                        f"Failed to decode image at {path}, skipping."
                    )
                    return

                # Pipeline:
                #   grayscale
                #       -> denoise
                #       -> illumination normalization
                #       -> Sauvola binarization
                #       -> deskew
                if raw_arr.ndim == 3:
                    if raw_arr.shape[2] == 4:
                        gray = cv2.cvtColor(
                            raw_arr,
                            cv2.COLOR_BGRA2GRAY,
                        )
                    elif raw_arr.shape[2] == 3:
                        gray = self._to_grayscale(raw_arr)
                    else:
                        gray = raw_arr[..., 0]
                else:
                    gray = raw_arr

                del raw_arr

                gray = self._denoise_gray(gray)
                gray = self._normalize_illumination(gray)

                binary: MatLike = self._binarize_adaptive(gray)
                del gray

                binary = self._deskew(binary)

                # Encode the processed image as JPEG 2000.
                encoded_bytes, _ = self._encode_to_image(
                    binary,
                    ".jp2",
                )
                del binary

                self.save_to_s3(
                    f"s3://{bucket}/{output_key}",
                    encoded_bytes,
                )

                logger.success(
                    f"Uploaded transformed image: "
                    f"s3://{bucket}/{output_key}"
                )

                with keys_lock:
                    new_keys.append(output_key)

            except Exception as exc:
                # Never let one bad page kill the whole run.
                logger.exception(
                    f"Failed to process {path}: {exc}"
                )
                return

            with keys_lock:
                new_keys.append(key)

        # Bounded worker pool. The pipeline is CPU + RAM heavy (Sauvola
        # allocates several float64 arrays the size of the image, and
        # deskew runs a Hough transform), so we deliberately keep the
        # concurrency low. Env var lets ops tune per-host.
        max_workers = int(os.environ.get("IMPULSE_IMG_WORKERS", "4"))
        max_workers = max(1, max_workers)
        logger.info(
            f"Processing {len(path_array)} images with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(process_one, i, path)
                for i, path in enumerate(path_array)
            ]
            for fut in as_completed(futures):
                # Surface any unexpected exception that escaped process_one.
                fut.result()

        new_keys = natsorted(new_keys)

        return FWAction(update_spec={"keys": new_keys})


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

    @staticmethod
    def _predict_chandra_batch(batch_input_items, manager):
        """Send an already-assembled list of BatchInputItems to vLLM in one call.

        Args:
            batch_input_items: list[BatchInputItem] — all images for this batch.
            manager: A pre-built chandra ``InferenceManager`` instance.

        Returns:
            list[BatchOutputItem] — one result per input item, same order.
        """
        return manager.generate(batch_input_items)

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

    def save_to_s3(self, items: list[ImpulseOutputItem]) -> bool:
        """Save each item's OCR/layout output as a single JSON file in S3.

        Args:
            items: ImpulseOutputItem instances, each with a source_path
                pointing at the S3 key to write the JSON to.
        """
        session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )

        s3 = session.client("s3")

        for item in items:
            s3_path = item.source_path
            logger.debug(f"s3_path: {s3_path}")
            bucket, key = self.parse_s3_path(s3_path)

            payload = {
                "ocr_data": item.ocr_data,
                "layout_data": item.layout_data,
                "extraction_model": item.extraction_model,
            }

            s3.put_object(
                Bucket=S3_BUCKET,
                Key="jobs" + "/" + key.split(".")[0] + ".json",
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json",
            )
            logger.success(f"Successfully saved file to s3: {key}")

        return True


    @override
    def run_task(self, fw_spec: dict[str, list[str]]) -> FWAction:
        """Run batched OCR extraction via vLLM with prefetched batches.

        A background producer thread downloads and decodes the *next*
        batch from S3 while the main thread runs vLLM inference on the
        *current* batch.  This keeps the GPU busy instead of idling
        during S3 downloads and image decoding.

        The producer pushes ``(batch_input_items, item_meta)`` tuples
        onto a bounded queue (``maxsize=1``, i.e. at most one batch
        ahead).  The main thread pulls from the queue, runs inference,
        and persists results to MongoDB.
        """
        from bs4 import BeautifulSoup
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        from surya.layout import LayoutPredictor
        from itertools import batched
        from tasks.common.s3 import download_s3_file        
        from urllib.parse import urlparse
        
        manager = SuryaInferenceManager()
        recognition_predictor = RecognitionPredictor(manager)
        layout_predictor = LayoutPredictor(manager)
        find_path_array_in: str = fw_spec["find_path_array_in"]
        path_array: list[str] = fw_spec[find_path_array_in]
        path_array = natsorted(path_array)
        
        impulse_identifier: str = fw_spec["impulse_identifier"]
        impulse_identifier = (
            impulse_identifier.replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .lower()
        )

        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{type(path_array)}")
        
        impulse_input_items: list[ImpulseInputItem] = []
        
        def handle_txt_format(item: ImpulseOutputItem):
            session = boto3.Session(
                profile_name=AWS_PROFILE,
                region_name=AWS_REGION,
            )

            s3 = session.client("s3")

            ocr_data: dict = item.ocr_data
            payload = []
            for block in ocr_data["blocks"]:
                html_raw = block.get("html", "")
                soup = BeautifulSoup(html_raw, "html.parser")
                soup2 = soup.get_text()
                payload.append(soup2)
            

            payload = "\n".join(payload)

            s3.put_object(
                Bucket=S3_BUCKET,
                Key="jobs" + "/" + item.source_path.split(".")[0] + ".txt",
                Body=payload.encode("utf-8"),
                ContentType="application/json",
            )



        def prepare_input_items(i, image_path):
            if s3_key_exists(S3_BUCKET, image_path):
                print("found s3 item")
                item = ImpulseInputItem(
                    impulse_identifier=impulse_identifier,
                    page_number=i + 1,
                    image_data=download_s3_file(f"s3://{S3_BUCKET}/{image_path}"),
                    source_path=image_path,
                )
                return item

        for i, path in enumerate(natsorted(path_array)):
            impulse_input_items.append(prepare_input_items(i, path))


        for batch in batched(impulse_input_items, 64):
                
            impulse_output_items: list[ImpulseOutputItem] = []

            batch_images = [item.image_data for item in batch]  # extract once
            batch_layout = layout_predictor(batch_images)
            batch_ocr = recognition_predictor(batch_images, batch_layout)

            for item, layout, ocr in zip(impulse_input_items, batch_layout, batch_ocr):
                path_parts = item.source_path.split(".")
                save_path = path_parts[0] + ".json"



                impulse_output_items.append(
                    ImpulseOutputItem(
                        image_data = item.image_data,
                        source_path = item.source_path,
                        impulse_identifier=item.impulse_identifier,
                        page_number=item.page_number,
                        layout_data=layout.model_dump(),
                        ocr_data=ocr.model_dump(),
                        data_save_path = save_path
                    )
                )


            contents: list[dict] = [asdict(output_item_dict) for output_item_dict in impulse_output_items]
            print(contents)

            self.save_to_s3(
                impulse_output_items,
            )
        FWAction()

