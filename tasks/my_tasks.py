import io
import json
import queue
import re
import threading
from typing import Generator, override
from uuid import uuid4

from natsort import natsorted
import boto3
from bs4 import BeautifulSoup
import certifi
import cv2
from cv2.typing import MatLike
from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger
import numpy as np
from pymongo import UpdateOne
from pymongo import MongoClient

from blingfire import text_to_sentences
from tasks import config
from tasks.helpers import _get_db, funcs, get_s3_content

SENTENCE_SPLIT = re.compile(r"(?<=[a-z0-9]{2}[.!?])\s+(?=[A-Z])")


SENTENCE_FILTER = re.compile(r"[a-zA-Z]{4,}")


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

    # ----------------------------
    # MAIN PIPELINE
    # ----------------------------
    def get_documents(self, impulse_identifier: str, coll) -> list[str]:
        stream = self.extract_stream(impulse_identifier, coll)

        full_text, char_map = self.build_document(stream)

        sentences = self.split_sentences(full_text)
        sentences = self.merge_broken_sentences(sentences)

        mapped = self.map_pages(sentences, full_text, char_map)

        logger.info(f"Extracted {len(mapped)} sentences")
        return mapped

    # ----------------------------
    # MAIN PIPELINE (BATCHED GENERATOR)
    # ----------------------------
    def get_documents_batched(
        self, impulse_identifier: str, coll, batch_size: int = 16
    ) -> Generator[list[dict], None, None]:
        """Yield sentence batches from the extraction pipeline.

        This is the streaming counterpart of ``get_documents``.  Instead of
        materialising the full list up-front, it yields lists of at most
        *batch_size* mapped-sentence dicts, allowing the caller to start
        embedding while extraction is still in progress.
        """
        stream = self.extract_stream(impulse_identifier, coll)
        full_text, char_map = self.build_document(stream)
        sentences = self.split_sentences(full_text)
        sentences = self.merge_broken_sentences(sentences)
        mapped = self.map_pages(sentences, full_text, char_map)

        logger.info(
            f"Extracted {len(mapped)} sentences (streaming in batches of {batch_size})"
        )

        for i in range(0, len(mapped), batch_size):
            yield mapped[i : i + batch_size]

    def embed(self, items, batch_size: int = 16, k=4):
        from sentence_transformers import SentenceTransformer
        from collections import deque
        from itertools import islice

        def sliding_window(iterable, k):
            "Collect data into overlapping fixed-length chunks or blocks."
            # sliding_window('ABCDEFG', 3) → ABC BCD CDE DEF EFG
            iterator = iter(iterable)
            window = deque(islice(iterator, k - 1), maxlen=k)
            for x in iterator:
                window.append(x)
                yield tuple(window)

        model = SentenceTransformer(
            "microsoft/harrier-oss-v1-27b", model_kwargs={"dtype": "auto"}
        )

        sentences = [x["sentence"] for x in items]
        chunks = sliding_window(sentences, k)
        chunks = [" ".join([ci for ci in c]) for c in chunks]
        embs = model.encode(
            chunks, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )

        items = []
        for item, emb in zip(chunks, embs):
            items.append({"chunk": item, "embedding": emb.tolist()})

        return items

    def store(self, items, coll):
        ops = []
        for item in items:
            print(item)
            ops.append(
                UpdateOne(
                    {
                        "impulse_identifier": item.get("impulse_identifier"),
                        "chunk": item["chunk"],
                    },
                    {"$set": item},
                    upsert=True,
                )
            )

        if ops:
            coll.bulk_write(ops)

        logger.success(f"Stored {len(ops)} embeddings")

    def _run_pipeline(self, impulse_identifier: str, db, batch_size: int = 16) -> int:
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
        from sentence_transformers import SentenceTransformer

        # -- shared queue (bounded so producer doesn't run too far ahead) --
        _SENTINEL = object()
        q: queue.Queue = queue.Queue(maxsize=2)

        # -- producer: extraction pipeline (CPU / IO bound) --
        def producer():
            try:
                for batch in self.get_documents_batched(
                    impulse_identifier, db["colt"], batch_size
                ):
                    for item in batch:
                        item["impulse_identifier"] = impulse_identifier
                    q.put(batch)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        total = 0

        while True:
            batch = q.get()

            if batch is _SENTINEL:
                break

            # Re-raise any exception thrown by the producer thread
            if isinstance(batch, BaseException):
                producer_thread.join()
                raise batch

            batch = self.embed(batch, batch_size=batch_size, k=8)
            self.store(batch, coll=db["embeddings"])
            total += len(batch)

        producer_thread.join()
        logger.success(f"Pipeline complete — {total} embeddings total")
        return total

    @override
    def run_task(self, fw_spec: dict) -> FWAction:
        client = MongoClient(config.MONGO_URI, tlsCAFile=certifi.where())
        db = client["praxis"]

        impulse_identifier = fw_spec.get("impulse_identifier")
        if not impulse_identifier:
            raise ValueError("Missing impulse_identifier")

        total = self._run_pipeline(impulse_identifier, db, batch_size=128)

        return FWAction(stored_data={"num_embeddings": total})


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

        array_denoise = cv2.fastNlMeansDenoising(array, None, 10, 7, 21)

        return array_denoise

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
        logger.debug(f"s3_path: {s3_path}")
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
                raw_arr = cv2.imdecode(
                    np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED
                )
                if raw_arr is None:
                    logger.error(f"Failed to decode image at {path}, skipping.")
                    continue

                if self._is_RGB(raw_arr):
                    raw_arr = self._to_grayscale(raw_arr)

                bin_arr: MatLike = self._binarize(raw_arr)
                dst_arr: MatLike = self._denoise(bin_arr)
                buffer, filetype = self._encode_to_image(dst_arr, ".jp2")

                output_s3_path = "/".join(
                    [
                        "nu-impulse-production",
                        "DATA",
                        str(impulse_identifier).upper(),
                        str(filestem.with_suffix(filetype)),
                    ]
                )
                logger.info("Meow")
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
        logger.debug(f"s3_path: {s3_path}")
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
    def _load_images(image_bytes: bytes, file_ext: str):
        """Decode raw bytes into a list of PIL images.

        A single-page image produces a one-element list; a PDF is expanded
        into one image per page.

        Args:
            image_bytes: Raw file bytes (image or PDF).
            file_ext: File extension (e.g. ``'png'``, ``'pdf'``, ``'jpg'``).

        Returns:
            list[PIL.Image.Image]
        """
        from chandra.input import load_pdf_images
        from PIL import Image

        if file_ext == "pdf":
            return load_pdf_images(io.BytesIO(image_bytes))
        return [Image.open(io.BytesIO(image_bytes)).convert("RGB")]

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

    def save_to_mongo(self, model, collection, s3_base_path: str):
        """Save any Pydantic model to MongoDB, with images stored in S3.

        Args:
            s3_base_path: S3 URI prefix e.g. s3://your-bucket/images
        """
        operations = []
        for i, page in enumerate(model):
            operations.append(
                UpdateOne(
                    {
                        "page_number": page["page_number"],
                        "impulse_identifier": page["impulse_identifier"],
                    },
                    {"$set": page},
                    upsert=True,
                )
            )

        if operations:
            collection.bulk_write(operations)
        logger.success("Successfully uploaded all documents!")
        return True

    def _fetch_batch(
        self,
        path_batch: tuple[str, ...],
        impulse_identifier: str,
        page_offset: int,
    ) -> tuple[list, list[dict], int]:
        """Download and decode all images for a single batch.

        This encapsulates the I/O-bound Phase 1 (S3 download + image
        decoding) so it can be run in a background thread while vLLM
        processes the previous batch on the GPU.

        Args:
            path_batch: Tuple of S3 paths for this batch.
            impulse_identifier: The impulse identifier for metadata.
            page_offset: Starting page number for this batch.

        Returns:
            (batch_input_items, item_meta, next_page_offset)
        """
        from chandra.model.schema import BatchInputItem

        batch_input_items: list = []
        item_meta: list[dict] = []
        page_counter = page_offset

        for path in path_batch:
            page_counter += 1
            filename = path.split("/")[-1]
            logger.info(f"Downloading {filename} from S3")
            image_bytes = get_s3_content(path)
            file_ext = self.filetype(image_bytes) or "png"

            images = self._load_images(image_bytes, file_ext)

            for img in images:
                batch_input_items.append(
                    BatchInputItem(image=img, prompt_type="ocr_layout")
                )
                item_meta.append(
                    {
                        "filename": filename,
                        "page_number": page_counter,
                        "source_image": path,
                        "impulse_identifier": impulse_identifier,
                    }
                )

        logger.info(
            f"Batch ready: {len(path_batch)} files → "
            f"{len(batch_input_items)} images to vLLM"
        )

        return batch_input_items, item_meta, page_counter

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
        from chandra.model import InferenceManager
        from itertools import batched

        find_path_array_in: str = fw_spec["find_path_array_in"]
        path_array: list[str] = fw_spec[find_path_array_in]
        path_array = natsorted(path_array)
        batch_size: int = fw_spec.get("batch_size", 1)
        impulse_identifier: str = fw_spec["impulse_identifier"]
        impulse_identifier = (
            impulse_identifier.replace("{", "")
            .replace("}", "")
            .replace("'", "")
            .lower()
        )

        logger.debug(f"Value of `path_array`:{path_array}")
        logger.debug(f"Type of `path_array`:{type(path_array)}")

        # Initialise chandra inference once and reuse across all images
        logger.info("Initialising chandra InferenceManager...")
        manager = InferenceManager(method="vllm")
        logger.success("InferenceManager ready.")

        # ------------------------------------------------------------------
        # Producer / consumer setup
        # ------------------------------------------------------------------
        _SENTINEL = object()
        prefetch_queue: queue.Queue = queue.Queue(maxsize=1)

        def producer():
            """Download & decode batches in a background thread."""
            page_counter = 0
            try:
                for path_batch in batched(path_array, n=batch_size):
                    batch_input_items, item_meta, page_counter = self._fetch_batch(
                        path_batch, impulse_identifier, page_counter
                    )
                    prefetch_queue.put((batch_input_items, item_meta))
            except Exception as exc:
                prefetch_queue.put(exc)
            finally:
                prefetch_queue.put(_SENTINEL)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        # ------------------------------------------------------------------
        # Main loop: consume prefetched batches → vLLM → MongoDB
        # ------------------------------------------------------------------
        while True:
            batch = prefetch_queue.get()

            if batch is _SENTINEL:
                break

            # Re-raise any exception from the producer thread
            if isinstance(batch, BaseException):
                producer_thread.join()
                raise batch

            batch_input_items, item_meta = batch

            # ----------------------------------------------------------
            # Phase 2: Single batched vLLM inference call
            # ----------------------------------------------------------
            results = self._predict_chandra_batch(batch_input_items, manager)

            # ----------------------------------------------------------
            # Phase 3: Reassemble results with metadata & persist
            # ----------------------------------------------------------
            contents: list[dict] = []
            for meta, result in zip(item_meta, results):
                contents.append(
                    {
                        **meta,
                        "extraction_model": "chandra",
                        "extracted_data": funcs.stringify_keys(
                            {
                                "html": result.html,
                                "markdown": result.markdown,
                                "chunks": result.chunks,
                            }
                        ),
                        "confidence_summary": {
                            "error": result.error,
                            "token_count": result.token_count,
                            "text_extraction_method": "chandra",
                        },
                    }
                )

            self.save_to_mongo(
                contents,
                collection=_get_db()["colt"],
                s3_base_path="nu-impulse-production",
            )

        producer_thread.join()
        return FWAction()
