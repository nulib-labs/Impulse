"""Image processing functions: binarize, denoise, grayscale, encode."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from cv2.typing import MatLike


def to_array(content: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV array."""
    arr = np.frombuffer(content, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Failed to decode image bytes")
    return decoded


def is_rgb(arr: np.ndarray) -> bool:
    """Return *True* if *arr* has 3 colour channels."""
    return len(arr.shape) == 3 and arr.shape[2] == 3


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR image to single-channel grayscale."""
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def binarize(arr: np.ndarray) -> np.ndarray:
    """Apply Otsu binarisation.  Converts to grayscale first if needed."""
    if len(arr.shape) == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def denoise(arr: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)


def encode_to_image(arr: np.ndarray, filetype: str = ".jp2") -> bytes:
    """Encode an OpenCV array to image bytes in the given format.

    Supports ``.jp2`` (JPEG 2000) via Pillow and other formats via OpenCV.
    """
    from PIL import Image

    if filetype == ".jp2":
        rgb = (
            cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            if len(arr.shape) == 2
            else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        )
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG2000")
        return buf.getvalue()

    success, buffer = cv2.imencode(filetype, arr)
    if not success:
        raise RuntimeError(f"cv2.imencode failed for {filetype}")
    return buffer.tobytes()


def process_image(content: bytes, output_format: str = ".jp2") -> bytes:
    """Full image processing pipeline: decode -> grayscale -> binarize -> denoise -> encode.

    This is the high-level function called by the ECS handler.
    """
    raw_arr = to_array(content)

    if is_rgb(raw_arr):
        raw_arr = to_grayscale(raw_arr)

    bin_arr = binarize(raw_arr)
    dst_arr = denoise(bin_arr)
    return encode_to_image(dst_arr, output_format)
