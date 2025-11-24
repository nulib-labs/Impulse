import os
from pathlib import Path
from PIL import Image
from marker.schema import document
import numpy as np
import cv2
from numpy._typing import _UnknownType
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor


class DocumentPage:
    array = None

    def __init__(self, input_path: Path) -> None:
        self.array = np.array(Image.open(input_path))
        return None

    def binarize(self, array) -> np.ndarray[_UnknownType, _UnknownType]:
        """Binarizes the images to B/W"""
        binarized_array: np.ndarray[_UnknownType, _UnknownType]

        # Convert to grayscale if RGB
        if len(array.shape) == 3 and array.shape[2] == 3:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        else:
            gray = array

        # Denoise
        dst = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Otsu’s thresholding (black text on white)
        otsu_thresh, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = otsu_thresh * 1.2
        _, binarized_array = cv2.threshold(dst, adjusted_thresh, 255, cv2.THRESH_BINARY)
        return binarized_array

    pass


class Document:
    pages: None | list[DocumentPage] = None

    def __init__(self, pages: list[DocumentPage]):
        self.pages = pages
        pass

    def ocr(self, pages):
        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()
        predictions = recognition_predictor([pages], det_predictor=detection_predictor)
        return predictions
