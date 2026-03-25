"""Tests for the image processing module."""

import unittest

import numpy as np

from impulse.processing.images import (
    binarize,
    denoise,
    encode_to_image,
    is_rgb,
    to_array,
    to_grayscale,
)


class TestImageHelpers(unittest.TestCase):
    def test_is_rgb_true(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(is_rgb(arr))

    def test_is_rgb_false_grayscale(self):
        arr = np.zeros((100, 100), dtype=np.uint8)
        self.assertFalse(is_rgb(arr))

    def test_is_rgb_false_rgba(self):
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        self.assertFalse(is_rgb(arr))

    def test_to_grayscale(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 200  # red channel
        gray = to_grayscale(arr)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))

    def test_binarize(self):
        # Create a grayscale image with some variation
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = binarize(arr)
        # Binarized image should only contain 0 and 255
        unique = set(np.unique(result))
        self.assertTrue(unique.issubset({0, 255}))

    def test_binarize_rgb_input(self):
        # Should handle RGB input by converting to grayscale first
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = binarize(arr)
        self.assertEqual(len(result.shape), 2)

    def test_denoise(self):
        arr = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = denoise(arr)
        self.assertEqual(result.shape, arr.shape)

    def test_encode_to_png(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        encoded = encode_to_image(arr, ".png")
        self.assertIsInstance(encoded, bytes)
        self.assertTrue(encoded.startswith(b"\x89PNG"))

    def test_encode_to_jp2(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        encoded = encode_to_image(arr, ".jp2")
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)

    def test_to_array_invalid(self):
        with self.assertRaises(ValueError):
            to_array(b"not an image")


if __name__ == "__main__":
    unittest.main()
