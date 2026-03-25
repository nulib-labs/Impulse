"""Tests for the utility functions."""

import unittest

from impulse.utils import is_s3_path, parse_s3_path, detect_filetype


class TestS3PathParsing(unittest.TestCase):
    def test_is_s3_path(self):
        self.assertTrue(is_s3_path("s3://bucket/key"))
        self.assertTrue(is_s3_path("s3a://bucket/key"))
        self.assertFalse(is_s3_path("/local/path"))
        self.assertFalse(is_s3_path("https://example.com"))

    def test_parse_s3_path(self):
        bucket, key = parse_s3_path("s3://my-bucket/path/to/file.jp2")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.jp2")

    def test_parse_s3a_path(self):
        bucket, key = parse_s3_path("s3a://my-bucket/path/to/file.jp2")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "path/to/file.jp2")

    def test_parse_s3_path_no_key(self):
        bucket, key = parse_s3_path("s3://my-bucket")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(key, "")


class TestDetectFiletype(unittest.TestCase):
    def test_png(self):
        self.assertEqual(detect_filetype(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100), "png")

    def test_jpeg(self):
        self.assertEqual(detect_filetype(b"\xff\xd8\xff\xe0" + b"\x00" * 100), "jpg")

    def test_pdf(self):
        self.assertEqual(detect_filetype(b"%PDF-1.4" + b"\x00" * 100), "pdf")

    def test_jp2(self):
        header = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
        self.assertEqual(detect_filetype(header + b"\x00" * 100), "jp2")

    def test_too_short(self):
        self.assertIsNone(detect_filetype(b"\x00"))

    def test_empty(self):
        self.assertIsNone(detect_filetype(b""))

    def test_none(self):
        self.assertIsNone(detect_filetype(None))  # type: ignore

    def test_text(self):
        self.assertEqual(detect_filetype(b"Hello world, this is plain text."), "txt")


if __name__ == "__main__":
    unittest.main()
