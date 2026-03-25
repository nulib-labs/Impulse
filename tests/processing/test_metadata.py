"""Tests for metadata extraction utilities."""

import unittest

from impulse.processing.metadata import extract_valid_json


class TestExtractValidJson(unittest.TestCase):
    def test_plain_json(self):
        result = extract_valid_json(
            '{"main_place": "Chicago", "key_people": ["Alice"]}'
        )
        self.assertEqual(result["main_place"], "Chicago")
        self.assertEqual(result["key_people"], ["Alice"])

    def test_json_in_markdown_fences(self):
        text = '```json\n{"main_place": "Berlin", "key_people": []}\n```'
        result = extract_valid_json(text)
        self.assertEqual(result["main_place"], "Berlin")

    def test_json_with_surrounding_prose(self):
        text = 'Here is the result: {"main_place": "NYC", "key_people": ["Bob"]} end.'
        result = extract_valid_json(text)
        self.assertEqual(result["main_place"], "NYC")

    def test_garbage_returns_default(self):
        result = extract_valid_json("this is not json at all")
        self.assertIsNone(result["main_place"])
        self.assertEqual(result["key_people"], [])

    def test_empty_string(self):
        result = extract_valid_json("")
        self.assertIsNone(result["main_place"])


if __name__ == "__main__":
    unittest.main()
