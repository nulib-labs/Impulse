"""Tests for the API router and handlers."""

import json
import unittest
from unittest.mock import patch, MagicMock


class TestApiRouter(unittest.TestCase):
    """Test the API Gateway Lambda router dispatch logic."""

    def _make_event(self, method, resource, body=None, path_params=None):
        return {
            "httpMethod": method,
            "resource": resource,
            "body": json.dumps(body) if body else None,
            "pathParameters": path_params,
            "requestContext": {"authorizer": {"claims": {"sub": "test-user-123"}}},
            "queryStringParameters": None,
        }

    @patch("impulse.api.create_job.get_collection")
    def test_create_job(self, mock_get_coll):
        mock_coll = MagicMock()
        mock_coll.insert_one.return_value = MagicMock(inserted_id="abc")
        mock_get_coll.return_value = mock_coll

        from impulse.api.router import handler

        event = self._make_event(
            "POST", "/jobs", body={"task_type": "image_transform", "file_count": 5}
        )
        response = handler(event, None)
        self.assertEqual(response["statusCode"], 201)

        body = json.loads(response["body"])
        self.assertIn("job_id", body)
        self.assertEqual(body["status"], "PENDING")

    @patch("impulse.api.list_jobs.get_collection")
    def test_list_jobs(self, mock_get_coll):
        mock_coll = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = iter(
            [
                {"job_id": "j1", "status": "COMPLETED"},
                {"job_id": "j2", "status": "PROCESSING"},
            ]
        )
        mock_coll.find.return_value = mock_cursor
        mock_get_coll.return_value = mock_coll

        from impulse.api.router import handler

        event = self._make_event("GET", "/jobs")
        response = handler(event, None)
        self.assertEqual(response["statusCode"], 200)

        body = json.loads(response["body"])
        self.assertEqual(body["count"], 2)

    def test_not_found(self):
        from impulse.api.router import handler

        event = self._make_event("DELETE", "/nonexistent")
        response = handler(event, None)
        self.assertEqual(response["statusCode"], 404)

    @patch("impulse.api.get_job.get_collection")
    def test_get_job_not_found(self, mock_get_coll):
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = None
        mock_get_coll.return_value = mock_coll

        from impulse.api.router import handler

        event = self._make_event(
            "GET", "/jobs/{jobId}", path_params={"jobId": "nonexistent"}
        )
        response = handler(event, None)
        self.assertEqual(response["statusCode"], 404)


if __name__ == "__main__":
    unittest.main()
