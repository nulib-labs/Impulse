"""Tests for the database models and client."""

import unittest
from dataclasses import asdict

import mongomock

from impulse.db.models import Job, JobStatus, Result, Work


class TestJobModel(unittest.TestCase):
    def setUp(self):
        self.client = mongomock.MongoClient()
        self.coll = self.client.db.jobs

    def test_create_job(self):
        job = Job(
            job_id="test-123",
            user_id="user-abc",
            status=JobStatus.PENDING.value,
            task_type="full_pipeline",
            total_documents=10,
        )
        result = self.coll.insert_one(job.to_dict())
        self.assertIsNotNone(result.inserted_id)

        found = self.coll.find_one({"job_id": "test-123"})
        self.assertIsNotNone(found)
        self.assertEqual(found["user_id"], "user-abc")
        self.assertEqual(found["total_documents"], 10)

    def test_update_job_status(self):
        job = Job(job_id="test-456", user_id="user-abc")
        self.coll.insert_one(job.to_dict())

        self.coll.update_one(
            {"job_id": "test-456"},
            {"$set": {"status": JobStatus.PROCESSING.value}},
        )

        found = self.coll.find_one({"job_id": "test-456"})
        self.assertEqual(found["status"], "PROCESSING")

    def test_increment_progress(self):
        job = Job(
            job_id="test-789",
            user_id="user-abc",
            processed_documents=0,
            total_documents=5,
        )
        self.coll.insert_one(job.to_dict())

        self.coll.update_one(
            {"job_id": "test-789"},
            {"$inc": {"processed_documents": 1}},
        )

        found = self.coll.find_one({"job_id": "test-789"})
        self.assertEqual(found["processed_documents"], 1)


class TestResultModel(unittest.TestCase):
    def setUp(self):
        self.client = mongomock.MongoClient()
        self.coll = self.client.db.results

    def test_create_result(self):
        result = Result(
            result_id="res-001",
            job_id="test-123",
            document_key="uploads/test-123/page001.jp2",
            page_number=1,
            extraction_model="marker",
            extracted_text="Sample text",
        )
        insert = self.coll.insert_one(result.to_dict())
        self.assertIsNotNone(insert.inserted_id)

    def test_paginated_results_query(self):
        for i in range(25):
            r = Result(
                result_id=f"res-{i:03d}",
                job_id="test-123",
                document_key=f"uploads/test-123/page{i:03d}.jp2",
                page_number=i,
            )
            self.coll.insert_one(r.to_dict())

        page1 = list(
            self.coll.find({"job_id": "test-123"})
            .sort("page_number", 1)
            .skip(0)
            .limit(10)
        )
        self.assertEqual(len(page1), 10)
        self.assertEqual(page1[0]["page_number"], 0)

        page3 = list(
            self.coll.find({"job_id": "test-123"})
            .sort("page_number", 1)
            .skip(20)
            .limit(10)
        )
        self.assertEqual(len(page3), 5)


class TestLegacyWork(unittest.TestCase):
    def setUp(self):
        self.client = mongomock.MongoClient()
        self.coll = self.client.db.works

    def test_legacy_work_model(self):
        work = Work(
            _id="test-work",
            page_count=123,
            metadata={"author": "me"},
            status="PENDING",
        )
        result = self.coll.insert_one(asdict(work))
        self.assertIsNotNone(result.inserted_id)


if __name__ == "__main__":
    unittest.main()
