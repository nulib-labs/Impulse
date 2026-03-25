import logging
from turtle import update
import unittest
import mongomock
from impulse.db import Work
from dataclasses import asdict
logger = logging.getLogger(__name__)

mongo = mongomock.MongoClient()
coll = mongo.db.collection

class TestDatabase(unittest.TestCase):
    def test_db(self):
        data = Work(_id="The test work", page_count=123, metadata={"author": "me"}, status="PENDING")
        
        result = coll.insert_one(asdict(data))  
        
        self.assertIsNotNone(result.inserted_id)

    def test_update_status(self):

        outdated_data = Work(_id="The test work", page_count=123, metadata={"author": "me"}, status="PENDING")
        updated_data =  Work(_id="The test work", page_count=123, metadata={"author": "me"}, status="PROCESSING")
        update_result = coll.update_one(asdict(outdated_data), {"$set": {"status" : "PROCESSING"}})
        print(update_result)
        self.assertIsNotNone(update_result)

if __name__ == "__main__":
    unittest.main()
