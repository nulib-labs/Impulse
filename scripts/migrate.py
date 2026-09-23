"""
For every unique impulse_identifier in a MongoDB collection, write one JSON file
per page to S3:

    s3://<BUCKET>/jobs/<project_number>/<barcode>/<page_number, 10 digits>.json

impulse_identifier looks like "p1074_35556032756942":
    project_number = "p1074"            (before the first underscore)
    barcode        = "35556032756942"   (after the first underscore)
"""
import os

import boto3
from bson import json_util
from pymongo import MongoClient
from tqdm import tqdm
import re
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "my_db")
COLLECTION = os.environ.get("MONGO_COLLECTION", "my_collection")
BUCKET = os.environ.get("S3_BUCKET", "my-bucket")
PREFIX = os.environ.get("S3_PREFIX", "jobs").strip("/")
IDENTIFIER_STARTS_WITH = "p1274"
FIELDS = ("ocr_data", "layout_data", "extraction_model")


def unique_identifiers(coll):
    pipeline = [
        {"$match": {"impulse_identifier": {"$regex": "^" + re.escape(IDENTIFIER_STARTS_WITH)}}},
        {"$group": {"_id": "$impulse_identifier"}},
    ]
    for row in coll.aggregate(pipeline, allowDiskUse=True):
        yield row["_id"]


def split_identifier(identifier):
    project_number, sep, barcode = identifier.partition("_")
    if not sep or not project_number or not barcode:
        return None
    return project_number.lower(), barcode.lower()


def build_page(project_number, barcode, doc):
    return {
        "project_number": project_number,
        "barcode": barcode,
        "page_number": doc["page_number"],
        **{f: doc.get(f) for f in FIELDS},
    }


def main():
    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][COLLECTION]
    s3 = boto3.client("s3")

    coll.create_index([("impulse_identifier", 1), ("page_number", 1)])

    identifiers = pages_written = 0
    for identifier in tqdm(unique_identifiers(coll)):
        parts = split_identifier(identifier)
        if parts is None:
            print(f"SKIP {identifier!r}: expected '<project>_<barcode>'")
            continue
        project_number, barcode = parts

        projection = {"_id": 0, "page_number": 1, **{f: 1 for f in FIELDS}}
        docs = coll.find({"impulse_identifier": identifier}, projection).sort("page_number", 1)

        for doc in docs:
            page = doc.get("page_number")
            if page is None:
                print(f"SKIP a doc in {identifier}: no page_number")
                continue

            key = f"{PREFIX}/{project_number.lower()}/{barcode.lower()}/{int(page):010d}.json"
            body = json_util.dumps(build_page(project_number, barcode, doc)).encode("utf-8")
            s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
            pages_written += 1

        identifiers += 1
        print(f"Exported {identifier}")

    print(f"Done. {identifiers} identifiers, {pages_written} pages exported.")


if __name__ == "__main__":
    main()
