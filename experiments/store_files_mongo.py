#!/usr/bin/env python3
"""
Iterates over all documents in the MongoDB 'pages' collection,
fetches the corresponding JP2 image from S3, stores the raw JP2
bytes and a converted JPG as BSON Binary fields on each document.

Usage:
    pip install pymongo boto3 Pillow
    MONGO_URI="mongodb+srv://..." python migrate_images.py

Environment variables:
    MONGO_URI       (required) MongoDB connection string
    MONGO_DB        (optional) Database name, default: "impulse"
    AWS_PROFILE     (optional) Overrides the default "impulse" profile
    DRY_RUN         (optional) Set to "1" to log S3 keys without writing
"""

import io
import os
import sys
import logging
import boto3
from bson.binary import Binary
from PIL import Image
from pymongo import MongoClient, UpdateOne

# ── Configuration ────────────────────────────────────────────────────────────

MONGO_URI   = os.environ.get("MONGODB_OCR_DEVELOPMENT_CONN_STRING_IMPULSE")
MONGO_DB    = os.environ.get("DB", "praxis")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "impulse")
S3_BUCKET   = "nu-impulse-production"
S3_PREFIX   = "DATA"
DRY_RUN     = os.environ.get("DRY_RUN", "0") == "1"
BATCH_SIZE  = 50   # documents written per bulk_write call
JPG_QUALITY = 90   # JPEG quality (1-95)

# ── Logging ──────────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def build_s3_key(accession_number: str, page_number) -> str:
    """
    nu-impulse-production/DATA/<ACCESSION_UPPER>/<PAGE_8ZEROS>.JP2
    e.g. DATA/ABC1234/00000001.JP2
    """
    acc = str(accession_number).upper()
    page = str(page_number).zfill(8)
    return f"{S3_PREFIX}/{acc}/{page}.JP2"
 
 
def fetch_s3_bytes(s3_client, key: str) -> bytes:
    """Download object from S3 and return raw bytes."""
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    return response["Body"].read()
 
 
def jp2_to_jpg_bytes(jp2_bytes: bytes, quality: int = JPG_QUALITY) -> bytes:
    """Convert JP2 bytes → JPEG bytes using Pillow."""
    with Image.open(io.BytesIO(jp2_bytes)) as img:
        # JP2 can be RGBA; JPEG does not support alpha
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
 
 
# ── Main ─────────────────────────────────────────────────────────────────────
 
def main():
    if not MONGO_URI:
        log.error("MONGO_URI environment variable is not set.")
        sys.exit(1)
 
    # AWS session using the named profile
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client("s3")
    log.info("AWS session ready (profile: %s, bucket: %s)", AWS_PROFILE, S3_BUCKET)
 
    # MongoDB connection
    client = MongoClient(MONGO_URI)
    collection = client[MONGO_DB]["pages"]
    total_docs = collection.count_documents({})    
    pending_docs = collection.count_documents({"JP2": {"$exists": False}, "JPG": {"$exists": False}})
    log.info("MongoDB connected — %d total documents, %d pending in '%s.pages'", total_docs, pending_docs, MONGO_DB)
 
    if DRY_RUN:
        log.warning("DRY_RUN mode: S3 keys will be logged but nothing written to Mongo.")
 
    processed = skipped = errors = 0
    bulk_ops = []
 
    cursor = collection.find(
        {"JP2": {"$exists": False}, "JPG": {"$exists": False}},
        {"_id": 1, "accession_number": 1, "page_number": 1}
    )
 
    for doc in cursor:
        doc_id = doc["_id"]
        accession = doc.get("accession_number")
        page = doc.get("page_number")
 
        if accession is None or page is None:
            log.warning("Skipping _id=%s — missing accession_number or page_number", doc_id)
            skipped += 1
            continue
 
        s3_key = build_s3_key(accession, int(page) + 1)
 
        if DRY_RUN:
            log.info("[DRY RUN] _id=%s  →  s3://%s/%s", doc_id, S3_BUCKET, s3_key)
            processed += 1
            continue
 
        try:
            log.info("Fetching s3://%s/%s", S3_BUCKET, s3_key)
            jp2_bytes = fetch_s3_bytes(s3, s3_key)
            jpg_bytes = jp2_to_jpg_bytes(jp2_bytes)
 
            bulk_ops.append(UpdateOne(
                {"_id": doc_id},
                {"$set": {
                    "JP2": Binary(jp2_bytes),
                    "JPG": Binary(jpg_bytes),
                }}
            ))
            processed += 1
 
        except s3.exceptions.NoSuchKey:
            try:
                jp2_bytes = fetch_s3_bytes(s3, s3_key.replace(".JP2", ".jp2"))
                jpg_bytes = jp2_to_jpg_bytes(jp2_bytes)
    
                bulk_ops.append(UpdateOne(
                    {"_id": doc_id},
                    {"$set": {
                        "JP2": Binary(jp2_bytes),
                        "JPG": Binary(jpg_bytes),
                    }}
                ))
                processed += 1
            except s3.exceptions.NoSuchKey:
                log.error("S3 key not found: %s  (skipping)", s3_key)
                errors += 1
        except Exception as exc:
            log.error("Error on _id=%s: %s", doc_id, exc)
            errors += 1
 
        # Flush batch to MongoDB
        if len(bulk_ops) >= BATCH_SIZE:
            result = collection.bulk_write(bulk_ops, ordered=False)
            log.info("Wrote batch of %d  (modified: %d)", len(bulk_ops), result.modified_count)
            bulk_ops.clear()
 
    # Final flush
    if bulk_ops:
        result = collection.bulk_write(bulk_ops, ordered=False)
        log.info("Wrote final batch of %d  (modified: %d)", len(bulk_ops), result.modified_count)
 
    log.info(
        "Done — processed: %d  |  skipped: %d  |  errors: %d  |  total: %d",
        processed, skipped, errors, total_docs,
    )
    client.close()
 
 
if __name__ == "__main__":
    main()
