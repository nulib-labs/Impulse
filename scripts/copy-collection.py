#!/usr/bin/env python3
"""Copy every document in a MongoDB collection into a collection in another database.

The source and destination can be on the same server or on different servers.
Documents are streamed in batches, so memory use stays flat for large collections.

Install:  pip install pymongo
Example:  python copy_collection.py \
              --src-uri mongodb://localhost:27017 --src-db old_db --src-coll docs \
              --dst-db new_db --copy-indexes
"""
import argparse
import sys

from pymongo import MongoClient
from pymongo.errors import BulkWriteError


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-uri", default="mongodb://localhost:27017", help="Source connection string")
    p.add_argument("--src-db", required=True, help="Source database name")
    p.add_argument("--src-coll", required=True, help="Source collection name")
    p.add_argument("--dst-uri", help="Destination connection string (defaults to --src-uri)")
    p.add_argument("--dst-db", required=True, help="Destination database name")
    p.add_argument("--dst-coll", help="Destination collection name (defaults to --src-coll)")
    p.add_argument("--batch-size", type=int, default=1000, help="Documents per insert batch")
    p.add_argument("--drop-dst", action="store_true", help="Drop the destination collection first")
    p.add_argument("--copy-indexes", action="store_true", help="Recreate source indexes on the destination")
    return p.parse_args()


def insert_batch(dst, batch):
    """Insert a batch, tolerating duplicate _ids. Returns (inserted, duplicates)."""
    try:
        res = dst.insert_many(batch, ordered=False)
        return len(res.inserted_ids), 0
    except BulkWriteError as e:
        errors = e.details.get("writeErrors", [])
        non_dup = [w for w in errors if w.get("code") != 11000]
        if non_dup:
            raise
        return e.details.get("nInserted", 0), len(errors)


def copy_indexes(src, dst):
    count = 0
    for name, info in src.index_information().items():
        if name == "_id_":
            continue
        opts = {k: v for k, v in info.items() if k not in ("v", "key", "ns")}
        dst.create_index(info["key"], **opts)
        count += 1
    return count


def main():
    args = parse_args()

    src_client = MongoClient(args.src_uri)
    dst_client = MongoClient(args.dst_uri) if args.dst_uri else src_client

    src = src_client[args.src_db][args.src_coll]
    dst = dst_client[args.dst_db][args.dst_coll or args.src_coll]

    if src.database.name == dst.database.name and src.name == dst.name and src_client.address == dst_client.address:
        sys.exit("Source and destination are the same collection; refusing to continue.")

    if args.drop_dst:
        dst.drop()

    total = src.estimated_document_count()
    print(f"Copying ~{total} documents: {args.src_db}.{args.src_coll} -> {dst.database.name}.{dst.name}")

    copied = dupes = 0
    batch = []
    # no_cursor_timeout avoids errors on long copies; the cursor is closed by the with block
    with src.find({}, no_cursor_timeout=True, batch_size=args.batch_size) as cursor:
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= args.batch_size:
                ins, dup = insert_batch(dst, batch)
                copied += ins
                dupes += dup
                batch.clear()
                print(f"  {copied} copied", end="\r", flush=True)
        if batch:
            ins, dup = insert_batch(dst, batch)
            copied += ins
            dupes += dup

    print(f"\nDone. Inserted {copied} documents, skipped {dupes} duplicate _ids.")

    if args.copy_indexes:
        print(f"Recreated {copy_indexes(src, dst)} indexes.")

    print(f"Destination now has {dst.count_documents({})} documents.")


if __name__ == "__main__":
    main()
