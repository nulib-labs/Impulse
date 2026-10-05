#!/usr/bin/env python3
"""
Download every document from one MongoDB collection and write it to a local file.

Install:  pip install pymongo
Usage:    python mongo_dump.py --uri "mongodb+srv://user:pass@cluster.mongodb.net" \
              --db mydb --collection mycoll --out mycoll.jsonl

The URI can also come from the MONGODB_URI environment variable, which keeps
credentials out of your shell history.
"""
import argparse
import os
import sys

from bson import json_util
from pymongo import MongoClient


def main():
    p = argparse.ArgumentParser(description="Export a MongoDB collection to a file.")
    p.add_argument("--uri", default=os.environ.get("MONGO_URI"),
                   help="Mongo connection URI (or set MONGO_URI)")
    p.add_argument("--db", required=True, help="Database name")
    p.add_argument("--collection", required=True, help="Collection name")
    p.add_argument("--out", help="Output path (default: <collection>.jsonl or .json)")
    p.add_argument("--format", choices=["jsonl", "json"], default="jsonl",
                   help="jsonl = one document per line (default, streams well); "
                        "json = a single JSON array")
    p.add_argument("--batch-size", type=int, default=1000, help="Cursor batch size")
    p.add_argument("--filter", default="{}",
                   help='Optional Mongo filter as JSON, e.g. \'{"status": "active"}\'')
    args = p.parse_args()

    if not args.uri:
        sys.exit("error: provide --uri or set MONGODB_URI")

    out_path = args.out or f"{args.collection}.{args.format}"
    query = json_util.loads(args.filter)

    client = MongoClient(args.uri, serverSelectionTimeoutMS=10_000)
    coll = client[args.db][args.collection]

    # Fail fast on bad credentials / network before starting the export.
    client.admin.command("ping")

    total = coll.estimated_document_count() if not query else coll.count_documents(query)
    print(f"Exporting ~{total:,} documents from {args.db}.{args.collection} -> {out_path}")

    written = 0
    # no_cursor_timeout avoids the 10-minute idle cursor expiry on large collections.
    with coll.find(query, no_cursor_timeout=True, batch_size=args.batch_size) as cursor, \
            open(out_path, "w", encoding="utf-8") as f:
        if args.format == "json":
            f.write("[\n")
        for doc in cursor:
            # json_util handles ObjectId, datetime, Decimal128, Binary, etc.
            line = json_util.dumps(doc, ensure_ascii=False)
            if args.format == "json":
                f.write(("," if written else "") + line + "\n")
            else:
                f.write(line + "\n")
            written += 1
            if written % 10_000 == 0:
                print(f"  {written:,} / ~{total:,}", end="\r", flush=True)
        if args.format == "json":
            f.write("]\n")

    print(f"\nDone. Wrote {written:,} documents to {out_path}")
    client.close()


if __name__ == "__main__":
    main()
