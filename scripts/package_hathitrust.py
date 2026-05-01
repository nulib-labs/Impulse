#!/usr/bin/env python3
"""Package HathiTrust submission files from Impulse pipeline data.

Queries MongoDB and S3 for a given impulse_identifier, then assembles a zip
file with the HathiTrust ingest structure:

    {identifier}.zip
    ├── JP2000/
    │   ├── 000000001.JP2
    │   ├── 000000002.JP2
    │   └── ...
    ├── TEXT/
    │   ├── 000000001.TXT
    │   ├── 000000002.TXT
    │   └── ...
    └── mets.yaml

Data sources:
    - JP2 images:  S3  nu-impulse-production/DATA/{IDENTIFIER}/*.jp2
    - Page text:   MongoDB  praxis.colt  (extracted_data.html → plain text)
    - METS YAML:   MongoDB  praxis.HathiTrust  (raw_yaml field)

Usage examples:
    python scripts/package_hathitrust.py p1274_35556036056489
    python scripts/package_hathitrust.py p1274_35556036056489 --prod
    python scripts/package_hathitrust.py p1274_35556036056489 --output my_package.zip
"""

import argparse
import os
import re
import sys
import textwrap
import zipfile
from io import BytesIO

import boto3
import certifi
from bs4 import BeautifulSoup
from natsort import natsorted
from pymongo import MongoClient

S3_BUCKET = "nu-impulse-production"
S3_DATA_PREFIX = "DATA"
DATABASE = "praxis"
HATHITRUST_COLLECTION = "HathiTrust"
COLT_COLLECTION = "colt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package HathiTrust submission files from Impulse pipeline data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s p1274_35556036056489
              %(prog)s p1274_35556036056489 --prod
              %(prog)s p1274_35556036056489 --output my_package.zip
        """),
    )
    parser.add_argument(
        "identifier",
        help="The impulse_identifier of the document to package.",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        default=False,
        help="Use the production database instead of the debug/staging database.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output zip file path. Defaults to {identifier}.zip in the current directory.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_mongo_uri(production: bool) -> str:
    """Resolve the MongoDB connection URI from environment variables."""
    if production:
        uri = os.getenv("IMPULSE_MONGODB_URI")
        label = "production"
    else:
        uri = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
        label = "debug/staging"

    if not uri:
        env_var = "IMPULSE_MONGODB_URI" if production else "IMPULSE_MONGODB_URI_DEBUG"
        print(
            f"Error: {env_var} is not set. Cannot connect to the {label} database.",
            file=sys.stderr,
        )
        sys.exit(1)

    return uri


def get_db(uri: str):
    """Return the praxis database handle."""
    client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
    return client[DATABASE]


# ---------------------------------------------------------------------------
# METS YAML
# ---------------------------------------------------------------------------

def fetch_mets_yaml(db, identifier: str) -> str:
    """Fetch the pre-generated METS YAML from the HathiTrust collection.

    Returns the raw_yaml string. Exits with an error if not found.
    """
    doc = db[HATHITRUST_COLLECTION].find_one(
        {"impulse_identifier": identifier},
        {"raw_yaml": 1},
    )
    if not doc or "raw_yaml" not in doc:
        print(
            f"Error: No HathiTrust record found for identifier '{identifier}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return doc["raw_yaml"]


# ---------------------------------------------------------------------------
# Page text
# ---------------------------------------------------------------------------

def strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace into plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["math", "script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def find_text_values(d, results=None):
    """Recursively extract HTML strings from extracted_data tree.

    Mirrors EmbeddingTask.find_text_values in tasks/my_tasks.py.
    """
    if results is None:
        results = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "html" and isinstance(v, str) and v.strip():
                results.append(v)
            else:
                find_text_values(v, results)
    elif isinstance(d, list):
        for item in d:
            find_text_values(item, results)
    return results


def fetch_page_texts(db, identifier: str) -> list[str]:
    """Fetch OCR text for every page from the colt collection.

    Returns a list of plain-text strings ordered by page_number.
    Exits with an error if no pages are found.
    """
    cursor = db[COLT_COLLECTION].find(
        {"impulse_identifier": identifier},
        {"page_number": 1, "extracted_data": 1},
    ).sort("page_number", 1)

    pages: list[str] = []
    for doc in cursor:
        html_blocks = find_text_values(doc.get("extracted_data", {}))
        combined = " ".join(strip_html(h) for h in html_blocks if h)
        pages.append(combined)

    if not pages:
        print(
            f"Error: No pages found in colt collection for identifier '{identifier}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return pages


# ---------------------------------------------------------------------------
# S3 JP2 discovery and download
# ---------------------------------------------------------------------------

def get_s3_client():
    """Create an S3 client using the impulse profile."""
    session = boto3.Session(profile_name="impulse")
    return session.client("s3", region_name="us-west-2")


def list_jp2_keys(s3_client, identifier: str) -> list[str]:
    """List JP2 file keys under DATA/{IDENTIFIER}/ in the S3 bucket.

    The ImageProcessingTask stores processed JP2s at
    s3://nu-impulse-production/DATA/{IDENTIFIER_UPPER}/{filename}.jp2

    Returns a naturally-sorted list of S3 keys. Exits if none found.
    """
    prefix = f"{S3_DATA_PREFIX}/{identifier.upper()}/"
    paginator = s3_client.get_paginator("list_objects_v2")

    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".jp2"):
                keys.append(key)

    keys = natsorted(keys)

    if not keys:
        print(
            f"Error: No JP2 files found in s3://{S3_BUCKET}/{prefix}",
            file=sys.stderr,
        )
        sys.exit(1)

    return keys


def download_s3_file(s3_client, key: str) -> bytes:
    """Download a single file from S3 and return its contents as bytes.

    Exits with an error if the download fails.
    """
    try:
        buf = BytesIO()
        s3_client.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        print(
            f"Error: Failed to download s3://{S3_BUCKET}/{key}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Zip assembly
# ---------------------------------------------------------------------------

def build_zip(
    jp2_data: list[bytes],
    texts: list[str],
    mets_yaml: str,
) -> bytes:
    """Assemble the HathiTrust zip package in memory.

    Structure:
        JP2000/000000001.JP2, 000000002.JP2, ...
        TEXT/000000001.TXT, 000000002.TXT, ...
        mets.yaml
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        # JP2 images
        for i, data in enumerate(jp2_data, start=1):
            filename = f"JP2000/{i:09d}.JP2"
            zf.writestr(filename, data)

        # Text files
        for i, text in enumerate(texts, start=1):
            filename = f"TEXT/{i:09d}.TXT"
            zf.writestr(filename, text)

        # METS YAML
        zf.writestr("mets.yaml", mets_yaml)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    identifier = args.identifier
    output_path = args.output or f"{identifier}.zip"
    env_label = "production" if args.prod else "debug/staging"

    print(f"Identifier:  {identifier}")
    print(f"Environment: {env_label}")
    print(f"Output:      {output_path}")
    print()

    # --- MongoDB ---
    uri = get_mongo_uri(args.prod)
    db = get_db(uri)

    # 1. METS YAML
    print("Fetching METS YAML from MongoDB...")
    mets_yaml = fetch_mets_yaml(db, identifier)
    print(f"  METS YAML: {len(mets_yaml)} bytes")

    # 2. Page texts
    print("Fetching page texts from MongoDB...")
    texts = fetch_page_texts(db, identifier)
    print(f"  Pages: {len(texts)}")

    # --- S3 ---
    s3_client = get_s3_client()

    # 3. Discover JP2 files
    print("Listing JP2 files in S3...")
    jp2_keys = list_jp2_keys(s3_client, identifier)
    print(f"  JP2 files: {len(jp2_keys)}")

    # 4. Download JP2 files
    print("Downloading JP2 files from S3...")
    jp2_data: list[bytes] = []
    for i, key in enumerate(jp2_keys, start=1):
        filename = key.rsplit("/", 1)[-1]
        print(f"  [{i}/{len(jp2_keys)}] {filename}")
        jp2_data.append(download_s3_file(s3_client, key))

    # --- Build zip ---
    print("\nAssembling zip file...")
    zip_bytes = build_zip(jp2_data, texts, mets_yaml)

    with open(output_path, "wb") as f:
        f.write(zip_bytes)

    size_mb = len(zip_bytes) / (1024 * 1024)
    print(f"\nDone. Wrote {output_path} ({size_mb:.1f} MB)")
    print(f"  JP2000/: {len(jp2_data)} files")
    print(f"  TEXT/:   {len(texts)} files")
    print(f"  mets.yaml: included")


if __name__ == "__main__":
    main()
