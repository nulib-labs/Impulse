import re
import boto3
from io import BytesIO
from pymongo import MongoClient
import os
import certifi

def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse S3 path into bucket and key.

    Args:
        s3_path: S3 URI in format s3://bucket/key or s3a://bucket/key

    Returns:
        Tuple of (bucket, key)
    """
    # Remove s3:// or s3a:// prefix
    path = re.sub(r"^s3a?://", "", s3_path)
    # Split into bucket and key
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def get_s3_content(s3_path: str) -> bytes:
    """
    Retrieve content from S3.

    Args:
        s3_path: S3 URI

    Returns:
        File content as bytes
    """
    bucket, key = parse_s3_path(s3_path)

    session = boto3.Session(profile_name="impulse")
    s3_client = session.client("s3")

    # Download file content
    buffer = BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)

    return buffer.read()


def _get_db():
    client = MongoClient(
        os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING_IMPULSE"),
        tls=True,
        tlsCAFile=certifi.where(),
    )
    db = client["praxis"]
    return db
