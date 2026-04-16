import re
import boto3
from io import BytesIO
from pymongo import MongoClient
import certifi
from tasks import config


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

def get_s3_text(s3_path: str):
    """
    Retrieve content from S3 as a string.
    
    Args:
        s3_path: S3 URI
    
    Returns:
        A string of contents
    """
    bucket, key = parse_s3_path(s3_path)
    session = boto3.Session(profile_name="impulse")
    s3_client = session.client("s3")
    buffer = BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    return buffer.read().decode("utf-8")

def _get_db():
    client = MongoClient(
        config.MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
    )
    db = client["praxis"]
    return db

def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return {str(i): stringify_keys(v) for i, v in enumerate(obj)}
    return obj
