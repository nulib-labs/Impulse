import re
from PIL import Image
import numpy as np
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


def get_s3_content(s3_path: str):
    """
    Retrieve content from S3 as a PIL Image.
    
    Args:
        s3_path: S3 URI
    
    Returns:
        PIL Image object
    """
    bucket, key = parse_s3_path(s3_path)
    session = boto3.Session(profile_name="impulse")
    s3_client = session.client("s3")

    buffer = BytesIO()
    s3_client.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    from PIL import Image
    image = Image.open(buffer)
    w, h = image.size
    new_w = 800
    new_h = int(h * (new_w / w))
    out = image.resize((new_w, new_h), Image.LANCZOS)
    print(out.size)
    out.show()
    return image

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
        os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING_IMPULSE"),
        tls=True,
        tlsCAFile=certifi.where(),
    )
    db = client["praxis"]
    return db
