from urllib.parse import urlparse
import io
import boto3
from io import BytesIO
from PIL import Image

def download_s3_file(s3_uri: str):
    """
        `download_s3_file` is a helper function that takes as input an S3 URI,
        extracts the bucket and key using urllib, and returns the response of
        the `get_object` method on the s3 client object.

        returns: bytes?
    """

    session = boto3.Session(profile_name='impulse')
    s3 = session.client('s3')
    parsed_path = urlparse(s3_uri)
    bucket = parsed_path.netloc
    object_key = parsed_path.path.lstrip('/')
    response = s3.get_object(Bucket=bucket, Key=object_key)
    image = Image.open(BytesIO(response['Body'].read()))

    return image

def upload_pil_image_to_s3(
    image: Image.Image,
    bucket: str,
    key: str,
    region: str = "us-east-1",
) -> str:
    """
    Upload a PIL Image to S3 as a JPEG.

    Args:
        image:   PIL Image object (any mode/format)
        bucket:  S3 bucket name
        key:     S3 object key, e.g. "images/photo.jpg"
        region:  AWS region (used to build the public URL)

    Returns:
        The S3 URL of the uploaded object.
    """
    # 1. Convert PIL Image → in-memory JPEG bytes
    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")   # JPEG doesn't support alpha channels
    rgb_image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    # 2. Upload to S3
    session = boto3.Session(profile_name='impulse')
    s3 = session.client("s3", region_name=region)
    s3.upload_fileobj(
        buffer,
        bucket,
        key,
        ExtraArgs={"ContentType": "image/jpeg"},
    )

    url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    print(f"Uploaded → {url}")
    return url

def s3_key_exists(bucket: str, key: str) -> bool:
    session = boto3.Session(profile_name='impulse')
    s3 = session.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise
