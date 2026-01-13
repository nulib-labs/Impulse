from pathlib import Path
import boto3
from PIL import Image
import io
import os
from tqdm import tqdm


def convert_jpg_to_jp2_in_s3(
    source_bucket,
    dest_bucket=None,
    prefix="",
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_region=None,
):
    """
    Convert all JPG files in an S3 bucket to JP2 format.

    Args:
        source_bucket (str): Source S3 bucket name
        dest_bucket (str): Destination S3 bucket name (defaults to source_bucket)
        prefix (str): Optional prefix to filter files (e.g., 'images/')
        aws_access_key_id (str): AWS Access Key ID
        aws_secret_access_key (str): AWS Secret Access Key
        aws_region (str): AWS region (e.g., 'us-east-1')
    """
    # Initialize S3 client with credentials if provided
    if aws_access_key_id and aws_secret_access_key:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
    else:
        # Fall back to default credentials (environment variables, ~/.aws/credentials, IAM role)
        s3 = boto3.client("s3", region_name=aws_region)

    # Use source bucket as destination if not specified
    if dest_bucket is None:
        dest_bucket = source_bucket

    print("Scanning S3 bucket for JPG files...")

    # First, collect all JPG files
    jpg_files = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=source_bucket, Prefix=prefix)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg")):
                jpg_files.append(key)

    if not jpg_files:
        print(f"No JPG files found in bucket '{source_bucket}' with prefix '{prefix}'")
        return

    print(f"Found {len(jpg_files)} JPG files to convert\n")

    converted_count = 0
    error_count = 0

    # Process files with progress bar
    for key in tqdm(jpg_files, desc="Converting JPGs to JP2", unit="file"):
        try:
            # Download the JPG file from S3
            response = s3.get_object(Bucket=source_bucket, Key=key)
            img_data = response["Body"].read()

            # Open image with PIL
            img = Image.open(io.BytesIO(img_data))

            # Convert to JP2
            jp2_buffer = io.BytesIO()
            img.save(jp2_buffer, format="JPEG2000")
            jp2_buffer.seek(0)

            accession_number: str = key.split("/")[0]
            filename: str = key.split("/")[-1].replace(".jpg", ".jp2")
            filename_page: str = filename.split("_")[-1]
            # Create new key with .jp2 extension

            new_key = "/".join([accession_number, "JP2000", filename_page])
            # Upload JP2 file to destination bucket
            s3.put_object(
                Bucket=dest_bucket,
                Key=new_key,
                Body=jp2_buffer.getvalue(),
                ContentType="image/jp2",
            )
            converted_count += 1

        except Exception as e:
            tqdm.write(f"✗ Error processing {key}: {str(e)}")
            error_count += 1

    print(f"\n{'=' * 50}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Errors: {error_count} files")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # ============ AWS CREDENTIALS ============
    # Option 1: Provide credentials directly (less secure)
    AWS_ACCESS_KEY_ID = os.getenv("IMPULSE_ACCESS_KEY_ID")  # AWS Access Key ID
    AWS_SECRET_ACCESS_KEY = os.getenv(
        "IMPULSE_SECRET_ACCESS_KEY"
    )  # Your AWS Secret Access Key
    AWS_REGION = "us-east-2"  # Your AWS region

    # Option 2: Leave empty to use default credentials from:
    # - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    # - ~/.aws/credentials file
    # - IAM role (if running on EC2/ECS/Lambda)

    # ============ BUCKET CONFIGURATION ============
    SOURCE_BUCKET = "nu-impulse-production"

    # Run the conversion
    convert_jpg_to_jp2_in_s3(
        source_bucket=SOURCE_BUCKET,
        aws_access_key_id=AWS_ACCESS_KEY_ID if AWS_ACCESS_KEY_ID else None,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY if AWS_SECRET_ACCESS_KEY else None,
        aws_region=AWS_REGION if AWS_REGION else None,
    )
