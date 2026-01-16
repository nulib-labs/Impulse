import boto3
import random
import os
import argparse
from pathlib import Path


def get_txt_keys(bucket_name, s3_client, max_keys=None):
    """
    Retrieve all keys ending in .TXT from the S3 bucket.

    Args:
        bucket_name: Name of the S3 bucket
        s3_client: Boto3 S3 client
        max_keys: Maximum number of keys to retrieve (None for all)

    Returns:
        List of S3 keys ending in .TXT
    """
    s3 = s3_client
    txt_keys = []

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.upper().endswith(".TXT"):
                txt_keys.append(key)

                if max_keys and len(txt_keys) >= max_keys:
                    return txt_keys

    return txt_keys


def download_jpg_files(bucket_name, txt_keys, s3_client, download_dir="downloads"):
    """
    Download JPG files corresponding to the TXT keys.

    Args:
        bucket_name: Name of the S3 bucket
        txt_keys: List of TXT file keys
        s3_client: Boto3 S3 client
        download_dir: Local directory to save downloads
    """
    s3 = s3_client

    # Create download directory if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0

    for txt_key in txt_keys:
        # Replace .TXT extension with .JPG
        jgp_key = txt_key.replace("TXT", "jpg")
        jpg_key = (
            txt_key[:-4] + ".JPG"
            if txt_key.upper().endswith(".TXT")
            else txt_key + ".JPG"
        )

        # Create local file path
        local_filename = os.path.join(download_dir, os.path.basename(jpg_key))

        try:
            print(f"Downloading: {jpg_key}")
            s3.download_file(bucket_name, jpg_key, local_filename)
            downloaded += 1
            print(f"  ✓ Saved to: {local_filename}")
        except s3.exceptions.NoSuchKey:
            print(f"  ✗ JPG file not found: {jpg_key}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error downloading {jpg_key}: {str(e)}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Total attempted: {len(txt_keys)}")
    print(f"{'=' * 50}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download random JPG files from S3 based on TXT file keys"
    )
    parser.add_argument("--bucket", "-b", required=True, help="S3 bucket name")
    parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help="AWS profile name to use for authentication",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=100,
        help="Number of random files to select (default: 100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="downloaded_images",
        help="Output directory for downloads (default: downloaded_images)",
    )

    args = parser.parse_args()

    # Create boto3 session with profile if provided
    if args.profile:
        print(f"Using AWS profile: {args.profile}")
        session = boto3.Session(profile_name=args.profile)
    else:
        print("Using default AWS credentials")
        session = boto3.Session()

    s3_client = session.client("s3")

    print(f"Fetching .TXT keys from bucket: {args.bucket}")

    # Get all TXT keys
    all_txt_keys = get_txt_keys(args.bucket, s3_client)

    print(f"Found {len(all_txt_keys)} .TXT files in the bucket")

    if len(all_txt_keys) == 0:
        print("No .TXT files found in the bucket!")
        return

    # Select random keys
    num_to_select = min(args.count, len(all_txt_keys))
    selected_keys = random.sample(all_txt_keys, num_to_select)

    print(f"Selected {num_to_select} random .TXT keys")
    print(f"\nDownloading corresponding .JPG files to '{args.output}/'...\n")

    # Download corresponding JPG files
    download_jpg_files(args.bucket, selected_keys, s3_client, args.output)


if __name__ == "__main__":
    main()
