from fnmatch import fnmatch
from fireworks.core.launchpad import LaunchPad
from fireworks.fw_config import os
from fireworks.utilities.filepad import FilePad
from fireworks.core.firework import Firework, Workflow
from fireworks.user_objects.firetasks.script_task import PyTask
from my_pads import fp, lp
import pandas as pd
import os
import boto3
from tqdm import tqdm
from pathlib import Path

# Global vars
conn_str: str
conn_str = str(os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING"))
url = os.getenv("MONGODB_OCR_DEVELOPMENT_CONN_STRING")
fp = fp
lp = lp

s3_impulse = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("IMPULSE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("IMPULSE_SECRET_ACCESS_KEY"),
)


def find_txt_files(s3_client, bucket_name):
    """
    Find all files matching */TXT/*.txt pattern in an S3 bucket.

    Args:
        s3_client: boto3 S3 client instance
        bucket_name: Name of the S3 bucket to search

    Returns:
        List of S3 object keys matching the pattern
    """
    matching_files = []

    # Use paginator to handle buckets with many objects
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            # Check if the key matches the pattern */TXT/*.txt
            if fnmatch(key, "*/TXT/*.txt"):
                matching_files.append(key)

    return matching_files


def create_fireworks(s3_keys):
    fws = []
    for i, f in tqdm(
        enumerate(sorted(s3_keys)), desc="Creating fireworks", total=len(s3_keys)
    ):
        keys_list = f.split("/")
        accession_number = keys_list[0]
        filename = keys_list[-1]

        # If f is an image path
        spec = {
            "source_s3_key": f,
            "file_name": filename,
            "accession_number": accession_number,
        }
        fw = Firework(
            tasks=PyTask(
                func="auxiliary.spacy_experiment",
                inputs=["source_s3_key", "file_name", "accession_number"],
            ),
            spec=spec,
            name=f"Convert METSXML to YAML",
        )
        fws.append(fw)
        print(f"Created firework with spec: {spec}")

    wf = Workflow(
        fws,
        metadata={},
        name="Run bulk NER",
        links_dict={},
    )

    lp.add_wf(wf)


# Example usage
if __name__ == "__main__":
    # Assuming you have s3_impulse client already configured
    # s3_impulse = boto3.client('s3')

    bucket_name = "nu-impulse-production"

    try:
        txt_files = find_txt_files(s3_impulse, bucket_name)
        print(f"Found {len(txt_files)} matching files:")
        create_fireworks(txt_files)
    except Exception as e:
        print(f"Error: {e}")
