#!/usr/bin/env python3
"""
submit_ien.py -- queue DocumentExtraction + ImageProcessing for every ien
barcode found in S3.

Expected input layout (same as the web app / impulse_cli.py):

    s3://<bucket>/jobs/ien/<barcode>/uploaded_images/<file>.tif|jp2

For each <barcode> that has at least one .tif/.jp2 in uploaded_images/, ONE
FireWorks workflow is inserted with two parallel Fireworks:

    impulse_identifier = ien_<barcode>
    spec.uploaded_files = every key in jobs/ien/<barcode>/uploaded_images/
    outputs            = jobs/ien/<barcode>/processed_images/   (image processing)
                         jobs/ien/<barcode>/outputs/document_extraction/

Nothing is executed here -- workers pick the jobs up later.

Barcodes whose impulse_identifier already exists in the FireWorks database
are skipped unless --force is given.

Environment: S3_BUCKET, AWS_PROFILE, AWS_REGION, MONGO_URI

Usage:
    submit_ien.py --dry-run          # show what would be submitted
    submit_ien.py                    # submit
    submit_ien.py --force            # resubmit barcodes that already exist
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import boto3
import certifi
from fireworks import Firework, LaunchPad, Workflow
from natsort import natsorted

import tasks

PROJECT_ID = "ien"
EXTENSIONS = (".tif", ".jp2")
IMAGES_DIR = "uploaded_images"

# (job_type, label, task class) -- both run in parallel, no links between them.
JOBS = [
    ("document_extraction", "Document Extraction", tasks.DocumentExtractionTask),
    ("image_processing", "Image Processing", tasks.ImageProcessingTask),
]


def output_prefix(project_id: str, barcode: str, job_type: str) -> str:
    """Same output layout as the web app / impulse_cli.py."""
    base = f"jobs/{project_id}/{barcode}"
    if job_type == "image_processing":
        return f"{base}/processed_images/"
    return f"{base}/outputs/{job_type}/"


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def group_keys_by_barcode(keys: list[str]) -> dict[str, list[str]]:
    """
    Group image keys by barcode. A key looks like

        jobs/ien/<barcode>/uploaded_images/<file>.tif
          [0]  [1]   [2]         [3]           [4]

    so the barcode is item 2 of the key split on "/". Only files directly in
    uploaded_images/ count, so processed_images/, outputs/ and anything nested
    deeper are never picked up as inputs.
    """
    groups: dict[str, list[str]] = defaultdict(list)

    for key in keys:
        parts = key.split("/")
        if len(parts) != 5 or parts[3] != IMAGES_DIR:
            continue
        if not parts[2] or not parts[4].lower().endswith(EXTENSIONS):
            continue

        groups[parts[2]].append(key)

    return {bc: natsorted(ks) for bc, ks in sorted(groups.items())}


def build_spec(bucket: str, barcode: str, job_type: str, label: str,
               image_keys: list[str]) -> dict:
    return {
        "job_type": job_type,
        "job_name": label,
        "s3_bucket": bucket,
        "project_id": PROJECT_ID,
        "barcode": barcode,
        "impulse_identifier": f"{PROJECT_ID}_{barcode}",
        "output_prefix": output_prefix(PROJECT_ID, barcode, job_type),
        "find_path_array_in": "uploaded_files",
        "uploaded_files": image_keys,
    }


def submit_barcode(lpad: LaunchPad, bucket: str, barcode: str,
                   image_keys: list[str]) -> dict[str, int]:
    """Insert one workflow (two parallel Fireworks). Returns {job_type: fw_id}."""
    identifier = f"{PROJECT_ID}_{barcode}"

    fireworks: dict[str, Firework] = {}
    for job_type, label, task_class in JOBS:
        fireworks[job_type] = Firework(
            task_class(),
            spec=build_spec(bucket, barcode, job_type, label, image_keys),
            name=f"job-{identifier}-{job_type}",
        )

    workflow = Workflow(list(fireworks.values()), name=f"impulse-{identifier}")

    # add_wf returns {temporary fw_id: database fw_id} and reassigns each
    # fw.fw_id in place, so capture the temporary ids first.
    temp_ids = {jt: fw.fw_id for jt, fw in fireworks.items()}
    id_map = lpad.add_wf(workflow)
    return {jt: id_map[temp_id] for jt, temp_id in temp_ids.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print what would be submitted; touch nothing")
    parser.add_argument("--force", action="store_true",
                        help="Submit even if ien_<barcode> already exists in FireWorks")
    args = parser.parse_args()

    prefix = f"jobs/{PROJECT_ID}/"

    bucket = os.environ["S3_BUCKET"]
    session = boto3.Session(
        profile_name=os.environ["AWS_PROFILE"],
        region_name=os.environ["AWS_REGION"],
    )
    s3 = session.client("s3")

    keys = list_keys(s3, bucket, prefix)
    groups = group_keys_by_barcode(keys)
    print(f"Found {sum(map(len, groups.values()))} image(s) across "
          f"{len(groups)} barcode(s) under s3://{bucket}/{prefix}")

    if not groups:
        return 0

    lpad = LaunchPad(
        uri_mode=True,
        host=os.environ["MONGO_URI"],
        name="fireworks",
        mongoclient_kwargs={"tlsCAFile": certifi.where()},
    )

    # One query up front instead of one per barcode.
    existing = set(lpad.fireworks.distinct(
        "spec.impulse_identifier", {"spec.project_id": PROJECT_ID}
    ))

    submitted = skipped = failed = 0

    for barcode, image_keys in groups.items():
        identifier = f"{PROJECT_ID}_{barcode}"

        if identifier in existing and not args.force:
            print(f"[{identifier}] skipped: already exists in FireWorks "
                  f"(--force to override)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[{identifier}] would submit {len(image_keys)} image(s)")
            submitted += 1
            continue

        try:
            fw_ids = submit_barcode(lpad, bucket, barcode, image_keys)
        except Exception as exc:  # keep going; report at the end
            print(f"[{identifier}] FAILED: {exc}", file=sys.stderr)
            failed += 1
            continue

        print(f"[{identifier}] submitted {len(image_keys)} image(s): {fw_ids}")
        submitted += 1

    verb = "would be submitted" if args.dry_run else "submitted"
    print(f"Done: {submitted} {verb}, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
