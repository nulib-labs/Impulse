"""Backfill IOTask + ImageProcessingTask workflows for every impulse_identifier
that already has DocumentExtractionTask output in praxis.colt.

For each such identifier we:
  1. Look up the identifier's S3 prefix under nu-impulse-production.
  2. Collect its .jpg source keys.
  3. Submit a Workflow: IOTask -> ImageProcessingTask.

DocumentExtractionTask is NOT submitted, because the extraction data is
already persisted in praxis.colt.
"""

from fireworks import Firework, LaunchPad, Workflow
from pymongo import MongoClient
from tasks.my_tasks import IOTask, ImageProcessingTask
import boto3
import os
import certifi

DEBUG = False

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

# ---------------------------------------------------------------------------
# LaunchPad, Mongo, S3 setup
# ---------------------------------------------------------------------------
launchpad: LaunchPad = LaunchPad(
    uri_mode=True,
    host=MONGO_URI,
    name="fireworks",
    mongoclient_kwargs={"tlsCAFile": certifi.where()},
)

praxis_db = MongoClient(MONGO_URI, tlsCAFile=certifi.where())["praxis"]

impulse_identifiers: list[str] = [
    x
    for x in praxis_db["colt"].distinct("impulse_identifier")
    if isinstance(x, str) and x.strip()
]
print(f"Found {len(impulse_identifiers)} identifiers in praxis.colt")

session = boto3.Session(profile_name="impulse")
s3_client = session.client("s3", region_name="us-west-2")
paginator = s3_client.get_paginator("list_objects_v2")

# ---------------------------------------------------------------------------
# Resolve each identifier to its S3 prefix under nu-impulse-production
# ---------------------------------------------------------------------------
# The identifiers in praxis.colt are lowercased "<project>_<accession>", but
# the S3 prefixes under nu-impulse-production are uppercase (e.g.
# "P0491_35556036063543/"). We list the top-level prefixes once and match
# case-insensitively.
top_level_prefixes: list[str] = []
paginator_delim = s3_client.get_paginator("list_objects_v2")
for page in paginator_delim.paginate(
    Bucket="nu-impulse-production", Delimiter="/"
):
    for cp in page.get("CommonPrefixes", []) or []:
        top_level_prefixes.append(cp["Prefix"])  # e.g. "P0491_35556036063543/"

prefix_by_identifier: dict[str, str] = {}
for p in top_level_prefixes:
    key = (
        p.rstrip("/").replace("{", "").replace("}", "").replace("'", "").lower()
    )
    prefix_by_identifier[key] = p

print(f"Discovered {len(prefix_by_identifier)} top-level prefixes on S3")

# ---------------------------------------------------------------------------
# Submit backfill workflows
# ---------------------------------------------------------------------------
submitted = 0
skipped_no_prefix = 0
skipped_no_keys = 0

for impulse_identifier in impulse_identifiers:
    prefix = prefix_by_identifier.get(impulse_identifier)
    if prefix is None:
        print(f"[skip] {impulse_identifier}: no matching S3 prefix")
        skipped_no_prefix += 1
        continue

    impulse_keys: list[str] = []
    for page in paginator.paginate(
        Bucket="nu-impulse-production", Prefix=prefix
    ):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].endswith("jpg"):
                impulse_keys.append(f"s3://nu-impulse-production/{obj['Key']}")

    if not impulse_keys:
        print(f"[skip] {impulse_identifier}: no .jpg source keys under {prefix}")
        skipped_no_keys += 1
        continue

    common_spec = {
        "impulse_identifier": impulse_identifier,
        "find_path_array_in": "keys",
        "keys": impulse_keys,
    }

    io_fw: Firework = Firework(
        IOTask(),
        spec=common_spec,
        name="I/O Workflow",
    )
    ip_fw: Firework = Firework(
        ImageProcessingTask(),
        spec=common_spec,
        name="Image Processing Task",
    )

    wf = Workflow(
        [io_fw, ip_fw],
        {io_fw: [ip_fw]},
        name=impulse_identifier,
    )
    launchpad.add_wf(wf)
    submitted += 1
    print(
        f"[BACKFILL] {impulse_identifier}: IO -> ImageProc "
        f"({len(impulse_keys)} keys)"
    )

print(
    f"\nDone. submitted={submitted} "
    f"skipped_no_prefix={skipped_no_prefix} "
    f"skipped_no_keys={skipped_no_keys}"
)
