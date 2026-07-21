from fireworks import Firework, LaunchPad, Workflow
from pymongo import MongoClient
from tasks.my_tasks import DocumentExtractionTask, IOTask, ImageProcessingTask
from tasks.config import MONGO_URI
import boto3
import subprocess
import os
import certifi

DEBUG = False

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

# ---------------------------------------------------------------------------
# One-time setup: LaunchPad, Mongo, S3
# ---------------------------------------------------------------------------
launchpad: LaunchPad = LaunchPad(
    uri_mode=True,
    host=MONGO_URI,
    name="fireworks",
    mongoclient_kwargs={"tlsCAFile": certifi.where()},
)

# Set of impulse_identifiers that already have DocumentExtractionTask output
# persisted in praxis.colt. Anything in this set is treated as an "existing"
# document and only receives the IO -> ImageProcessing backfill pipeline.
praxis_db = MongoClient(MONGO_URI, tlsCAFile=certifi.where())["praxis"]
already_extracted: set[str] = {
    x
    for x in praxis_db["colt"].distinct("impulse_identifier")
    if isinstance(x, str) and x.strip()
}
print(f"Found {len(already_extracted)} identifiers already in praxis.colt")

session = boto3.Session(profile_name="impulse")
client = session.client("s3", region_name="us-west-2")
paginator = client.get_paginator("list_objects_v2")

# ---------------------------------------------------------------------------
# Enumerate top-level identifier prefixes under nu-impulse-production
# ---------------------------------------------------------------------------
out = subprocess.Popen(
    ["aws", "s3", "ls", "--profile", "impulse", "s3://nu-impulse-production"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
stdout, stderr = out.communicate()

if stderr:
    print("Error:", stderr.decode("utf-8"))
    output = []
else:
    output = stdout.decode("utf-8").splitlines()
    output = [i.strip().replace("PRE ", "") for i in output if i.strip()]

# ---------------------------------------------------------------------------
# Per-identifier submission
# ---------------------------------------------------------------------------
for prefix in output:
    # Skip the top-of-listing header line (empty PRE) if any snuck through
    if not prefix or prefix in ("PRE",):
        continue

    impulse_identifier = (
        prefix.replace("/", "").replace("}", "").replace("{", "").lower()
    )

    operation_parameters = {
        "Bucket": "nu-impulse-production",
        "Prefix": prefix,
    }
    page_iterator = paginator.paginate(**operation_parameters)

    impulse_keys: list[str] = []
    for page in page_iterator:
        try:
            for j in page["Contents"]:
                key = f"s3://nu-impulse-production/{j['Key']}"
                if key.endswith("jpg"):
                    impulse_keys.append(key)
        except KeyError:
            continue

    if not impulse_keys:
        print(f"[skip] {impulse_identifier}: no .jpg source keys")
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

    is_new = impulse_identifier not in already_extracted
    if is_new:
        ocr_fw: Firework = Firework(
            DocumentExtractionTask(),
            spec=common_spec,
            name="Document Extraction Workflow",
        )
        wf = Workflow(
            [io_fw, ocr_fw, ip_fw],
            {io_fw: [ocr_fw], ocr_fw: [ip_fw]},
            name=impulse_identifier,
        )
        print(
            f"[NEW] {impulse_identifier}: IO -> DocExtract -> ImageProc "
            f"({len(impulse_keys)} keys)"
        )
    else:
        wf = Workflow(
            [io_fw, ip_fw],
            {io_fw: [ip_fw]},
            name=impulse_identifier,
        )
        print(
            f"[BACKFILL] {impulse_identifier}: IO -> ImageProc "
            f"({len(impulse_keys)} keys)"
        )

    launchpad.add_wf(wf)

print("Done.")
