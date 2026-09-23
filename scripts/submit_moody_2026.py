"""Submit a DocumentExtractionTask FireWorks job for every subdir under
``s3://nu-impulse-data/moody_2026/``.

Each subdir (e.g. ``1928_BANKFINANCE_VOL1/``) is a flat set of
``page_XXXXXXXXXX.png`` images. For each subdir we:

  1. Enumerate its ``.png`` object keys.
  2. Derive a 2-part ``impulse_identifier`` (see note below).
  3. Submit a single-node Workflow containing one DocumentExtractionTask.

Identifier note
---------------
``DocumentExtractionTask.run_task`` splits ``impulse_identifier`` on ``_`` into
exactly ``[project_number, accession_number]`` to build its S3 ``raw_images``
key. Subdir names like ``1928_BANKFINANCE_VOL1`` have three underscore-parts,
so we collapse them: ``project`` = first part (year), ``accession`` = the rest
joined and lowercased. Example:

    1928_BANKFINANCE_VOL1  ->  impulse_identifier = "1928_bankfinancevol1"

This guarantees a clean, unique raw_images key per volume.

Run with:  uv run python scripts/submit_moody_2026.py
Requires:  env IMPULSE_MONGODB_URI, AWS profile "impulse".
"""

from fireworks import Firework, LaunchPad, Workflow
from pymongo import MongoClient
from tasks.my_tasks import DocumentExtractionTask
import boto3
import os
import certifi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = False
# When True, resolve + print everything but DO NOT submit any workflow.
DRY_RUN = False

BUCKET = "nu-impulse-data"
ROOT_PREFIX = "evaluation_moody_2026/"

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

# Identifiers that already have DocumentExtractionTask output in praxis.colt.
praxis_db = MongoClient(MONGO_URI, tlsCAFile=certifi.where())["praxis"]
already_extracted: set[str] = {
    x
    for x in praxis_db["colt"].distinct("impulse_identifier")
    if isinstance(x, str) and x.strip()
}
print(f"Found {len(already_extracted)} identifiers already in praxis.colt")

session = boto3.Session(profile_name="impulse")
s3_client = session.client("s3", region_name="us-east-1")
paginator = s3_client.get_paginator("list_objects_v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def derive_impulse_identifier(subdir_prefix: str) -> str:
    """Collapse a subdir name into a 2-part ``project_accession`` identifier.

    "moody_2026/1928_BANKFINANCE_VOL1/" -> "1928_bankfinancevol1"
    """
    name = subdir_prefix[len(ROOT_PREFIX):].rstrip("/")
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name.lower()
    project = parts[0]
    accession = "".join(parts[1:]).lower()
    if not accession:
        return project.lower()
    return f"{project.lower()}_{accession}"


# ---------------------------------------------------------------------------
# Enumerate subdirs under moody_2026/
# ---------------------------------------------------------------------------
subdir_prefixes: list[str] = []
for page in paginator.paginate(
    Bucket=BUCKET, Prefix=ROOT_PREFIX, Delimiter="/"
):
    for cp in page.get("CommonPrefixes", []) or []:
        subdir_prefixes.append(cp["Prefix"])  # e.g. "moody_2026/1928_BANKFINANCE_VOL1/"

print(f"Discovered {len(subdir_prefixes)} subdirs under s3://{BUCKET}/{ROOT_PREFIX}")

# ---------------------------------------------------------------------------
# Per-subdir submission
# ---------------------------------------------------------------------------
submitted = 0
skipped_existing = 0
skipped_no_keys = 0

for subdir in subdir_prefixes:
    impulse_identifier = derive_impulse_identifier(subdir)

    if impulse_identifier in already_extracted:
        print(f"[skip] {impulse_identifier}: already in praxis.colt")
        skipped_existing += 1
        continue

    impulse_keys: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=subdir):
        for obj in page.get("Contents", []) or []:
            if obj["Key"].endswith(".png"):
                impulse_keys.append(f"s3://{BUCKET}/{obj['Key']}")

    if not impulse_keys:
        print(f"[skip] {impulse_identifier}: no .png keys under {subdir}")
        skipped_no_keys += 1
        continue

    spec = {
        "impulse_identifier": impulse_identifier,
        "find_path_array_in": "keys",
        "keys": impulse_keys,
    }

    if DRY_RUN:
        print(
            f"[DRY-RUN] {impulse_identifier}: DocExtract "
            f"({len(impulse_keys)} pngs) <- {subdir}"
        )
        continue

    ocr_fw: Firework = Firework(
        DocumentExtractionTask(),
        spec=spec,
        name="Document Extraction Workflow",
    )
    wf = Workflow([ocr_fw], name=impulse_identifier)
    launchpad.add_wf(wf)
    submitted += 1
    print(
        f"[NEW] {impulse_identifier}: DocExtract "
        f"({len(impulse_keys)} pngs) <- {subdir}"
    )
    exit()

mode = "DRY-RUN (nothing submitted)" if DRY_RUN else "submitted"
print(
    f"\nDone [{mode}]. submitted={submitted} "
    f"skipped_existing={skipped_existing} skipped_no_keys={skipped_no_keys}"
)

