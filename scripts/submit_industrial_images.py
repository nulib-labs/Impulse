"""Upload the extracted industrial_images PNGs to S3 and submit one
DocumentExtractionTask job per source PDF.

Grouping: one job per PDF stem (e.g. ``1920_ex1``), which is already a clean
2-part ``project_accession`` identifier as required by DocumentExtractionTask.

S3 layout:
    s3://nu-impulse-data/eval_moody_2026/<stem>/<original-filename>.png

Run with:  uv run python scripts/submit_industrial_images.py
Requires:  env IMPULSE_MONGODB_URI, AWS profile "impulse".
"""

import glob
import os

import boto3
import certifi
from fireworks import Firework, LaunchPad, Workflow
from pymongo import MongoClient

from tasks.my_tasks import DocumentExtractionTask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = False
DRY_RUN = False

SRC_ROOT = "industrial_images"
BUCKET = "nu-impulse-data"
DEST_PREFIX = "eval_moody_2026"

if DEBUG:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI_DEBUG")
else:
    MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")


def stem_from_filename(path: str) -> str:
    """industrial_images/single_column/1920_ex1_page003.png -> 1920_ex1"""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    # strip trailing _pageNNN(_imgN)
    idx = name.find("_page")
    return name[:idx] if idx != -1 else name


def main() -> None:
    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    pngs = sorted(glob.glob(os.path.join(SRC_ROOT, "**", "*.png"), recursive=True))
    if not pngs:
        raise SystemExit(f"No PNGs found under {SRC_ROOT}/")

    # Group local files by PDF stem (= impulse_identifier).
    groups: dict[str, list[str]] = {}
    for p in pngs:
        groups.setdefault(stem_from_filename(p), []).append(p)
    for stem in groups:
        groups[stem].sort()

    print(f"Found {len(pngs)} PNGs in {len(groups)} groups: {sorted(groups)}")

    # ------------------------------------------------------------------
    # LaunchPad, Mongo, S3 setup
    # ------------------------------------------------------------------
    launchpad = LaunchPad(
        uri_mode=True,
        host=MONGO_URI,
        name="fireworks",
        mongoclient_kwargs={"tlsCAFile": certifi.where()},
    )
    praxis_db = MongoClient(MONGO_URI, tlsCAFile=certifi.where())["praxis"]
    already_extracted: set[str] = {
        x
        for x in praxis_db["colt"].distinct("impulse_identifier")
        if isinstance(x, str) and x.strip()
    }
    print(f"Found {len(already_extracted)} identifiers already in praxis.colt")

    session = boto3.Session(profile_name="impulse")
    s3 = session.client("s3", region_name="us-east-1")

    submitted = 0
    skipped_existing = 0

    for stem in sorted(groups):
        impulse_identifier = stem.lower()

        if impulse_identifier in already_extracted:
            print(f"[skip] {impulse_identifier}: already in praxis.colt")
            skipped_existing += 1
            continue

        local_files = groups[stem]

        # Upload each page PNG to S3 and collect the s3:// keys.
        impulse_keys: list[str] = []
        for local_path in local_files:
            fname = os.path.basename(local_path)
            key = f"{DEST_PREFIX}/{stem}/{fname}"
            s3_uri = f"s3://{BUCKET}/{key}"
            if not DRY_RUN:
                s3.upload_file(
                    local_path,
                    BUCKET,
                    key,
                    ExtraArgs={"ContentType": "image/png"},
                )
            impulse_keys.append(s3_uri)

        spec = {
            "impulse_identifier": impulse_identifier,
            "find_path_array_in": "keys",
            "keys": impulse_keys,
        }

        if DRY_RUN:
            print(
                f"[DRY-RUN] {impulse_identifier}: would upload+submit "
                f"({len(impulse_keys)} pngs) -> {DEST_PREFIX}/{stem}/"
            )
            continue

        ocr_fw = Firework(
            DocumentExtractionTask(),
            spec=spec,
            name="Document Extraction Workflow",
        )
        wf = Workflow([ocr_fw], name=impulse_identifier)
        launchpad.add_wf(wf)
        submitted += 1
        print(
            f"[NEW] {impulse_identifier}: DocExtract "
            f"({len(impulse_keys)} pngs) <- {DEST_PREFIX}/{stem}/"
        )

    mode = "DRY-RUN (nothing uploaded/submitted)" if DRY_RUN else "submitted"
    print(
        f"\nDone [{mode}]. submitted={submitted} "
        f"skipped_existing={skipped_existing}"
    )


if __name__ == "__main__":
    main()
