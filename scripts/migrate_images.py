# Migrates old image data from the old S3 bucket/account to the new (mellon it) bucket.
#
# Every key containing `raw_images` is parsed for project_id / barcode, converted
# to PNG (lossless), and written to:
#   jobs/<project_id>/<barcode>/uploaded_images/<name>.png
#
# Usage:
#   python migrate_raw_images.py          # dry run (default)
#   python migrate_raw_images.py --run    # actually migrate

import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
from PIL import Image

OLD_S3_BUCKET = os.environ["OLD_S3_BUCKET"]
OLD_AWS_PROFILE = os.environ["OLD_AWS_PROFILE"]
AWS_REGION = os.environ["AWS_REGION"]

NEW_S3_BUCKET = os.environ["S3_BUCKET"]
NEW_AWS_PROFILE = os.environ["AWS_PROFILE"]

# Optional: narrow the listing (e.g. "some/prefix/") to avoid scanning the whole bucket.
OLD_PREFIX = os.environ.get("OLD_PREFIX", "p1274")

DRY_RUN = "--run" not in sys.argv
MAX_WORKERS = 8  # images are held in memory, so keep this modest

# ASSUMPTION: old keys look like  <anything>/raw_images/<project_id>/<barcode>/<filename>.<ext>
# Adjust this regex if the real layout differs. Named groups are required.
KEY_PATTERN = re.compile(
    r"(?P<project_id>[^/]+)/(?P<barcode>[^/]+)/raw_images/(?P<name>[^/]+)$"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Some scans are huge; don't let Pillow refuse them.
Image.MAX_IMAGE_PIXELS = None

old_session = boto3.Session(profile_name=OLD_AWS_PROFILE, region_name=AWS_REGION)
old_s3 = old_session.client("s3")

new_session = boto3.Session(profile_name=NEW_AWS_PROFILE, region_name=AWS_REGION)
new_s3 = new_session.client("s3")


def all_keys():
    """Yield every key in the old bucket that contains `raw_images`."""
    paginator = old_s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=OLD_S3_BUCKET, Prefix=OLD_PREFIX):
        for obj in page.get("Contents", []):
            if "raw_images" in obj["Key"]:
                yield obj["Key"]


def plan():
    """Return ({old_key: new_key}, [skipped_keys], [collisions])."""
    mapping, skipped = {}, []
    for key in all_keys():
        m = KEY_PATTERN.search(key)
        stem, ext = os.path.splitext(m.group("name")) if m else ("", "")
        if not m or ext.lower() not in IMAGE_EXTS:
            skipped.append(key)
            continue
        new_key = (
            f"jobs/{m.group('project_id')}/{m.group('barcode')}"
            f"/uploaded_images/{stem}.png"
        )
        mapping[key] = new_key

    # Two sources mapping to the same destination (e.g. a.jpg and a.tif) would
    # silently overwrite each other, so flag and skip them.
    by_dest = {}
    for old, new in mapping.items():
        by_dest.setdefault(new, []).append(old)
    collisions = {d: srcs for d, srcs in by_dest.items() if len(srcs) > 1}
    for srcs in collisions.values():
        for s in srcs:
            del mapping[s]
    return mapping, skipped, collisions


def exists_in_new(key):
    try:
        new_s3.head_object(Bucket=NEW_S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def to_png_bytes(data, source_key):
    """Convert image bytes to PNG without altering pixel values."""
    if source_key.lower().endswith(".png"):
        return data  # already lossless; don't re-encode

    im = Image.open(io.BytesIO(data))
    im.load()

    # PNG can't store CMYK (or other exotic modes). Convert only when necessary.
    if im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")

    out = io.BytesIO()
    save_kwargs = {"format": "PNG"}
    if im.info.get("icc_profile"):
        save_kwargs["icc_profile"] = im.info["icc_profile"]  # keep color profile
    im.save(out, **save_kwargs)
    return out.getvalue()


def migrate_one(old_key, new_key):
    if exists_in_new(new_key):
        return old_key, new_key, "exists"
    if DRY_RUN:
        return old_key, new_key, "dryrun"

    body = old_s3.get_object(Bucket=OLD_S3_BUCKET, Key=old_key)["Body"].read()
    png = to_png_bytes(body, old_key)
    new_s3.upload_fileobj(
        io.BytesIO(png),
        NEW_S3_BUCKET,
        new_key,
        ExtraArgs={"ContentType": "image/png"},
    )
    return old_key, new_key, "migrated"


def main():
    mapping, skipped, collisions = plan()
    print(f"{len(mapping)} images to migrate")
    print(f"{len(skipped)} keys skipped (no pattern match or not an image)")
    print(f"{len(collisions)} destination collisions (skipped)")
    for s in skipped[:10]:
        print(f"  skipped: {s}")
    for dest, srcs in list(collisions.items())[:10]:
        print(f"  collision -> {dest}: {srcs}")
    if DRY_RUN:
        print("DRY RUN: nothing will be written. Pass --run to migrate.\n")

    counts, failures = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(migrate_one, o, n): o for o, n in mapping.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                old, new, status = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if DRY_RUN and i <= 20:
                    print(f"(dryrun) {old} -> {new}")
            except Exception as e:
                failures.append((futures[fut], repr(e)))
            if i % 500 == 0:
                print(f"  {i}/{len(mapping)} processed")

    print(f"\nSummary: {counts}, failures: {len(failures)}")
    for key, err in failures[:20]:
        print(f"  FAILED {key}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
