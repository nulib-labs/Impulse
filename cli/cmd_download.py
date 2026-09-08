"""``impulse download`` — pull processed outputs for one or more identifiers.

Artifact selection is flag-driven:

* ``--text``       OCR text from praxis.colt  -> ``<out>/<name>/TXT/<page>.txt``
* ``--raw``        source images              -> ``<out>/<name>/raw_images/``
* ``--binarized``  binarized images           -> ``<out>/<name>/binarized_images/``
* ``--jp2``        JP2s from production        -> ``<out>/<name>/JP2000/``
* ``--mets``       mets.xml / mets.yaml       -> ``<out>/<name>/``
* ``--all``-artifacts shorthand for everything above

With no artifact flag, ``--text`` is assumed.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from natsort import natsorted

from cli import common


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Download processed outputs (text, images, METS) by identifier.",
        description=(
            "Fetch pipeline outputs for one or more identifiers into a local "
            "directory tree. Defaults to OCR text only."
        ),
    )
    common.add_identifier_selection_args(parser)
    parser.add_argument(
        "-o",
        "--out",
        default="impulse_download",
        help="Output directory (default: impulse_download).",
    )
    parser.add_argument(
        "--name-by",
        choices=("identifier", "accession"),
        default="identifier",
        help=(
            "Name each output directory by the full identifier (default) or by "
            "the bare accession/barcode."
        ),
    )

    group = parser.add_argument_group("artifacts")
    group.add_argument("--text", action="store_true", help="OCR text from praxis.colt.")
    group.add_argument("--raw", action="store_true", help="Source images from S3.")
    group.add_argument(
        "--binarized", action="store_true", help="Binarized images from S3."
    )
    group.add_argument(
        "--jp2", action="store_true", help="JP2 images from the production bucket."
    )
    group.add_argument("--mets", action="store_true", help="mets.xml and mets.yaml.")
    group.add_argument(
        "--all-artifacts",
        action="store_true",
        help="Shorthand for --text --raw --binarized --jp2 --mets.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel downloads (default: 8).",
    )
    parser.add_argument(
        "--skip-empty-text",
        action="store_true",
        help=(
            "Do not write .txt files for pages with no extractable text. "
            "By default an empty file is written so file count matches page count."
        ),
    )
    common.add_connection_args(parser)
    common.add_dry_run_arg(parser)
    parser.set_defaults(func=run)


# --------------------------------------------------------------------------


def _selected(args) -> dict[str, bool]:
    sel = {
        "text": args.text,
        "raw": args.raw,
        "binarized": args.binarized,
        "jp2": args.jp2,
        "mets": args.mets,
    }
    if args.all_artifacts:
        return {k: True for k in sel}
    if not any(sel.values()):
        sel["text"] = True
    return sel


def _out_name(identifier: str, name_by: str) -> str:
    if name_by == "accession":
        return common.split_identifier(identifier)[1]
    return identifier


def _download_text(db, identifier: str, dest: str, args) -> tuple[int, int]:
    """Write one .txt per page. Returns (written, empty)."""
    cursor = (
        db[common.COLT_COLLECTION]
        .find(
            {"impulse_identifier": identifier},
            {"page_number": 1, "ocr_data": 1, "extracted_data": 1, "_id": 0},
        )
        .sort("page_number", 1)
    )

    txt_dir = os.path.join(dest, "TXT")
    if not args.dry_run:
        os.makedirs(txt_dir, exist_ok=True)

    written = 0
    empty = 0
    for doc in cursor:
        page = doc.get("page_number")
        if page is None:
            continue
        text = common.page_text(doc)
        if not text:
            empty += 1
            if args.skip_empty_text:
                continue
        if not args.dry_run:
            path = os.path.join(txt_dir, f"{int(page):010d}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        written += 1
    return written, empty


def _download_prefix(
    s3, bucket: str, prefix: str, dest: str, args, label: str
) -> tuple[int, int]:
    """Download every object under ``prefix`` into ``dest``. Returns (files, bytes)."""
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                objects.append((obj["Key"], obj["Size"]))

    if not objects:
        return 0, 0

    objects = natsorted(objects, key=lambda t: t[0])
    total_bytes = sum(sz for _, sz in objects)

    if args.dry_run:
        print(
            f"    {label}: would download {len(objects)} file(s) "
            f"({common.humanize_bytes(total_bytes)})"
        )
        return len(objects), total_bytes

    os.makedirs(dest, exist_ok=True)

    def fetch(key: str) -> None:
        target = os.path.join(dest, os.path.basename(key))
        s3.download_file(bucket, key, target)

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch, k): k for k, _ in objects}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((futures[fut], str(exc)))

    for key, err in failures:
        print(f"[error] failed to download {key}: {err}", file=sys.stderr)

    return len(objects) - len(failures), total_bytes


def _download_mets(s3, bucket: str, identifier: str, dest: str, args) -> int:
    got = 0
    for name in (common.METS_XML_NAME, common.METS_YAML_NAME):
        key = f"{common.data_prefix(identifier)}/{name}"
        if not common.key_exists(s3, bucket, key):
            continue
        if args.dry_run:
            print(f"    mets: would download {name}")
            got += 1
            continue
        os.makedirs(dest, exist_ok=True)
        s3.download_file(bucket, key, os.path.join(dest, name))
        got += 1
    return got


def _production_prefix_for(s3, identifier: str, subdir: str) -> str | None:
    """Resolve the production-bucket prefix, tolerating identifier casing."""
    for candidate in (identifier, identifier.upper(), identifier.lower()):
        prefix = f"{candidate}/{subdir}/"
        resp = s3.list_objects_v2(
            Bucket=common.PRODUCTION_BUCKET, Prefix=prefix, MaxKeys=1
        )
        if resp.get("KeyCount"):
            return prefix
    return None


# --------------------------------------------------------------------------


def run(args) -> int:
    sel = _selected(args)
    needs_s3 = any(sel[k] for k in ("raw", "binarized", "jp2", "mets"))

    # S3 is built lazily so that `--text` (Mongo-only) works for users who
    # have database access but no configured AWS profile.
    s3_get = common.s3_factory(args)
    db = common.get_praxis_db(args.prod) if sel["text"] else None

    identifiers = common.resolve_identifiers(
        args, s3=s3_get, db=db, all_from_db=not needs_s3
    )
    if not identifiers:
        raise SystemExit(
            "Error: no identifiers to process. Pass identifiers, "
            "--identifiers-file, or --all."
        )

    s3 = s3_get() if needs_s3 else None

    active = ", ".join(k for k, v in sel.items() if v)
    print(f"Identifiers: {len(identifiers)}")
    print(f"Artifacts:   {active}")
    print(f"Output:      {args.out}")
    if args.dry_run:
        print("Mode:        DRY RUN (no writes)")
    print()

    totals = {
        "text": 0,
        "text_empty": 0,
        "raw": 0,
        "binarized": 0,
        "jp2": 0,
        "mets": 0,
        "bytes": 0,
    }

    for identifier in identifiers:
        try:
            name = _out_name(identifier, args.name_by)
        except ValueError as exc:
            print(f"[warn] skipping {identifier}: {exc}", file=sys.stderr)
            continue

        dest = os.path.join(args.out, name)
        print(f"{identifier}:")

        if sel["text"]:
            written, empty = _download_text(db, identifier, dest, args)
            totals["text"] += written
            totals["text_empty"] += empty
            if written:
                print(f"    text: {written} page(s) ({empty} empty)")
            else:
                print("    text: no pages found in praxis.colt")

        if sel["raw"]:
            prefix = f"{common.data_prefix(identifier, common.RAW_IMAGES)}/"
            n, b = _download_prefix(
                s3, args.bucket, prefix, os.path.join(dest, common.RAW_IMAGES), args, "raw"
            )
            totals["raw"] += n
            totals["bytes"] += b
            if n and not args.dry_run:
                print(f"    raw: {n} file(s) ({common.humanize_bytes(b)})")
            elif not n:
                print("    raw: none found")

        if sel["binarized"]:
            prefix = f"{common.data_prefix(identifier, common.BINARIZED_IMAGES)}/"
            n, b = _download_prefix(
                s3,
                args.bucket,
                prefix,
                os.path.join(dest, common.BINARIZED_IMAGES),
                args,
                "binarized",
            )
            totals["binarized"] += n
            totals["bytes"] += b
            if n and not args.dry_run:
                print(f"    binarized: {n} file(s) ({common.humanize_bytes(b)})")
            elif not n:
                print("    binarized: none found")

        if sel["jp2"]:
            prefix = _production_prefix_for(s3, identifier, "JP2000")
            if prefix is None:
                print("    jp2: none found")
            else:
                n, b = _download_prefix(
                    s3,
                    common.PRODUCTION_BUCKET,
                    prefix,
                    os.path.join(dest, "JP2000"),
                    args,
                    "jp2",
                )
                totals["jp2"] += n
                totals["bytes"] += b
                if n and not args.dry_run:
                    print(f"    jp2: {n} file(s) ({common.humanize_bytes(b)})")

        if sel["mets"]:
            n = _download_mets(s3, args.bucket, identifier, dest, args)
            totals["mets"] += n
            print(f"    mets: {n} file(s)")

    print("\n=== TOTALS ===")
    if sel["text"]:
        print(
            f"text pages:       {totals['text']} "
            f"({totals['text_empty']} empty)"
        )
    for key in ("raw", "binarized", "jp2", "mets"):
        if sel[key]:
            print(f"{key + ' files:':<18}{totals[key]}")
    if totals["bytes"]:
        print(f"{'downloaded:':<18}{common.humanize_bytes(totals['bytes'])}")
    return 0
