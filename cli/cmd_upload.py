"""``impulse upload`` — push files into S3 and register a workflow.

Handles both halves of getting work into the system:

1. Uploading local image files (and optionally a METS XML) into
   ``nu-impulse-data/<project>/<accession>/``.
2. Adding the corresponding FireWorks ``Workflow`` to the launchpad.

Either half can be skipped (``--no-upload`` / ``--workflow none``).
"""

from __future__ import annotations

import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from natsort import natsorted

from cli import common

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2")

WORKFLOW_CHOICES = (
    "auto",
    "io-ocr-image",
    "io-image",
    "ocr",
    "image",
    "none",
)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "upload",
        help="Push files to S3 and/or add a workflow to the fireworks launchpad.",
        description=(
            "Upload local images into nu-impulse-data and register a FireWorks "
            "workflow for them. Can also upload a METS XML for the mets-yaml job."
        ),
    )
    parser.add_argument(
        "identifier",
        help="Target impulse_identifier, e.g. p1274_35556039349519.",
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help=(
            "Local directory (or single file) of images to upload. "
            "Omit to register a workflow over images already in S3."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Only upload files whose name matches this glob (e.g. '*.jpg').",
    )
    parser.add_argument(
        "--mets",
        default=None,
        help=(
            "Local METS XML to upload to "
            "<bucket>/<project>/<accession>/mets.xml (input for `jobs mets-yaml`)."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=WORKFLOW_CHOICES,
        default="auto",
        help=(
            "Which workflow to add. 'auto' picks io-ocr-image for documents not "
            "yet in praxis.colt, io-image otherwise. 'none' uploads only."
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the S3 upload step; only add the workflow.",
    )
    parser.add_argument(
        "--keep-names",
        action="store_true",
        help=(
            "Preserve original filenames instead of renaming to the pipeline's "
            "<project>_<accession>_<0000000001> convention."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel S3 uploads (default: 8).",
    )
    common.add_connection_args(parser)
    common.add_dry_run_arg(parser)
    parser.set_defaults(func=run)


# --------------------------------------------------------------------------


def _collect_local_files(source: str, pattern: str | None) -> list[str]:
    import fnmatch

    if os.path.isfile(source):
        return [source]
    if not os.path.isdir(source):
        raise SystemExit(f"Error: --from path does not exist: {source}")

    found: list[str] = []
    for root, _dirs, files in os.walk(source):
        for name in files:
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            if not pattern and not name.lower().endswith(IMAGE_EXTS):
                continue
            found.append(os.path.join(root, name))
    return natsorted(found)


def _target_key(
    identifier: str, index: int, local_path: str, keep_names: bool
) -> str:
    project, accession = common.split_identifier(identifier)
    prefix = f"{project}/{accession}/{common.RAW_IMAGES}"
    if keep_names:
        return f"{prefix}/{os.path.basename(local_path)}"
    ext = os.path.splitext(local_path)[1].lower() or ".jpg"
    return f"{prefix}/{project}_{accession}_{index:010d}{ext}"


def _upload_files(
    s3, bucket: str, identifier: str, files: list[str], args
) -> list[str]:
    """Upload local files, returning the resulting s3:// URIs."""
    plan: list[tuple[str, str]] = []
    for i, path in enumerate(files, start=1):
        plan.append((path, _target_key(identifier, i, path, args.keep_names)))

    non_jpg = [k for _, k in plan if not k.lower().endswith((".jpg", ".jpeg"))]
    if non_jpg:
        print(
            f"[note] {len(non_jpg)} file(s) are not .jpg. Some existing pipeline "
            "code filters source keys on a 'jpg' suffix; verify downstream "
            "discovery if you rely on it.",
            file=sys.stderr,
        )

    if args.dry_run:
        for path, key in plan[:10]:
            print(f"  would upload {path} -> {common.s3_uri(bucket, key)}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")
        return [common.s3_uri(bucket, key) for _, key in plan]

    uploaded: list[str] = []
    failures: list[tuple[str, str]] = []

    def put(path: str, key: str) -> str:
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            s3.put_object(Bucket=bucket, Key=key, Body=fh.read(), ContentType=ctype)
        return common.s3_uri(bucket, key)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(put, p, k): (p, k) for p, k in plan}
        done = 0
        for fut in as_completed(futures):
            path, key = futures[fut]
            try:
                uploaded.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                failures.append((path, str(exc)))
            done += 1
            if done % 50 == 0 or done == len(plan):
                print(f"  uploaded {done}/{len(plan)}")

    for path, err in failures:
        print(f"[error] upload failed for {path}: {err}", file=sys.stderr)

    return natsorted(uploaded)


def _upload_mets(s3, bucket: str, identifier: str, path: str, dry_run: bool) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"Error: --mets file does not exist: {path}")
    key = f"{common.data_prefix(identifier)}/{common.METS_XML_NAME}"
    if dry_run:
        print(f"  would upload METS {path} -> {common.s3_uri(bucket, key)}")
        return
    with open(path, "rb") as fh:
        s3.put_object(
            Bucket=bucket, Key=key, Body=fh.read(), ContentType="application/xml"
        )
    print(f"  METS XML -> {common.s3_uri(bucket, key)}")


def _discover_existing(s3, bucket: str, identifier: str) -> list[str]:
    prefix = f"{common.data_prefix(identifier, common.RAW_IMAGES)}/"
    keys = list(common.iter_keys(s3, bucket, prefix, IMAGE_EXTS))
    return natsorted(common.s3_uri(bucket, k) for k in keys)


def _build_workflow(kind: str, spec: dict, identifier: str):
    from fireworks import Firework, Workflow

    from tasks.my_tasks import (
        DocumentExtractionTask,
        ImageProcessingTask,
        IOTask,
    )

    io_fw = Firework(IOTask(), spec=spec, name="I/O Workflow")
    ocr_fw = Firework(
        DocumentExtractionTask(), spec=spec, name="Document Extraction Workflow"
    )
    ip_fw = Firework(ImageProcessingTask(), spec=spec, name="Image Processing Task")

    if kind == "io-ocr-image":
        return Workflow(
            [io_fw, ocr_fw, ip_fw],
            {io_fw: [ocr_fw], ocr_fw: [ip_fw]},
            name=identifier,
        )
    if kind == "io-image":
        return Workflow([io_fw, ip_fw], {io_fw: [ip_fw]}, name=identifier)
    if kind == "ocr":
        return Workflow([ocr_fw], name=identifier)
    if kind == "image":
        return Workflow([ip_fw], name=identifier)
    raise ValueError(f"unknown workflow kind: {kind}")


# --------------------------------------------------------------------------


def run(args) -> int:
    identifier = common.normalize_identifier(args.identifier)
    try:
        project, accession = common.split_identifier(identifier)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Identifier: {identifier}  (project={project}, accession={accession})")
    print(f"Bucket:     {args.bucket}")
    if args.dry_run:
        print("Mode:       DRY RUN (no writes)")
    print()

    s3 = common.get_s3(args.aws_profile, args.region)

    # --- 1. files -> S3 -------------------------------------------------
    keys: list[str] = []
    if args.source and not args.no_upload:
        files = _collect_local_files(args.source, args.pattern)
        if not files:
            raise SystemExit(f"Error: no matching files under {args.source}")
        print(f"Uploading {len(files)} file(s) to S3...")
        keys = _upload_files(s3, args.bucket, identifier, files, args)
    else:
        print("Discovering existing images in S3...")
        keys = _discover_existing(s3, args.bucket, identifier)
        print(f"  found {len(keys)} image(s)")

    if args.mets:
        _upload_mets(s3, args.bucket, identifier, args.mets, args.dry_run)

    # --- 2. workflow -> launchpad ---------------------------------------
    if args.workflow == "none":
        print("\nWorkflow: none requested. Done.")
        return 0

    if not keys:
        print(
            "\n[warn] no image keys resolved — refusing to add a workflow with an "
            "empty key list.",
            file=sys.stderr,
        )
        return 1

    kind = args.workflow
    if kind == "auto":
        db = common.get_praxis_db(args.prod)
        already = set(common.list_colt_identifiers(db))
        kind = "io-image" if identifier in already else "io-ocr-image"
        print(f"\nWorkflow: auto -> {kind}")
    else:
        print(f"\nWorkflow: {kind}")

    spec = {
        "impulse_identifier": identifier,
        "find_path_array_in": "keys",
        "keys": keys,
    }

    if args.dry_run:
        print(f"  would add workflow {kind!r} with {len(keys)} keys")
        return 0

    wf = _build_workflow(kind, spec, identifier)
    launchpad = common.get_launchpad(args.prod)
    result = launchpad.add_wf(wf)
    print(f"  added workflow ({len(keys)} keys) -> fw_ids {list(result.values())}")
    return 0
