"""``impulse jobs`` — create FireWorks jobs for existing S3 data.

Three job types:

* ``image-processing``    -> ``ImageProcessingTask``  (binarize + deskew)
* ``document-extraction`` -> ``DocumentExtractionTask`` (surya OCR + layout)
* ``mets-yaml``           -> ``METSXMLToHathiTrustManifestTask`` (METS XML -> HathiTrust YAML)

Unlike ``impulse upload``, these commands never touch S3 contents; they only
enumerate what is already there and enqueue work.
"""

from __future__ import annotations

import sys

from natsort import natsorted

from cli import common

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "jobs",
        help="Create image-processing, document-extraction, or mets-yaml jobs.",
        description="Enqueue FireWorks jobs over data already present in S3.",
    )
    job_subs = parser.add_subparsers(dest="job_type", metavar="JOB_TYPE")
    job_subs.required = True

    _register_image_processing(job_subs)
    _register_document_extraction(job_subs)
    _register_mets_yaml(job_subs)


def _add_shared(parser) -> None:
    common.add_identifier_selection_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Enqueue even if the output already appears complete.",
    )
    common.add_connection_args(parser)
    common.add_dry_run_arg(parser)


# --------------------------------------------------------------------------
# image-processing
# --------------------------------------------------------------------------


def _register_image_processing(subs) -> None:
    parser = subs.add_parser(
        "image-processing",
        help="Binarize + deskew source images (ImageProcessingTask).",
        description=(
            "Enqueue ImageProcessingTask over <project>/<accession>/raw_images/. "
            "By default, identifiers whose binarized_images/ already has one .png "
            "per source image are skipped."
        ),
    )
    parser.add_argument(
        "--source",
        default=common.RAW_IMAGES,
        help=f"Source subdirectory in the bucket (default: {common.RAW_IMAGES}).",
    )
    _add_shared(parser)
    parser.set_defaults(func=run_image_processing)


def run_image_processing(args) -> int:
    from fireworks import Firework, Workflow

    from tasks.my_tasks import ImageProcessingTask

    s3 = common.get_s3(args.aws_profile, args.region)
    identifiers = common.resolve_identifiers(args, s3=s3)
    if not identifiers:
        raise SystemExit("Error: no identifiers selected.")

    print(f"Job type:    image-processing")
    print(f"Identifiers: {len(identifiers)}")
    if args.dry_run:
        print("Mode:        DRY RUN")
    print()

    launchpad = None if args.dry_run else common.get_launchpad(args.prod)
    submitted = skipped = empty = 0

    for identifier in identifiers:
        try:
            src_prefix = f"{common.data_prefix(identifier, args.source)}/"
        except ValueError as exc:
            print(f"[warn] {identifier}: {exc}", file=sys.stderr)
            continue

        keys = natsorted(common.iter_keys(s3, args.bucket, src_prefix, IMAGE_EXTS))
        if not keys:
            print(f"  {identifier}: no source images — skip")
            empty += 1
            continue

        if not args.force:
            bin_prefix = f"{common.data_prefix(identifier, common.BINARIZED_IMAGES)}/"
            n_bin = sum(
                1 for _ in common.iter_keys(s3, args.bucket, bin_prefix, (".png",))
            )
            if n_bin >= len(keys):
                print(
                    f"  {identifier}: {n_bin} binarized .png >= {len(keys)} source "
                    "— skip (use --force to re-run)"
                )
                skipped += 1
                continue

        spec = {
            "impulse_identifier": identifier,
            "find_path_array_in": "keys",
            "keys": [common.s3_uri(args.bucket, k) for k in keys],
        }

        if args.dry_run:
            print(f"  {identifier}: would enqueue ({len(keys)} images)")
            submitted += 1
            continue

        fw = Firework(ImageProcessingTask(), spec=spec, name="Image Processing Task")
        launchpad.add_wf(Workflow([fw], name=identifier))
        print(f"  {identifier}: enqueued ({len(keys)} images)")
        submitted += 1

    print(
        f"\nSubmitted {submitted}, skipped {skipped} (already done), "
        f"{empty} with no source images."
    )
    return 0


# --------------------------------------------------------------------------
# document-extraction
# --------------------------------------------------------------------------


def _register_document_extraction(subs) -> None:
    parser = subs.add_parser(
        "document-extraction",
        help="Run surya OCR + layout (DocumentExtractionTask).",
        description=(
            "Enqueue DocumentExtractionTask. By default, identifiers that already "
            "have documents in praxis.colt are skipped."
        ),
    )
    parser.add_argument(
        "--source",
        default=common.RAW_IMAGES,
        choices=(common.RAW_IMAGES, common.BINARIZED_IMAGES),
        help=f"Which images to OCR (default: {common.RAW_IMAGES}).",
    )
    _add_shared(parser)
    parser.set_defaults(func=run_document_extraction)


def run_document_extraction(args) -> int:
    from fireworks import Firework, Workflow

    from tasks.my_tasks import DocumentExtractionTask

    s3 = common.get_s3(args.aws_profile, args.region)
    db = common.get_praxis_db(args.prod)
    identifiers = common.resolve_identifiers(args, s3=s3, db=db)
    if not identifiers:
        raise SystemExit("Error: no identifiers selected.")

    already = set(common.list_colt_identifiers(db))

    print(f"Job type:    document-extraction")
    print(f"Identifiers: {len(identifiers)}")
    print(f"Source:      {args.source}")
    if args.dry_run:
        print("Mode:        DRY RUN")
    print()

    launchpad = None if args.dry_run else common.get_launchpad(args.prod)
    submitted = skipped = empty = 0

    for identifier in identifiers:
        if identifier in already and not args.force:
            print(f"  {identifier}: already in praxis.colt — skip (--force to re-run)")
            skipped += 1
            continue

        try:
            src_prefix = f"{common.data_prefix(identifier, args.source)}/"
        except ValueError as exc:
            print(f"[warn] {identifier}: {exc}", file=sys.stderr)
            continue

        keys = natsorted(common.iter_keys(s3, args.bucket, src_prefix, IMAGE_EXTS))
        if not keys:
            print(f"  {identifier}: no source images — skip")
            empty += 1
            continue

        spec = {
            "impulse_identifier": identifier,
            "find_path_array_in": "keys",
            "keys": [common.s3_uri(args.bucket, k) for k in keys],
        }

        if args.dry_run:
            print(f"  {identifier}: would enqueue ({len(keys)} images)")
            submitted += 1
            continue

        fw = Firework(
            DocumentExtractionTask(), spec=spec, name="Document Extraction Workflow"
        )
        launchpad.add_wf(Workflow([fw], name=identifier))
        print(f"  {identifier}: enqueued ({len(keys)} images)")
        submitted += 1

    print(
        f"\nSubmitted {submitted}, skipped {skipped} (already extracted), "
        f"{empty} with no source images."
    )
    return 0


# --------------------------------------------------------------------------
# mets-yaml
# --------------------------------------------------------------------------


def _register_mets_yaml(subs) -> None:
    parser = subs.add_parser(
        "mets-yaml",
        help="Convert METS XML to a HathiTrust manifest YAML.",
        description=(
            "Enqueue METSXMLToHathiTrustManifestTask. Reads "
            "<bucket>/<project>/<accession>/mets.xml and writes mets.yaml "
            "alongside it, also persisting the result to praxis.HathiTrust. "
            "Identifiers without a mets.xml in the bucket are skipped."
        ),
    )
    _add_shared(parser)
    parser.set_defaults(func=run_mets_yaml)


def run_mets_yaml(args) -> int:
    from fireworks import Firework, Workflow

    from tasks.mets import METSXMLToHathiTrustManifestTask

    s3 = common.get_s3(args.aws_profile, args.region)
    identifiers = common.resolve_identifiers(args, s3=s3)
    if not identifiers:
        raise SystemExit("Error: no identifiers selected.")

    print(f"Job type:    mets-yaml")
    print(f"Identifiers: {len(identifiers)}")
    print(f"Bucket:      {args.bucket}")
    if args.dry_run:
        print("Mode:        DRY RUN")
    print()

    launchpad = None if args.dry_run else common.get_launchpad(args.prod)
    submitted = skipped = missing = 0

    for identifier in identifiers:
        try:
            prefix = common.data_prefix(identifier)
        except ValueError as exc:
            print(f"[warn] {identifier}: {exc}", file=sys.stderr)
            continue

        xml_key = f"{prefix}/{common.METS_XML_NAME}"
        yaml_key = f"{prefix}/{common.METS_YAML_NAME}"

        if not common.key_exists(s3, args.bucket, xml_key):
            print(
                f"  {identifier}: no {common.METS_XML_NAME} at "
                f"{common.s3_uri(args.bucket, xml_key)} — skip"
            )
            missing += 1
            continue

        if not args.force and common.key_exists(s3, args.bucket, yaml_key):
            print(
                f"  {identifier}: {common.METS_YAML_NAME} already exists — skip "
                "(--force to regenerate)"
            )
            skipped += 1
            continue

        spec = {
            "impulse_identifier": identifier,
            "s3_xml_path": common.s3_uri(args.bucket, xml_key),
            "s3_yaml_path": common.s3_uri(args.bucket, yaml_key),
        }

        if args.dry_run:
            print(f"  {identifier}: would enqueue ({xml_key} -> {yaml_key})")
            submitted += 1
            continue

        fw = Firework(
            METSXMLToHathiTrustManifestTask(),
            spec=spec,
            name="METS XML to HathiTrust Manifest",
        )
        launchpad.add_wf(Workflow([fw], name=identifier))
        print(f"  {identifier}: enqueued")
        submitted += 1

    print(
        f"\nSubmitted {submitted}, skipped {skipped} (yaml exists), "
        f"{missing} without a {common.METS_XML_NAME}."
    )
    if missing:
        print(
            f"\nNote: mets.xml is expected at "
            f"s3://{args.bucket}/<project>/<accession>/{common.METS_XML_NAME}. "
            f"Upload one with:  impulse upload <identifier> "
            f"--mets path/to/mets.xml --workflow none"
        )
    return 0
