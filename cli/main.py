#!/usr/bin/env python3
"""
impulse_cli.py -- command-line equivalent of the Flask job UI.

Same S3 layout, same FireWorks workflow construction, same env vars:

    S3_BUCKET, AWS_PROFILE, AWS_REGION            (required)
    MONGO_URI                                     (default mongodb://localhost:27017)
    HATHITRUST_MANIFEST_NAME                      (default manifest.json)

Examples
--------
    # Upload images and queue two jobs in one workflow
    impulse_cli.py submit -p proj1 -b 39015012345678 \\
        -j document_extraction -j image_processing --files ./scans/

    # HathiTrust (runs after any other selected jobs)
    impulse_cli.py submit -p proj1 -b 39015012345678 \\
        -j hathitrust --xml ./input.xml

    # Start jobs for inputs already in S3
    impulse_cli.py start -p proj1 -b 39015012345678 -j document_extraction

    impulse_cli.py status  -p proj1 -b 39015012345678 --json
    impulse_cli.py wait    -p proj1 -b 39015012345678 --timeout 3600
    impulse_cli.py files   -p proj1 -b 39015012345678
    impulse_cli.py manifest -p proj1 -b 39015012345678 -o manifest.json
    impulse_cli.py health
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Heavy / environment-dependent imports are deferred where practical so that
# `--help` works without AWS or Mongo configured.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = "fireworks"
HATHITRUST_MANIFEST_NAME = os.environ.get("HATHITRUST_MANIFEST_NAME", "manifest.json")

ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

TERMINAL_STATES = {"COMPLETED", "FIZZLED", "DEFUSED", "ARCHIVED"}
FAILED_STATES = {"FIZZLED", "DEFUSED"}


class CliError(Exception):
    """User-facing error; printed to stderr with exit code 1."""


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise CliError(f"Missing required environment variable: {name}") from None


@lru_cache(maxsize=1)
def get_s3():
    import boto3

    session = boto3.Session(
        profile_name=_require_env("AWS_PROFILE"),
        region_name=_require_env("AWS_REGION"),
    )
    return session.client("s3")


@lru_cache(maxsize=1)
def get_bucket() -> str:
    return _require_env("S3_BUCKET")


@lru_cache(maxsize=1)
def get_lpad():
    from fireworks import LaunchPad

    return LaunchPad(host=MONGO_URI, name=MONGO_DB, uri_mode=True)


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobDefinition:
    name: str
    label: str
    description: str
    task_class_name: str            # resolved lazily from the `tasks` module
    requires_files: bool = True
    requires_xml: bool = False
    requires_impulse_identifier: bool = False

    @property
    def task_class(self) -> type:
        import tasks

        return getattr(tasks, self.task_class_name)


JOB_DEFINITIONS: dict[str, JobDefinition] = {
    "document_extraction": JobDefinition(
        name="document_extraction",
        label="Document Extraction",
        description="OCR/layout extraction on uploaded images or PDFs.",
        task_class_name="DocumentExtractionTask",
        requires_files=True,
        requires_impulse_identifier=True,
    ),
    "image_processing": JobDefinition(
        name="image_processing",
        label="Image Processing",
        description=(
            "Grayscale, denoising, illumination normalization, Sauvola "
            "binarization, and deskewing."
        ),
        task_class_name="ImageProcessingTask",
        requires_files=True,
        requires_impulse_identifier=True,
    ),
    "hathitrust": JobDefinition(
        name="hathitrust",
        label="HathiTrust Create Manifest",
        description="Upload an XML document and build a HathiTrust manifest.",
        task_class_name="CreateHathiTrustManifest",
        requires_files=False,
        requires_xml=True,
        requires_impulse_identifier=True,
    ),
}


def get_job_definition(job_type: str) -> JobDefinition:
    try:
        return JOB_DEFINITIONS[job_type]
    except KeyError:
        raise CliError(
            f"Unknown job type: {job_type} "
            f"(choose from: {', '.join(JOB_DEFINITIONS)})"
        ) from None


# ---------------------------------------------------------------------------
# Identifier / S3 key layout  (identical to the web app)
# ---------------------------------------------------------------------------
#
#   jobs/<project_id>/<barcode>/uploaded_images/0000000001.png ...
#   jobs/<project_id>/<barcode>/hathitrust/input.xml
#   jobs/<project_id>/<barcode>/processed_images/               (image_processing)
#   jobs/<project_id>/<barcode>/outputs/<job_type>/...          (other outputs)

def validate_ids(project_id: str, barcode: str) -> None:
    if not (
        project_id
        and barcode
        and ID_PATTERN.match(project_id)
        and ID_PATTERN.match(barcode)
    ):
        raise CliError(
            "Project ID and barcode are required and may only contain "
            "letters, numbers, '.', '_' and '-'."
        )


def make_impulse_identifier(project_id: str, barcode: str) -> str:
    return f"{project_id}_{barcode}"


def job_prefix(project_id: str, barcode: str) -> str:
    return f"jobs/{project_id}/{barcode}"


def images_prefix(project_id: str, barcode: str) -> str:
    return f"{job_prefix(project_id, barcode)}/uploaded_images/"


def xml_key(project_id: str, barcode: str) -> str:
    return f"{job_prefix(project_id, barcode)}/hathitrust/input.xml"


def output_prefix(project_id: str, barcode: str, job_type: str) -> str:
    if job_type == "image_processing":
        return f"{job_prefix(project_id, barcode)}/processed_images/"
    return f"{job_prefix(project_id, barcode)}/outputs/{job_type}/"


def hathitrust_manifest_key(project_id: str, barcode: str) -> str:
    return output_prefix(project_id, barcode, "hathitrust") + HATHITRUST_MANIFEST_NAME


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_key_exists(key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        get_s3().head_object(Bucket=get_bucket(), Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def expand_local_files(paths: Iterable[str]) -> list[Path]:
    """Expand files / directories (recursively) into a naturally sorted list."""
    from natsort import natsorted

    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found.extend(
                f for f in p.rglob("*")
                if f.is_file() and not f.name.startswith(".")
            )
        elif p.is_file():
            found.append(p)
        else:
            raise CliError(f"Not a file or directory: {raw}")

    return natsorted(found, key=lambda f: str(f))


def upload_files_to_s3(project_id: str, barcode: str, files: list[Path]) -> list[str]:
    """
    Upload local files to uploaded_images/<n><ext>, zero-padded and naturally
    sorted -- same renaming scheme as the web app.
    """
    s3 = get_s3()
    keys: list[str] = []

    for i, path in enumerate(files, start=1):
        key = f"{images_prefix(project_id, barcode)}{i:010d}{path.suffix.lower()}"

        extra_args: dict[str, Any] = {}
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type:
            extra_args["ContentType"] = content_type

        _log(f"  uploading {path} -> s3://{get_bucket()}/{key}")
        s3.upload_file(str(path), get_bucket(), key, ExtraArgs=extra_args or None)
        keys.append(key)

    return keys


def validate_xml(data: bytes) -> None:
    """
    TODO: fill in real validation (XSD, required elements, barcode match...).
    Uses defusedxml when installed, since uploads may be untrusted.
    """
    try:
        try:
            from defusedxml import ElementTree as SafeET

            SafeET.fromstring(data)
        except ImportError:
            ET.fromstring(data)
    except ET.ParseError as exc:
        raise CliError(f"Malformed XML: {exc}") from exc


def upload_xml_to_s3(project_id: str, barcode: str, path: Path) -> str:
    if not path.is_file():
        raise CliError(f"XML file not found: {path}")

    data = path.read_bytes()
    if not data:
        raise CliError("The XML file is empty.")

    validate_xml(data)

    key = xml_key(project_id, barcode)
    _log(f"  uploading {path} -> s3://{get_bucket()}/{key}")
    get_s3().put_object(
        Bucket=get_bucket(), Key=key, Body=data, ContentType="application/xml"
    )
    return key


def list_job_keys(project_id: str, barcode: str, prefix: str | None = None) -> list[str]:
    prefix = prefix or f"{job_prefix(project_id, barcode)}/"
    paginator = get_s3().get_paginator("list_objects_v2")

    keys: list[str] = []
    for page in paginator.paginate(Bucket=get_bucket(), Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def list_uploaded_image_keys(project_id: str, barcode: str) -> list[str]:
    return list_job_keys(project_id, barcode, prefix=images_prefix(project_id, barcode))


def list_job_files(project_id: str, barcode: str) -> list[str]:
    prefix = f"{job_prefix(project_id, barcode)}/"
    return [k[len(prefix):] for k in list_job_keys(project_id, barcode)]


# ---------------------------------------------------------------------------
# FireWorks spec construction / submission
# ---------------------------------------------------------------------------

def build_spec(project_id: str, barcode: str, job_type: str) -> dict[str, Any]:
    impulse_identifier = make_impulse_identifier(project_id, barcode)
    definition = get_job_definition(job_type)

    spec: dict[str, Any] = {
        "job_type": job_type,
        "job_name": definition.label,
        "s3_bucket": get_bucket(),
        "project_id": project_id,
        "barcode": barcode,
        "output_prefix": output_prefix(project_id, barcode, job_type),
    }

    if definition.requires_files:
        uploaded = list_uploaded_image_keys(project_id, barcode)
        if not uploaded:
            raise CliError(f"No uploaded files found for job {impulse_identifier}")
        spec["find_path_array_in"] = "uploaded_files"
        spec["uploaded_files"] = uploaded

    if definition.requires_xml:
        key = xml_key(project_id, barcode)
        if not s3_key_exists(key):
            raise CliError(f"No XML document found for job {impulse_identifier}")
        spec["xml_key"] = key
        # TODO: add any extra HathiTrust-specific spec values here.

    if definition.requires_impulse_identifier:
        spec["impulse_identifier"] = impulse_identifier

    return spec


def build_specs(
    project_id: str, barcode: str, job_types: list[str]
) -> dict[str, dict[str, Any]]:
    return {jt: build_spec(project_id, barcode, jt) for jt in job_types}


def submit_fireworks_jobs(
    impulse_identifier: str, specs: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """
    Insert ONE workflow with one Firework per job type. HathiTrust (if
    selected) is linked to run after every other Firework.

    Returns {job_type: database fw_id}.
    """
    from fireworks import Firework, Workflow

    fireworks: dict[str, Firework] = {}
    for job_type, spec in specs.items():
        definition = get_job_definition(job_type)
        fireworks[job_type] = Firework(
            definition.task_class(),
            spec=spec,
            name=f"job-{impulse_identifier}-{job_type}",
        )

    # HathiTrust runs only after every other job has completed.
    links: dict[Firework, list[Firework]] = {}
    hathitrust_fw = fireworks.get("hathitrust")
    if hathitrust_fw is not None:
        for fw in fireworks.values():
            if fw is not hathitrust_fw:
                links[fw] = [hathitrust_fw]

    workflow = Workflow(
        list(fireworks.values()),
        links_dict=links,
        name=f"impulse-{impulse_identifier}",
    )

    # add_wf returns {temporary fw_id: database fw_id}, NOT {job_type: fw_id}.
    id_map = get_lpad().add_wf(workflow)
    return {jt: id_map[fw.fw_id] for jt, fw in fireworks.items()}


def find_fireworks_for_job(impulse_identifier: str) -> list[dict[str, Any]]:
    docs = get_lpad().fireworks.find(
        {"spec.impulse_identifier": impulse_identifier}
    ).sort([("created_on", -1)])

    return [
        {
            "fw_id": d.get("fw_id"),
            "name": d.get("name"),
            "job_type": d.get("spec", {}).get("job_type"),
            "state": d.get("state"),
        }
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Progress messages go to stderr so stdout stays machine-readable."""
    print(msg, file=sys.stderr)


def _emit(data: Any, as_json: bool, human: str | None = None) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    elif human is not None:
        print(human)


def _format_fireworks(fws: list[dict[str, Any]]) -> str:
    if not fws:
        return "(no FireWorks found)"
    width = max(len(str(f["job_type"])) for f in fws)
    return "\n".join(
        f"{f['fw_id']:>6}  {str(f['job_type']):<{width}}  {f['state']}" for f in fws
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _wait_for_completion(impulse_identifier: str, timeout: float, interval: float) -> int:
    """Poll until all FireWorks are terminal. Returns a process exit code."""
    deadline = time.monotonic() + timeout if timeout > 0 else None
    while True:
        fws = find_fireworks_for_job(impulse_identifier)
        states = {f["state"] for f in fws}

        if fws and states <= TERMINAL_STATES:
            _log(_format_fireworks(fws))
            return 1 if states & FAILED_STATES else 0

        if deadline and time.monotonic() > deadline:
            _log(_format_fireworks(fws))
            _log("Timed out waiting for jobs to finish.")
            return 2

        time.sleep(interval)


def cmd_submit(args: argparse.Namespace) -> int:
    validate_ids(args.project_id, args.barcode)
    definitions = [get_job_definition(jt) for jt in args.job]
    impulse_identifier = make_impulse_identifier(args.project_id, args.barcode)

    needs_xml = any(d.requires_xml for d in definitions)

    if args.files:
        files = expand_local_files(args.files)
        if not files:
            raise CliError("No files found to upload.")
        _log(f"Uploading {len(files)} file(s)...")
        upload_files_to_s3(args.project_id, args.barcode, files)

    if args.xml:
        upload_xml_to_s3(args.project_id, args.barcode, Path(args.xml).expanduser())
    elif needs_xml and not args.skip_upload_check:
        pass  # build_spec() will verify the XML already exists in S3

    return _submit(args, impulse_identifier)


def cmd_start(args: argparse.Namespace) -> int:
    """Submit jobs for inputs that are already in S3."""
    validate_ids(args.project_id, args.barcode)
    for jt in args.job:
        get_job_definition(jt)
    return _submit(args, make_impulse_identifier(args.project_id, args.barcode))


def _submit(args: argparse.Namespace, impulse_identifier: str) -> int:
    specs = build_specs(args.project_id, args.barcode, args.job)
    fw_ids = submit_fireworks_jobs(impulse_identifier, specs)

    _emit(
        {"impulse_identifier": impulse_identifier, "fw_ids": fw_ids},
        args.json,
        human="\n".join(
            f"Submitted {JOB_DEFINITIONS[jt].label}: fw_id={fw_id}"
            for jt, fw_id in fw_ids.items()
        ),
    )

    if getattr(args, "wait", False):
        return _wait_for_completion(impulse_identifier, args.timeout, args.interval)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    validate_ids(args.project_id, args.barcode)
    impulse_identifier = make_impulse_identifier(args.project_id, args.barcode)

    fws = find_fireworks_for_job(impulse_identifier)
    files = list_job_files(args.project_id, args.barcode)
    if not fws and not files:
        raise CliError(f"No such job: {impulse_identifier}")

    manifest = s3_key_exists(hathitrust_manifest_key(args.project_id, args.barcode))
    _emit(
        {
            "impulse_identifier": impulse_identifier,
            "fireworks": fws,
            "file_count": len(files),
            "hathitrust_manifest_available": manifest,
        },
        args.json,
        human=(
            f"{impulse_identifier}\n"
            f"{_format_fireworks(fws)}\n"
            f"files in S3: {len(files)}\n"
            f"HathiTrust manifest: {'available' if manifest else 'not available'}"
        ),
    )
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    validate_ids(args.project_id, args.barcode)
    return _wait_for_completion(
        make_impulse_identifier(args.project_id, args.barcode),
        args.timeout,
        args.interval,
    )


def cmd_files(args: argparse.Namespace) -> int:
    validate_ids(args.project_id, args.barcode)
    files = list_job_files(args.project_id, args.barcode)
    if not files:
        raise CliError("No files found for this job.")
    _emit(files, args.json, human="\n".join(files))
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    validate_ids(args.project_id, args.barcode)
    key = hathitrust_manifest_key(args.project_id, args.barcode)

    if not s3_key_exists(key):
        raise CliError("Manifest not found (has the job finished?)")

    download_name = f"{args.project_id}_{args.barcode}_{HATHITRUST_MANIFEST_NAME}"

    if args.url:
        url = get_s3().generate_presigned_url(
            "get_object",
            Params={
                "Bucket": get_bucket(),
                "Key": key,
                "ResponseContentDisposition": f'attachment; filename="{download_name}"',
            },
            ExpiresIn=args.expires,
        )
        print(url)
        return 0

    if args.output == "-":
        obj = get_s3().get_object(Bucket=get_bucket(), Key=key)
        for chunk in obj["Body"].iter_chunks():
            sys.stdout.buffer.write(chunk)
        return 0

    dest = Path(args.output or download_name)
    get_s3().download_file(get_bucket(), key, str(dest))
    _log(f"Saved manifest to {dest}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    try:
        get_lpad().connection.admin.command("ping")
    except Exception as exc:
        _emit({"status": "error", "error": str(exc)}, True)
        return 1
    _emit({"status": "ok", "fireworks": "connected", "s3_bucket": get_bucket()}, True)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_job_id_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-p", "--project-id", required=True)
    p.add_argument("-b", "--barcode", required=True)


def _add_job_type_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-j", "--job",
        action="append",
        required=True,
        choices=list(JOB_DEFINITIONS),
        metavar="JOB",
        help="Job type; repeat to run several in one workflow. "
             f"Choices: {', '.join(JOB_DEFINITIONS)}",
    )


def _add_wait_args(p: argparse.ArgumentParser, include_flag: bool = True) -> None:
    if include_flag:
        p.add_argument("--wait", action="store_true",
                       help="Block until all jobs finish (exit 1 if any failed)")
    p.add_argument("--timeout", type=float, default=0,
                   help="Max seconds to wait (0 = forever)")
    p.add_argument("--interval", type=float, default=5.0,
                   help="Polling interval in seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impulse_cli",
        description="Submit and inspect Impulse FireWorks jobs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="Upload inputs and submit jobs")
    _add_job_id_args(p)
    _add_job_type_arg(p)
    p.add_argument("--files", nargs="+", metavar="PATH",
                   help="Images/PDFs or directories to upload")
    p.add_argument("--xml", metavar="PATH", help="HathiTrust XML document to upload")
    p.add_argument("--skip-upload-check", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true")
    _add_wait_args(p)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("start", help="Submit jobs for inputs already in S3")
    _add_job_id_args(p)
    _add_job_type_arg(p)
    p.add_argument("--json", action="store_true")
    _add_wait_args(p)
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("status", help="Show FireWorks state and S3 summary")
    _add_job_id_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("wait", help="Block until a job's FireWorks finish")
    _add_job_id_args(p)
    _add_wait_args(p, include_flag=False)
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("files", help="List S3 files for a job")
    _add_job_id_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_files)

    p = sub.add_parser("manifest", help="Download the HathiTrust manifest")
    _add_job_id_args(p)
    p.add_argument("-o", "--output",
                   help="Destination path, or '-' for stdout "
                        "(default: <project>_<barcode>_<manifest name>)")
    p.add_argument("--url", action="store_true",
                   help="Print a presigned URL instead of downloading")
    p.add_argument("--expires", type=int, default=300,
                   help="Presigned URL lifetime in seconds (with --url)")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("health", help="Check MongoDB connectivity")
    p.set_defaults(func=cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
