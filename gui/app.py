import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import boto3
from botocore.exceptions import ClientError
from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from fireworks import Firework, LaunchPad, Workflow
from natsort import natsorted

import tasks

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 500 * 1024 * 1024)
)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = "fireworks"

S3_BUCKET = os.environ["S3_BUCKET"]
AWS_PROFILE = os.environ["AWS_PROFILE"]
AWS_REGION = os.environ["AWS_REGION"]

# TODO: set to whatever filename your HathiTrust task writes its manifest to.
HATHITRUST_MANIFEST_NAME = os.environ.get(
    "HATHITRUST_MANIFEST_NAME", "manifest.json"
)

# Project IDs / barcodes end up in S3 keys, so keep them boring.
ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# FireWorks LaunchPad. This application only inserts workflows into MongoDB.
# It does NOT run FireWorks locally.
lpad = LaunchPad(host=MONGO_URI, name=MONGO_DB, uri_mode=True)
print(lpad.host)
print(lpad.name)

session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
s3 = session.client("s3")


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobDefinition:
    """Configuration describing a FireWorks task exposed by the web UI."""

    name: str
    label: str
    description: str
    task_class: type
    requires_files: bool = True          # uploaded images / PDFs
    requires_xml: bool = False           # a single uploaded XML document
    requires_impulse_identifier: bool = False


JOB_DEFINITIONS: dict[str, JobDefinition] = {
    "document_extraction": JobDefinition(
        name="document_extraction",
        label="Document Extraction",
        description=(
            "Run OCR/layout extraction on uploaded images or PDFs "
            "using the DocumentExtractionTask."
        ),
        task_class=tasks.DocumentExtractionTask,
        requires_files=True,
        requires_impulse_identifier=True,
    ),
    "image_processing": JobDefinition(
        name="image_processing",
        label="Image Processing",
        description=(
            "Process uploaded S3 images with grayscale, denoising, "
            "illumination normalization, Sauvola binarization, and deskewing."
        ),
        task_class=tasks.ImageProcessingTask,
        requires_files=True,
        requires_impulse_identifier=True,
    ),
    "hathitrust": JobDefinition(
        name="hathitrust",
        label="HathiTrust Create Manifest",
        description=(
            "Upload an XML document and run the HathiTrust processing job. "
            "The resulting manifest can be downloaded from the job page."
        ),
        # TODO: implement tasks.HathiTrustProcessingTask (stub in the notes).
        task_class=tasks.CreateHathiTrustManifest,
        requires_files=False,
        requires_xml=True,
        requires_impulse_identifier=True,
    ),
}


def get_job_definition(job_type: str) -> JobDefinition:
    """Return a configured job definition or abort with HTTP 404."""

    try:
        return JOB_DEFINITIONS[job_type]
    except KeyError:
        abort(404, description=f"Unknown job type: {job_type}")


# ---------------------------------------------------------------------------
# Identifier / S3 key layout
# ---------------------------------------------------------------------------
#
#   jobs/<project_id>/<barcode>/uploaded_images/0000000001.png ...
#   jobs/<project_id>/<barcode>/processed_images/...
#   jobs/<project_id>/<barcode>/hathitrust/input.xml
#   jobs/<project_id>/<barcode>/outputs/<job_type>/...   (task outputs)
#
# Each job type writes to its own outputs/ prefix so that jobs running
# concurrently on the same images never clobber each other.

def validate_ids(project_id: str, barcode: str) -> None:
    if not ID_PATTERN.match(project_id) or not ID_PATTERN.match(barcode):
        raise ValueError(
            "Project ID and barcode may only contain letters, numbers, "
            "'.', '_' and '-'."
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
    else:
        return f"{job_prefix(project_id, barcode)}/outputs/{job_type}/"


def hathitrust_manifest_key(project_id: str, barcode: str) -> str:
    return output_prefix(project_id, barcode, "hathitrust") + HATHITRUST_MANIFEST_NAME


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """
    Safely convert a browser-supplied filename into an S3-relative path.

    We allow nested directories, but reject absolute paths and '..'
    traversal components.
    """

    if not filename:
        raise ValueError("Empty filename")

    filename = filename.replace("\\", "/")

    if filename.startswith("/"):
        raise ValueError(f"Absolute path not allowed: {filename}")

    if re.match(r"^[A-Za-z]:/", filename):
        raise ValueError(f"Absolute path not allowed: {filename}")

    parts = []
    for part in filename.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"Path traversal not allowed: {filename}")
        parts.append(part)

    if not parts:
        raise ValueError(f"Invalid filename: {filename}")

    return "/".join(parts)


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_to_s3(project_id: str, barcode: str, files) -> list[str]:
    """
    Upload Flask FileStorage objects to:

        s3://<bucket>/jobs/<project_id>/<barcode>/uploaded_images/<n>.<ext>

    Returns the full S3 keys. Files are naturally sorted and renamed to
    zero-padded sequence numbers.
    """

    keys: list[str] = []
    i = 1

    for file_storage in natsorted(files, key=lambda f: f.filename or ""):
        if not file_storage or not file_storage.filename:
            continue

        try:
            relative_path = sanitize_filename(file_storage.filename)
        except ValueError:
            continue

        ext = Path(relative_path).suffix.lower()
        key = f"{images_prefix(project_id, barcode)}{i:010d}{ext}"

        extra_args: dict[str, Any] = {}
        content_type = getattr(file_storage, "content_type", None)
        if content_type:
            extra_args["ContentType"] = content_type

        file_storage.stream.seek(0)
        s3.upload_fileobj(
            file_storage.stream,
            S3_BUCKET,
            key,
            ExtraArgs=extra_args or None,
        )

        keys.append(key)
        i += 1

    return keys


def validate_xml(data: bytes) -> None:
    """
    Validate an uploaded XML document. Raise ValueError if it's unacceptable.

    TODO: fill in your logic (schema/XSD validation, required elements,
    barcode matches the job, etc.).

    NOTE: for untrusted uploads, prefer `defusedxml.ElementTree` over the
    stdlib parser to avoid entity-expansion attacks.
    """

    try:
        ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML: {exc}") from exc


def upload_xml_to_s3(project_id: str, barcode: str, file_storage) -> str:
    """Validate and upload the HathiTrust XML. Returns the S3 key."""

    if not file_storage or not file_storage.filename:
        raise ValueError("Please select an XML file.")

    data = file_storage.read()
    if not data:
        raise ValueError("The XML file is empty.")

    validate_xml(data)

    key = xml_key(project_id, barcode)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=data,
        ContentType="application/xml",
    )
    return key


def list_job_keys(project_id: str, barcode: str, prefix: str | None = None) -> list[str]:
    """
    Return every S3 key under `prefix` (defaults to the whole job).

    Uses a paginator because a job can contain more than 1,000 objects.
    """

    prefix = prefix or f"{job_prefix(project_id, barcode)}/"
    paginator = s3.get_paginator("list_objects_v2")

    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    return keys


def list_uploaded_image_keys(project_id: str, barcode: str) -> list[str]:
    """Only the uploaded images -- NOT the XML or any task outputs."""

    return list_job_keys(project_id, barcode, prefix=images_prefix(project_id, barcode))


def list_job_files(project_id: str, barcode: str) -> list[str]:
    """Return filenames relative to the job's S3 directory."""

    prefix = f"{job_prefix(project_id, barcode)}/"
    return [key[len(prefix):] for key in list_job_keys(project_id, barcode)]


# ---------------------------------------------------------------------------
# Downloadable artifact categories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactDefinition:
    """A category of job files the user can pick for download."""

    label: str
    prefix: str                              # relative to jobs/<project>/<barcode>/
    suffixes: tuple[str, ...] = ()           # empty = every file under prefix
    exclude_prefixes: tuple[str, ...] = ()   # relative to the job prefix


ARTIFACT_DEFINITIONS: dict[str, ArtifactDefinition] = {
    "uploaded_images": ArtifactDefinition(
        label="Uploaded images",
        prefix="uploaded_images/",
    ),
    "processed_images": ArtifactDefinition(
        label="Processed images",
        prefix="processed_images/",
    ),
    "txt": ArtifactDefinition(
        label="Text files (.txt)",
        prefix="txt/",
        suffixes=(".txt",),
    ),
    "json": ArtifactDefinition(
        label="JSON files (.json)",
        prefix="json/",
        suffixes=(".json",),
        # The HathiTrust manifest may be JSON; it has its own categories.
        exclude_prefixes=("outputs/hathitrust/",),
    ),
    "hathitrust_xml": ArtifactDefinition(
        label="HathiTrust XML manifest",
        prefix="",
        suffixes=(".xml",),
    ),
    "hathitrust_yaml": ArtifactDefinition(
        label="HathiTrust YAML manifest",
        prefix="",
        suffixes=(".yaml", ".yml"),
    ),
    "hathitrust_zip": ArtifactDefinition(
        label="Zipped HathiTrust manifest",
        prefix="outputs/hathitrust/",
        suffixes=(".zip",),
    ),
}


def collect_artifact_keys(
    project_id: str, barcode: str, artifact_types: list[str]
) -> list[str]:
    """Return the de-duplicated, sorted S3 keys for the selected artifact types."""

    base = f"{job_prefix(project_id, barcode)}/"
    found: set[str] = set()

    for artifact_type in artifact_types:
        definition = ARTIFACT_DEFINITIONS[artifact_type]
        for key in list_job_keys(project_id, barcode, prefix=base + definition.prefix):
            relative = key[len(base):]
            if relative.endswith("/"):  # S3 "folder" placeholder objects
                continue
            if definition.suffixes and not relative.lower().endswith(definition.suffixes):
                continue
            if any(relative.startswith(p) for p in definition.exclude_prefixes):
                continue
            found.add(key)

    return natsorted(found)


# ---------------------------------------------------------------------------
# FireWorks spec construction
# ---------------------------------------------------------------------------

def build_spec(project_id: str, barcode: str, job_type: str) -> dict[str, Any]:
    """
    Build the FireWorks spec for ONE job type. Nothing is executed here.
    """

    impulse_identifier = make_impulse_identifier(project_id, barcode)
    definition = get_job_definition(job_type)

    spec: dict[str, Any] = {
        "job_type": job_type,
        "job_name": definition.label,
        "s3_bucket": S3_BUCKET,
        "project_id": project_id,
        "barcode": barcode,
        # Each job type writes to its own prefix so concurrent jobs
        # sharing the same input images don't overwrite each other.
        "output_prefix": output_prefix(project_id, barcode, job_type),
    }

    if definition.requires_files:
        uploaded_files = list_uploaded_image_keys(project_id, barcode)
        if not uploaded_files:
            raise ValueError(f"No uploaded files found for job {impulse_identifier}")

        spec["find_path_array_in"] = "uploaded_files"
        spec["uploaded_files"] = uploaded_files

    if definition.requires_xml:
        key = xml_key(project_id, barcode)
        if not s3_key_exists(key):
            raise ValueError(f"No XML document found for job {impulse_identifier}")

        spec["xml_key"] = key
        # TODO: add any extra HathiTrust-specific spec values here.

    if definition.requires_impulse_identifier:
        spec["impulse_identifier"] = impulse_identifier

    return spec


def build_specs(
    project_id: str, barcode: str, job_types: list[str]
) -> dict[str, dict[str, Any]]:
    return {jt: build_spec(project_id, barcode, jt) for jt in job_types}


# ---------------------------------------------------------------------------
# FireWorks submission
# ---------------------------------------------------------------------------

def submit_fireworks_jobs(
    impulse_identifier: str,
    specs: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """
    Insert ONE workflow containing one Firework per selected job type.

    Jobs are independent (and run in parallel given enough workers, e.g.
    multiple `rlaunch` processes or a queue adapter), except that HathiTrust
    waits for every other selected job to finish.

    Returns the id map from add_wf.
    """

    fireworks = {}
    for job_type, spec in specs.items():
        definition = get_job_definition(job_type)
        fireworks[job_type] = Firework(
            definition.task_class(),
            spec=spec,
            name=f"job-{impulse_identifier}-{job_type}",
        )

    # HathiTrust runs only after every other job has completed
    links = {}
    hathitrust_fw = next(
        (
            fw
            for job_type, fw in fireworks.items()
            if get_job_definition(job_type).name == "hathitrust"
        ),
        None,
    )
    if hathitrust_fw is not None:
        for fw in fireworks.values():
            if fw is not hathitrust_fw:
                links[fw] = [hathitrust_fw]

    workflow = Workflow(
        list(fireworks.values()),
        links_dict=links,
        name=f"impulse-{impulse_identifier}",
    )

    # add_wf maps the original (temporary) fw_ids to the database ids.
    id_map = lpad.add_wf(workflow)

    return id_map


# ---------------------------------------------------------------------------
# FireWorks inspection
# ---------------------------------------------------------------------------

def find_fireworks_for_job(impulse_identifier: str) -> list[dict[str, Any]]:
    """Find FireWorks associated with a job (read-only Mongo query)."""

    documents = lpad.fireworks.find(
        {"spec.impulse_identifier": impulse_identifier}
    ).sort([("created_on", -1)])

    return [
        {
            "fw_id": d.get("fw_id"),
            "name": d.get("name"),
            "state": d.get("state"),
            "spec": d.get("spec", {}),
        }
        for d in documents
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_index(error: str | None = None, status: int = 200):
    return (
        render_template(
            "index.html",
            jobs=JOB_DEFINITIONS,
            artifacts=ARTIFACT_DEFINITIONS,
            error=error,
        ),
        status,
    )


def render_job(project_id: str, barcode: str, status: int = 200, **context):
    impulse_identifier = make_impulse_identifier(project_id, barcode)

    context.setdefault("fireworks", find_fireworks_for_job(impulse_identifier))

    return (
        render_template(
            "job.html",
            project_id=project_id,
            barcode=barcode,
            impulse_identifier=impulse_identifier,
            files=list_job_files(project_id, barcode),
            hathitrust_manifest_available=s3_key_exists(
                hathitrust_manifest_key(project_id, barcode)
            ),
            artifacts=ARTIFACT_DEFINITIONS,
            **context,
        ),
        status,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload page.

    The user can:
      1. Select ONE OR MORE jobs (they share the same uploaded files).
      2. Enter the project ID + barcode.
      3. Upload images/PDFs (for image-based jobs) and/or an XML document
         (for the HathiTrust job).
      4. Create all selected jobs in a single FireWorks workflow.
    """

    if request.method != "POST":
        return render_index()

    job_types = [jt.strip() for jt in request.form.getlist("job_type") if jt.strip()]
    if not job_types:
        return render_index("Please select at least one job.", 400)

    definitions = [get_job_definition(jt) for jt in job_types]

    project_id = request.form.get("project_id", "").strip()
    barcode = request.form.get("barcode", "").strip()

    if any(d.requires_impulse_identifier for d in definitions) and not (
        project_id and barcode
    ):
        return render_index("A project ID and barcode are required.", 400)

    try:
        validate_ids(project_id, barcode)
    except ValueError as exc:
        return render_index(str(exc), 400)

    impulse_identifier = make_impulse_identifier(project_id, barcode)

    needs_images = any(d.requires_files for d in definitions)
    needs_xml = any(d.requires_xml for d in definitions)

    # Upload inputs ONCE, before creating any workflow. Both image-based
    # jobs then reference the same S3 keys.
    if needs_images:
        files = request.files.getlist("file")
        keys = upload_to_s3(project_id, barcode, files)
        if not keys:
            return render_index("Please select at least one valid file.", 400)

    if needs_xml:
        try:
            upload_xml_to_s3(project_id, barcode, request.files.get("xml_file"))
        except ValueError as exc:
            return render_index(str(exc), 400)

    try:
        specs = build_specs(project_id, barcode, job_types)
        fw_ids = submit_fireworks_jobs(impulse_identifier, specs)
    except Exception as exc:
        app.logger.exception("Failed to submit FireWorks jobs %s", impulse_identifier)
        return render_index(f"Failed to create FireWorks jobs: {exc}", 500)

    app.logger.info("Submitted %s: %s", impulse_identifier, fw_ids)

    return redirect(url_for("job", project_id=project_id, barcode=barcode))


@app.route("/jobs/<project_id>/<barcode>")
def job(project_id: str, barcode: str):
    """Display the uploaded files, FireWorks state, and manifest link."""

    try:
        validate_ids(project_id, barcode)
    except ValueError:
        abort(404)

    files = list_job_files(project_id, barcode)
    fireworks = find_fireworks_for_job(make_impulse_identifier(project_id, barcode))

    if not files and not fireworks:
        abort(404)

    return render_job(project_id, barcode, fireworks=fireworks)


@app.route("/jobs/<project_id>/<barcode>/start", methods=["POST"])
def start_job(project_id: str, barcode: str):
    """
    Submit one or more jobs for inputs that are already in S3.

    The `job_type` form field may be repeated to start several jobs at once.
    """

    try:
        validate_ids(project_id, barcode)
    except ValueError:
        abort(404)

    job_types = [
        jt.strip() for jt in request.form.getlist("job_type") if jt.strip()
    ] or ["document_extraction"]

    for jt in job_types:
        get_job_definition(jt)  # 404 on unknown types

    impulse_identifier = make_impulse_identifier(project_id, barcode)

    try:
        specs = build_specs(project_id, barcode, job_types)
        fw_ids = submit_fireworks_jobs(impulse_identifier, specs)
    except Exception as exc:
        app.logger.exception("Failed to submit FireWorks jobs %s", impulse_identifier)
        return render_job(project_id, barcode, 500, error=f"Failed to submit job: {exc}")

    labels = ", ".join(JOB_DEFINITIONS[jt].label for jt in job_types)
    return render_job(
        project_id,
        barcode,
        message=f"Submitted {labels} to FireWorks (fw_ids: {fw_ids})",
    )


@app.route("/jobs/<project_id>/<barcode>/hathitrust/manifest")
def download_hathitrust_manifest(project_id: str, barcode: str):
    """
    Download the HathiTrust manifest produced by the HathiTrust job.

    Redirects to a short-lived presigned S3 URL so the file doesn't stream
    through Flask. The manifest is expected at
    hathitrust_manifest_key(project_id, barcode).
    """

    try:
        validate_ids(project_id, barcode)
    except ValueError:
        abort(404)

    key = hathitrust_manifest_key(project_id, barcode)

    if not s3_key_exists(key):
        abort(404, description="Manifest not found (has the job finished?)")

    download_name = f"{project_id}_{barcode}_{HATHITRUST_MANIFEST_NAME}"

    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
            "ResponseContentDisposition": f'attachment; filename="{download_name}"',
        },
        ExpiresIn=300,
    )

    # TODO (optional): instead of redirecting, stream/transform the manifest:
    #   obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    #   return Response(obj["Body"].iter_chunks(), mimetype=..., headers=...)
    return redirect(url)


@app.route("/download", methods=["POST"])
def download_artifacts():
    """Zip the selected artifact categories for a job and send them."""

    project_id = request.form.get("project_id", "").strip()
    barcode = request.form.get("barcode", "").strip()

    try:
        validate_ids(project_id, barcode)
    except ValueError as exc:
        return render_index(str(exc), 400)

    selected = [
        a for a in request.form.getlist("artifact") if a in ARTIFACT_DEFINITIONS
    ]
    if not selected:
        return render_index("Please select at least one artifact to download.", 400)

    keys = collect_artifact_keys(project_id, barcode, selected)
    if not keys:
        return render_index("No matching files found for that job.", 404)

    base = f"{job_prefix(project_id, barcode)}/"

    # Spools to disk past 100 MB so big image sets don't sit in RAM.
    tmp = tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in keys:
            # Mirrors the S3 layout inside the zip, so categories sharing a
            # prefix (txt/json under outputs/) never collide.
            with zf.open(key[len(base):], "w", force_zip64=True) as dest:
                s3.download_fileobj(S3_BUCKET, key, dest)
    tmp.seek(0)

    return send_file(
        tmp,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{project_id}_{barcode}_artifacts.zip",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Verify that the LaunchPad's MongoDB connection is usable."""

    try:
        lpad.connection.admin.command("ping")
    except Exception as exc:
        return {"status": "error", "error": str(exc)}, 503

    return {"status": "ok", "fireworks": "connected", "s3_bucket": S3_BUCKET}


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "0.0.0.0"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
