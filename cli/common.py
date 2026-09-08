"""Shared helpers for the Impulse CLI.

Centralises the boilerplate that was previously duplicated across every
``scripts/submit_*.py``: LaunchPad construction, Mongo/S3 clients,
impulse_identifier normalisation, and OCR-text extraction.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Iterable, Iterator

# --------------------------------------------------------------------------
# Constants — the conventions the pipeline already follows
# --------------------------------------------------------------------------

DATA_BUCKET = "nu-impulse-data"
PRODUCTION_BUCKET = "nu-impulse-production"

RAW_IMAGES = "raw_images"
BINARIZED_IMAGES = "binarized_images"

PRAXIS_DB = "praxis"
COLT_COLLECTION = "colt"
HATHITRUST_COLLECTION = "HathiTrust"
LAUNCHPAD_DB = "fireworks"

DEFAULT_PROFILE = "impulse"
DEFAULT_REGION = "us-east-1"

METS_XML_NAME = "mets.xml"
METS_YAML_NAME = "mets.yaml"


# --------------------------------------------------------------------------
# Argument helpers
# --------------------------------------------------------------------------


def add_connection_args(parser) -> None:
    """Flags shared by every subcommand that touches Mongo or S3."""
    parser.add_argument(
        "--debug",
        dest="prod",
        action="store_false",
        default=True,
        help="Use IMPULSE_MONGODB_URI_DEBUG instead of the production URI.",
    )
    parser.add_argument(
        "--aws-profile",
        default=DEFAULT_PROFILE,
        help=f"boto3 profile name (default: {DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--bucket",
        default=DATA_BUCKET,
        help=f"Impulse data bucket (default: {DATA_BUCKET}).",
    )


def add_dry_run_arg(parser) -> None:
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing to S3 or the launchpad.",
    )


def add_identifier_selection_args(parser) -> None:
    """Flags for choosing which identifiers a command operates on."""
    parser.add_argument(
        "identifiers",
        nargs="*",
        help="One or more impulse_identifiers (e.g. p1274_35556039349519).",
    )
    parser.add_argument(
        "--identifiers-file",
        help="Path to a text file with one identifier (or bare barcode) per line.",
    )
    parser.add_argument(
        "--all",
        dest="select_all",
        action="store_true",
        help="Operate on every identifier discovered in the data bucket.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many identifiers.",
    )


# --------------------------------------------------------------------------
# Identifier handling
# --------------------------------------------------------------------------


def normalize_identifier(raw: str) -> str:
    """Apply the same normalisation the FireTasks apply to fw_spec values."""
    return (
        raw.strip()
        .replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace('"', "")
        .rstrip("/")
        .lower()
    )


def split_identifier(identifier: str) -> tuple[str, str]:
    """Split ``<project>_<accession>`` into its two parts.

    The FireTasks do ``identifier.split("_")[0]`` / ``[1]`` unguarded, so an
    identifier without an underscore would crash inside the worker. We fail
    fast here with a readable message instead.
    """
    parts = identifier.split("_")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"identifier {identifier!r} must look like <project>_<accession> "
            "(e.g. p1274_35556039349519)"
        )
    return parts[0].lower(), parts[1].lower()


def data_prefix(identifier: str, *subdir: str) -> str:
    """Build the nu-impulse-data key prefix for an identifier."""
    project, accession = split_identifier(identifier)
    return "/".join([project, accession, *subdir])


def production_prefix(identifier: str, *subdir: str) -> str:
    """Build the nu-impulse-production key prefix (``<PROJECT>_<ACCESSION>/``).

    The production bucket uses the joined identifier as a single path
    segment, and its casing is inconsistent (``P0491_...`` vs ``p1074_...``),
    so callers should treat a miss as "try the other casing".
    """
    return "/".join([identifier, *subdir])


def resolve_identifiers(args, s3=None, db=None, all_from_db: bool = False) -> list[str]:
    """Resolve the identifier list from CLI args.

    Accepts explicit identifiers, a file of identifiers, or ``--all``.
    Bare barcodes (no underscore) are resolved against the known identifier
    list so users can paste accession numbers directly.

    ``s3`` may be a client or a zero-arg factory (see :func:`s3_factory`); it
    is only invoked when S3 is actually required, so commands that need only
    Mongo never touch AWS.

    ``all_from_db`` makes ``--all`` enumerate identifiers from ``praxis.colt``
    instead of the S3 bucket, which lets Mongo-only users run bulk text
    downloads without AWS credentials.
    """

    def _s3():
        return s3() if callable(s3) else s3

    raw: list[str] = [normalize_identifier(x) for x in (args.identifiers or [])]

    if getattr(args, "identifiers_file", None):
        try:
            with open(args.identifiers_file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        raw.append(normalize_identifier(line))
        except OSError as exc:
            raise SystemExit(
                f"Error: cannot read --identifiers-file {args.identifiers_file!r}: {exc}"
            ) from exc

    if getattr(args, "select_all", False):
        if all_from_db and db is not None:
            raw.extend(sorted(list_colt_identifiers(db)))
        else:
            raw.extend(list_data_identifiers(_s3(), args.bucket))

    # Resolve bare barcodes (no underscore) against known identifiers.
    needs_lookup = [x for x in raw if "_" not in x]
    resolved: list[str] = [x for x in raw if "_" in x]
    if needs_lookup:
        # Prefer Mongo for the lookup so a text-only download never needs AWS.
        known: set[str] = set()
        if db is not None:
            known |= set(list_colt_identifiers(db))
        if not known and s3 is not None:
            known |= set(list_data_identifiers(_s3(), args.bucket))

        for bare in needs_lookup:
            hits = sorted(k for k in known if k.endswith("_" + bare))
            if len(hits) == 1:
                resolved.append(hits[0])
            elif not hits:
                print(f"[warn] no identifier found for barcode {bare!r}", file=sys.stderr)
            else:
                print(
                    f"[warn] barcode {bare!r} is ambiguous: {hits} — skipping",
                    file=sys.stderr,
                )

    # De-duplicate, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for x in resolved:
        if x not in seen:
            seen.add(x)
            out.append(x)

    limit = getattr(args, "limit", None)
    if limit is not None:
        out = out[:limit]
    return out


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------


def get_mongo_uri(prod: bool) -> str:
    var = "IMPULSE_MONGODB_URI" if prod else "IMPULSE_MONGODB_URI_DEBUG"
    uri = os.getenv(var)
    if not uri:
        raise SystemExit(f"Error: environment variable {var} is not set.")
    return uri


def get_praxis_db(prod: bool):
    import certifi
    from pymongo import MongoClient

    client = MongoClient(get_mongo_uri(prod), tls=True, tlsCAFile=certifi.where())
    return client[PRAXIS_DB]


def get_launchpad(prod: bool):
    """Build a LaunchPad using the exact kwargs the submit scripts use."""
    import certifi
    from fireworks import LaunchPad

    return LaunchPad(
        uri_mode=True,
        host=get_mongo_uri(prod),
        name=LAUNCHPAD_DB,
        mongoclient_kwargs={"tlsCAFile": certifi.where()},
    )


def get_s3(profile: str = DEFAULT_PROFILE, region: str = DEFAULT_REGION):
    """Build an S3 client, failing with a readable message on bad credentials."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import NoCredentialsError, ProfileNotFound

    try:
        session = boto3.Session(profile_name=profile)
        return session.client(
            "s3",
            region_name=region,
            config=Config(max_pool_connections=64, retries={"max_attempts": 3}),
        )
    except ProfileNotFound:
        raise SystemExit(
            f"Error: AWS profile {profile!r} not found.\n"
            "  Configure it with:  aws configure --profile impulse\n"
            "  Or select another:  --aws-profile <name>"
        ) from None
    except NoCredentialsError:
        raise SystemExit(
            f"Error: no AWS credentials available for profile {profile!r}."
        ) from None


def s3_factory(args):
    """Return a zero-arg callable that builds the S3 client on first use.

    Lets commands that may not need S3 at all (e.g. ``download --text``)
    avoid requiring AWS credentials.
    """
    cache: dict[str, object] = {}

    def get():
        if "client" not in cache:
            cache["client"] = get_s3(args.aws_profile, args.region)
        return cache["client"]

    return get


# --------------------------------------------------------------------------
# S3 helpers
# --------------------------------------------------------------------------


def iter_keys(s3, bucket: str, prefix: str, suffixes: tuple[str, ...] = ()) -> Iterator[str]:
    """Yield every object key under ``prefix``, optionally filtered by suffix."""
    paginator = s3.get_paginator("list_objects_v2")
    lowered = tuple(s.lower() for s in suffixes)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if lowered and not key.lower().endswith(lowered):
                continue
            yield key


def list_common_prefixes(s3, bucket: str, prefix: str = "") -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"])
    return out


def list_data_identifiers(s3, bucket: str = DATA_BUCKET) -> list[str]:
    """Discover ``<project>_<accession>`` identifiers in the data bucket."""
    if s3 is None:
        return []
    out: list[str] = []
    for project_prefix in list_common_prefixes(s3, bucket):
        project = project_prefix.rstrip("/")
        for acc_prefix in list_common_prefixes(s3, bucket, project_prefix):
            accession = acc_prefix.rstrip("/").split("/")[-1]
            out.append(f"{project}_{accession}")
    return sorted(out)


def list_colt_identifiers(db) -> list[str]:
    return [
        x
        for x in db[COLT_COLLECTION].distinct("impulse_identifier")
        if isinstance(x, str) and x.strip()
    ]


def key_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "403"):
            return False
        raise


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


# --------------------------------------------------------------------------
# OCR text extraction
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def clean_html(html: str) -> str:
    """Convert an HTML fragment to a single line of plain text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["math", "script", "style"]):
        tag.decompose()
    return _WS.sub(" ", soup.get_text(separator=" ", strip=True)).strip()


def _find_html_values(node, out: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "html" and isinstance(value, str) and value.strip():
                out.append(value)
            else:
                _find_html_values(value, out)
    elif isinstance(node, list):
        for item in node:
            _find_html_values(item, out)


def page_text(doc: dict) -> str:
    """Extract plain text for one colt document.

    ``colt`` contains two schema variants:

    * surya-2 documents store blocks under ``ocr_data.blocks[]`` with a
      ``reading_order`` field, which we sort on.
    * older chandra documents store a nested ``extracted_data`` tree; we walk
      it for ``html`` values.

    Blocks are joined with newlines. Returns "" for pages with no text.
    """
    ocr = doc.get("ocr_data")
    if isinstance(ocr, dict) and isinstance(ocr.get("blocks"), list):
        def order(block):
            ro = block.get("reading_order") if isinstance(block, dict) else None
            return ro if isinstance(ro, (int, float)) else float("inf")

        lines: list[str] = []
        for block in sorted(ocr["blocks"], key=order):
            if not isinstance(block, dict):
                continue
            html = block.get("html")
            if isinstance(html, str) and html.strip():
                text = clean_html(html)
                if text:
                    lines.append(text)
        return "\n".join(lines)

    extracted = doc.get("extracted_data")
    if extracted:
        found: list[str] = []
        _find_html_values(extracted, found)
        lines = [clean_html(h) for h in found]
        return "\n".join(x for x in lines if x)

    return ""


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def humanize_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def chunked(items: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
