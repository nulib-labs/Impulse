"""Impulse command-line interface.

    impulse upload    <identifier> --from DIR      push files to S3 + add a workflow
    impulse download  <identifier> [--text ...]    pull processed outputs back down
    impulse jobs      <job-type>   [...]           enqueue work over existing S3 data

Run with no arguments, or with ``--help`` on any subcommand, for details.

Examples:
    # Upload a folder of page images and queue the full pipeline
    impulse upload p1274_35556039349519 --from ./scans

    # Grab the OCR text for a set of barcodes, one directory per barcode
    impulse download --identifiers-file barcodes.txt --text --name-by accession

    # Binarize + deskew everything that still needs it
    impulse jobs image-processing --all

    # Convert METS XML to HathiTrust YAML
    impulse jobs mets-yaml p1274_35556039349519
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="impulse",
        description=(
            "Impulse pipeline CLI — move data in and out of the FireWorks "
            "launchpad and the Impulse S3 buckets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Imported lazily inside register() calls so that `--help` stays fast and
    # does not pull in torch/surya via tasks.*.
    from cli import cmd_download, cmd_jobs, cmd_upload

    cmd_upload.register(subparsers)
    cmd_download.register(subparsers)
    cmd_jobs.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
