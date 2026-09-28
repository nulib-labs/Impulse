#!/usr/bin/env bash
#
# submit_jobs.sh -- queue document_extraction + image_processing jobs.
#
# Each argument is a directory named <project_id>_<barcode> containing the
# images/PDFs to process:
#
#     data/
#       projA_39015012345678/  page1.png page2.png ...
#       projA_39015087654321/  scan1.tif scan2.tif ...
#
# The project ID and barcode are parsed from the directory name (split at the
# LAST underscore, so project IDs may contain underscores but barcodes may
# not), then impulse_cli.py uploads the files and inserts one workflow per
# directory into the FireWorks database. Nothing is run here -- workers pick
# the jobs up later.
#
# Usage:
#     submit_jobs.sh [-n] DIR [DIR ...]
#
#     submit_jobs.sh data/*/            # every job directory under data/
#     submit_jobs.sh -n data/*/         # dry run: show what would be submitted
#
# Options:
#     -n    Dry run (parse and print, but upload/submit nothing)
#     -h    Show this help
#
# Environment:
#     S3_BUCKET, AWS_PROFILE, AWS_REGION   required (used by impulse_cli.py)
#     MONGO_URI                            optional (default in impulse_cli.py)
#     IMPULSE_CLI                          path to impulse_cli.py
#                                          (default: next to this script)
#     PYTHON                               interpreter (default: python3)
#
# Exit status: 0 if every directory was submitted, 1 if any failed.

set -uo pipefail

JOBS=(document_extraction image_processing)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cli="${IMPULSE_CLI:-$script_dir/impulse_cli.py}"
python_bin="${PYTHON:-python3}"
dry_run=0

usage() {
    sed -n '2,/^$/p;/^# Usage:/,/^# Exit status/p' "${BASH_SOURCE[0]}" \
        | sed 's/^# \{0,1\}//' | sed '/^!/d'
}

while getopts ":nh" opt; do
    case "$opt" in
        n) dry_run=1 ;;
        h) usage; exit 0 ;;
        *) echo "error: unknown option -$OPTARG" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -eq 0 ]]; then
    echo "error: no directories given" >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$cli" ]]; then
    echo "error: impulse_cli.py not found at $cli (set IMPULSE_CLI)" >&2
    exit 1
fi

if (( ! dry_run )); then
    for var in S3_BUCKET AWS_PROFILE AWS_REGION; do
        if [[ -z "${!var:-}" ]]; then
            echo "error: required environment variable $var is not set" >&2
            exit 1
        fi
    done
fi

job_args=()
for job in "${JOBS[@]}"; do
    job_args+=(-j "$job")
done

submitted=0
failed=0

for dir in "$@"; do
    if [[ ! -d "$dir" ]]; then
        echo "error: not a directory: $dir" >&2
        failed=$((failed + 1))
        continue
    fi

    # Resolve to an absolute path so "." and trailing slashes behave.
    name="$(basename "$(cd "$dir" && pwd)")"

    if [[ "$name" != *_* ]]; then
        echo "error: $dir: directory name must be <project_id>_<barcode>" >&2
        failed=$((failed + 1))
        continue
    fi

    barcode="${name##*_}"
    project_id="${name%_*}"

    if [[ -z "$project_id" || -z "$barcode" ]]; then
        echo "error: $dir: empty project ID or barcode in '$name'" >&2
        failed=$((failed + 1))
        continue
    fi

    if [[ -z "$(find "$dir" -type f ! -name '.*' -print -quit)" ]]; then
        echo "error: $dir: no files found" >&2
        failed=$((failed + 1))
        continue
    fi

    echo "[$name] project_id=$project_id barcode=$barcode jobs=${JOBS[*]}"

    if (( dry_run )); then
        submitted=$((submitted + 1))
        continue
    fi

    if "$python_bin" "$cli" submit \
            -p "$project_id" -b "$barcode" \
            "${job_args[@]}" \
            --files "$dir"; then
        submitted=$((submitted + 1))
    else
        echo "error: $dir: submission failed" >&2
        failed=$((failed + 1))
    fi
done

if (( dry_run )); then verb="would be submitted"; else verb="submitted"; fi
echo "Done: $submitted $verb, $failed failed." >&2

(( failed == 0 ))
