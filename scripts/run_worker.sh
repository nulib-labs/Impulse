#!/usr/bin/env bash
# Launch a FireWorks worker with the repo root on PYTHONPATH so that the
# `tasks` package can be imported when fireworks are deserialized.
#
# Usage:
#   scripts/run_worker.sh                    # rapidfire (default)
#   scripts/run_worker.sh singleshot         # run one firework and exit
#   scripts/run_worker.sh rapidfire --nlaunches 5
#
# Assumes:
#   - Repo is cloned somewhere and this script lives in <repo>/scripts/.
#   - A venv exists at <repo>/.venv (skipped if it doesn't).
#   - my_launchpad.yaml and my_fireworker.yaml live at the repo root.

set -euo pipefail

# Resolve <repo> = parent dir of this script, regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Put the repo root on PYTHONPATH so `import tasks` works from any CWD.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Tell FireWorks where to find FW_config.yaml (holds ADD_USER_PACKAGES: [tasks]).
export FW_CONFIG_FILE="${REPO_ROOT}/FW_config.yaml"

# Point Surya at the locally-running vLLM Docker container instead of letting it
# spawn its own. Must be the OpenAI-compatible base (ends in /v1); Surya derives
# the /health probe by stripping /v1. AUTOSTART=False makes an unreachable URL
# fail fast with a clear error instead of trying to spawn a container.
export SURYA_INFERENCE_URL="http://127.0.0.1:8000/v1"
export SURYA_INFERENCE_AUTOSTART=False

# Activate the local venv if present.
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

cd "$REPO_ROOT"

MODE="${1:-rapidfire}"
shift || true

exec rlaunch \
    -w "${REPO_ROOT}/my_fireworker.yaml" \
    -l "${REPO_ROOT}/my_launchpad.yaml" \
    "$MODE" "$@"
