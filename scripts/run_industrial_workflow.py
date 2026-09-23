"""Poll the industrial DocumentExtraction jobs until complete, then run the
full output workflow (bbox+OCR images, combined HTML, table CSVs) for each.

Polls FireWorks in the ``fireworks`` LaunchPad db for the given identifiers.
Once every job has reached a terminal state, runs the three output scripts
per identifier:

  * scripts/render_bboxes.py   -> annotated page images
  * scripts/build_html.py      -> combined HTML
  * scripts/extract_tables.py  -> per-table CSVs

Outputs go to ``eval_output/<identifier>/``.

Run with:  uv run python scripts/run_industrial_workflow.py
Requires:  env IMPULSE_MONGODB_URI, AWS profile "impulse".
"""

import os
import subprocess
import sys
import time

import certifi
from pymongo import MongoClient

IDENTIFIERS = [
    "1920_ex1",
    "1920_ex2",
    "1920_ex3",
    "1920_ex4",
    "1928_ex5",
    "1929_ex6",
]
OUT_ROOT = "eval_output"
POLL_SECONDS = 30

MONGO_URI = os.getenv("IMPULSE_MONGODB_URI")

TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FIZZLED", "DEFUSED", "ARCHIVED"}


def get_fireworks_db():
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    return client["fireworks"]


def poll_until_done(fw_db) -> dict[str, dict]:
    """Block until every identifier's firework hits a terminal state.

    Returns a map identifier -> {"state", "fw_id", "n_pages"}.
    """
    while True:
        cursor = fw_db["fireworks"].find(
            {"spec.impulse_identifier": {"$in": IDENTIFIERS}},
            {"fw_id": 1, "state": 1, "spec.impulse_identifier": 1, "spec.keys": 1},
        )
        by_id: dict[str, dict] = {}
        for fw in cursor:
            ident = fw["spec"]["impulse_identifier"]
            by_id[ident] = {
                "state": fw.get("state"),
                "fw_id": fw.get("fw_id"),
                "n_pages": len(fw.get("spec", {}).get("keys", []) or []),
            }

        missing = [i for i in IDENTIFIERS if i not in by_id]
        states = {i: by_id.get(i, {}).get("state") for i in IDENTIFIERS}
        pending = [
            i
            for i in IDENTIFIERS
            if states.get(i) not in (TERMINAL_OK | TERMINAL_BAD)
        ]

        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] states: {states}")
        if missing:
            print(f"  [warn] not found in LaunchPad: {missing}")

        if not pending:
            return by_id

        time.sleep(POLL_SECONDS)


def run_step(script: str, identifier: str, env: dict) -> bool:
    print(f"  -> {script} {identifier}")
    result = subprocess.run(
        ["uv", "run", "python", script, identifier],
        env=env,
    )
    if result.returncode != 0:
        print(f"  [error] {script} failed for {identifier} (rc={result.returncode})")
        return False
    return True


def main() -> None:
    if not MONGO_URI:
        raise SystemExit("IMPULSE_MONGODB_URI is not set")

    fw_db = get_fireworks_db()
    print(f"Polling {len(IDENTIFIERS)} jobs every {POLL_SECONDS}s: {IDENTIFIERS}")
    status = poll_until_done(fw_db)

    completed = [i for i in IDENTIFIERS if status.get(i, {}).get("state") in TERMINAL_OK]
    failed = [i for i in IDENTIFIERS if status.get(i, {}).get("state") in TERMINAL_BAD]
    print(f"\nAll jobs terminal. completed={completed} failed={failed}")

    if failed:
        print("[warn] some jobs did not complete successfully; "
              "outputs will only reflect data actually persisted.")

    ran, errored = [], []
    for ident in completed:
        n_pages = status[ident]["n_pages"] or 50
        out_dir = os.path.join(OUT_ROOT, ident)
        os.makedirs(out_dir, exist_ok=True)
        env = {
            **os.environ,
            "IMPULSE_IDENTIFIER": ident,
            "IMPULSE_MAX_PAGES": str(n_pages),
            "IMPULSE_OUT_DIR": out_dir,
        }
        print(f"\n=== {ident} ({n_pages} pages) -> {out_dir}/ ===")
        ok = True
        ok &= run_step("scripts/render_bboxes.py", ident, env)
        ok &= run_step("scripts/build_html.py", ident, env)
        ok &= run_step("scripts/extract_tables.py", ident, env)
        (ran if ok else errored).append(ident)

    print(f"\nDone. outputs_ran={ran} errored={errored} skipped_failed_jobs={failed}")


if __name__ == "__main__":
    main()
