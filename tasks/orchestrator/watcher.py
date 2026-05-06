"""
MongoDB change-stream watcher that launches EC2 instances for new fireworks.

Opens a change stream on the ``fireworks`` collection in the LaunchPad
database and reacts to new READY fireworks by spinning up EC2 workers
(respecting the configured ``max_instances`` cap).

The watcher is designed to be long-running.  It persists a resume token so
that restarts do not miss events that occurred while it was down.

Usage::

    uv run python -m tasks.orchestrator.watcher          # uses ec2_config.yaml
    uv run python -m tasks.orchestrator.watcher -c my.yaml  # custom config
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import certifi
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from loguru import logger

from tasks.orchestrator.config import OrchestratorConfig, load_config
from tasks.orchestrator.instance_manager import InstanceManager


# ── Graceful shutdown ─────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning("Received {} -- shutting down watcher", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Resume token persistence ─────────────────────────────────────────────


def _save_resume_token(
    client: MongoClient,
    db_name: str,
    token: dict,
) -> None:
    """Persist the change-stream resume token to a MongoDB collection."""
    db = client[db_name]
    db["_orchestrator_resume_tokens"].replace_one(
        {"_id": "watcher"},
        {"_id": "watcher", "token": token},
        upsert=True,
    )


def _load_resume_token(
    client: MongoClient,
    db_name: str,
) -> dict | None:
    """Load a previously saved resume token, or return None."""
    db = client[db_name]
    doc = db["_orchestrator_resume_tokens"].find_one({"_id": "watcher"})
    if doc and "token" in doc:
        logger.info("Resuming change stream from saved token")
        return doc["token"]
    return None


# ── Task-name extraction ─────────────────────────────────────────────────


def _extract_task_name(firework_doc: dict) -> str:
    """Best-effort extraction of the FireWorks task name from a firework document.

    The ``spec._tasks`` array contains dicts with a ``_fw_name`` key.
    Falls back to ``"unknown"`` if the structure is unexpected.
    """
    try:
        spec = firework_doc.get("spec", {})
        tasks = spec.get("_tasks", [])
        if tasks:
            return tasks[0].get("_fw_name", "unknown")
    except (KeyError, IndexError, TypeError):
        pass
    return "unknown"


# ── Core watcher logic ────────────────────────────────────────────────────


class Watcher:
    """Watches the FireWorks LaunchPad and launches EC2 workers on demand."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._instance_mgr = InstanceManager(config)
        self._client = MongoClient(
            config.mongo_uri,
            tls=True,
            tlsCAFile=certifi.where(),
        )
        self._db = self._client[config.mongo_db_name]
        self._collection = self._db["fireworks"]

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start watching -- blocks until a signal is received."""
        logger.info(
            "Watcher starting (db={}, max_instances={})",
            self._config.mongo_db_name,
            self._config.max_instances,
        )

        # On startup, check if there are already READY fireworks that need
        # workers.  This handles the case where jobs were submitted while
        # the orchestrator was down.
        self._check_backlog()

        while not _shutdown_requested:
            try:
                self._watch_loop()
            except PyMongoError as exc:
                if _shutdown_requested:
                    break
                logger.error("Change stream error: {} -- retrying in 10s", exc)
                time.sleep(10)

        logger.info("Watcher stopped")
        self._client.close()

    # ------------------------------------------------------------------

    def _check_backlog(self) -> None:
        """Launch workers for any READY fireworks that already exist."""
        ready_count = self._collection.count_documents({"state": "READY"})
        if ready_count == 0:
            logger.info("No backlog -- waiting for new fireworks")
            return

        logger.info("Found {} READY fireworks in backlog", ready_count)

        # Group by task type and launch instances up to the cap
        ready_fws = self._collection.find(
            {"state": "READY"},
            {"spec._tasks": 1},
        )

        task_names_seen: set[str] = set()
        for fw in ready_fws:
            task_name = _extract_task_name(fw)
            task_names_seen.add(task_name)

        for task_name in task_names_seen:
            if not self._instance_mgr.has_capacity():
                logger.warning(
                    "Instance cap ({}) reached -- remaining backlog will be "
                    "handled by running workers",
                    self._config.max_instances,
                )
                break
            self._launch_if_needed(task_name)

    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        """Open a change stream and process events until shutdown or error."""
        resume_token = _load_resume_token(
            self._client, self._config.mongo_db_name
        )

        # We watch for two patterns:
        # 1. Insert of a new firework with state READY
        # 2. Update that sets state to READY (e.g. a re-run)
        pipeline = [
            {
                "$match": {
                    "$or": [
                        {
                            "operationType": "insert",
                            "fullDocument.state": "READY",
                        },
                        {
                            "operationType": "update",
                            "updateDescription.updatedFields.state": "READY",
                        },
                    ]
                }
            }
        ]

        stream_kwargs: dict = {
            "pipeline": pipeline,
            "full_document": "updateLookup",
        }
        if resume_token:
            stream_kwargs["resume_after"] = resume_token

        logger.info("Opening change stream on {}.fireworks", self._config.mongo_db_name)

        with self._collection.watch(**stream_kwargs) as stream:
            for change in stream:
                if _shutdown_requested:
                    break

                # Persist resume token after every event
                _save_resume_token(
                    self._client,
                    self._config.mongo_db_name,
                    stream.resume_token,
                )

                doc = change.get("fullDocument", {})
                task_name = _extract_task_name(doc)
                fw_id = doc.get("fw_id", "?")

                logger.info(
                    "New READY firework detected: fw_id={}, task={}",
                    fw_id,
                    task_name,
                )

                self._launch_if_needed(task_name)

    # ------------------------------------------------------------------

    def _launch_if_needed(self, task_name: str) -> None:
        """Launch an EC2 instance if we are under the cap."""
        if not self._instance_mgr.has_capacity():
            active = self._instance_mgr.count_active_instances()
            logger.info(
                "At instance cap ({}/{}) -- job will be picked up by an "
                "existing worker's drain loop",
                active,
                self._config.max_instances,
            )
            return

        try:
            instance_id = self._instance_mgr.launch_instance(task_name)
            logger.info("Launched worker {} for task '{}'", instance_id, task_name)
        except Exception:
            logger.exception("Failed to launch EC2 instance for task '{}'", task_name)


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Impulse EC2 orchestrator -- watches for new fireworks "
        "and launches EC2 workers",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to the EC2 orchestrator YAML config file "
        "(default: ec2_config.yaml or $EC2_CONFIG_PATH)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if not config.mongo_uri:
        logger.error(
            "No MongoDB URI configured.  Set IMPULSE_MONGODB_URI or "
            "add mongo_uri to your ec2_config.yaml."
        )
        sys.exit(1)

    if not config.ami_id:
        logger.error(
            "No AMI ID configured.  Set EC2_AMI_ID or add ami_id to "
            "your ec2_config.yaml."
        )
        sys.exit(1)

    watcher = Watcher(config)
    watcher.run()


if __name__ == "__main__":
    main()
