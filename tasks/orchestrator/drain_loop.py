"""
Drain-loop worker that runs on each EC2 instance.

Repeatedly pulls and executes READY fireworks from the LaunchPad.
When no work remains for ``--idle-timeout`` seconds the process exits,
allowing the instance's user-data script to trigger a shutdown.

Usage (on the EC2 instance)::

    uv run python -m tasks.orchestrator.drain_loop \
        --launchpad /opt/impulse/worker_launchpad.yaml \
        --idle-timeout 300
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import certifi
from fireworks import FWorker, LaunchPad
from loguru import logger


# ── Graceful shutdown handling ────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning("Received {} -- will exit after current job completes", sig_name)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Helpers ───────────────────────────────────────────────────────────────


def _pending_count(launchpad: LaunchPad) -> int:
    """Return the number of READY fireworks in the LaunchPad."""
    return launchpad.fireworks.count_documents({"state": "READY"})


def _launch_one_rocket(launchpad: LaunchPad) -> bool:
    """Attempt to pull and run a single READY firework.

    Returns True if a firework was executed (regardless of outcome),
    False if no READY firework was available.
    """
    from fireworks.core.rocket_launcher import launch_rocket

    launched = launch_rocket(launchpad, FWorker())
    return launched is not None


# ── Main loop ─────────────────────────────────────────────────────────────


def drain(launchpad: LaunchPad, idle_timeout: int) -> None:
    """Execute fireworks until the queue is empty and idle timeout elapses."""
    idle_since: float | None = None

    logger.info(
        "Drain loop started (idle_timeout={}s, pending={})",
        idle_timeout,
        _pending_count(launchpad),
    )

    while not _shutdown_requested:
        if _launch_one_rocket(launchpad):
            # Successfully ran a job -- reset idle timer
            idle_since = None
            logger.info(
                "Job complete. Pending fireworks remaining: {}",
                _pending_count(launchpad),
            )
            continue

        # No job was available
        now = time.monotonic()

        if idle_since is None:
            idle_since = now
            logger.info("No pending jobs -- starting idle countdown")

        elapsed_idle = now - idle_since

        if elapsed_idle >= idle_timeout:
            logger.info(
                "Idle for {:.0f}s (timeout={}s) -- exiting drain loop",
                elapsed_idle,
                idle_timeout,
            )
            break

        # Brief sleep before checking again
        remaining = idle_timeout - elapsed_idle
        sleep_time = min(10.0, remaining)
        logger.debug("Idle for {:.0f}s, sleeping {:.0f}s", elapsed_idle, sleep_time)
        time.sleep(sleep_time)

    if _shutdown_requested:
        logger.info("Drain loop exiting due to shutdown signal")


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Impulse drain-loop worker for EC2 instances",
    )
    parser.add_argument(
        "--launchpad",
        "-l",
        required=True,
        help="Path to the FireWorks LaunchPad YAML config file",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=300,
        help="Seconds to wait with no pending jobs before exiting (default: 300)",
    )
    args = parser.parse_args()

    launchpad = LaunchPad.from_file(args.launchpad)

    logger.info(
        "Connecting to LaunchPad at {} (db={})",
        launchpad.host,
        launchpad.name,
    )

    try:
        drain(launchpad, args.idle_timeout)
    except Exception:
        logger.exception("Drain loop crashed")
        sys.exit(1)

    logger.info("Drain loop finished cleanly")


if __name__ == "__main__":
    main()
