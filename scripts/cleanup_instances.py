#!/usr/bin/env python3
"""
Emergency kill switch: terminate ALL orchestrator-managed EC2 instances.

Usage::

    uv run python scripts/cleanup_instances.py
    uv run python scripts/cleanup_instances.py --config path/to/ec2_config.yaml
    uv run python scripts/cleanup_instances.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from tasks.orchestrator.config import load_config
from tasks.orchestrator.instance_manager import InstanceManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminate all Impulse orchestrator-managed EC2 instances",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to the EC2 orchestrator YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List instances that would be terminated without actually terminating them",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mgr = InstanceManager(config)

    instances = mgr.get_active_instances()

    if not instances:
        logger.info("No active orchestrator-managed instances found")
        return

    logger.info("Found {} active managed instance(s):", len(instances))
    for inst in instances:
        inst_id = inst["InstanceId"]
        inst_type = inst["InstanceType"]
        state = inst["State"]["Name"]
        launch_time = inst.get("LaunchTime", "?")

        # Extract the task type tag
        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
        task_type = tags.get("ImpulseTaskType", "?")

        logger.info(
            "  {} | {} | {} | task={} | launched={}",
            inst_id,
            inst_type,
            state,
            task_type,
            launch_time,
        )

    if args.dry_run:
        logger.info("Dry run -- no instances terminated")
        return

    # Confirm before proceeding
    print()
    answer = input(
        f"Terminate {len(instances)} instance(s)? [y/N] "
    ).strip().lower()

    if answer != "y":
        logger.info("Aborted")
        sys.exit(0)

    terminated = mgr.terminate_all()
    logger.info("Terminated {} instance(s): {}", len(terminated), terminated)


if __name__ == "__main__":
    main()
