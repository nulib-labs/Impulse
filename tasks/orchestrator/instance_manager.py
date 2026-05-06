"""
EC2 instance lifecycle management for Impulse workers.

Handles launching, listing, and terminating worker instances.
Each instance is tagged so the orchestrator can track its fleet.
"""

from __future__ import annotations

import base64
import textwrap
from typing import Any

import boto3
from loguru import logger

from tasks.orchestrator.config import OrchestratorConfig


# Instance states we consider "active" (consuming a slot in max_instances)
_ACTIVE_STATES = {"pending", "running"}

# Tag key used to identify orchestrator-managed instances
_MANAGED_TAG_KEY = "ManagedBy"
_MANAGED_TAG_VALUE = "impulse-orchestrator"


class InstanceManager:
    """Launch, query, and terminate EC2 instances for Impulse workers."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config
        self._ec2_client = boto3.client("ec2", region_name=config.region)
        self._ec2_resource = boto3.resource("ec2", region_name=config.region)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch_instance(self, task_name: str) -> str:
        """Launch a single EC2 worker instance for *task_name*.

        Returns the instance ID of the new instance.
        """
        instance_type = self._config.instance_type_for_task(task_name)
        user_data = self._build_user_data()
        tags = self._build_tags(task_name)

        logger.info(
            "Launching EC2 instance: type={}, task={}, ami={}",
            instance_type,
            task_name,
            self._config.ami_id,
        )

        kwargs: dict[str, Any] = {
            "ImageId": self._config.ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": base64.b64encode(user_data.encode()).decode(),
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": tags,
                }
            ],
            "IamInstanceProfile": {
                "Name": self._config.iam_instance_profile,
            },
            "InstanceInitiatedShutdownBehavior": "terminate",
        }

        if self._config.security_group_ids:
            kwargs["SecurityGroupIds"] = self._config.security_group_ids

        if self._config.subnet_id:
            kwargs["SubnetId"] = self._config.subnet_id

        if self._config.key_name:
            kwargs["KeyName"] = self._config.key_name

        response = self._ec2_client.run_instances(**kwargs)
        instance_id: str = response["Instances"][0]["InstanceId"]

        logger.info("Launched instance {} ({})", instance_id, instance_type)
        return instance_id

    def get_active_instances(self) -> list[dict[str, Any]]:
        """Return metadata for all orchestrator-managed instances in active states."""
        filters = [
            {
                "Name": f"tag:{_MANAGED_TAG_KEY}",
                "Values": [_MANAGED_TAG_VALUE],
            },
            {
                "Name": "instance-state-name",
                "Values": list(_ACTIVE_STATES),
            },
        ]

        paginator = self._ec2_client.get_paginator("describe_instances")
        instances: list[dict[str, Any]] = []

        for page in paginator.paginate(Filters=filters):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    instances.append(inst)

        return instances

    def count_active_instances(self) -> int:
        """Return the number of active orchestrator-managed instances."""
        return len(self.get_active_instances())

    def has_capacity(self) -> bool:
        """Return True if we can launch at least one more instance."""
        return self.count_active_instances() < self._config.max_instances

    def terminate_instance(self, instance_id: str) -> None:
        """Terminate a specific EC2 instance."""
        logger.info("Terminating instance {}", instance_id)
        self._ec2_client.terminate_instances(InstanceIds=[instance_id])

    def terminate_all(self) -> list[str]:
        """Terminate ALL orchestrator-managed instances.  Emergency kill switch.

        Returns a list of terminated instance IDs.
        """
        instances = self.get_active_instances()
        if not instances:
            logger.info("No active managed instances to terminate")
            return []

        instance_ids = [i["InstanceId"] for i in instances]
        logger.warning("Terminating {} managed instances: {}", len(instance_ids), instance_ids)
        self._ec2_client.terminate_instances(InstanceIds=instance_ids)
        return instance_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tags(self, task_name: str) -> list[dict[str, str]]:
        """Build EC2 instance tags."""
        tags: dict[str, str] = {
            **self._config.instance_tags,
            "Name": f"impulse-worker",
            "ImpulseTaskType": task_name,
        }
        return [{"Key": k, "Value": v} for k, v in tags.items()]

    def _build_user_data(self) -> str:
        """Build the cloud-init user data script.

        This script runs on first boot of the EC2 instance.  It:
        1. Retrieves MongoDB credentials from AWS Secrets Manager
        2. Writes a LaunchPad YAML config
        3. Starts the drain-loop worker
        4. Schedules a hard walltime shutdown as a safety net
        """
        idle_timeout = self._config.idle_timeout_seconds
        max_walltime = self._config.max_walltime_seconds
        secret_id = self._config.mongodb_secret_id
        mongo_db = self._config.mongo_db_name
        region = self._config.region

        # The user data script is a bash script that runs as root on first boot.
        # The AMI is expected to have: Python 3.13, uv, the Impulse repo at
        # /opt/impulse, and all Python dependencies pre-installed.
        script = textwrap.dedent(f"""\
            #!/bin/bash
            set -euo pipefail

            exec > /var/log/impulse-worker.log 2>&1
            echo "$(date -Iseconds) Impulse worker starting"

            # ── Safety net: hard shutdown after max walltime ──────────────
            (sleep {max_walltime} && shutdown -h now "Max walltime reached") &
            WALLTIME_PID=$!

            # ── Retrieve MongoDB URI from Secrets Manager ─────────────────
            MONGO_URI=$(aws secretsmanager get-secret-value \\
                --secret-id "{secret_id}" \\
                --region "{region}" \\
                --query SecretString \\
                --output text)

            export IMPULSE_MONGODB_URI="$MONGO_URI"

            # ── Write FireWorks LaunchPad config ──────────────────────────
            cat > /opt/impulse/worker_launchpad.yaml <<LPEOF
            host: "$MONGO_URI"
            logdir: null
            name: {mongo_db}
            port: 27017
            strm_lvl: INFO
            uri_mode: true
            LPEOF

            # ── Run the drain loop ────────────────────────────────────────
            cd /opt/impulse
            uv run python -m tasks.orchestrator.drain_loop \\
                --launchpad /opt/impulse/worker_launchpad.yaml \\
                --idle-timeout {idle_timeout}

            echo "$(date -Iseconds) Drain loop exited -- shutting down"

            # ── Clean shutdown ────────────────────────────────────────────
            kill $WALLTIME_PID 2>/dev/null || true
            shutdown -h now "Drain loop complete"
        """)

        return script
