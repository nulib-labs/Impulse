"""
Configuration for the EC2 orchestrator.

Loads settings from a YAML config file (default: ec2_config.yaml) with
environment variable overrides where noted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


DEFAULT_CONFIG_PATH = Path("ec2_config.yaml")
DEFAULT_IDLE_TIMEOUT = 300  # seconds
DEFAULT_MAX_INSTANCES = 5
DEFAULT_MAX_WALLTIME = 43200  # 12 hours hard limit
DEFAULT_INSTANCE_TYPE = "m5.xlarge"


@dataclass
class OrchestratorConfig:
    """All settings required to launch and manage EC2 worker instances."""

    # AWS / EC2
    ami_id: str
    region: str
    subnet_id: str
    security_group_ids: list[str]
    iam_instance_profile: str
    key_name: str | None = None

    # Instance scaling
    max_instances: int = DEFAULT_MAX_INSTANCES

    # Maps FireWorks task name -> EC2 instance type.
    # Falls back to ``default`` key when a task name is not listed.
    instance_type_map: dict[str, str] = field(
        default_factory=lambda: {"default": DEFAULT_INSTANCE_TYPE}
    )

    # Lifecycle
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT
    max_walltime_seconds: int = DEFAULT_MAX_WALLTIME

    # MongoDB (LaunchPad) -- can be overridden by env vars
    mongo_uri: str = ""
    mongo_db_name: str = "fireworks"

    # S3 bucket for the Impulse project (passed to worker env)
    impulse_bucket: str = "nu-impulse-production"

    # AWS Secrets Manager secret ID that holds the MongoDB URI.
    # Workers retrieve credentials at boot via their IAM instance profile.
    mongodb_secret_id: str = "impulse/mongodb-uri"

    # Tags applied to every managed EC2 instance
    instance_tags: dict[str, str] = field(
        default_factory=lambda: {
            "Project": "Impulse",
            "ManagedBy": "impulse-orchestrator",
        }
    )

    def instance_type_for_task(self, task_name: str) -> str:
        """Return the EC2 instance type appropriate for *task_name*."""
        return self.instance_type_map.get(
            task_name,
            self.instance_type_map.get("default", DEFAULT_INSTANCE_TYPE),
        )


def load_config(path: str | Path | None = None) -> OrchestratorConfig:
    """Load orchestrator config from a YAML file with env-var overrides.

    Resolution order (highest priority first):
        1. Environment variables (``EC2_AMI_ID``, ``EC2_REGION``, etc.)
        2. Values in the YAML file
        3. Dataclass defaults

    Parameters
    ----------
    path:
        Path to the YAML config file.  Falls back to the
        ``EC2_CONFIG_PATH`` environment variable, then to
        ``ec2_config.yaml`` in the current directory.
    """
    if path is None:
        path = Path(os.getenv("EC2_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    else:
        path = Path(path)

    raw: dict[str, Any] = {}
    if path.exists():
        logger.info("Loading EC2 orchestrator config from {}", path)
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        logger.warning(
            "Config file {} not found -- falling back to env vars / defaults",
            path,
        )

    def _env_or(yaml_key: str, env_key: str, default: Any = "") -> Any:
        return os.getenv(env_key, raw.get(yaml_key, default))

    mongo_uri = _env_or("mongo_uri", "IMPULSE_MONGODB_URI", "")
    if not mongo_uri:
        # Try the debug URI as a last resort for local dev
        mongo_uri = os.getenv("IMPULSE_MONGODB_URI_DEBUG", "")

    security_group_ids = raw.get("security_group_ids", [])
    env_sgs = os.getenv("EC2_SECURITY_GROUP_IDS")
    if env_sgs:
        security_group_ids = [s.strip() for s in env_sgs.split(",")]

    instance_type_map = raw.get("instance_type_map", {"default": DEFAULT_INSTANCE_TYPE})

    instance_tags = raw.get("instance_tags", {})
    # Always ensure the management tag is present
    instance_tags.setdefault("Project", "Impulse")
    instance_tags.setdefault("ManagedBy", "impulse-orchestrator")

    return OrchestratorConfig(
        ami_id=_env_or("ami_id", "EC2_AMI_ID"),
        region=_env_or("region", "EC2_REGION", "us-east-2"),
        subnet_id=_env_or("subnet_id", "EC2_SUBNET_ID"),
        security_group_ids=security_group_ids,
        iam_instance_profile=_env_or(
            "iam_instance_profile", "EC2_IAM_INSTANCE_PROFILE"
        ),
        key_name=_env_or("key_name", "EC2_KEY_NAME", None) or None,
        max_instances=int(
            _env_or("max_instances", "EC2_MAX_INSTANCES", DEFAULT_MAX_INSTANCES)
        ),
        instance_type_map=instance_type_map,
        idle_timeout_seconds=int(
            _env_or(
                "idle_timeout_seconds",
                "EC2_IDLE_TIMEOUT",
                DEFAULT_IDLE_TIMEOUT,
            )
        ),
        max_walltime_seconds=int(
            _env_or(
                "max_walltime_seconds",
                "EC2_MAX_WALLTIME",
                DEFAULT_MAX_WALLTIME,
            )
        ),
        mongo_uri=mongo_uri,
        mongo_db_name=_env_or("mongo_db_name", "EC2_MONGO_DB_NAME", "fireworks"),
        impulse_bucket=_env_or(
            "impulse_bucket", "IMPULSE_BUCKET", "nu-impulse-production"
        ),
        mongodb_secret_id=_env_or(
            "mongodb_secret_id", "MONGODB_SECRET_ID", "impulse/mongodb-uri"
        ),
        instance_tags=instance_tags,
    )
