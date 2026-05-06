#!/usr/bin/env python3
"""
Start the Impulse EC2 orchestrator.

Watches the FireWorks LaunchPad for new READY fireworks and
automatically launches EC2 worker instances to process them.

Usage::

    uv run python scripts/start_orchestrator.py
    uv run python scripts/start_orchestrator.py -c path/to/ec2_config.yaml

Or via the installed entry point::

    impulse-orchestrator
    impulse-orchestrator -c path/to/ec2_config.yaml
"""

from tasks.orchestrator.watcher import main

if __name__ == "__main__":
    main()
