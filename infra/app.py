#!/usr/bin/env python3
"""CDK app entry point for the Impulse pipeline infrastructure."""

import aws_cdk as cdk

from stacks.storage import StorageStack
from stacks.network import NetworkStack
from stacks.auth import AuthStack
from stacks.processing import ProcessingStack
from stacks.api import ApiStack
from stacks.frontend import FrontendStack

app = cdk.App()

env = cdk.Environment(
    account=app.node.try_get_context("account") or None,
    region=app.node.try_get_context("region") or "us-east-1",
)

# ── Foundation stacks ────────────────────────────────────────────────────────

storage = StorageStack(app, "ImpulseStorage", env=env)
network = NetworkStack(app, "ImpulseNetwork", env=env)
auth = AuthStack(app, "ImpulseAuth", env=env)

# ── Processing stack (Step Functions, ECS, Lambda) ───────────────────────────

processing = ProcessingStack(
    app,
    "ImpulseProcessing",
    vpc=network.vpc,
    bucket=storage.bucket,
    image_repo=storage.image_processing_repo,
    extraction_repo=storage.extraction_repo,
    env=env,
)

# ── API stack ────────────────────────────────────────────────────────────────

api = ApiStack(
    app,
    "ImpulseApi",
    user_pool=auth.user_pool,
    bucket=storage.bucket,
    state_machine=processing.state_machine,
    env=env,
)

# ── Frontend stack ───────────────────────────────────────────────────────

frontend = FrontendStack(app, "ImpulseFrontend", env=env)

app.synth()
