"""Shared authentication and authorization helpers."""

from __future__ import annotations

import json

ADMIN_GROUP = "admins"


def is_admin(claims: dict) -> bool:
    """Return True if the Cognito claims indicate the user is in the admins group."""
    groups = claims.get("cognito:groups", "")
    # Cognito encodes groups as a comma-separated string or a single value
    if isinstance(groups, list):
        return ADMIN_GROUP in groups
    return ADMIN_GROUP in [g.strip() for g in groups.split(",") if g.strip()]


def require_admin(claims: dict) -> dict | None:
    """Return a 403 response dict if the user is not an admin, else None."""
    if is_admin(claims):
        return None
    return {
        "statusCode": 403,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps({"error": "Admin privileges required to delete resources"}),
    }
