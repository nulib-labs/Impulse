"""Admin API handlers for user management via Cognito."""

from __future__ import annotations

import json
import os

import boto3
from loguru import logger

# Admin emails -- only these users can access the admin panel.
# In production, store this in a DB or config; for now, the first user
# who created the pool is the admin.
ADMIN_GROUP = "admins"


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(body, default=str),
    }


def _get_cognito():
    return boto3.client("cognito-idp")


def _pool_id() -> str:
    pool_id = os.environ.get("USER_POOL_ID", "")
    if not pool_id:
        raise RuntimeError("USER_POOL_ID not configured")
    return pool_id


# ── GET /admin/users ─────────────────────────────────────────────────────


def list_users(user_id: str) -> dict:
    """List all users in the Cognito User Pool."""
    cognito = _get_cognito()
    pool_id = _pool_id()

    try:
        result = cognito.list_users(UserPoolId=pool_id, Limit=60)
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return _response(500, {"error": str(e)})

    users = []
    for u in result.get("Users", []):
        attrs = {a["Name"]: a["Value"] for a in u.get("Attributes", [])}
        users.append(
            {
                "username": u["Username"],
                "email": attrs.get("email", ""),
                "status": u["UserStatus"],
                "enabled": u["Enabled"],
                "created_at": u.get("UserCreateDate", ""),
                "last_modified": u.get("UserLastModifiedDate", ""),
            }
        )

    return _response(200, {"users": users})


# ── POST /admin/users ────────────────────────────────────────────────────


def create_user(body: dict, user_id: str) -> dict:
    """Create a new user in the Cognito User Pool."""
    email = body.get("email", "").strip().lower()
    if not email or "@" not in email:
        return _response(400, {"error": "Valid email address is required"})

    cognito = _get_cognito()
    pool_id = _pool_id()

    try:
        result = cognito.admin_create_user(
            UserPoolId=pool_id,
            Username=email,
            UserAttributes=[
                {"Name": "email", "Value": email},
                {"Name": "email_verified", "Value": "true"},
            ],
            DesiredDeliveryMediums=["EMAIL"],
        )
        user = result.get("User", {})
        logger.info(f"Created user {email} in Cognito")

        return _response(
            201,
            {
                "username": user.get("Username", ""),
                "email": email,
                "status": user.get("UserStatus", ""),
                "message": f"User {email} created. A temporary password has been sent to their email.",
            },
        )
    except cognito.exceptions.UsernameExistsException:
        return _response(409, {"error": f"User {email} already exists"})
    except Exception as e:
        logger.error(f"Failed to create user {email}: {e}")
        return _response(500, {"error": str(e)})


# ── DELETE /admin/users/{username} ───────────────────────────────────────


def delete_user(username: str, user_id: str) -> dict:
    """Delete a user from the Cognito User Pool."""
    cognito = _get_cognito()
    pool_id = _pool_id()

    try:
        cognito.admin_delete_user(
            UserPoolId=pool_id,
            Username=username,
        )
        logger.info(f"Deleted user {username} from Cognito")
        return _response(200, {"message": f"User {username} deleted"})
    except cognito.exceptions.UserNotFoundException:
        return _response(404, {"error": "User not found"})
    except Exception as e:
        logger.error(f"Failed to delete user {username}: {e}")
        return _response(500, {"error": str(e)})


# ── POST /admin/users/{username}/disable ─────────────────────────────────


def disable_user(username: str, user_id: str) -> dict:
    """Disable a user in the Cognito User Pool."""
    cognito = _get_cognito()
    pool_id = _pool_id()

    try:
        cognito.admin_disable_user(
            UserPoolId=pool_id,
            Username=username,
        )
        logger.info(f"Disabled user {username}")
        return _response(200, {"message": f"User {username} disabled"})
    except cognito.exceptions.UserNotFoundException:
        return _response(404, {"error": "User not found"})
    except Exception as e:
        return _response(500, {"error": str(e)})


# ── POST /admin/users/{username}/enable ──────────────────────────────────


def enable_user(username: str, user_id: str) -> dict:
    """Enable a disabled user in the Cognito User Pool."""
    cognito = _get_cognito()
    pool_id = _pool_id()

    try:
        cognito.admin_enable_user(
            UserPoolId=pool_id,
            Username=username,
        )
        logger.info(f"Enabled user {username}")
        return _response(200, {"message": f"User {username} enabled"})
    except cognito.exceptions.UserNotFoundException:
        return _response(404, {"error": "User not found"})
    except Exception as e:
        return _response(500, {"error": str(e)})
