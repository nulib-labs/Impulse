"""Collections API handlers.

CRUD for collections + add/remove documents + ZIP download.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import boto3
from loguru import logger

from impulse.config import S3_BUCKET
from impulse.db.client import get_collection as get_mongo_collection
from impulse.db.models import Collection
from impulse.utils import generate_presigned_download_url


# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _col():
    return get_mongo_collection("collections")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── POST /collections ───────────────────────────────────────────────────────


def create_collection(body: dict, user_id: str) -> dict:
    name = body.get("name", "").strip()
    if not name:
        return _response(400, {"error": "Collection name is required"})

    collection = Collection(
        collection_id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        description=body.get("description", "").strip(),
    )

    _col().insert_one(collection.to_dict())
    logger.info(f"Created collection {collection.collection_id} for user {user_id}")

    return _response(
        201,
        {
            "collection_id": collection.collection_id,
            "name": collection.name,
        },
    )


# ── GET /collections ────────────────────────────────────────────────────────


def list_collections(user_id: str) -> dict:
    cursor = (
        _col()
        .find(
            {"user_id": user_id},
            {
                "_id": 0,
                "collection_id": 1,
                "name": 1,
                "description": 1,
                "documents": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
        .sort("updated_at", -1)
    )

    collections = []
    for c in cursor:
        c["document_count"] = len(c.get("documents", []))
        # Don't send full document list in the listing
        del c["documents"]
        collections.append(c)

    return _response(200, {"collections": collections, "count": len(collections)})


# ── GET /collections/{collectionId} ─────────────────────────────────────────


def get_collection_detail(collection_id: str, user_id: str) -> dict:
    col = _col().find_one(
        {"collection_id": collection_id, "user_id": user_id},
        {"_id": 0},
    )
    if not col:
        return _response(404, {"error": "Collection not found"})

    # Generate presigned URLs for each document
    for doc in col.get("documents", []):
        doc["url"] = generate_presigned_download_url(doc["s3_key"], expires_in=3600)

    col["document_count"] = len(col.get("documents", []))
    return _response(200, {"collection": col})


# ── PUT /collections/{collectionId} ─────────────────────────────────────────


def update_collection(collection_id: str, body: dict, user_id: str) -> dict:
    col = _col().find_one(
        {"collection_id": collection_id, "user_id": user_id},
        {"_id": 0, "collection_id": 1},
    )
    if not col:
        return _response(404, {"error": "Collection not found"})

    updates: dict = {"updated_at": _now()}
    if "name" in body:
        name = body["name"].strip()
        if not name:
            return _response(400, {"error": "Collection name cannot be empty"})
        updates["name"] = name
    if "description" in body:
        updates["description"] = body["description"].strip()

    _col().update_one(
        {"collection_id": collection_id, "user_id": user_id},
        {"$set": updates},
    )
    return _response(200, {"message": "Collection updated"})


# ── DELETE /collections/{collectionId} ───────────────────────────────────────


def delete_collection(
    collection_id: str, user_id: str, claims: dict | None = None
) -> dict:
    from impulse.api.auth import require_admin

    denied = require_admin(claims or {})
    if denied:
        return denied

    result = _col().delete_one(
        {"collection_id": collection_id, "user_id": user_id},
    )
    if result.deleted_count == 0:
        return _response(404, {"error": "Collection not found"})

    logger.info(f"Deleted collection {collection_id}")
    return _response(200, {"message": "Collection deleted"})


# ── POST /collections/{collectionId}/documents ──────────────────────────────


def modify_collection_documents(
    collection_id: str,
    body: dict,
    user_id: str,
    claims: dict | None = None,
) -> dict:
    col = _col().find_one(
        {"collection_id": collection_id, "user_id": user_id},
        {"_id": 0},
    )
    if not col:
        return _response(404, {"error": "Collection not found"})

    action = body.get("action", "")
    documents = body.get("documents", [])

    if action == "add":
        # Validate each document has required fields
        existing_keys = {d["s3_key"] for d in col.get("documents", [])}
        new_docs = []
        for doc in documents:
            s3_key = doc.get("s3_key", "")
            if not s3_key or s3_key in existing_keys:
                continue
            new_docs.append(
                {
                    "s3_key": s3_key,
                    "filename": doc.get("filename", s3_key.split("/")[-1]),
                    "job_id": doc.get("job_id", ""),
                    "source_type": doc.get("source_type", "input"),
                    "size": doc.get("size", 0),
                }
            )
            existing_keys.add(s3_key)

        if new_docs:
            _col().update_one(
                {"collection_id": collection_id, "user_id": user_id},
                {
                    "$push": {"documents": {"$each": new_docs}},
                    "$set": {"updated_at": _now()},
                },
            )

        return _response(
            200,
            {
                "message": f"Added {len(new_docs)} document(s)",
                "added": len(new_docs),
            },
        )

    elif action == "remove":
        from impulse.api.auth import require_admin

        denied = require_admin(claims or {})
        if denied:
            return denied

        keys_to_remove = {d.get("s3_key", "") for d in documents}
        remaining = [
            d for d in col.get("documents", []) if d["s3_key"] not in keys_to_remove
        ]
        removed = len(col.get("documents", [])) - len(remaining)

        _col().update_one(
            {"collection_id": collection_id, "user_id": user_id},
            {
                "$set": {"documents": remaining, "updated_at": _now()},
            },
        )

        return _response(
            200,
            {
                "message": f"Removed {removed} document(s)",
                "removed": removed,
            },
        )

    else:
        return _response(400, {"error": "action must be 'add' or 'remove'"})


# ── GET /collections/{collectionId}/download ─────────────────────────────────


def download_collection(collection_id: str, user_id: str) -> dict:
    """Create a ZIP of all documents in the collection, upload to S3,
    and return a presigned download URL."""
    col = _col().find_one(
        {"collection_id": collection_id, "user_id": user_id},
        {"_id": 0},
    )
    if not col:
        return _response(404, {"error": "Collection not found"})

    documents = col.get("documents", [])
    if not documents:
        return _response(400, {"error": "Collection has no documents"})

    s3 = boto3.client("s3")
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for doc in documents:
            s3_key = doc["s3_key"]
            filename = doc.get("filename", s3_key.split("/")[-1])
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                zf.writestr(filename, obj["Body"].read())
            except Exception as e:
                logger.warning(f"Skipping {s3_key}: {e}")

    buf.seek(0)
    zip_key = f"downloads/{collection_id}/{col['name']}.zip"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=zip_key,
        Body=buf.getvalue(),
        ContentType="application/zip",
    )

    url = generate_presigned_download_url(zip_key, expires_in=3600)
    logger.info(f"Created ZIP for collection {collection_id}: {zip_key}")

    return _response(200, {"download_url": url, "zip_key": zip_key})
