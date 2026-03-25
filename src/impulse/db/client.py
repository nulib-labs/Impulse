"""
Singleton MongoDB client for the Impulse pipeline.

Reuses a single ``MongoClient`` per process so Lambda containers and
ECS tasks do not create a new connection on every invocation.
"""

from __future__ import annotations

import certifi
from pymongo import MongoClient
from pymongo.database import Database

from impulse.config import MONGODB_DATABASE, get_mongodb_uri

_client: MongoClient | None = None


def get_db() -> Database:
    """Return the default Impulse database handle (cached)."""
    global _client
    if _client is None:
        _client = MongoClient(
            get_mongodb_uri(),
            tls=True,
            tlsCAFile=certifi.where(),
        )
    return _client[MONGODB_DATABASE]


def get_collection(name: str):
    """Shorthand for ``get_db()[name]``."""
    return get_db()[name]
