"""Geocoding via OpenStreetMap Nominatim."""

from __future__ import annotations

import time

import requests
from loguru import logger

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "Impulse-Pipeline/1.0 (Northwestern University Library)"


def geocode(place: str) -> dict | None:
    """Geocode a place name and return ``{lat, lon, display_name}`` or *None*.

    Rate-limited to respect Nominatim's 1-request-per-second policy.
    """
    if not place:
        return None

    params = {
        "q": place,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": NOMINATIM_UA}

    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if results:
            r = results[0]
            return {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "display_name": r.get("display_name", place),
            }
    except Exception as e:
        logger.warning(f"Geocode failed for '{place}': {e}")

    return None


def geocode_batch(places: list[str]) -> list[dict | None]:
    """Geocode a list of place names, respecting rate limits."""
    results: list[dict | None] = []
    for place in places:
        results.append(geocode(place))
        time.sleep(1)  # Nominatim rate limit
    return results
