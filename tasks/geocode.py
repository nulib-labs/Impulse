from fireworks.core.firework import FWAction, FireTaskBase
from loguru import logger

class GeocodeTask(FireTaskBase):
    """
    Geocode each document's main_place using Nominatim (OpenStreetMap).
    Adds lat/lon coordinates to the metadata results.
    Rate-limited to 1 request/second per Nominatim's usage policy.

    fw_spec keys:
      metadata_path — S3 path to GetMetadataTask output
      output_path   — S3 path to write geocoded results JSON
      debug         — if True, read/write local files
    """

    _fw_name = "Geocode Task"

    @staticmethod
    def _geocode(place: str) -> dict | None:
        """Call Nominatim and return {lat, lon, display_name} or None."""
        import requests

        if not place:
            return None

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "NOMINATIM_UA"}

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
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
