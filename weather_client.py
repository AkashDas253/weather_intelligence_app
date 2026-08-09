"""
Client for the National Weather Service (NWS) API.

Harvests unstructured weather text (alerts and multi-day forecasts)
and normalizes them into document records for Lakebase inside the
`weather.weather_documents` table.
"""

import hashlib
import os
import re
from typing import Any

import psycopg2
from psycopg2.extras import execute_values, Json
import requests

_NWS_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_DEFAULT_TIMEOUT = 30
_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "(WeatherIntelligenceApp, contact@example.com)"
)


class WeatherClient:
    """Thin wrapper around the NWS API with session setup and document normalization."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _NWS_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a GET request against NWS API (handles relative and full URLs)."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def resolve_coordinates(self, location_str: str) -> tuple[float, float]:
        """Resolve city/state or lat/lon string to (latitude, longitude) floats."""
        # Check if already passed as 'lat,lon'
        coords_match = re.match(
            r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", location_str
        )
        if coords_match:
            return float(coords_match.group(1)), float(coords_match.group(2))

        # Fallback dictionary for common test locations
        city_coords = {
            "chicago, il": (41.8781, -87.6298),
            "austin, tx": (30.2672, -97.7431),
            "new york, ny": (40.7128, -74.0060),
            "seattle, wa": (47.6062, -122.3321),
            "miami, fl": (25.7617, -80.1918),
        }

        cleaned = location_str.strip().lower()
        if cleaned in city_coords:
            return city_coords[cleaned]

        raise ValueError(
            f"Could not resolve coordinates for '{location_str}'. Provide coordinates as 'lat,lon'."
        )

    def get_point_metadata(self, lat: float, lon: float) -> dict:
        """Fetch NWS grid point metadata given latitude and longitude."""
        data = self.get(f"/points/{lat:.4f},{lon:.4f}")
        return data.get("properties", {})

    def get_active_alerts(self, lat: float, lon: float) -> list[dict]:
        """Fetch active weather alert features for a coordinate point."""
        data = self.get(f"/alerts/active?point={lat:.4f},{lon:.4f}")
        return data.get("features", [])

    def get_forecast_periods(self, forecast_url: str) -> list[dict]:
        """Fetch narrative forecast periods from the forecast endpoint."""
        data = self.get(forecast_url)
        return data.get("properties", {}).get("periods", [])

    def normalize_alert(self, alert: dict, location_name: str) -> dict | None:
        """Normalize an alert feature into a weather_documents record dict."""
        props = alert.get("properties", {})
        alert_id = alert.get("id") or props.get("id")
        if not alert_id:
            return None

        description = props.get("description") or ""
        instruction = props.get("instruction") or ""
        narrative = f"{description}\n\nInstruction: {instruction}".strip()

        if not narrative:
            return None

        return {
            "id": f"alert_{alert_id}",
            "location": location_name,
            "source_type": "alert",
            "headline": props.get("headline") or props.get("event"),
            "narrative_text": narrative,
            "issued_at": props.get("sent") or props.get("effective"),
            "effective_at": props.get("effective"),
            "payload": alert,
        }

    def normalize_forecast(self, period: dict, location_name: str) -> dict | None:
        """Normalize a forecast period into a weather_documents record dict."""
        detailed = period.get("detailedForecast") or ""
        if not detailed:
            return None

        period_name = period.get("name", "Forecast Period")
        issued = period.get("startTime")

        # Stable hash ID to prevent duplicate forecast entries across syncs
        raw_key = f"{location_name}_{period_name}_{issued}_{detailed}"
        stable_id = "forecast_" + hashlib.md5(raw_key.encode("utf-8")).hexdigest()

        return {
            "id": stable_id,
            "location": location_name,
            "source_type": "forecast",
            "headline": f"{period_name}: {period.get('shortForecast', '')}",
            "narrative_text": detailed,
            "issued_at": issued,
            "effective_at": period.get("startTime"),
            "payload": period,
        }

    def harvest(self, locations: list[str], limit: int = 50) -> list[dict]:
        """Harvest active alerts and forecast narratives for given locations."""
        documents: list[dict] = []

        for loc in locations:
            try:
                lat, lon = self.resolve_coordinates(loc)

                # 1. Active Alerts
                alerts = self.get_active_alerts(lat, lon)
                for alert in alerts:
                    doc = self.normalize_alert(alert, loc)
                    if doc:
                        documents.append(doc)

                # 2. Multi-Day Narrative Forecasts
                point_meta = self.get_point_metadata(lat, lon)
                forecast_url = point_meta.get("forecast")
                if forecast_url:
                    periods = self.get_forecast_periods(forecast_url)
                    for period in periods:
                        doc = self.normalize_forecast(period, loc)
                        if doc:
                            documents.append(doc)

            except Exception as err:
                print(f"[WeatherClient] Error harvesting '{loc}': {err}")
                continue

            if len(documents) >= limit:
                break

        return documents[:limit]


def sync_weather_documents_to_db(conn, documents: list[dict]) -> int:
    """Upsert harvested weather documents into `weather.weather_documents` using psycopg2."""
    if not documents:
        return 0

    sql = """
        INSERT INTO weather.weather_documents (
            id, location, source_type, headline, narrative_text, issued_at, effective_at, payload
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            payload = EXCLUDED.payload,
            synced_at = CURRENT_TIMESTAMP;
    """

    tuples = [
        (
            doc["id"],
            doc["location"],
            doc["source_type"],
            doc["headline"],
            doc["narrative_text"],
            doc["issued_at"],
            doc["effective_at"],
            Json(doc["payload"]),
        )
        for doc in documents
    ]

    with conn.cursor() as cur:
        execute_values(cur, sql, tuples)
    conn.commit()

    return len(documents)