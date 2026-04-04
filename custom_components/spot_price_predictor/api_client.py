"""Async API clients for external data sources."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from aiohttp import ClientSession

from .const import (
    API_SAHKOTIN,
    API_OPENMETEO_FORECAST,
    API_ELERING,
    API_ELPRISET,
    API_FINGRID,
    FINGRID_NUCLEAR,
    FINGRID_FLOW_SE1,
    FINGRID_FLOW_SE3,
    FINGRID_FLOW_EE,
    FINGRID_MAX_VALUES,
    FINLAND_LOCATIONS,
    FORECAST_HOURS,
)

_LOGGER = logging.getLogger(__name__)


class ApiClientError(Exception):
    """Raised when an API call fails."""


class SpotPriceApiClient:
    """Async API client for all external data sources."""

    def __init__(self, session: ClientSession, fingrid_api_key: str | None = None) -> None:
        self._session = session
        self._fingrid_api_key = fingrid_api_key

    # ------------------------------------------------------------------
    # Open-Meteo weather forecast
    # ------------------------------------------------------------------

    async def fetch_weather(self) -> list[dict[str, float]]:
        """Fetch weather forecasts from Open-Meteo for all Finnish locations.

        Returns a list of dicts (one per hour) with keys:
        wind_weighted, solar_weighted, temp_weighted.
        """
        location_data: list[dict[str, list[float]]] = []

        for loc in FINLAND_LOCATIONS:
            params = {
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "hourly": "wind_speed_120m,global_tilted_irradiance_instant,temperature_2m",
                "wind_speed_unit": "ms",
                "tilt": 45,
                "forecast_days": 8,
                "timezone": "UTC",
            }
            try:
                async with self._session.get(API_OPENMETEO_FORECAST, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    hourly = data.get("hourly", {})
                    location_data.append({
                        "wind": hourly.get("wind_speed_120m", []),
                        "solar": hourly.get("global_tilted_irradiance_instant", []),
                        "temp": hourly.get("temperature_2m", []),
                        "loc": loc,
                    })
            except Exception as err:
                _LOGGER.warning("Open-Meteo fetch failed for %s: %s", loc["name"], err)
                continue

        if not location_data:
            raise ApiClientError("Failed to fetch weather data from any location")

        # Determine number of hours from shortest series
        n_hours = min(
            len(ld["wind"]) for ld in location_data
        )
        n_hours = min(n_hours, FORECAST_HOURS + 24)  # cap

        result: list[dict[str, float]] = []
        for i in range(n_hours):
            wind_w = 0.0
            solar_w = 0.0
            temp_w = 0.0
            for ld in location_data:
                loc = ld["loc"]
                wind_val = ld["wind"][i] if i < len(ld["wind"]) else 0.0
                solar_val = ld["solar"][i] if i < len(ld["solar"]) else 0.0
                temp_val = ld["temp"][i] if i < len(ld["temp"]) else 0.0
                wind_w += wind_val * loc["wind_weight"]
                solar_w += solar_val * loc["solar_weight"]
                temp_w += temp_val * loc["temp_weight"]
            result.append({
                "wind_weighted": wind_w,
                "solar_weighted": solar_w,
                "temp_weighted": temp_w,
            })

        _LOGGER.info("Weather data: %d hours from %d locations", len(result), len(location_data))
        return result

    # ------------------------------------------------------------------
    # Sahkotin spot prices (Finland)
    # ------------------------------------------------------------------

    async def fetch_spot_prices(self) -> list[dict[str, Any]]:
        """Fetch Finnish spot prices from Sahkotin.

        Returns list of dicts with keys: timestamp (ISO), price_eur_mwh.
        """
        try:
            params = {"hours": 48}
            async with self._session.get(API_SAHKOTIN, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                prices = []
                if isinstance(data, dict) and "prices" in data:
                    raw = data["prices"]
                elif isinstance(data, list):
                    raw = data
                else:
                    raw = []
                for entry in raw:
                    prices.append({
                        "timestamp": entry.get("date") or entry.get("timestamp"),
                        "price_eur_mwh": float(entry.get("value", 0.0)) / 10.0,
                    })
                _LOGGER.info("Sahkotin: fetched %d price entries", len(prices))
                return prices
        except Exception as err:
            _LOGGER.error("Sahkotin fetch failed: %s", err)
            raise ApiClientError(f"Sahkotin: {err}") from err

    # ------------------------------------------------------------------
    # Cross-border prices: Elering (Estonia) and Elpriset (Sweden)
    # ------------------------------------------------------------------

    async def fetch_neighbor_prices(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch cross-border prices for Tier 2 features.

        Returns dict with keys se1, se3, ee mapping to price lists.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=8)).strftime("%Y-%m-%d")
        end = (now + timedelta(days=2)).strftime("%Y-%m-%d")

        # Estonia via Elering
        try:
            params = {"start": f"{start}T00:00:00.000Z", "end": f"{end}T00:00:00.000Z"}
            async with self._session.get(API_ELERING, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                ee_data = data.get("data", {}).get("ee", [])
                result["ee"] = [
                    {"timestamp": e.get("timestamp"), "price_eur_mwh": float(e.get("price", 0.0))}
                    for e in ee_data
                ]
        except Exception as err:
            _LOGGER.warning("Elering fetch failed: %s", err)
            result["ee"] = []

        # Sweden SE1 and SE3 via elpriset
        for zone in ("SE1", "SE3"):
            key = zone.lower()
            try:
                prices_list: list[dict[str, Any]] = []
                # Fetch day by day for last 8 days
                for day_offset in range(-8, 2):
                    d = now + timedelta(days=day_offset)
                    url = f"{API_ELPRISET}/{d.strftime('%Y/%m-%d')}_{zone}.json"
                    try:
                        async with self._session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                for entry in data:
                                    prices_list.append({
                                        "timestamp": entry.get("time_start"),
                                        "price_eur_mwh": float(entry.get("EUR_per_kWh", 0.0)) * 1000,
                                    })
                    except Exception:
                        continue
                result[key] = prices_list
            except Exception as err:
                _LOGGER.warning("Elpriset fetch failed for %s: %s", zone, err)
                result[key] = []

        _LOGGER.info(
            "Neighbor prices: EE=%d, SE1=%d, SE3=%d",
            len(result.get("ee", [])),
            len(result.get("se1", [])),
            len(result.get("se3", [])),
        )
        return result

    # ------------------------------------------------------------------
    # Fingrid grid data (Tier 3)
    # ------------------------------------------------------------------

    async def fetch_fingrid_data(self) -> dict[str, float]:
        """Fetch latest Fingrid grid data (nuclear, cross-border flows).

        Returns normalized values (0-1) for use as features.
        """
        if not self._fingrid_api_key:
            return {}

        headers = {"x-api-key": self._fingrid_api_key}
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        datasets = {
            "nuclear_mw": FINGRID_NUCLEAR,
            "flow_fi_se1": FINGRID_FLOW_SE1,
            "flow_fi_se3": FINGRID_FLOW_SE3,
            "flow_fi_ee": FINGRID_FLOW_EE,
        }

        result: dict[str, float] = {}
        for name, dataset_id in datasets.items():
            try:
                url = f"{API_FINGRID}/{dataset_id}/data"
                params = {"startTime": start, "endTime": end, "format": "json", "pageSize": 1, "sortOrder": "desc"}
                async with self._session.get(url, headers=headers, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    entries = data.get("data", [])
                    if entries:
                        raw_val = float(entries[0].get("value", 0.0))
                        max_val = FINGRID_MAX_VALUES.get(name, 1.0)
                        result[name] = raw_val / max_val if max_val else 0.0
                    else:
                        result[name] = 0.0
            except Exception as err:
                _LOGGER.warning("Fingrid fetch failed for %s: %s", name, err)
                result[name] = 0.0

        _LOGGER.info("Fingrid data: %s", result)
        return result

    async def validate_fingrid_key(self) -> bool:
        """Validate Fingrid API key by making a test request."""
        if not self._fingrid_api_key:
            return False
        try:
            headers = {"x-api-key": self._fingrid_api_key}
            url = f"{API_FINGRID}/{FINGRID_NUCLEAR}/data"
            params = {"format": "json", "pageSize": 1}
            async with self._session.get(url, headers=headers, params=params) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Utility: compute rolling spread for Tier 2
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rolling_spreads(
        fi_prices: list[dict[str, Any]],
        neighbor_prices: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Compute 7-day rolling mean spread (FI - neighbor) for each zone.

        Returns a dict with keys se1, se3, ee mapping to average spread values.
        Uses only simple averaging of available overlapping hours.
        """
        # Build FI price lookup by hour
        fi_lookup: dict[str, float] = {}
        for p in fi_prices:
            ts = p.get("timestamp", "")
            if ts:
                key = ts[:13]  # YYYY-MM-DDTHH
                fi_lookup[key] = p["price_eur_mwh"]

        result: dict[str, float] = {}
        for zone, prices in neighbor_prices.items():
            spreads: list[float] = []
            for p in prices:
                ts = p.get("timestamp", "")
                if ts:
                    key = ts[:13]
                    if key in fi_lookup:
                        spreads.append(fi_lookup[key] - p["price_eur_mwh"])
            if spreads:
                result[zone] = sum(spreads) / len(spreads)
            else:
                result[zone] = 0.0

        return result
