"""Statnett Nordic hydro reservoir client.

Reads weekly Nordic reservoir fill data from the Norwegian TSO's public REST
endpoint at `driftsdata.statnett.no/restapi/Reservoir/`. No authentication,
no API key, JSON response, weekly resolution, ~1-2 week publication lag.

Norwegian hydro storage drives Nordic electricity prices via cross-border
flows (Norway holds ~50% of European hydro capacity). The fill level
provides a slow-moving exogenous indicator that the v2.4.4 FI model will
consume as a "reservoir offset from seasonal baseline" feature.

This is the v2.4.1 infrastructure release — the client is wired up and its
output is cached in the coordinator, but it does NOT yet feed any model
input. That happens in v2.4.4 once the SE/EE de-seasonalization in
v2.4.2/v2.4.3 has been validated via the NPK-CVaR hedge methodology.

Endpoints used:
    /restapi/Reservoir/LastWeekData/{n}    last n weeks summary by zone
    /restapi/Reservoir/                    full historical graph data

Output shape (LastWeekData):
    {
      "currentYear": [{ "week": int, "year": int, "total": float (% fill),
                        "no1": float, ..., "no5": float }, ...],
      "lastYear":    [...]
    }

Stability invariant note: per the v2.3 architecture, baseload reads of HA
entities would create feedback loops with the optimizer. The Statnett feed
is a weather-driven external observation independent of any optimizer
decision, so reading it does NOT violate the invariant.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid hard runtime dependency for unit tests
    import aiohttp

_LOGGER = logging.getLogger(__name__)

STATNETT_BASE_URL = "http://driftsdata.statnett.no/restapi"
RESERVOIR_LAST_N_WEEKS_PATH = "/Reservoir/LastWeekData/{n}"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_LOOKBACK_WEEKS = 4

# Cache filename inside <config>/.storage/
CACHE_FILENAME = "spot_price_predictor_hydro_cache.json"

# Min interval between refreshes; hydro data updates ~weekly so daily is plenty.
MIN_REFRESH_INTERVAL_SECONDS = 24 * 3600


class StatnettClientError(Exception):
    """Raised when Statnett returns an unparseable response or times out."""


class StatnettReservoirClient:
    """Async client for Nordic reservoir level data.

    Usage in coordinator:
        client = StatnettReservoirClient(session, cache_dir)
        data = await client.async_get_latest()
        # data['weeks'] is a list of recent weekly observations
        # data['latest']['total_pct'] is the most recent Nordic-wide fill %
    """

    def __init__(
        self,
        session: "aiohttp.ClientSession",
        cache_dir: Path | None = None,
    ) -> None:
        self._session = session
        self._cache_path: Path | None = (
            cache_dir / CACHE_FILENAME if cache_dir is not None else None
        )
        self._last_response: dict[str, Any] | None = None
        self._last_fetched_at: datetime | None = None

    async def async_get_latest(
        self, lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    ) -> dict[str, Any] | None:
        """Return latest reservoir data, fetching if cache is stale.

        Returns a normalized dict:
            {
              'fetched_at': ISO8601 UTC string,
              'source': 'statnett',
              'unit': 'percent_fill',
              'weeks': [{ 'year', 'week', 'total_pct',
                          'zones': {'no1': pct, 'no2': pct, ...} }, ...],
              'latest': { 'year', 'week', 'total_pct', 'zones': {...} },
            }
        Returns None if both the fetch and cache fail.
        """
        # Use cache if it's fresh enough
        if self._is_cache_fresh():
            return self._last_response

        # Try to fetch live. Catch broad Exception so we don't need aiohttp
        # imported at the top of the module (it's an HA-runtime dependency).
        try:
            raw = await self._fetch_raw(lookback_weeks)
            normalized = self._normalize(raw)
            self._last_response = normalized
            self._last_fetched_at = datetime.now(timezone.utc)
            self._persist_cache(normalized)
            _LOGGER.info(
                "Statnett: fetched %d weekly observations; latest total fill %.2f%% "
                "(week %d %d)",
                len(normalized["weeks"]),
                normalized["latest"]["total_pct"],
                normalized["latest"]["week"],
                normalized["latest"]["year"],
            )
            return normalized
        except (asyncio.TimeoutError, StatnettClientError) as err:
            _LOGGER.warning("Statnett fetch failed: %s; trying cache", err)
            return self._load_cache()
        except Exception as err:  # noqa: BLE001 — aiohttp.ClientError or similar
            # aiohttp.ClientError isn't imported eagerly; rely on the bare
            # Exception catch with a class-name check to keep narrow intent.
            cls = type(err).__name__
            if "ClientError" in cls or "Disconnect" in cls or "OSError" in cls:
                _LOGGER.warning("Statnett fetch failed: %s; trying cache", err)
                return self._load_cache()
            raise

    def _is_cache_fresh(self) -> bool:
        if self._last_response is None or self._last_fetched_at is None:
            return False
        age_seconds = (
            datetime.now(timezone.utc) - self._last_fetched_at
        ).total_seconds()
        return age_seconds < MIN_REFRESH_INTERVAL_SECONDS

    async def _fetch_raw(self, lookback_weeks: int) -> dict[str, Any]:
        """GET the LastWeekData/{n} endpoint and return parsed JSON.

        aiohttp is imported lazily so unit tests don't need it installed.
        """
        import aiohttp  # lazy import (HA runtime dep, not test dep)

        url = STATNETT_BASE_URL + RESERVOIR_LAST_N_WEEKS_PATH.format(
            n=int(lookback_weeks)
        )
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with self._session.get(url, timeout=timeout) as response:
            if response.status != 200:
                raise StatnettClientError(
                    f"Statnett returned HTTP {response.status} for {url}"
                )
            try:
                return await response.json(content_type=None)
            except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
                raise StatnettClientError(
                    f"Statnett response was not valid JSON: {exc}"
                ) from exc

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert Statnett's currentYear/lastYear shape into a flat list."""
        if not isinstance(raw, dict):
            raise StatnettClientError(
                f"Statnett response is not a dict: type={type(raw).__name__}"
            )
        current_year = raw.get("currentYear", []) or []
        if not current_year:
            raise StatnettClientError(
                "Statnett response missing 'currentYear' data"
            )

        zone_keys = ("no1", "no2", "no3", "no4", "no5")
        weeks_out: list[dict[str, Any]] = []
        for entry in current_year:
            try:
                weeks_out.append(
                    {
                        "year": int(entry["year"]),
                        "week": int(entry["week"]),
                        "total_pct": float(entry.get("total", 0.0)),
                        "zones": {
                            zk: float(entry[zk])
                            for zk in zone_keys
                            if zk in entry
                        },
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                _LOGGER.warning(
                    "Statnett: skipping malformed entry %r (%s)", entry, exc
                )
                continue
        if not weeks_out:
            raise StatnettClientError(
                "Statnett returned no parseable weekly entries"
            )

        # Sort newest-first; "latest" is the most recent (year, week) tuple
        weeks_out.sort(key=lambda w: (w["year"], w["week"]), reverse=True)
        latest = weeks_out[0]

        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "statnett",
            "unit": "percent_fill",
            "weeks": weeks_out,
            "latest": latest,
        }

    def _persist_cache(self, data: dict[str, Any]) -> None:
        """Write last successful response to disk for restart-resilience."""
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as err:
            _LOGGER.warning(
                "Statnett: failed to write cache to %s: %s",
                self._cache_path, err,
            )

    def _load_cache(self) -> dict[str, Any] | None:
        """Read last persisted response from disk."""
        if self._cache_path is None or not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            self._last_response = data
            self._last_fetched_at = datetime.fromisoformat(
                data.get("fetched_at", datetime.now(timezone.utc).isoformat())
            )
            _LOGGER.info("Statnett: loaded %d weeks from cache", len(data["weeks"]))
            return data
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as err:
            _LOGGER.warning(
                "Statnett: failed to load cache from %s: %s",
                self._cache_path, err,
            )
            return None
