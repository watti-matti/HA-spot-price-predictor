"""DataUpdateCoordinator for Spot Price Predictor."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api_client import ApiClientError, SpotPriceApiClient
from .const import (
    DOMAIN,
    CONF_FINGRID_API_KEY,
    CONF_ENABLE_TIER2,
    CONF_OPERATOR,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    DEFAULT_SELLER_MARGIN,
    CONF_SEARCH_START_HOURS,
    CONF_SEARCH_DURATION_HOURS,
    DEFAULT_SEARCH_START_HOURS,
    DEFAULT_SEARCH_DURATION_HOURS,
    OPERATORS,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
    DEMAND_DEFAULTS,
    UPDATE_INTERVAL_WEATHER,
    FORECAST_HOURS,
    DEFAULT_TIMEZONE,
)

from .features import build_forecast_features
from .holidays import build_holiday_set
from .model import SpotPriceModel

_LOGGER = logging.getLogger(__name__)

RETRY_INTERVAL_SECONDS = 900  # 15 minutes after failure


class SpotPriceCoordinator(DataUpdateCoordinator):
    """Coordinator that fetches data and runs model inference."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=UPDATE_INTERVAL_WEATHER),
            always_update=True,
        )
        self.entry = entry
        session = async_get_clientsession(hass)
        fingrid_key = entry.data.get(CONF_FINGRID_API_KEY)
        self.api = SpotPriceApiClient(session, fingrid_key)
        self.model = SpotPriceModel.load()

        # Operator tariff config
        operator_id = entry.data.get(CONF_OPERATOR, "elenia")
        if operator_id == "custom":
            self.day_rate = entry.data.get(CONF_CUSTOM_DAY_RATE, 0.05)
            self.night_rate = entry.data.get(CONF_CUSTOM_NIGHT_RATE, 0.04)
            self.vat_multiplier = entry.data.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER)
            self.energy_tax = entry.data.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX)
        else:
            op = OPERATORS.get(operator_id, OPERATORS["elenia"])
            self.day_rate = op["day_rate"]
            self.night_rate = op["night_rate"]
            self.vat_multiplier = DEFAULT_VAT_MULTIPLIER
            self.energy_tax = DEFAULT_ENERGY_TAX

        self.seller_margin = entry.data.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN)

        self.enable_tier2 = entry.data.get(CONF_ENABLE_TIER2, False)
        self.has_fingrid = bool(fingrid_key)
        self.search_start_hours = entry.data.get(
            CONF_SEARCH_START_HOURS, DEFAULT_SEARCH_START_HOURS
        )
        self.search_duration_hours = entry.data.get(
            CONF_SEARCH_DURATION_HOURS, DEFAULT_SEARCH_DURATION_HOURS
        )

        # Build holiday set
        now = datetime.now(timezone.utc)
        self.holidays = build_holiday_set(now.year - 1, now.year + 2)

        # Cache last successful result so sensors stay available during API failures
        self._last_successful_data: dict[str, Any] | None = None
        self._last_successful_time: datetime | None = None

        # Rolling forecast history: keeps past predictions so charts
        # can show data from the beginning of the day, not just from
        # the last refresh time. Key = ISO timestamp, value = forecast entry.
        self._forecast_history: dict[str, dict] = {}
        self._consumer_history: dict[str, dict] = {}

    def _return_cached_or_fail(self, err: Exception) -> dict[str, Any]:
        """Return cached data on failure, or raise UpdateFailed if no cache."""
        if self._last_successful_data is not None:
            now = datetime.now(timezone.utc)
            age_minutes = int(
                (now - self._last_successful_time).total_seconds() / 60
            ) if self._last_successful_time else 0
            _LOGGER.warning(
                "Update failed (%s), serving cached data (%d min old). "
                "Retrying in %d minutes",
                err, age_minutes, RETRY_INTERVAL_SECONDS // 60,
            )
            self.update_interval = timedelta(seconds=RETRY_INTERVAL_SECONDS)
            cached = dict(self._last_successful_data)
            cached["stale"] = True
            cached["data_age_minutes"] = age_minutes
            return cached

        raise UpdateFailed(f"API error (no cached data available): {err}") from err

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from APIs and run model inference."""
        _LOGGER.info("Update started")
        try:
            # Fetch weather (always)
            weather = await self.api.fetch_weather()

            # Fetch spot prices
            spot_prices = await self.api.fetch_spot_prices()

            # Tier 2: cross-border prices
            tier2_spreads: dict[str, float] | None = None
            if self.enable_tier2:
                try:
                    neighbor = await self.api.fetch_neighbor_prices()
                    tier2_spreads = self.api.compute_rolling_spreads(spot_prices, neighbor)
                except Exception as err:
                    _LOGGER.warning("Tier 2 data fetch failed: %s", err)

            # Tier 3: Fingrid data
            tier3_data: dict[str, float] | None = None
            if self.has_fingrid:
                try:
                    tier3_data = await self.api.fetch_fingrid_data()
                except Exception as err:
                    _LOGGER.warning("Tier 3 data fetch failed: %s", err)

            # Nuclear outage schedule (Nord Pool UMM, public, no key required)
            tier3_hourly: dict[str, list[float]] | None = None
            if tier3_data and "nuclear_mw" in tier3_data:
                try:
                    outage_schedule = await self.api.fetch_nuclear_outage_schedule()
                    if outage_schedule:
                        now_utc = datetime.now(timezone.utc).replace(
                            minute=0, second=0, microsecond=0)
                        nuclear_hourly = self.api.compute_hourly_nuclear_mw(
                            current_nuclear_mw=tier3_data["nuclear_mw"],
                            outage_schedule=outage_schedule,
                            start_utc=now_utc,
                            hours=min(FORECAST_HOURS, len(weather)),
                        )
                        tier3_hourly = {"nuclear_mw": nuclear_hourly}
                        _LOGGER.info(
                            "Nuclear outage schedule: %d entries, "
                            "nuclear_mw range [%.3f, %.3f]",
                            len(outage_schedule),
                            min(nuclear_hourly),
                            max(nuclear_hourly),
                        )
                except Exception as err:
                    _LOGGER.warning(
                        "UMM outage fetch failed, using constant nuclear_mw: %s", err)

            # Build features for forecast window
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            feature_rows = build_forecast_features(
                start_utc=now,
                hours=min(FORECAST_HOURS, len(weather)),
                weather_data=weather,
                holidays=self.holidays,
                demand=DEMAND_DEFAULTS,
                tier2_spreads=tier2_spreads,
                tier3_data=tier3_data,
                tier3_hourly=tier3_hourly,
            )

            # Run model inference
            predictions = self.model.predict_batch(feature_rows)

            # Build forecast list with timestamps and weather context
            forecast = []
            for i, pred in enumerate(predictions):
                ts = now + timedelta(hours=i)
                entry = {
                    "timestamp": ts.isoformat(),
                    "price_eur_mwh": round(pred, 2),
                }
                # Include weather data for dashboard charts
                if i < len(weather):
                    entry["wind_weighted"] = round(weather[i].get("wind_weighted", 0), 1)
                    entry["solar_weighted"] = round(weather[i].get("solar_weighted", 0), 0)
                    entry["temp_weighted"] = round(weather[i].get("temp_weighted", 0), 1)
                forecast.append(entry)

            # Compute consumer prices (EUR/kWh) with tariff
            consumer_forecast = []
            for entry_item in forecast:
                ts = datetime.fromisoformat(entry_item["timestamp"])
                try:
                    from zoneinfo import ZoneInfo
                    local_hour = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(DEFAULT_TIMEZONE)).hour
                except Exception:
                    local_hour = (ts + timedelta(hours=3)).hour
                is_night = local_hour < 7 or local_hour >= 22
                transfer = self.night_rate if is_night else self.day_rate
                spot_kwh = entry_item["price_eur_mwh"] / 1000.0
                consumer_price = (spot_kwh + self.seller_margin + transfer + self.energy_tax) * self.vat_multiplier
                consumer_forecast.append({
                    "timestamp": entry_item["timestamp"],
                    "price_eur_kwh": round(consumer_price, 5),
                })

            # Cheapest hours calculation (user-configurable search window)
            cheapest_hours = self._find_cheapest_hours(
                forecast, now,
                start_offset_hours=self.search_start_hours,
                duration_hours=self.search_duration_hours,
            )

            # Merge into rolling history (keeps past predictions for charts)
            # New predictions overwrite older ones for the same timestamp
            for f in forecast:
                self._forecast_history[f["timestamp"]] = f
            for c in consumer_forecast:
                self._consumer_history[c["timestamp"]] = c

            # Prune history older than 7 days
            cutoff = (now - timedelta(days=7)).isoformat()
            self._forecast_history = {
                k: v for k, v in self._forecast_history.items() if k >= cutoff
            }
            self._consumer_history = {
                k: v for k, v in self._consumer_history.items() if k >= cutoff
            }

            # Build combined forecast from history (sorted by timestamp)
            combined_forecast = sorted(
                self._forecast_history.values(), key=lambda x: x["timestamp"]
            )
            combined_consumer = sorted(
                self._consumer_history.values(), key=lambda x: x["timestamp"]
            )

            # Current hour values
            current_spot = forecast[0]["price_eur_mwh"] if forecast else 0.0
            current_consumer = consumer_forecast[0]["price_eur_kwh"] if consumer_forecast else 0.0

            # Tiers active description
            tiers = ["Tier 1 (weather)"]
            if tier2_spreads:
                tiers.append("Tier 2 (cross-border)")
            if tier3_data:
                tiers.append("Tier 3 (Fingrid)")

            result = {
                "spot_price": current_spot,
                "spot_forecast": combined_forecast,
                "consumer_price": current_consumer,
                "consumer_forecast": combined_consumer,
                "cheapest_hours": cheapest_hours,
                "tiers_active": " + ".join(tiers),
                "last_update": now.isoformat(),
                "stale": False,
                "data_age_minutes": 0,
            }

            # Cache successful result and restore normal interval
            self._last_successful_data = result
            self._last_successful_time = now
            self.update_interval = timedelta(seconds=UPDATE_INTERVAL_WEATHER)
            _LOGGER.info(
                "Update completed: %d forecast hours, tiers: %s",
                len(forecast), " + ".join(tiers),
            )
            return result

        except ApiClientError as err:
            return self._return_cached_or_fail(err)
        except Exception as err:
            _LOGGER.exception("Unexpected error during update")
            return self._return_cached_or_fail(err)

    @staticmethod
    def _format_offset(hours: int) -> str:
        """Format hours as 'Nd Nh' string."""
        d, h = divmod(hours, 24)
        return f"{d}d {h}h"

    @staticmethod
    def _find_cheapest_hours(
        forecast: list[dict[str, Any]],
        now: datetime,
        start_offset_hours: int = 24,
        duration_hours: int = 48,
    ) -> dict[str, Any]:
        """Find cheapest consecutive hour blocks in a configurable window.

        Args:
            forecast: Full forecast list with timestamp + price_eur_mwh.
            now: Current UTC time.
            start_offset_hours: Hours from now to start of search window.
                Default 24 = tomorrow midnight (approximately).
            duration_hours: Length of search window in hours.
                Default 48 = two days.

        Returns start timestamps and average prices for blocks of
        1, 2, 3, 4, 6, and 8 consecutive hours, plus a list of all
        hours with below-average price (useful for flexible loads).
        """
        window_start = now + timedelta(hours=start_offset_hours)
        window_end = window_start + timedelta(hours=duration_hours)

        # Filter forecast to the search window
        upcoming = []
        for f in forecast:
            try:
                ts = datetime.fromisoformat(f["timestamp"])
                if window_start <= ts < window_end:
                    upcoming.append(f)
            except (ValueError, TypeError):
                continue

        result: dict[str, Any] = {
            "search_start": window_start.isoformat(),
            "search_end": window_end.isoformat(),
            "search_window": (
                f"start {SpotPriceCoordinator._format_offset(start_offset_hours)}"
                f" + duration {SpotPriceCoordinator._format_offset(duration_hours)}"
            ),
            "hours_in_window": len(upcoming),
        }

        if not upcoming:
            return result

        prices = [f["price_eur_mwh"] for f in upcoming]
        avg_price = sum(prices) / len(prices)
        result["avg_price_in_window"] = round(avg_price, 2)

        # Find cheapest N consecutive hours
        def cheapest_block(n: int) -> tuple[str | None, float | None]:
            if len(upcoming) < n:
                return None, None
            best_avg = float("inf")
            best_start = None
            for i in range(len(upcoming) - n + 1):
                block_avg = sum(prices[i:i + n]) / n
                if block_avg < best_avg:
                    best_avg = block_avg
                    best_start = upcoming[i]["timestamp"]
            return best_start, round(best_avg, 2) if best_start else None

        for n in (1, 2, 3, 4, 6, 8):
            start, avg = cheapest_block(n)
            result[f"cheapest_{n}h_start"] = start
            key = "cheapest_1h_price" if n == 1 else f"cheapest_{n}h_avg_price"
            result[key] = avg

        # All hours with price below window average
        result["hours_below_avg"] = [
            f["timestamp"] for f in upcoming
            if f["price_eur_mwh"] < avg_price
        ]

        return result
