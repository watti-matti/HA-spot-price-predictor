"""DataUpdateCoordinator for Spot Price Predictor."""

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
)
from .features import build_forecast_features
from .holidays import build_holiday_set
from .model import SpotPriceModel

_LOGGER = logging.getLogger(__name__)


class SpotPriceCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator that fetches data and runs model inference."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=UPDATE_INTERVAL_WEATHER),
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

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from APIs and run model inference."""
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
            )

            # Run model inference
            predictions = self.model.predict_batch(feature_rows)

            # Build forecast list with timestamps
            forecast = []
            for i, pred in enumerate(predictions):
                ts = now + timedelta(hours=i)
                forecast.append({
                    "timestamp": ts.isoformat(),
                    "price_eur_mwh": round(pred, 2),
                })

            # Compute consumer prices (EUR/kWh) with tariff
            consumer_forecast = []
            for entry_item in forecast:
                ts = datetime.fromisoformat(entry_item["timestamp"])
                local_hour = (ts + timedelta(hours=2)).hour  # Finland approx
                is_night = local_hour < 7 or local_hour >= 22
                transfer = self.night_rate if is_night else self.day_rate
                spot_kwh = entry_item["price_eur_mwh"] / 1000.0
                consumer_price = (spot_kwh + transfer + self.energy_tax) * self.vat_multiplier
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

            # Current hour values
            current_spot = forecast[0]["price_eur_mwh"] if forecast else 0.0
            current_consumer = consumer_forecast[0]["price_eur_kwh"] if consumer_forecast else 0.0

            # Tiers active description
            tiers = ["Tier 1 (weather)"]
            if tier2_spreads:
                tiers.append("Tier 2 (cross-border)")
            if tier3_data:
                tiers.append("Tier 3 (Fingrid)")

            return {
                "spot_price": current_spot,
                "spot_forecast": forecast,
                "consumer_price": current_consumer,
                "consumer_forecast": consumer_forecast,
                "cheapest_hours": cheapest_hours,
                "tiers_active": " + ".join(tiers),
                "last_update": now.isoformat(),
            }

        except ApiClientError as err:
            raise UpdateFailed(f"API error: {err}") from err
        except Exception as err:
            _LOGGER.exception("Unexpected error during update")
            raise UpdateFailed(f"Unexpected error: {err}") from err

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
        upcoming = [
            f for f in forecast
            if window_start <= datetime.fromisoformat(f["timestamp"]) < window_end
        ]

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
