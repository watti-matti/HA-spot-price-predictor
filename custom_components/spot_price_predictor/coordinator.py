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

            # Power control factor: normalize price to [-1, +1]
            prices_list = [f["price_eur_mwh"] for f in forecast]
            if prices_list:
                p_min = min(prices_list)
                p_max = max(prices_list)
                p_range = p_max - p_min if p_max > p_min else 1.0
                p_median = sorted(prices_list)[len(prices_list) // 2]
            else:
                p_min = p_max = p_range = p_median = 0.0

            control_forecast = []
            for f in forecast:
                if p_range > 0:
                    factor = -1.0 + 2.0 * (f["price_eur_mwh"] - p_min) / p_range
                    # Invert: cheap = +1, expensive = -1
                    factor = -factor
                else:
                    factor = 0.0
                control_forecast.append({
                    "timestamp": f["timestamp"],
                    "factor": round(max(-1.0, min(1.0, factor)), 3),
                })

            # Cheapest hours calculation
            cheapest_hours = self._find_cheapest_hours(forecast, now)

            # Windowed average (24h sliding window of top N values)
            window_hours = 24
            windowed_forecast = []
            for i in range(len(control_forecast)):
                start_idx = max(0, i - window_hours // 2)
                end_idx = min(len(control_forecast), i + window_hours // 2)
                window = [control_forecast[j]["factor"] for j in range(start_idx, end_idx)]
                if window:
                    avg = sum(window) / len(window)
                else:
                    avg = 0.0
                windowed_forecast.append({
                    "timestamp": control_forecast[i]["timestamp"],
                    "factor": round(max(-1.0, min(1.0, avg)), 3),
                })

            # Current hour values
            current_spot = forecast[0]["price_eur_mwh"] if forecast else 0.0
            current_consumer = consumer_forecast[0]["price_eur_kwh"] if consumer_forecast else 0.0
            current_control = control_forecast[0]["factor"] if control_forecast else 0.0
            current_windowed = windowed_forecast[0]["factor"] if windowed_forecast else 0.0

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
                "control_factor_pm1": current_control,
                "control_forecast_pm1": control_forecast,
                "windowed_avg_pm1": current_windowed,
                "windowed_forecast_pm1": windowed_forecast,
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
    def _find_cheapest_hours(
        forecast: list[dict[str, Any]], now: datetime
    ) -> dict[str, Any]:
        """Find cheapest consecutive hour blocks in the next 24h."""
        # Filter to next 24 hours
        cutoff = now + timedelta(hours=24)
        upcoming = [
            f for f in forecast
            if datetime.fromisoformat(f["timestamp"]) >= now
            and datetime.fromisoformat(f["timestamp"]) < cutoff
        ]

        if not upcoming:
            return {"next_cheapest": None, "cheapest_4h": None, "cheapest_8h": None}

        # Single cheapest hour
        cheapest = min(upcoming, key=lambda x: x["price_eur_mwh"])

        # Cheapest N consecutive hours
        def cheapest_block(n: int) -> dict[str, Any] | None:
            if len(upcoming) < n:
                return None
            best_avg = float("inf")
            best_start = None
            for i in range(len(upcoming) - n + 1):
                block = upcoming[i:i + n]
                avg = sum(b["price_eur_mwh"] for b in block) / n
                if avg < best_avg:
                    best_avg = avg
                    best_start = block[0]["timestamp"]
            if best_start is None:
                return None
            return {"start": best_start, "avg_price": round(best_avg, 2)}

        return {
            "next_cheapest": cheapest["timestamp"],
            "cheapest_4h": cheapest_block(4),
            "cheapest_8h": cheapest_block(8),
        }
