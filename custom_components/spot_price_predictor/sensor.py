"""Sensor entities for Spot Price Predictor.

Provides two forecast sensors:
  - Price Forecast: 170h hourly price array (spot EUR/MWh + consumer c/kWh)
  - Duration Forecast: 7-day D(k) duration curves (CVaR of daily prices)

Optimization functions (cheapest hours, load scheduling) belong in a
separate thermal optimization integration that consumes these forecasts.

Optional Nordpool sensors provide actual spot prices for comparison.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    CONF_NORDPOOL_ENTITY,
    CONF_ENABLE_PV_SELLING,
    CONF_PV_SELL_COMMISSION,
    DEFAULT_PV_SELL_COMMISSION,
    CONF_OPERATOR,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
    DEFAULT_SELLER_MARGIN,
    OPERATORS,
    DEFAULT_TIMEZONE,
)
from .coordinator import SpotPriceCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor entities from a config entry."""
    coordinator: SpotPriceCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        PriceForecastSensor(coordinator, entry),
        DurationForecastSensor(coordinator, entry),
    ]

    # Add Nordpool-derived sensors if entity is configured
    nordpool_entity = entry.data.get(CONF_NORDPOOL_ENTITY, "")
    if nordpool_entity:
        entities.append(SpotElectricityPriceSensor(coordinator, entry, hass))
        if entry.data.get(CONF_ENABLE_PV_SELLING, False):
            entities.append(SpotElectricitySellingPriceSensor(coordinator, entry, hass))

    async_add_entities(entities)


def _device_info(entry: ConfigEntry) -> dict[str, Any]:
    """Shared device info for all sensors."""
    return {
        "identifiers": {(DOMAIN, entry.entry_id)},
        "name": "Spot Price Predictor",
        "manufacturer": "watti-matti",
        "model": "Spot Price Predictor",
        "sw_version": "2.0.0",
    }


def _status_attributes(data: dict[str, Any] | None) -> dict[str, Any]:
    """Shared status attributes for all sensors."""
    if not data:
        return {}
    return {
        "last_update": data.get("last_update"),
        "data_sources_active": data.get("data_sources_active", ""),
        "stale": data.get("stale", False),
        "data_age_minutes": data.get("data_age_minutes", 0),
    }


# ── Forecast sensors (always created) ──────────────────────────────


class PriceForecastSensor(CoordinatorEntity, SensorEntity):
    """Electricity price forecast — 170 hours ahead.

    State: current consumer price (c/kWh) including transfer tariff,
    energy tax, seller margin, and VAT.

    Attributes:
      forecast: 170h array [{timestamp, spot_eur_mwh, consumer_ckwh,
                wind, solar, temp}, ...]
      current_spot_eur_mwh: current hour spot price
      forecast_hours: number of forecast entries
      week_min/avg/max_ckwh: statistics over forecast window
      operator: configured operator name

    This sensor provides the raw forecast data. Optimization decisions
    (cheapest hours, load scheduling) should be computed downstream by
    a thermal optimization integration or HA automation templates.
    """

    _attr_has_entity_name = True
    _attr_name = "Price Forecast"
    _attr_native_unit_of_measurement = "c/kWh"
    _attr_icon = "mdi:chart-line"
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_price_forecast"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("current_consumer_ckwh")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        data = self.coordinator.data
        forecast = data.get("forecast", [])

        # Compute week statistics from forecast
        attrs: dict[str, Any] = {
            "current_spot_eur_mwh": data.get("current_spot_eur_mwh"),
            "forecast": forecast,
            "forecast_hours": len(forecast),
            "operator": self._entry.data.get(CONF_OPERATOR, ""),
        }

        if forecast:
            prices = [f["consumer_ckwh"] for f in forecast if "consumer_ckwh" in f]
            if prices:
                attrs["week_min_ckwh"] = round(min(prices), 2)
                attrs["week_avg_ckwh"] = round(sum(prices) / len(prices), 2)
                attrs["week_max_ckwh"] = round(max(prices), 2)

        attrs.update(_status_attributes(data))
        return attrs

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class DurationForecastSensor(CoordinatorEntity, SensorEntity):
    """D(k) duration curve forecast — 7 days ahead.

    State: first forecast day's D(4) = average consumer price for the
    cheapest 4 hours (c/kWh). D(k) = CVaR at level k/24.

    Attributes:
      daily_forecast: 7-day array, each entry:
        {date, weekday, dk_consumer_cent_kwh[24], dk_spot_eur_mwh[24]}
        dk_consumer_cent_kwh[k-1] = D(k) consumer price in c/kWh for k=1..24
        dk_spot_eur_mwh[k-1] = D(k) spot price in EUR/MWh for k=1..24
      forecast_days: number of forecast days

    All D(k) vectors are guaranteed length 24 (only complete days
    are included). Thermal optimization can read any D(k) directly
    from the vector: dk_consumer_cent_kwh[k-1] for k=1..24.
    """

    _attr_has_entity_name = True
    _attr_name = "Duration Forecast"
    _attr_native_unit_of_measurement = "c/kWh"
    _attr_icon = "mdi:chart-timeline-variant-shimmer"
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_duration_forecast"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """State = first forecast day's D(4) in c/kWh."""
        if not self.coordinator.data:
            return None
        dk_list = self.coordinator.data.get("duration_forecast", [])
        if not dk_list:
            return None
        dk_vec = dk_list[0].get("dk_consumer_cent_kwh", [])
        return dk_vec[3] if len(dk_vec) >= 4 else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        dk_list = self.coordinator.data.get("duration_forecast", [])
        if not dk_list:
            return {}

        return {
            "daily_forecast": dk_list,
            "forecast_days": len(dk_list),
            **_status_attributes(self.coordinator.data),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


# ── Nordpool sensors (optional, for actual price comparison) ────────


def _get_tariff_config(entry: ConfigEntry) -> dict[str, float]:
    """Extract tariff parameters from config entry.

    Always reads from config entry data so user edits in the options
    flow take effect. Operator defaults used only as fallback.
    """
    operator_id = entry.data.get(CONF_OPERATOR, "elenia")
    op = OPERATORS.get(operator_id, OPERATORS["elenia"])
    return {
        "day_rate": entry.data.get(CONF_CUSTOM_DAY_RATE, op["day_rate"]),
        "night_rate": entry.data.get(CONF_CUSTOM_NIGHT_RATE, op["night_rate"]),
        "vat": entry.data.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER),
        "energy_tax": entry.data.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX),
        "seller_margin": entry.data.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN),
    }


def _spot_to_consumer(spot_eur_kwh: float, hour: int, tariff: dict[str, float]) -> float:
    """Convert spot price to consumer price with full overhead."""
    is_night = hour < 7 or hour >= 22
    transfer = tariff["night_rate"] if is_night else tariff["day_rate"]
    return (spot_eur_kwh + tariff["seller_margin"] + transfer + tariff["energy_tax"]) * tariff["vat"]


def _process_nordpool_data(hass: HomeAssistant, entity_id: str) -> list[dict[str, Any]]:
    """Process Nordpool sensor attributes into a deduplicated continuous timeline.

    Tries 'data' attribute first, then falls back to today/tomorrow.
    Returns sorted, deduplicated list with one entry per hour.
    """
    state = hass.states.get(entity_id)
    if not state:
        return []

    entries: dict[str, float] = {}
    attrs = state.attributes

    data_attr = attrs.get("data")
    if data_attr and isinstance(data_attr, list):
        for d in data_attr:
            ts = d.get("Timestamp") or d.get("timestamp")
            price = d.get("TotalPrice") or d.get("total_price") or d.get("price")
            if ts is not None and price is not None:
                if isinstance(ts, (int, float)) and ts > 1e9:
                    ts_key = datetime.fromtimestamp(ts).isoformat()
                else:
                    ts_key = str(ts)
                entries[ts_key] = float(price)
    else:
        for attr_name in ("raw_today", "raw_tomorrow", "today", "tomorrow"):
            prices = attrs.get(attr_name)
            if prices and isinstance(prices, list):
                for p in prices:
                    if isinstance(p, dict):
                        ts = p.get("start") or p.get("timestamp")
                        price = p.get("value") or p.get("price")
                        if ts and price is not None:
                            ts_key = str(ts)
                            if ts_key not in entries:
                                entries[ts_key] = float(price)

    return sorted(
        [{"timestamp": k, "price_eur_kwh": v} for k, v in entries.items()],
        key=lambda x: x["timestamp"],
    )


class SpotElectricityPriceSensor(CoordinatorEntity, SensorEntity):
    """Actual total consumer electricity price from Nordpool.

    Applies the same overhead as the forecast consumer price:
    (spot + seller_margin + transfer + energy_tax) x VAT
    """

    _attr_has_entity_name = True
    _attr_name = "Spot Electricity Price"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_icon = "mdi:flash"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator, entry: ConfigEntry, hass: HomeAssistant) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_spot_electricity_price"
        self._entry = entry
        self._hass = hass

    @property
    def native_value(self) -> float | None:
        entity_id = self._entry.data.get(CONF_NORDPOOL_ENTITY, "")
        if not entity_id:
            return None
        state = self._hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                spot = float(state.state)
                tariff = _get_tariff_config(self._entry)
                now_hour = datetime.now().hour
                return round(_spot_to_consumer(spot, now_hour, tariff), 5)
            except (ValueError, TypeError):
                pass
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        entity_id = self._entry.data.get(CONF_NORDPOOL_ENTITY, "")
        if not entity_id:
            return {}
        tariff = _get_tariff_config(self._entry)
        raw_timeline = _process_nordpool_data(self._hass, entity_id)
        consumer_timeline = []
        for entry_item in raw_timeline:
            try:
                ts = datetime.fromisoformat(entry_item["timestamp"])
                try:
                    from zoneinfo import ZoneInfo
                    local_hour = ts.astimezone(ZoneInfo(DEFAULT_TIMEZONE)).hour
                except Exception:
                    local_hour = (ts.hour + 3) % 24
                consumer = _spot_to_consumer(entry_item["price_eur_kwh"], local_hour, tariff)
                consumer_timeline.append({
                    "timestamp": entry_item["timestamp"],
                    "price_eur_kwh": round(consumer, 5),
                })
            except (ValueError, TypeError):
                continue
        return {
            "timeline": consumer_timeline,
            "source_entity": entity_id,
            "unit": "EUR/kWh",
            "includes": "spot + margin + transfer + tax + VAT",
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class SpotElectricitySellingPriceSensor(CoordinatorEntity, SensorEntity):
    """Spot electricity selling price (spot minus PV commission)."""

    _attr_has_entity_name = True
    _attr_name = "Spot Electricity Selling Price"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_icon = "mdi:solar-power"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator, entry: ConfigEntry, hass: HomeAssistant) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_spot_electricity_selling_price"
        self._entry = entry
        self._hass = hass

    @property
    def native_value(self) -> float | None:
        entity_id = self._entry.data.get(CONF_NORDPOOL_ENTITY, "")
        commission = self._entry.data.get(CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION)
        if not entity_id:
            return None
        state = self._hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                return max(0.0, float(state.state) - commission)
            except (ValueError, TypeError):
                pass
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        entity_id = self._entry.data.get(CONF_NORDPOOL_ENTITY, "")
        commission = self._entry.data.get(CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION)
        if not entity_id:
            return {}

        raw_timeline = _process_nordpool_data(self._hass, entity_id)
        timeline = [
            {
                "timestamp": e["timestamp"],
                "price_eur_kwh": round(max(0.0, e["price_eur_kwh"] - commission), 5),
            }
            for e in raw_timeline
        ]

        return {
            "timeline": timeline,
            "commission_eur_kwh": commission,
            "source_entity": entity_id,
            "unit": "EUR/kWh",
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)
