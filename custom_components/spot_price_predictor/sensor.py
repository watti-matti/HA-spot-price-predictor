"""Sensor entities for Spot Price Predictor."""

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
        SpotPriceForecastSensor(coordinator, entry),
        ConsumerPriceSensor(coordinator, entry),
        CheapestHoursSensor(coordinator, entry),
        WeekStatsSensor(coordinator, entry),
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
        "sw_version": "1.4.0",
    }


def _status_attributes(data: dict[str, Any] | None) -> dict[str, Any]:
    """Shared status attributes for all sensors."""
    if not data:
        return {}
    return {
        "last_update": data.get("last_update"),
        "tiers_active": data.get("tiers_active", ""),
        "stale": data.get("stale", False),
        "data_age_minutes": data.get("data_age_minutes", 0),
    }


class SpotPriceForecastSensor(CoordinatorEntity, SensorEntity):
    """Predicted spot price for current hour (EUR/MWh)."""

    _attr_has_entity_name = True
    _attr_name = "Spot Price Forecast"
    _attr_native_unit_of_measurement = "EUR/MWh"
    _attr_icon = "mdi:chart-line"
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_spot_price_forecast"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("spot_price")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        return {
            "forecast": self.coordinator.data.get("spot_forecast", []),
            "unit": "EUR/MWh",
            "forecast_hours": len(self.coordinator.data.get("spot_forecast", [])),
            **_status_attributes(self.coordinator.data),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class ConsumerPriceSensor(CoordinatorEntity, SensorEntity):
    """Consumer price including transfer tariff, VAT, energy tax (EUR/kWh)."""

    _attr_has_entity_name = True
    _attr_name = "Consumer Price"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_icon = "mdi:currency-eur"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_consumer_price"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("consumer_price")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        return {
            "forecast": self.coordinator.data.get("consumer_forecast", []),
            "operator": self._entry.data.get("operator", ""),
            "unit": "EUR/kWh",
            **_status_attributes(self.coordinator.data),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class CheapestHoursSensor(CoordinatorEntity, SensorEntity):
    """Cheapest upcoming hours for load scheduling.

    State: start time of the single cheapest hour in the next 24h.
    Attributes: cheapest consecutive blocks (1h-8h) with start times
    and average prices, plus list of all hours below average price.
    """

    _attr_has_entity_name = True
    _attr_name = "Cheapest Hours"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:clock-check-outline"

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_cheapest_hours"
        self._entry = entry

    @property
    def native_value(self) -> datetime | None:
        if self.coordinator.data:
            ch = self.coordinator.data.get("cheapest_hours", {})
            ts = ch.get("cheapest_1h_start")
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    return None
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        ch = self.coordinator.data.get("cheapest_hours", {})
        return {
            "cheapest_1h_start": ch.get("cheapest_1h_start"),
            "cheapest_1h_price": ch.get("cheapest_1h_price"),
            "cheapest_2h_start": ch.get("cheapest_2h_start"),
            "cheapest_2h_avg_price": ch.get("cheapest_2h_avg_price"),
            "cheapest_3h_start": ch.get("cheapest_3h_start"),
            "cheapest_3h_avg_price": ch.get("cheapest_3h_avg_price"),
            "cheapest_4h_start": ch.get("cheapest_4h_start"),
            "cheapest_4h_avg_price": ch.get("cheapest_4h_avg_price"),
            "cheapest_6h_start": ch.get("cheapest_6h_start"),
            "cheapest_6h_avg_price": ch.get("cheapest_6h_avg_price"),
            "cheapest_8h_start": ch.get("cheapest_8h_start"),
            "cheapest_8h_avg_price": ch.get("cheapest_8h_avg_price"),
            "hours_below_avg": ch.get("hours_below_avg", []),
            "window_hours": ch.get("window_hours", 24),
            "avg_price_in_window": ch.get("avg_price_in_window"),
            **_status_attributes(self.coordinator.data),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class WeekStatsSensor(CoordinatorEntity, SensorEntity):
    """Weekly consumer price forecast statistics (min, avg, max in EUR/kWh)."""

    _attr_has_entity_name = True
    _attr_name = "Week Price Stats"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_icon = "mdi:chart-box-outline"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_week_stats"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """State = weekly average consumer price."""
        if self.coordinator.data:
            forecast = self.coordinator.data.get("consumer_forecast", [])
            if forecast:
                prices = [f["price_eur_kwh"] for f in forecast]
                return round(sum(prices) / len(prices), 4)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        forecast = self.coordinator.data.get("consumer_forecast", [])
        if not forecast:
            return {}
        prices = [f["price_eur_kwh"] for f in forecast]
        return {
            "week_min": round(min(prices), 4),
            "week_avg": round(sum(prices) / len(prices), 4),
            "week_max": round(max(prices), 4),
            "unit": "EUR/kWh",
            "forecast_hours": len(prices),
            "operator": self._entry.data.get("operator", ""),
            **_status_attributes(self.coordinator.data),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


def _get_tariff_config(entry: ConfigEntry) -> dict[str, float]:
    """Extract tariff parameters from config entry."""
    operator_id = entry.data.get(CONF_OPERATOR, "elenia")
    if operator_id == "custom":
        day_rate = entry.data.get(CONF_CUSTOM_DAY_RATE, 0.0361)
        night_rate = entry.data.get(CONF_CUSTOM_NIGHT_RATE, 0.0220)
    else:
        op = OPERATORS.get(operator_id, OPERATORS["elenia"])
        day_rate = op["day_rate"]
        night_rate = op["night_rate"]
    return {
        "day_rate": day_rate,
        "night_rate": night_rate,
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

    # Use dict keyed by timestamp to deduplicate
    entries: dict[str, float] = {}
    attrs = state.attributes

    # Try 'data' attribute first (some Nordpool integrations)
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
        # Fallback: try today/tomorrow attributes
        # Use raw_today/raw_tomorrow first (more reliable), then today/tomorrow
        for attr_name in ("raw_today", "raw_tomorrow", "today", "tomorrow"):
            prices = attrs.get(attr_name)
            if prices and isinstance(prices, list):
                for p in prices:
                    if isinstance(p, dict):
                        ts = p.get("start") or p.get("timestamp")
                        price = p.get("value") or p.get("price")
                        if ts and price is not None:
                            ts_key = str(ts)
                            if ts_key not in entries:  # Don't overwrite
                                entries[ts_key] = float(price)

    # Return full resolution (15-min or hourly depending on source)
    # sorted and deduplicated
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
        # Apply consumer price overhead to each hour
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

        # Selling price = spot price minus commission (no overhead added)
        # Same time range as the buying price sensor (Nordpool today + tomorrow only)
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
