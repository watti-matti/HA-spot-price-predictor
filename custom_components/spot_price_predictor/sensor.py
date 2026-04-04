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
        "model": "ML Price Predictor v3.1",
        "sw_version": "1.1.0",
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
            "last_update": self.coordinator.data.get("last_update"),
            "tiers_active": self.coordinator.data.get("tiers_active", ""),
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
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


def _process_nordpool_data(hass: HomeAssistant, entity_id: str) -> list[dict[str, Any]]:
    """Process Nordpool sensor today+tomorrow attributes into continuous timeline."""
    state = hass.states.get(entity_id)
    if not state:
        return []

    result = []
    attrs = state.attributes

    # Try 'data' attribute (some Nordpool integrations)
    data_attr = attrs.get("data")
    if data_attr and isinstance(data_attr, list):
        for d in data_attr:
            ts = d.get("Timestamp") or d.get("timestamp")
            price = d.get("TotalPrice") or d.get("total_price") or d.get("price")
            if ts is not None and price is not None:
                if isinstance(ts, (int, float)) and ts > 1e9:
                    ts_iso = datetime.fromtimestamp(ts).isoformat()
                else:
                    ts_iso = str(ts)
                result.append({"timestamp": ts_iso, "price_eur_kwh": float(price)})
        return result

    # Try today/tomorrow and raw_today/raw_tomorrow
    for attr_name in ("today", "tomorrow", "raw_today", "raw_tomorrow"):
        prices = attrs.get(attr_name)
        if prices and isinstance(prices, list):
            for p in prices:
                if isinstance(p, dict):
                    ts = p.get("start") or p.get("timestamp")
                    price = p.get("value") or p.get("price")
                    if ts and price is not None:
                        result.append({"timestamp": str(ts), "price_eur_kwh": float(price)})

    return result


class SpotElectricityPriceSensor(CoordinatorEntity, SensorEntity):
    """Actual spot electricity buying price from Nordpool as continuous timeline."""

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
                return float(state.state)
            except (ValueError, TypeError):
                pass
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        entity_id = self._entry.data.get(CONF_NORDPOOL_ENTITY, "")
        if not entity_id:
            return {}
        return {
            "timeline": _process_nordpool_data(self._hass, entity_id),
            "source_entity": entity_id,
            "unit": "EUR/kWh",
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
        timeline = _process_nordpool_data(self._hass, entity_id)
        return {
            "timeline": [{"timestamp": e["timestamp"],
                          "price_eur_kwh": round(max(0.0, e["price_eur_kwh"] - commission), 5)}
                         for e in timeline],
            "commission_eur_kwh": commission,
            "source_entity": entity_id,
            "unit": "EUR/kWh",
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)
