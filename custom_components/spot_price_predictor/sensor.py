"""Sensor entities for Spot Price Predictor."""

import logging
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

from .const import DOMAIN
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
        PowerControlFactorPM1Sensor(coordinator, entry),
        PowerControlFactor01Sensor(coordinator, entry),
        WindowedAveragePM1Sensor(coordinator, entry),
        WindowedAverage01Sensor(coordinator, entry),
        CheapestHoursSensor(coordinator, entry),
        WeekStatsSensor(coordinator, entry),
    ]
    async_add_entities(entities)


def _device_info(entry: ConfigEntry) -> dict[str, Any]:
    """Shared device info for all sensors."""
    return {
        "identifiers": {(DOMAIN, entry.entry_id)},
        "name": "Spot Price Predictor",
        "manufacturer": "watti-matti",
        "model": "ML Price Predictor v3.1",
        "sw_version": "1.0.0",
    }


class SpotPriceForecastSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Predicted spot price for current hour (EUR/MWh)."""

    _attr_has_entity_name = True
    _attr_name = "Spot Price Forecast"
    _attr_native_unit_of_measurement = "EUR/MWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
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


class ConsumerPriceSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Consumer price including transfer tariff, VAT, energy tax (EUR/kWh)."""

    _attr_has_entity_name = True
    _attr_name = "Consumer Price"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
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


class PowerControlFactorPM1Sensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Power control factor [-1, +1]: cheapest(+1) to most expensive(-1)."""

    _attr_has_entity_name = True
    _attr_name = "Power Control Factor"
    _attr_native_unit_of_measurement = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:tune-vertical"
    _attr_suggested_display_precision = 3

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_power_control_pm1"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("control_factor_pm1")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        return {
            "forecast": self.coordinator.data.get("control_forecast_pm1", []),
            "range": "[-1, +1]",
            "description": "+1 = cheapest, -1 = most expensive",
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class PowerControlFactor01Sensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Power control factor [0, 1]: ON/OFF threshold control."""

    _attr_has_entity_name = True
    _attr_name = "Power Control 0-1"
    _attr_native_unit_of_measurement = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:toggle-switch-outline"
    _attr_suggested_display_precision = 3

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_power_control_01"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            pm1 = self.coordinator.data.get("control_factor_pm1")
            if pm1 is not None:
                return round((pm1 + 1.0) / 2.0, 3)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        forecast_pm1 = self.coordinator.data.get("control_forecast_pm1", [])
        forecast_01 = []
        for f in forecast_pm1:
            forecast_01.append({
                "timestamp": f["timestamp"],
                "factor": round((f["factor"] + 1.0) / 2.0, 3),
            })
        return {
            "forecast": forecast_01,
            "range": "[0, 1]",
            "description": "1 = cheapest, 0 = most expensive",
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class WindowedAveragePM1Sensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Windowed average power control [-1, +1] (smoothed sliding window)."""

    _attr_has_entity_name = True
    _attr_name = "Windowed Control Factor"
    _attr_native_unit_of_measurement = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve"
    _attr_suggested_display_precision = 3

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_windowed_avg_pm1"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("windowed_avg_pm1")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        return {
            "forecast": self.coordinator.data.get("windowed_forecast_pm1", []),
            "range": "[-1, +1]",
            "window_hours": 24,
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class WindowedAverage01Sensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Windowed average power control [0, 1] (smoothed sliding window)."""

    _attr_has_entity_name = True
    _attr_name = "Windowed Control 0-1"
    _attr_native_unit_of_measurement = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve-cumulative"
    _attr_suggested_display_precision = 3

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_windowed_avg_01"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            pm1 = self.coordinator.data.get("windowed_avg_pm1")
            if pm1 is not None:
                return round((pm1 + 1.0) / 2.0, 3)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        forecast_pm1 = self.coordinator.data.get("windowed_forecast_pm1", [])
        forecast_01 = []
        for f in forecast_pm1:
            forecast_01.append({
                "timestamp": f["timestamp"],
                "factor": round((f["factor"] + 1.0) / 2.0, 3),
            })
        return {
            "forecast": forecast_01,
            "range": "[0, 1]",
            "window_hours": 24,
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class CheapestHoursSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Next cheapest hour timestamp with block attributes."""

    _attr_has_entity_name = True
    _attr_name = "Cheapest Hours"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:clock-check-outline"

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_cheapest_hours"
        self._entry = entry

    @property
    def native_value(self) -> str | None:
        if self.coordinator.data:
            ch = self.coordinator.data.get("cheapest_hours", {})
            return ch.get("next_cheapest")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        ch = self.coordinator.data.get("cheapest_hours", {})
        return {
            "cheapest_4h": ch.get("cheapest_4h"),
            "cheapest_8h": ch.get("cheapest_8h"),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class WeekStatsSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Weekly forecast statistics (min, avg, max)."""

    _attr_has_entity_name = True
    _attr_name = "Week Forecast Stats"
    _attr_native_unit_of_measurement = "EUR/MWh"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-box-outline"
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_week_stats"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """State = weekly average predicted price."""
        if self.coordinator.data:
            forecast = self.coordinator.data.get("spot_forecast", [])
            if forecast:
                prices = [f["price_eur_mwh"] for f in forecast]
                return round(sum(prices) / len(prices), 1)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        forecast = self.coordinator.data.get("spot_forecast", [])
        if not forecast:
            return {}
        prices = [f["price_eur_mwh"] for f in forecast]
        return {
            "week_min": round(min(prices), 2),
            "week_avg": round(sum(prices) / len(prices), 2),
            "week_max": round(max(prices), 2),
            "forecast_hours": len(prices),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)
