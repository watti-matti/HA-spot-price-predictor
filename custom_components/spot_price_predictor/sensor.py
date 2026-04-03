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
        PowerControlFactorSensor(coordinator, entry),
        CheapestHoursSensor(coordinator, entry),
    ]
    async_add_entities(entities)


class SpotPriceForecastSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Current spot price with forecast attribute."""

    _attr_has_entity_name = True
    _attr_name = "Spot Price Forecast"
    _attr_native_unit_of_measurement = "EUR/MWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-line"

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
            "last_update": self.coordinator.data.get("last_update"),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "Spot Price Predictor",
            "manufacturer": "watti-matti",
            "model": "ML Price Predictor",
        }


class ConsumerPriceSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Consumer price including tariff, VAT, and energy tax."""

    _attr_has_entity_name = True
    _attr_name = "Consumer Price"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:currency-eur"

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
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
        }


class PowerControlFactorSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Power control factor [-1, +1] for automation."""

    _attr_has_entity_name = True
    _attr_name = "Power Control Factor"
    _attr_native_unit_of_measurement = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:tune-vertical"

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_power_control_factor"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("control_factor")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        return {
            "forecast": self.coordinator.data.get("control_forecast", []),
        }

    @property
    def device_info(self) -> dict[str, Any]:
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
        }


class CheapestHoursSensor(CoordinatorEntity[SpotPriceCoordinator], SensorEntity):
    """Next cheapest hour timestamp."""

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
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
        }
