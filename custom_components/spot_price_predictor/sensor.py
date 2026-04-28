"""Sensor entities for Spot Price Predictor.

Provides two forecast sensors:
  - Price Forecast: 170h hourly price array (spot EUR/MWh + consumer EUR/kWh)
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

    State: current consumer price (EUR/kWh) including transfer tariff,
    energy tax, seller margin, and VAT.

    Attributes:
      forecast: 170h array [{timestamp, spot_eur_mwh, consumer_eur_kwh,
                wind, solar, temp}, ...]
      current_spot_eur_mwh: current hour spot price
      forecast_hours: number of forecast entries
      week_min/avg/max_eur_kwh: statistics over forecast window
      operator: configured operator name

    This sensor provides the raw forecast data. Optimization decisions
    (cheapest hours, load scheduling) should be computed downstream by
    a thermal optimization integration or HA automation templates.
    """

    _attr_has_entity_name = True
    _attr_name = "Price Forecast"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-line"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_price_forecast"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("current_consumer_eur_kwh")
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
            "week_min_eur_kwh": None,
            "week_avg_eur_kwh": None,
            "week_max_eur_kwh": None,
        }

        if forecast:
            prices = [f["consumer_eur_kwh"] for f in forecast if "consumer_eur_kwh" in f]
            if prices:
                attrs["week_min_eur_kwh"] = round(min(prices), 4)
                attrs["week_avg_eur_kwh"] = round(sum(prices) / len(prices), 4)
                attrs["week_max_eur_kwh"] = round(max(prices), 4)

        attrs.update(_status_attributes(data))
        return attrs

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class DurationForecastSensor(CoordinatorEntity, SensorEntity):
    """D(k) duration curve forecast — 7 days ahead.

    State: first forecast day's `dk_cheap_eur_kwh[3]` = average consumer
    price for the cheapest 4 hours of that day (EUR/kWh). This is the
    most-used D(k) value for thermal scheduling of deferrable loads.

    Attributes:
      daily_forecast: 7-day array, each entry:
        {date, weekday, source,
         dk_cheap_eur_kwh[12], dk_peak_eur_kwh[12],
         dk_cheap_spot_eur_mwh[12], dk_peak_spot_eur_mwh[12],
         dk_consumer_eur_kwh[24], dk_spot_eur_mwh[24]}

        Phase A schema (use these for new consumers):
          dk_cheap_eur_kwh[k-1] = mean consumer price (EUR/kWh) of the
                                  CHEAPEST k hours, k=1..12, monotone
                                  non-decreasing.
          dk_peak_eur_kwh[k-1]  = mean consumer price (EUR/kWh) of the
                                  PRICIEST k hours, k=1..12, monotone
                                  non-increasing.
          dk_cheap_spot_eur_mwh / dk_peak_spot_eur_mwh: same for spot prices.

        Legacy schema (deprecated, kept for one transition release; will
        be removed in a future version):
          dk_consumer_eur_kwh[24] / dk_spot_eur_mwh[24] — single cumulative
          D(k) sorted ascending. Indices k=13..24 are not decision-relevant
          for scheduling; use the cheap/peak split above instead.

      forecast_days: number of forecast days
    """

    _attr_has_entity_name = True
    _attr_name = "Duration Forecast"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-timeline-variant-shimmer"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_duration_forecast"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """State = first forecast day's cheapest-4h average (EUR/kWh).

        Prefers the new `dk_cheap_eur_kwh[3]` attribute; falls back to
        the legacy `dk_consumer_eur_kwh[3]` if the coordinator has not
        yet been updated with the new schema.
        """
        if not self.coordinator.data:
            return None
        dk_list = self.coordinator.data.get("duration_forecast", [])
        if not dk_list:
            return None
        first = dk_list[0]
        cheap_vec = first.get("dk_cheap_eur_kwh") or []
        if len(cheap_vec) >= 4:
            return cheap_vec[3]
        # Fallback to legacy 24-array
        legacy_vec = first.get("dk_consumer_eur_kwh") or []
        return legacy_vec[3] if len(legacy_vec) >= 4 else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        dk_list = self.coordinator.data.get("duration_forecast", [])
        if not dk_list:
            return {}

        first = dk_list[0]
        cheap_vec = first.get("dk_cheap_eur_kwh") or []
        peak_vec = first.get("dk_peak_eur_kwh") or []

        # Convenience scalars for common scheduling decisions
        attrs: dict[str, Any] = {
            "daily_forecast": dk_list,
            "forecast_days": len(dk_list),
        }
        if len(cheap_vec) >= 12:
            attrs["today_cheap_1h_eur_kwh"] = cheap_vec[0]
            attrs["today_cheap_4h_eur_kwh"] = cheap_vec[3]
            attrs["today_cheap_8h_eur_kwh"] = cheap_vec[7]
            attrs["today_cheap_12h_eur_kwh"] = cheap_vec[11]
        if len(peak_vec) >= 12:
            attrs["today_peak_1h_eur_kwh"] = peak_vec[0]
            attrs["today_peak_4h_eur_kwh"] = peak_vec[3]
            attrs["today_peak_8h_eur_kwh"] = peak_vec[7]
            attrs["today_peak_12h_eur_kwh"] = peak_vec[11]

        # DtACI calibrated bands for today (consumer EUR/kWh). Only
        # populated when the layer is enabled and warmed up; missing
        # otherwise so card templates can branch cleanly.
        cheap_lo = first.get("dk_cheap_lower_eur_kwh") or []
        cheap_hi = first.get("dk_cheap_upper_eur_kwh") or []
        peak_lo = first.get("dk_peak_lower_eur_kwh") or []
        peak_hi = first.get("dk_peak_upper_eur_kwh") or []
        if len(cheap_lo) == 12 and len(cheap_hi) == 12:
            attrs["today_cheap_4h_lower_eur_kwh"] = cheap_lo[3]
            attrs["today_cheap_4h_upper_eur_kwh"] = cheap_hi[3]
        if len(peak_lo) == 12 and len(peak_hi) == 12:
            attrs["today_peak_1h_lower_eur_kwh"] = peak_lo[0]
            attrs["today_peak_1h_upper_eur_kwh"] = peak_hi[0]

        # DtACI diagnostic block (per-zone, per-(direction, k)).
        # Drives the diagnostics Lovelace card.
        diag = self.coordinator.data.get("dtaci_diagnostics") or {}
        if diag:
            attrs["dtaci_diagnostics"] = diag
            # Top-level scalars for header/state badges
            fi = (diag.get("zones") or {}).get("fi") or {}
            attrs["dtaci_target_coverage"] = diag.get("target_coverage")
            attrs["dtaci_fi_mean_coverage"] = fi.get("mean_coverage")
            attrs["dtaci_fi_mean_width_eur_kwh"] = fi.get("mean_width")
            attrs["dtaci_fi_warm_instances"] = fi.get("n_warm_instances")
            attrs["dtaci_fi_total_instances"] = fi.get("n_total_instances")

        attrs.update(_status_attributes(self.coordinator.data))
        return attrs

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
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
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
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
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
