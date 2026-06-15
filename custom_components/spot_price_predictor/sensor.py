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
from homeassistant.const import UnitOfSpeed
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
        SpotPriceForecastSensor(coordinator, entry),
        DurationForecastSensor(coordinator, entry),
        EffectiveWindSpeedSensor(coordinator, entry),
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
        "sw_version": "2.11.9-beta.1",
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

        # PV-aware attributes (only when PV is configured). All keys are
        # absent when pv_enabled=False so card templates can branch cleanly.
        if data.get("pv_enabled"):
            attrs["pv_capacity_kwp"] = data.get("pv_capacity_kwp")
            attrs["pv_source"] = data.get("pv_source")
            attrs["baseload_kwh_per_hour"] = data.get("baseload_kwh_per_hour")
            attrs["current_effective_eur_kwh"] = data.get(
                "current_effective_eur_kwh")
            if forecast:
                eff = [f["effective_eur_kwh"] for f in forecast
                       if "effective_eur_kwh" in f]
                if eff:
                    attrs["week_min_effective_eur_kwh"] = round(min(eff), 4)
                    attrs["week_avg_effective_eur_kwh"] = round(
                        sum(eff) / len(eff), 4)
                    attrs["week_max_effective_eur_kwh"] = round(max(eff), 4)

        attrs.update(_status_attributes(data))
        return attrs

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class EffectiveWindSpeedSensor(CoordinatorEntity, SensorEntity):
    """Model's capacity-weighted effective wind speed (FI wind regions).

    This is Open-Meteo `wind_speed_120m` (turbine hub height) aggregated
    capacity-weighted across the Finnish wind regions and used as a
    price-model feature — NOT local surface wind. Exposed as its own
    sensor so downstream consumers (dashboards, optimisers like ENP) can
    read the model's effective wind without making their own Open-Meteo
    calls.

    State: current-hour effective wind speed (m/s).
    Attributes:
      forecast: [{timestamp, wind}, ...] over the forecast horizon (m/s).
    """

    _attr_has_entity_name = True
    _attr_name = "Effective Wind Speed"
    _attr_native_unit_of_measurement = UnitOfSpeed.METERS_PER_SECOND
    _attr_device_class = SensorDeviceClass.WIND_SPEED
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:weather-windy"
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_effective_wind_speed"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        if self.coordinator.data:
            return self.coordinator.data.get("current_wind")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        data = self.coordinator.data
        forecast = data.get("forecast", [])
        wind_series = [
            {"timestamp": f.get("timestamp"), "wind": f.get("wind")}
            for f in forecast
            if f.get("wind") is not None
        ]
        attrs: dict[str, Any] = {
            "forecast": wind_series,
            "forecast_hours": len(wind_series),
            "height_m": 120,
            "aggregation": "capacity-weighted over FI wind regions",
            "source": "open-meteo wind_speed_120m",
        }
        attrs.update(_status_attributes(data))
        return attrs

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


class SpotPriceForecastSensor(CoordinatorEntity, SensorEntity):
    """Spot-price-forecast time series in Nordpool-compatible format.

    State: current-hour forecast spot price (EUR/kWh — converted from
    the pipeline's EUR/MWh output). Matches the unit that the Nordpool
    integration's `state` field uses, so EMHASS / ApexCharts / any
    Nordpool-aware automation can swap to this sensor without changes.

    Attributes mirror the Nordpool schema (`raw_today`, `raw_tomorrow`
    as lists of `{start, end, value}` triples) plus the integration's
    unique value-add `raw_extended` covering the full 170 h forecast
    horizon. The L4 fan-chart bands are exposed under
    `confidence_band` for consumers that want risk-aware decisions.

    This sensor is the time-series interpretation of what the pipeline
    already computes internally — L1 seasonal + L2 Ridge + L3 AR(1) +
    softplus floor → spot_eur_mwh per hour. It surfaces the same
    information in a far easier-to-consume shape than the existing
    `forecast` attribute on `sensor.price_forecast`.
    """

    _attr_has_entity_name = True
    _attr_name = "Spot Price Forecast FI"
    _attr_native_unit_of_measurement = "EUR/kWh"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:cash-multiple"
    _attr_suggested_display_precision = 4

    def __init__(self, coordinator: SpotPriceCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        # Entity id will be `sensor.spot_price_forecast_fi` if HA renders
        # the name into a slug; explicit suggestion below.
        self._attr_unique_id = f"{entry.entry_id}_spot_price_forecast_fi"
        self.entity_id = "sensor.spot_price_forecast_fi"
        self._entry = entry

    @property
    def native_value(self) -> float | None:
        """Current-hour spot forecast in EUR/kWh."""
        if not self.coordinator.data:
            return None
        spot_mwh = self.coordinator.data.get("current_spot_eur_mwh")
        if spot_mwh is None:
            return None
        return round(float(spot_mwh) / 1000.0, 6)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        if not self.coordinator.data:
            return {}
        forecast = self.coordinator.data.get("forecast", []) or []
        if not forecast:
            return {}

        try:
            from zoneinfo import ZoneInfo
            local_tz = ZoneInfo(DEFAULT_TIMEZONE)
        except Exception:
            local_tz = None

        now_utc = datetime.utcnow().replace(tzinfo=None)
        today_local_date = None
        tomorrow_local_date = None
        try:
            if local_tz is not None:
                local_now = datetime.now(tz=local_tz)
                today_local_date = local_now.date()
                from datetime import timedelta as _td
                tomorrow_local_date = (local_now + _td(days=1)).date()
        except Exception:
            today_local_date = None
            tomorrow_local_date = None

        raw_today: list[dict[str, Any]] = []
        raw_tomorrow: list[dict[str, Any]] = []
        raw_extended: list[dict[str, Any]] = []
        confidence_p5: list[float] = []
        confidence_p95: list[float] = []
        prices_today: list[float] = []
        prices_tomorrow: list[float] = []

        for f in forecast:
            ts_iso = f.get("timestamp")
            spot_mwh = f.get("spot_eur_mwh")
            if ts_iso is None or spot_mwh is None:
                continue
            try:
                ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            except Exception:
                continue
            value_kwh = round(float(spot_mwh) / 1000.0, 6)
            try:
                start_local = (ts.astimezone(local_tz)
                               if local_tz is not None else ts)
            except Exception:
                start_local = ts
            end_local_iso = None
            try:
                from datetime import timedelta as _td
                end_local = start_local + _td(hours=1)
                end_local_iso = end_local.isoformat()
            except Exception:
                end_local_iso = None

            entry_item = {
                "start":     start_local.isoformat(),
                "end":       end_local_iso,
                "value":     value_kwh,
            }
            raw_extended.append(entry_item)

            if today_local_date and start_local.date() == today_local_date:
                raw_today.append(entry_item)
                prices_today.append(value_kwh)
            elif tomorrow_local_date and start_local.date() == tomorrow_local_date:
                raw_tomorrow.append(entry_item)
                prices_tomorrow.append(value_kwh)

            # L4 fan-chart bands in EUR/kWh.
            p5 = f.get("P5_eur_mwh")
            p95 = f.get("P95_eur_mwh")
            if p5 is not None and p95 is not None:
                confidence_p5.append(round(float(p5) / 1000.0, 6))
                confidence_p95.append(round(float(p95) / 1000.0, 6))

        def _stats(arr: list[float]) -> dict[str, float | None]:
            if not arr:
                return {"min": None, "avg": None, "max": None}
            return {
                "min": round(min(arr), 6),
                "avg": round(sum(arr) / len(arr), 6),
                "max": round(max(arr), 6),
            }

        today_stats = _stats(prices_today)
        tomorrow_stats = _stats(prices_tomorrow)

        attrs: dict[str, Any] = {
            "raw_today":    raw_today,
            "raw_tomorrow": raw_tomorrow,
            "raw_extended": raw_extended,
            "today_min":    today_stats["min"],
            "today_avg":    today_stats["avg"],
            "today_max":    today_stats["max"],
            "tomorrow_min": tomorrow_stats["min"],
            "tomorrow_avg": tomorrow_stats["avg"],
            "tomorrow_max": tomorrow_stats["max"],
            "forecast_horizon_h": len(raw_extended),
            "currency":     "EUR",
            "unit":         "kWh",
            "source":       "spot_price_predictor L1+L2+L3+L4",
            "last_updated": self.coordinator.data.get("last_update"),
        }
        if confidence_p5 and confidence_p95:
            attrs["confidence_band"] = {
                "p5":  confidence_p5,
                "p95": confidence_p95,
            }
        attrs.update(_status_attributes(self.coordinator.data))
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
         dk_cheap_eur_mwh[24], dk_peak_eur_mwh[24],
         dk_cheap_eur_kwh[24], dk_peak_eur_kwh[24]}

        dk_cheap_eur_mwh[i] = mean spot price (EUR/MWh) of the (i+1)
                              CHEAPEST hours of the day, i=0..23
                              (monotone non-decreasing).
        dk_peak_eur_mwh[i]  = mean spot price (EUR/MWh) of the (i+1)
                              PRICIEST hours of the day, i=0..23
                              (monotone non-increasing).
        dk_cheap_eur_kwh / dk_peak_eur_kwh: same shape, in consumer
                              price (EUR/kWh) with per-hour tariff
                              conversion applied.

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
        """State = first forecast day's cheapest-4h average (EUR/kWh)."""
        if not self.coordinator.data:
            return None
        dk_list = self.coordinator.data.get("duration_forecast", [])
        if not dk_list:
            return None
        cheap_vec = dk_list[0].get("dk_cheap_eur_kwh") or []
        return cheap_vec[3] if len(cheap_vec) >= 4 else None

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

        # PV-aware D(k) scalars (only when PV is configured). The
        # 24-element vectors `dk_cheap_pv_eur_kwh` / `dk_peak_pv_eur_kwh`
        # live inside each `daily_forecast[i]` dict from the coordinator.
        cheap_pv_vec = first.get("dk_cheap_pv_eur_kwh") or []
        peak_pv_vec = first.get("dk_peak_pv_eur_kwh") or []
        if len(cheap_pv_vec) >= 12:
            attrs["today_cheap_pv_1h_eur_kwh"] = cheap_pv_vec[0]
            attrs["today_cheap_pv_4h_eur_kwh"] = cheap_pv_vec[3]
            attrs["today_cheap_pv_8h_eur_kwh"] = cheap_pv_vec[7]
            attrs["today_cheap_pv_12h_eur_kwh"] = cheap_pv_vec[11]
        if len(peak_pv_vec) >= 12:
            attrs["today_peak_pv_1h_eur_kwh"] = peak_pv_vec[0]
            attrs["today_peak_pv_4h_eur_kwh"] = peak_pv_vec[3]
            attrs["today_peak_pv_8h_eur_kwh"] = peak_pv_vec[7]
            attrs["today_peak_pv_12h_eur_kwh"] = peak_pv_vec[11]
        # Surface PV metadata at the duration sensor too for dashboard
        # convenience (matches PriceForecastSensor)
        if self.coordinator.data.get("pv_enabled"):
            attrs["pv_capacity_kwp"] = self.coordinator.data.get(
                "pv_capacity_kwp")
            attrs["pv_source"] = self.coordinator.data.get("pv_source")

        # DtACI calibrated bands for today (consumer EUR/kWh). Only
        # populated when the layer is enabled and warmed up; missing
        # otherwise so card templates can branch cleanly.
        cheap_lo = first.get("dk_cheap_lower_eur_kwh") or []
        cheap_hi = first.get("dk_cheap_upper_eur_kwh") or []
        peak_lo = first.get("dk_peak_lower_eur_kwh") or []
        peak_hi = first.get("dk_peak_upper_eur_kwh") or []
        if len(cheap_lo) >= 4 and len(cheap_hi) >= 4:
            attrs["today_cheap_4h_lower_eur_kwh"] = cheap_lo[3]
            attrs["today_cheap_4h_upper_eur_kwh"] = cheap_hi[3]
        if len(peak_lo) >= 1 and len(peak_hi) >= 1:
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

            # v2.1.1: human-readable warmup status string the Lovelace
            # cards can show as a badge. Examples:
            #   "ready (24/24 warm)"
            #   "warming up (3/24 warm — needs ~2 more daily updates)"
            #   "cold start (0/24 warm)"
            # Logic: scan the configured zone(s) — FI only since v2.11.8 —
            # reporting the min(n_updates) seen and how many of the
            # 2 × 24 = 48 instances are warm. The most-lagging instance
            # dictates whether the global state is "ready" or "warming".
            zones = diag.get("zones") or {}
            min_updates: int | None = None
            warm_total = 0
            inst_total = 0
            min_warmup_threshold: int | None = None
            for zone_state in zones.values():
                per_k = (zone_state.get("per_k") or {})
                for direction in ("cheap", "peak"):
                    block = per_k.get(direction) or {}
                    for inst in block.values():
                        n = inst.get("n_updates")
                        if n is not None:
                            min_updates = (n if min_updates is None
                                           else min(min_updates, n))
                        # Heuristic: if the diagnostic block reports
                        # a non-zero half_width, the instance is past
                        # its min_warmup gate.
                        hw = inst.get("half_width", 0)
                        if hw and hw > 0:
                            warm_total += 1
                        inst_total += 1
            # Derive the threshold from the most-warm instance we know:
            # if any instance's `bias_warm` is true, bias_warmup_steps
            # has been crossed. We don't have direct access to the
            # bundle's min_warmup here, so fall back to v2.1.1 defaults.
            min_warmup_threshold = 5  # v2.1.1 DkDtACIBundle.min_warmup
            bias_threshold = 7         # v2.1.1 bias_warmup_steps
            if min_updates is None or inst_total == 0:
                attrs["dtaci_warmup_status"] = "no data"
            elif min_updates >= max(min_warmup_threshold, bias_threshold):
                attrs["dtaci_warmup_status"] = (
                    f"ready ({warm_total}/{inst_total} warm)"
                )
            elif warm_total > 0:
                remaining = max(min_warmup_threshold,
                                bias_threshold) - min_updates
                attrs["dtaci_warmup_status"] = (
                    f"warming up ({warm_total}/{inst_total} warm "
                    f"— needs ~{remaining} more daily updates)"
                )
            else:
                attrs["dtaci_warmup_status"] = (
                    f"cold start ({warm_total}/{inst_total} warm; "
                    f"{min_warmup_threshold - min_updates} updates "
                    f"to first interval band, "
                    f"{bias_threshold - min_updates} to bias correction)"
                )
            attrs["dtaci_min_n_updates"] = min_updates if min_updates is not None else 0

        attrs.update(_status_attributes(self.coordinator.data))
        return attrs

    @property
    def device_info(self) -> dict[str, Any]:
        return _device_info(self._entry)


# ── Nordpool sensors (optional, for actual price comparison) ────────


def _normalize_entity_id(entity_id: str) -> str:
    """Return a valid HA entity ID, auto-prepending `sensor.` if the
    domain prefix is missing.

    Users sometimes paste the bare object ID (`nordpool_kwh_fi_eur_3_10_0`)
    into the Nordpool-entity config field instead of the full entity ID
    (`sensor.nordpool_kwh_fi_eur_3_10_0`). HA's state lookup fails
    silently in that case; this helper makes the integration forgive
    that.
    """
    if not entity_id or "." in entity_id:
        return entity_id
    return f"sensor.{entity_id}"


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


def _normalize_ts_to_iso(ts: Any) -> str | None:
    """Coerce any reasonable timestamp representation to canonical ISO-8601
    with a `T` separator. Returns None when the input can't be parsed.

    Handles:
      - datetime.datetime objects (the format Nordpool's HACS integration
        produces as of the 15-minute-resolution rollout in 2025+);
      - int/float UNIX seconds (legacy 'data' schema);
      - ISO strings with either 'T' or space separator, or trailing 'Z'.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, (int, float)):
        if ts > 1e9:
            return datetime.fromtimestamp(ts).isoformat()
        return None
    if isinstance(ts, str):
        s = ts.replace(" ", "T").replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).isoformat()
        except (ValueError, TypeError):
            return s if "T" in s else None
    return None


def _process_nordpool_data(hass: HomeAssistant, entity_id: str) -> list[dict[str, Any]]:
    """Process Nordpool sensor attributes into a deduplicated continuous timeline.

    Tries 'data' attribute first, then falls back to today/tomorrow.
    Returns sorted, deduplicated list with one entry per source timestep
    (24 entries when Nordpool publishes hourly; 96 when it publishes
    quarter-hourly per the EU 15-minute-settlement rollout 2025+).

    All timestamps are normalised to canonical ISO 8601 with 'T' separator
    so downstream consumers can `datetime.fromisoformat()` them reliably
    on any supported Python version.
    """
    state = hass.states.get(entity_id)
    if not state:
        return []

    entries: dict[str, float] = {}
    attrs = state.attributes

    data_attr = attrs.get("data")
    if data_attr and isinstance(data_attr, list):
        for d in data_attr:
            ts = (d.get("Timestamp") or d.get("timestamp")
                  or d.get("start"))
            price = (d.get("TotalPrice") or d.get("total_price")
                     or d.get("price") or d.get("value"))
            ts_iso = _normalize_ts_to_iso(ts)
            if ts_iso and price is not None:
                entries[ts_iso] = float(price)
    else:
        for attr_name in ("raw_today", "raw_tomorrow", "today", "tomorrow"):
            prices = attrs.get(attr_name)
            if prices and isinstance(prices, list):
                for p in prices:
                    if isinstance(p, dict):
                        ts = p.get("start") or p.get("timestamp")
                        price = p.get("value") or p.get("price")
                        ts_iso = _normalize_ts_to_iso(ts)
                        if (ts_iso and price is not None
                                and ts_iso not in entries):
                            entries[ts_iso] = float(price)

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
        entity_id = _normalize_entity_id(
            self._entry.data.get(CONF_NORDPOOL_ENTITY, ""))
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
        entity_id = _normalize_entity_id(
            self._entry.data.get(CONF_NORDPOOL_ENTITY, ""))
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
        entity_id = _normalize_entity_id(
            self._entry.data.get(CONF_NORDPOOL_ENTITY, ""))
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
        entity_id = _normalize_entity_id(
            self._entry.data.get(CONF_NORDPOOL_ENTITY, ""))
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
