"""DataUpdateCoordinator for Spot Price Predictor."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api_client import ApiClientError, SpotPriceApiClient
from .const import (
    DOMAIN,
    CONF_FINGRID_API_KEY,
    CONF_ENABLE_NEIGHBOR_PRICES,
    CONF_ENABLE_DTACI_DK,
    CONF_OPERATOR,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    CONF_PV_SELL_COMMISSION,
    CONF_PV_CAPACITY_KWP,
    CONF_PV_TILT_DEG,
    CONF_PV_AZIMUTH_DEG,
    CONF_PV_SYSTEM_EFFICIENCY,
    CONF_PV_EXTERNAL_ENTITY,
    CONF_PV_EXPORT_GRID_FEE,
    CONF_BASELOAD_KWH_PER_HOUR,
    CONF_BASELOAD_DAY_FACTOR,
    CONF_BASELOAD_NIGHT_FACTOR,
    CONF_ANNUAL_CONSUMPTION_KWH,
    CONF_CONSUMPTION_ENTITY,
    CONSUMPTION_HYSTERESIS_PCT,
    CONSUMPTION_SMOOTHING_DAYS,
    FINLAND_RESIDENTIAL_MONTHLY_FACTORS,
    DEFAULT_ENABLE_DTACI_DK,
    DEFAULT_SELLER_MARGIN,
    DEFAULT_PV_SELL_COMMISSION,
    DEFAULT_PV_CAPACITY_KWP,
    DEFAULT_PV_TILT_DEG,
    DEFAULT_PV_AZIMUTH_DEG,
    DEFAULT_PV_SYSTEM_EFFICIENCY,
    DEFAULT_PV_EXPORT_GRID_FEE,
    DEFAULT_BASELOAD_KWH_PER_HOUR,
    DEFAULT_BASELOAD_DAY_FACTOR,
    DEFAULT_BASELOAD_NIGHT_FACTOR,
    DEFAULT_ANNUAL_CONSUMPTION_KWH,
    DEFAULT_CONSUMPTION_ENTITY,
    DTACI_TARGET_COVERAGE,
    DTACI_ZONES,
    OPERATORS,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
    DEMAND_DEFAULTS,
    UPDATE_INTERVAL_WEATHER,
    FORECAST_HOURS,
    DEFAULT_TIMEZONE,
)

from .dk_utils import compute_dk_cheap_peak
from .features import build_forecast_features
from .holidays import build_holiday_set
from .model import SpotPriceModel
from .pv_estimate import (
    estimate_pv_kwh_per_hour,
    marginal_effective_eur_kwh,
    net_household_cost_eur,
)
from .pipeline import Pipeline

_LOGGER = logging.getLogger(__name__)

RETRY_INTERVAL_SECONDS = 900  # 15 minutes after failure


def _iso_to_naive_ts(iso_str: str) -> datetime:
    """Parse an ISO-8601 string to a naive UTC datetime suitable for
    numpy datetime64 construction. Strips trailing 'Z' if present."""
    s = iso_str.rstrip("Z")
    if "+" in s:
        s = s.split("+", 1)[0]
    return datetime.fromisoformat(s).replace(tzinfo=None)


class SpotPriceCoordinator(DataUpdateCoordinator):
    """Coordinator that fetches data and runs model inference.

    Produces a unified forecast array with both spot (EUR/MWh) and
    consumer (EUR/kWh) prices for each hour, plus D(k) duration curves.
    Optimization functions (cheapest hours, load scheduling) are NOT
    included — they belong in a separate thermal optimization layer.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry,
                 model: SpotPriceModel | None = None) -> None:
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
        self.model = model or SpotPriceModel.load()

        # Operator tariff config — always read from config entry data so
        # user edits in the options flow propagate to D(k) consumer pricing.
        # Operator defaults used only as fallback for backward compatibility.
        operator_id = entry.data.get(CONF_OPERATOR, "elenia")
        op = OPERATORS.get(operator_id, OPERATORS["elenia"])
        self.day_rate = entry.data.get(CONF_CUSTOM_DAY_RATE, op["day_rate"])
        self.night_rate = entry.data.get(CONF_CUSTOM_NIGHT_RATE, op["night_rate"])
        self.vat_multiplier = entry.data.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER)
        self.energy_tax = entry.data.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX)

        self.seller_margin = entry.data.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN)

        self.enable_neighbor_prices = entry.data.get(CONF_ENABLE_NEIGHBOR_PRICES, False)
        self.has_fingrid = bool(fingrid_key)
        self.enable_dtaci_dk = entry.data.get(
            CONF_ENABLE_DTACI_DK, DEFAULT_ENABLE_DTACI_DK,
        )

        # ── Household PV system + baseload (Phase 1, post-prediction transform)
        # When `pv_capacity_kwp > 0` the coordinator augments forecast hours
        # with PV-aware marginal effective price `m_h` and per-day PV-aware
        # D(k) cheap/peak duration curves. Baseload is a constant (with
        # optional day/night shape) — coordinator NEVER reads HA energy
        # entities for baseload (stability invariant: open-loop wrt the
        # downstream optimizer; see TECHNICAL_GUIDE).
        self.pv_capacity_kwp = float(entry.data.get(
            CONF_PV_CAPACITY_KWP, DEFAULT_PV_CAPACITY_KWP))
        self.pv_tilt_deg = float(entry.data.get(
            CONF_PV_TILT_DEG, DEFAULT_PV_TILT_DEG))
        self.pv_azimuth_deg = float(entry.data.get(
            CONF_PV_AZIMUTH_DEG, DEFAULT_PV_AZIMUTH_DEG))
        self.pv_efficiency = float(entry.data.get(
            CONF_PV_SYSTEM_EFFICIENCY, DEFAULT_PV_SYSTEM_EFFICIENCY))
        self.pv_external_entity = entry.data.get(
            CONF_PV_EXTERNAL_ENTITY, "") or ""
        self.pv_export_grid_fee = float(entry.data.get(
            CONF_PV_EXPORT_GRID_FEE, DEFAULT_PV_EXPORT_GRID_FEE))
        self.pv_sell_commission = float(entry.data.get(
            CONF_PV_SELL_COMMISSION, DEFAULT_PV_SELL_COMMISSION))
        self.baseload_kwh_per_hour = float(entry.data.get(
            CONF_BASELOAD_KWH_PER_HOUR, DEFAULT_BASELOAD_KWH_PER_HOUR))
        self.baseload_day_factor = float(entry.data.get(
            CONF_BASELOAD_DAY_FACTOR, DEFAULT_BASELOAD_DAY_FACTOR))
        self.baseload_night_factor = float(entry.data.get(
            CONF_BASELOAD_NIGHT_FACTOR, DEFAULT_BASELOAD_NIGHT_FACTOR))

        # ── v2.4 baseload schema (annual_consumption_kwh + consumption_entity)
        # Migration: if the entry only carries v2.3 legacy fields and no
        # `annual_consumption_kwh`, infer it from the legacy values once and
        # log INFO. The legacy fields stay in entry.data untouched so a
        # downgrade to v2.3.x still works.
        self.annual_consumption_kwh = float(entry.data.get(
            CONF_ANNUAL_CONSUMPTION_KWH, 0.0))
        if self.annual_consumption_kwh <= 0.0:
            # Legacy v2.3 entry — derive from baseload_kwh_per_hour and the
            # day/night factors weighted by their hour shares.
            avg_per_hour = self.baseload_kwh_per_hour * (
                (self.baseload_day_factor * 15
                 + self.baseload_night_factor * 9) / 24.0
            )
            self.annual_consumption_kwh = avg_per_hour * 8760.0
            if entry.data.get(CONF_BASELOAD_KWH_PER_HOUR) is not None:
                _LOGGER.info(
                    "v2.4 migration: inferred annual_consumption_kwh = "
                    "%.0f kWh/yr from legacy baseload_kwh_per_hour = %.2f "
                    "(day_factor=%.2f, night_factor=%.2f). To re-tune, edit "
                    "the integration's Options dialog.",
                    self.annual_consumption_kwh, self.baseload_kwh_per_hour,
                    self.baseload_day_factor, self.baseload_night_factor,
                )
        self.consumption_entity = (entry.data.get(
            CONF_CONSUMPTION_ENTITY, DEFAULT_CONSUMPTION_ENTITY) or "")

        # Smoothed-daily-kWh cache for `consumption_entity`. Resolved once
        # per day (not every coordinator cycle); persisted in
        # `.storage/spot_price_predictor_consumption_cache.json`.
        self._consumption_cached_daily_kwh: float | None = None
        self._consumption_last_resolved_at: datetime | None = None

        self._pv_enabled = self.pv_capacity_kwp > 0.0 or bool(
            self.pv_external_entity)

        # DtACI per-D(i) bundles (one per zone). Lazy-loaded on first
        # use so cold-start doesn't block the initial coordinator
        # refresh. State files live under
        # `<config_dir>/.storage/spot_price_predictor_dtaci/`.
        self._dtaci_bundles: dict[str, Any] = {}
        self._dtaci_state_dir: Path | None = None
        # Rolling DK-forecast history per (date, zone) used to reconcile
        # forecasts against actuals once Sähkötin reports the day's
        # complete 24-hour window.
        # Key: date_str (YYYY-MM-DD)
        # Value: {"<zone>": {"cheap": [12], "peak": [12]}}
        self._dk_forecast_history: dict[str, dict[str, dict]] = {}
        # Days that have already been fed to the bundle, to avoid
        # double-counting on subsequent coordinator cycles.
        self._dk_reconciled_dates: set[tuple[str, str]] = set()

        # Build holiday set
        now = datetime.now(timezone.utc)
        self.holidays = build_holiday_set(now.year - 1, now.year + 2)

        # Timezone for local hour lookup
        try:
            from zoneinfo import ZoneInfo
            self._tz = ZoneInfo(DEFAULT_TIMEZONE)
        except Exception:
            self._tz = None

        # Cache last successful result so sensors stay available during API failures
        self._last_successful_data: dict[str, Any] | None = None
        self._last_successful_time: datetime | None = None

        # Rolling forecast history: keeps past predictions so charts
        # can show data from the beginning of the day, not just from
        # the last refresh time. Key = ISO timestamp, value = forecast entry.
        self._forecast_history: dict[str, dict] = {}

        # v2.6.0 — additive L1+L2+L3+L4+floor+calibrators pipeline.
        # Runs alongside the v2.2 model; its outputs land in additional
        # Prediction pipeline (L1+L2+L3+L4+softplus floor+DtACI calibrators).
        # Persistent calibrator state lives under
        # `<config>/.storage/spot_price_predictor_pipeline/`. One-time
        # migration: rename the legacy directory if it still exists, so
        # users keep the accumulated bias-corrector history across the
        # rename.
        self._pipeline = None
        try:
            data_dir = Path(__file__).resolve().parent / "data"
            storage_dir = Path(hass.config.path(
                ".storage", "spot_price_predictor_pipeline"))
            legacy_dir = Path(hass.config.path(
                ".storage", "spot_price_predictor_v26"))
            if legacy_dir.exists() and not storage_dir.exists():
                try:
                    legacy_dir.rename(storage_dir)
                    _LOGGER.info("Migrated calibrator state %s → %s",
                                 legacy_dir, storage_dir)
                except OSError as e:
                    _LOGGER.warning("Calibrator-state migration failed (%s); "
                                    "cold-starting under %s", e, storage_dir)
            self._pipeline = Pipeline(data_dir=data_dir,
                                       storage_dir=storage_dir)
            _LOGGER.info(
                "Prediction pipeline ready (Ridge β shape %s, AR(1) φ=%.3f)",
                self._pipeline._ridge_coef.shape,
                self._pipeline._ar1_phi,
            )
        except Exception as e:
            _LOGGER.warning("Prediction pipeline disabled — init failed: %s", e)
            self._pipeline = None

    def _get_local_hour(self, ts_utc: datetime) -> int:
        """Get local hour for a UTC timestamp."""
        if self._tz:
            try:
                from zoneinfo import ZoneInfo
                aware = ts_utc.replace(tzinfo=ZoneInfo("UTC")) if ts_utc.tzinfo is None else ts_utc
                return aware.astimezone(self._tz).hour
            except Exception:
                pass
        # Fallback: UTC+3 (Finland without DST)
        return (ts_utc.hour + 3) % 24

    def _local_date_str(self, ts_utc: datetime) -> str:
        """Return the local-time YYYY-MM-DD date for a UTC timestamp."""
        if self._tz:
            try:
                from zoneinfo import ZoneInfo
                aware = ts_utc.replace(tzinfo=ZoneInfo("UTC")) if ts_utc.tzinfo is None else ts_utc
                return aware.astimezone(self._tz).strftime("%Y-%m-%d")
            except Exception:
                pass
        return (ts_utc + timedelta(hours=3)).strftime("%Y-%m-%d")

    def _spot_to_consumer_eur_kwh(self, spot_eur_mwh: float, is_night: bool) -> float:
        """Convert spot EUR/MWh to consumer EUR/kWh using configured tariffs."""
        transfer = self.night_rate if is_night else self.day_rate
        spot_kwh = max(0.0, spot_eur_mwh) / 1000.0
        return (spot_kwh + self.seller_margin + transfer + self.energy_tax) \
            * self.vat_multiplier

    # ── PV-aware effective pricing (Phase 1) ──────────────────────

    def _spot_to_sell_eur_kwh(self, spot_eur_mwh: float) -> float:
        """Sell price (EUR/kWh) for excess PV exported to grid.

        Note: `spot_eur_mwh` is NOT clipped at zero — when spot is negative,
        `s_h` becomes negative and the user pays to export. The Phase 1
        marginal-cost model handles this case correctly (m_h ∈ [s_h, b_h]
        can include slightly negative values during deep oversupply).
        """
        return (float(spot_eur_mwh) / 1000.0
                - self.pv_sell_commission
                - self.pv_export_grid_fee)

    def _resolve_baseload(self, ts_utc: datetime) -> float:
        """Per-hour baseload (kWh/h) for the typical-total demand.

        Algorithm (v2.4 schema):
            daily_kwh   = smoothed_consumption_entity OR
                          annual_consumption_kwh / 365
            month_idx   = local-hour's month - 1
            baseload(h) = daily_kwh / 24 × monthly_factor[month_idx]

        STABILITY INVARIANT: when `consumption_entity` is empty (default),
        baseload is a deterministic function of (config, time) — no HA
        entity reads, fully open-loop with respect to the optimizer.
        When `consumption_entity` is set, the smoothed daily kWh comes
        from a long-window (14-day) EMA with 5 % hysteresis, so EMHASS's
        daily scheduling decisions don't propagate back into the forecast.

        Legacy v2.3 path: if `annual_consumption_kwh` was inferred from
        the legacy `baseload_kwh_per_hour` × day/night shape during
        migration, the same inferred value drives this resolver. Existing
        v2.3.x deployments continue to behave identically until the user
        explicitly updates their config to use the new schema.
        """
        # Resolve daily kWh — entity-smoothed if configured, else config.
        daily_kwh = None
        if self.consumption_entity:
            daily_kwh = self._smooth_consumption_entity(ts_utc)
        if daily_kwh is None or daily_kwh <= 0.0:
            daily_kwh = self.annual_consumption_kwh / 365.0

        # Apply monthly seasonal factor based on local time-of-year.
        local_dt = self._get_local_dt(ts_utc)
        month_idx = local_dt.month - 1  # 0..11
        try:
            month_factor = FINLAND_RESIDENTIAL_MONTHLY_FACTORS[month_idx]
        except IndexError:
            month_factor = 1.0

        # Per-hour value. The monthly factor is normalized so its mean is
        # 1.0 across the 12 months — daily_kwh × factor gives the typical
        # daily kWh for that month, divided by 24 hours.
        return max(0.05, daily_kwh / 24.0 * month_factor)

    def _get_local_dt(self, ts_utc: datetime) -> datetime:
        """Convert a UTC timestamp to the configured local timezone."""
        if self._tz:
            try:
                from zoneinfo import ZoneInfo
                aware = ts_utc.replace(tzinfo=ZoneInfo("UTC")) \
                    if ts_utc.tzinfo is None else ts_utc
                return aware.astimezone(self._tz)
            except Exception:
                pass
        # Fallback: UTC+3 (Finland, no DST)
        return ts_utc + timedelta(hours=3)

    def _smooth_consumption_entity(self, ts_utc: datetime) -> float | None:
        """Smoothed typical-daily-kWh from `self.consumption_entity`.

        Returns None on any failure (caller falls back to
        `annual_consumption_kwh / 365`). Recomputes the smoothed value at
        most once per day; the cached value persists across coordinator
        cycles. 5 % hysteresis on the cached value prevents tiny sensor
        fluctuations from re-triggering coordinator updates.

        Sensor type auto-detection (in priority order):

        1. Annual / cumulative-kWh counter (`unit = kWh`, `state_class =
           total_increasing`): query recorder for the value 14 days ago,
           subtract from current, divide by 14.
        2. Daily / monthly utility_meter (`state_class = total`):
           interpret current value × scale factor based on cycle, OR
           read history of past 14 daily totals and average.
        3. Instantaneous power (`unit = W` or `kW`, `device_class =
           power`): use HA's statistics_during_period (28 days, mean) →
           kWh per day.
        4. Unknown — returns None, caller falls back to config.
        """
        # If we already resolved within the last 23 hours, return cache.
        # (23h window so the recompute happens on a slightly drifting
        # boundary; avoids reading HA history on every coordinator tick.)
        if (
            self._consumption_cached_daily_kwh is not None
            and self._consumption_last_resolved_at is not None
            and (ts_utc - self._consumption_last_resolved_at)
                < timedelta(hours=23)
        ):
            return self._consumption_cached_daily_kwh

        try:
            new_value = self._fetch_consumption_daily_kwh(ts_utc)
        except Exception as err:
            _LOGGER.warning(
                "consumption_entity '%s' resolve failed: %s; "
                "falling back to annual_consumption_kwh config",
                self.consumption_entity, err,
            )
            return None
        if new_value is None or new_value <= 0.0:
            return None

        # Apply hysteresis — only update the cached value if the new
        # reading deviates by more than 5 % from the cached one.
        if self._consumption_cached_daily_kwh is not None:
            old = self._consumption_cached_daily_kwh
            if abs(new_value - old) / max(old, 1e-6) < CONSUMPTION_HYSTERESIS_PCT:
                # Within dead-band; keep the old cached value but bump the
                # resolved-at timestamp so we don't recompute again today.
                self._consumption_last_resolved_at = ts_utc
                return old

        self._consumption_cached_daily_kwh = new_value
        self._consumption_last_resolved_at = ts_utc
        _LOGGER.info(
            "consumption_entity smoothed daily kWh = %.2f (entity=%s, "
            "smoothing window %d days, hysteresis %.0f%%)",
            new_value, self.consumption_entity,
            CONSUMPTION_SMOOTHING_DAYS, CONSUMPTION_HYSTERESIS_PCT * 100,
        )
        return new_value

    def _fetch_consumption_daily_kwh(self, ts_utc: datetime) -> float | None:
        """Auto-detect sensor type and return smoothed daily kWh.

        Reading HA's recorder/history is allowed here because we apply
        14-day smoothing — long enough that EMHASS's daily scheduling
        decisions don't propagate back into our value (single-day
        variation is 1/14 ≈ 7 % of the average). Combined with the 5 %
        hysteresis dead-band, the closed-loop gain stays well below 1.
        """
        state = self.hass.states.get(self.consumption_entity)
        if state is None:
            return None
        unit = (state.attributes.get("unit_of_measurement") or "").lower()
        device_class = (state.attributes.get("device_class") or "").lower()
        state_class = (state.attributes.get("state_class") or "").lower()

        if unit in ("kwh",) and state_class in ("total_increasing", "total"):
            # Cumulative-kWh counter (smart meter, utility_meter, ...).
            # Use a 14-day delta from recorder history.
            return self._smooth_kwh_counter(state, ts_utc)
        if unit in ("w", "kw") and device_class == "power":
            # Instantaneous power. Use statistics_during_period for the
            # mean over the smoothing window.
            return self._smooth_power_sensor(state, ts_utc, unit)
        # Unknown sensor type — log and bail.
        _LOGGER.warning(
            "consumption_entity '%s' has unsupported attributes "
            "(unit=%r, state_class=%r, device_class=%r) — falling back "
            "to annual_consumption_kwh config. Supported: kWh counters "
            "(total_increasing or total) and power sensors (W/kW).",
            self.consumption_entity, unit, state_class, device_class,
        )
        return None

    def _smooth_kwh_counter(
        self, state, ts_utc: datetime,
    ) -> float | None:
        """14-day rolling average of daily kWh from a kWh counter."""
        try:
            current = float(state.state)
        except (TypeError, ValueError):
            return None
        try:
            from homeassistant.components.recorder import (
                history,
                get_instance,
            )
        except ImportError:
            return None
        start = ts_utc - timedelta(days=CONSUMPTION_SMOOTHING_DAYS)
        try:
            recorder = get_instance(self.hass)
            past = recorder.history.state_changes_during_period(
                self.hass, start, ts_utc, self.consumption_entity,
                no_attributes=True,
            )
        except Exception:
            try:
                past = history.state_changes_during_period(
                    self.hass, start, ts_utc, self.consumption_entity,
                    no_attributes=True,
                )
            except Exception:
                return None
        rows = past.get(self.consumption_entity) or []
        oldest_value: float | None = None
        for row in rows:
            try:
                v = float(row.state)
                if oldest_value is None:
                    oldest_value = v
                    break
            except (TypeError, ValueError):
                continue
        if oldest_value is None:
            # Not enough history — caller will fall back.
            return None
        if state_class := state.attributes.get("state_class"):
            if state_class.lower() == "total":
                # Daily/monthly utility_meter — counter resets within
                # window; can't take a simple delta. Fall back.
                return None
        delta = current - oldest_value
        if delta <= 0:
            return None
        return delta / float(CONSUMPTION_SMOOTHING_DAYS)

    def _smooth_power_sensor(
        self, state, ts_utc: datetime, unit: str,
    ) -> float | None:
        """28-day mean of instantaneous power → typical daily kWh."""
        try:
            from homeassistant.components.recorder import statistics
        except ImportError:
            return None
        start = ts_utc - timedelta(days=CONSUMPTION_SMOOTHING_DAYS * 2)
        try:
            stats = statistics.statistics_during_period(
                self.hass,
                start, ts_utc,
                statistic_ids={self.consumption_entity},
                period="day",
                units=None,
                types={"mean"},
            )
        except Exception:
            return None
        rows = stats.get(self.consumption_entity) or []
        means = [r.get("mean") for r in rows if r.get("mean") is not None]
        if not means:
            return None
        avg_power = sum(means) / len(means)
        # Convert W → kW if needed; multiply by 24 to get daily kWh.
        if unit == "w":
            avg_power = avg_power / 1000.0
        return float(avg_power) * 24.0

    def _read_external_pv_forecast(self) -> list[float] | None:
        """Read up to 168 h of PV forecast from a configured HA entity.

        Source-agnostic: auto-detects four common attribute conventions
        published by HA PV-forecast integrations and templates. Returns a
        list of hourly kWh values, or None if no convention matches —
        coordinator silently falls back to the internal estimator.

        Supported attribute conventions (checked in order):

        1. ``forecast`` — list[dict] with hourly entries; keys searched:
           ``pv_kwh``, ``kwh``, ``energy``, ``value``. Unit kWh.
        2. ``wh_hours`` — dict {ISO timestamp -> Wh}. Sorted by timestamp,
           divided by 1000 to convert to kWh.
        3. ``watts`` — dict {ISO timestamp -> W}. Sorted by timestamp; at
           1-hour granularity 1 W ≈ 0.001 kWh.
        4. ``irradiance`` — list[number] of pre-multiplied PV power.
           Unit auto-detected by magnitude: any value > 50 → assume W
           (divide by 1000); otherwise treat as kWh.

        Each value is clamped to ``[0, capacity_kwp · efficiency]`` so a
        broken template can't propagate unrealistic spikes downstream.

        Note: reading the entity itself is allowed because the PV forecast
        is weather-driven and independent of optimizer decisions — no
        feedback loop is created. (See stability invariant in
        ``_resolve_baseload``.)
        """
        if not self.pv_external_entity:
            return None
        try:
            state = self.hass.states.get(self.pv_external_entity)
            if state is None:
                return None
            attrs = state.attributes

            ceiling = (self.pv_capacity_kwp * self.pv_efficiency
                       if self.pv_capacity_kwp > 0 else 100.0)

            def _clamp(v: Any) -> float:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return 0.0
                if f != f:  # NaN
                    return 0.0
                return max(0.0, min(f, ceiling))

            # 1) forecast = list of dicts in kWh
            forecast_attr = attrs.get("forecast")
            if isinstance(forecast_attr, list) and forecast_attr:
                values: list[float] = []
                for entry in forecast_attr:
                    if isinstance(entry, dict):
                        v = (entry.get("pv_kwh")
                             or entry.get("kwh")
                             or entry.get("energy")
                             or entry.get("value")
                             or 0.0)
                    else:
                        v = entry
                    values.append(_clamp(v))
                if values:
                    return values

            # 2) wh_hours = dict {ISO ts: Wh}
            wh_hours = attrs.get("wh_hours")
            if isinstance(wh_hours, dict) and wh_hours:
                items = sorted(wh_hours.items(), key=lambda kv: kv[0])
                return [_clamp(float(v) / 1000.0) for _, v in items]

            # 3) watts = dict {ISO ts: W}; at 1-hour granularity 1 W ≈ 1 Wh
            watts = attrs.get("watts")
            if isinstance(watts, dict) and watts:
                items = sorted(watts.items(), key=lambda kv: kv[0])
                return [_clamp(float(v) / 1000.0) for _, v in items]

            # 4) irradiance = list[number] (pre-multiplied PV power);
            #    auto-detect W vs kWh by magnitude.
            irr_attr = attrs.get("irradiance")
            if isinstance(irr_attr, list) and irr_attr:
                # Skip non-numeric entries when probing magnitude
                numeric: list[float] = []
                for v in irr_attr:
                    try:
                        numeric.append(float(v))
                    except (TypeError, ValueError):
                        continue
                if not numeric:
                    return None
                # If the largest magnitude exceeds 50, assume Watts
                divisor = 1000.0 if max(numeric) > 50.0 else 1.0
                return [_clamp(v / divisor) for v in numeric]

            return None
        except Exception as err:
            _LOGGER.warning(
                "PV external entity '%s' read failed: %s; "
                "falling back to internal estimator",
                self.pv_external_entity, err)
            return None

    def _compute_pv_forecast(
        self,
        weather: list[dict[str, Any]],
        n_hours: int,
    ) -> list[float]:
        """Build per-hour PV production forecast (kWh).

        Returns a list of length `n_hours`. External-entity output is
        truncated/extended with zeros to match `n_hours`. When PV is
        disabled (capacity_kwp = 0 and no external entity), returns all
        zeros (caller should treat this as PV-aware path being inactive).
        """
        if not self._pv_enabled:
            return [0.0] * n_hours

        # Try external entity first
        external = self._read_external_pv_forecast()
        if external:
            out = list(external[:n_hours])
            while len(out) < n_hours:
                out.append(0.0)
            return out

        # Internal estimator from Open-Meteo solar irradiance
        out: list[float] = []
        for i in range(n_hours):
            irr = (weather[i].get("solar_weighted", 0.0)
                   if i < len(weather) else 0.0)
            out.append(estimate_pv_kwh_per_hour(
                irradiance_w_m2=float(irr),
                capacity_kwp=self.pv_capacity_kwp,
                tilt_deg=self.pv_tilt_deg,
                azimuth_deg=self.pv_azimuth_deg,
                efficiency=self.pv_efficiency,
            ))
        return out

    # ── DtACI per-D(i) calibration layer ──────────────────────────

    def _dtaci_init_bundles(self) -> None:
        """Lazy-init the four DtACI bundles (FI, SE1, SE3, EE).

        Each bundle is loaded from `<config_dir>/.storage/<DOMAIN>_dtaci/
        dtaci_dk_<zone>.json` if present, otherwise cold-started. Idempotent —
        safe to call every cycle.
        """
        if self._dtaci_bundles:
            return
        try:
            from .dtaci_integration import load_or_create_bundle
            base = Path(self.hass.config.path()) / ".storage" / f"{DOMAIN}_dtaci"
            base.mkdir(parents=True, exist_ok=True)
            self._dtaci_state_dir = base
            for zone in DTACI_ZONES:
                path = base / f"dtaci_dk_{zone}.json"
                self._dtaci_bundles[zone] = load_or_create_bundle(
                    path, target_coverage=DTACI_TARGET_COVERAGE,
                )
            _LOGGER.info(
                "DtACI: initialised %d zone bundles in %s",
                len(self._dtaci_bundles), base,
            )
        except Exception as err:
            _LOGGER.exception("DtACI: bundle init failed: %s", err)

    def _dtaci_record_forecasts(
        self, duration_forecast: list[dict[str, Any]],
    ) -> None:
        """Capture today's day-ahead FI D(i) forecast for later reconciliation.

        Each `duration_forecast` entry with `source == "forecast"` is stored
        by date_str so a subsequent cycle (which sees the same date as
        `source == "actual"`) can pair the forecast with the realised D(i).
        Pruned to the last 14 days to bound memory.
        """
        for d in duration_forecast:
            if d.get("source") != "forecast":
                continue
            date_str = d.get("date")
            cheap = d.get("dk_cheap_eur_kwh") or []
            peak = d.get("dk_peak_eur_kwh") or []
            if not (date_str and len(cheap) >= 24 and len(peak) >= 24):
                continue
            slot = self._dk_forecast_history.setdefault(date_str, {})
            slot["fi"] = {"cheap": list(cheap[:24]), "peak": list(peak[:24])}
        # Prune to last 14 days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)
                  ).strftime("%Y-%m-%d")
        for old in list(self._dk_forecast_history.keys()):
            if old < cutoff:
                del self._dk_forecast_history[old]

    def _dtaci_reconcile_actuals(
        self, duration_forecast: list[dict[str, Any]],
    ) -> int:
        """Feed (forecast, actual) D(i) pairs to the FI bundle.

        For each entry in `duration_forecast` whose source is "actual" and
        whose date appears in our forecast history, build the pair and
        update the FI bundle. Each (zone, date) is only fed once.
        Returns the number of pairs ingested.
        """
        bundle = self._dtaci_bundles.get("fi")
        if bundle is None:
            return 0
        n = 0
        for d in duration_forecast:
            if d.get("source") != "actual":
                continue
            date_str = d.get("date")
            if not date_str:
                continue
            key = ("fi", date_str)
            if key in self._dk_reconciled_dates:
                continue
            forecast_entry = self._dk_forecast_history.get(date_str, {}).get("fi")
            if not forecast_entry:
                # No stored forecast for this date — happens on first
                # ever startup or after long downtime. Skip without fail.
                continue
            actual_cheap = d.get("dk_cheap_eur_kwh") or []
            actual_peak = d.get("dk_peak_eur_kwh") or []
            if len(actual_cheap) < 24 or len(actual_peak) < 24:
                continue
            try:
                bundle.update(
                    forecast_dk_cheap=forecast_entry["cheap"],
                    forecast_dk_peak=forecast_entry["peak"],
                    actual_dk_cheap=list(actual_cheap[:24]),
                    actual_dk_peak=list(actual_peak[:24]),
                )
                self._dk_reconciled_dates.add(key)
                n += 1
            except (KeyError, ValueError) as exc:
                _LOGGER.warning(
                    "DtACI[FI]: reconcile failed for %s: %s", date_str, exc,
                )
        if n:
            _LOGGER.info("DtACI[FI]: reconciled %d new actual day(s)", n)
        return n

    def _dtaci_attach_bands(
        self, duration_forecast: list[dict[str, Any]],
    ) -> None:
        """Mutate forecast entries to add `dk_cheap_lower/upper_eur_kwh`
        and `dk_peak_lower/upper_eur_kwh`. No-op for actual entries
        (their bands collapse to the actual; meaningless to display)."""
        bundle = self._dtaci_bundles.get("fi")
        if bundle is None:
            return
        try:
            from .dtaci_integration import attach_dk_intervals
            forecast_only = [d for d in duration_forecast
                             if d.get("source") == "forecast"]
            attach_dk_intervals(bundle, forecast_only)
        except Exception as err:
            _LOGGER.exception("DtACI[FI]: attach bands failed: %s", err)

    def _dtaci_save(self) -> None:
        """Persist all bundles atomically. Best-effort — failures logged
        but don't propagate (sensor data should still be usable)."""
        if not self._dtaci_state_dir or not self._dtaci_bundles:
            return
        try:
            from .dtaci_integration import save_bundle
            for zone, bundle in self._dtaci_bundles.items():
                path = self._dtaci_state_dir / f"dtaci_dk_{zone}.json"
                save_bundle(path, bundle)
        except Exception as err:
            _LOGGER.warning("DtACI: save_bundle failed: %s", err)

    def _dtaci_diagnostics(self) -> dict[str, Any]:
        """Per-zone diagnostics for the duration-forecast sensor.

        Output structure mirrors the reference UI card's parameter set
        (mean coverage, mean width, dominant gamma, weight entropy, plus
        per-(direction, k) breakdown).
        """
        if not self._dtaci_bundles:
            return {}
        out: dict[str, Any] = {
            "enabled": True,
            "target_coverage": DTACI_TARGET_COVERAGE,
            "zones": {},
        }
        for zone, bundle in self._dtaci_bundles.items():
            try:
                out["zones"][zone] = bundle.diagnostics()
            except Exception as err:
                _LOGGER.warning(
                    "DtACI[%s]: diagnostics failed: %s", zone, err,
                )
        return out

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

            # Cross-border neighbor prices
            neighbor_spreads: dict[str, float] | None = None
            neighbor: dict[str, list] = {}
            if self.enable_neighbor_prices:
                try:
                    neighbor = await self.api.fetch_neighbor_prices()
                    # Use historical FI prices for spread computation (forward-only
                    # spot_prices may not overlap with neighbor historical data)
                    try:
                        fi_historical = await self.api.fetch_spot_prices_historical(days=2)
                    except Exception:
                        fi_historical = []
                    fi_for_spreads = fi_historical + spot_prices if fi_historical else spot_prices
                    neighbor_spreads = self.api.compute_rolling_spreads(fi_for_spreads, neighbor)
                    if neighbor_spreads:
                        _LOGGER.info("Cross-border spreads: %s",
                                     {k: f"{v:.1f}" for k, v in neighbor_spreads.items()})
                except Exception as err:
                    _LOGGER.warning("Cross-border data fetch failed: %s", err)

            # Fingrid nuclear data
            nuclear_data: dict[str, float] | None = None
            if self.has_fingrid:
                try:
                    nuclear_data = await self.api.fetch_fingrid_data()
                except Exception as err:
                    _LOGGER.warning("Fingrid nuclear data fetch failed: %s", err)

            # v2.2: Fingrid day-ahead consumption / wind / solar generation
            # forecasts for net-load feature. Empirically the strongest
            # single feature improvement (cor 0.80 with FI prices, 46 % R²
            # on AR(2) residuals — see studies/fingrid_netload_study.py).
            netload_hourly: dict[str, list[dict[str, Any]]] | None = None
            if self.has_fingrid:
                try:
                    netload_hourly = await self.api.fetch_fingrid_forecasts()
                    if netload_hourly:
                        _LOGGER.info(
                            "Fingrid net-load forecasts: %s",
                            {k: len(v) for k, v in netload_hourly.items()},
                        )
                except Exception as err:
                    _LOGGER.warning(
                        "Fingrid net-load forecast fetch failed: %s", err)

            # Nuclear outage schedule (Nord Pool UMM, public, no key required)
            nuclear_hourly_data: dict[str, list[float]] | None = None
            if nuclear_data and "nuclear_mw" in nuclear_data:
                try:
                    outage_schedule = await self.api.fetch_nuclear_outage_schedule()
                    if outage_schedule:
                        now_utc = datetime.now(timezone.utc).replace(
                            minute=0, second=0, microsecond=0)
                        nuclear_hourly = self.api.compute_hourly_nuclear_mw(
                            current_nuclear_mw=nuclear_data["nuclear_mw"],
                            outage_schedule=outage_schedule,
                            start_utc=now_utc,
                            hours=min(FORECAST_HOURS, len(weather)),
                        )
                        nuclear_hourly_data = {"nuclear_mw": nuclear_hourly}
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

            # AR neighbor price forecasts (uses stored AR models from training)
            ar_neighbor_hourly: dict[str, list[float]] | None = None
            ar_models = getattr(self.model, "ar_models", None)
            if ar_models and self.enable_neighbor_prices:
                try:
                    from .features import compute_ar_forecast
                    now_utc = datetime.now(timezone.utc).replace(
                        minute=0, second=0, microsecond=0)
                    n_hours = min(FORECAST_HOURS, len(weather))
                    ar_neighbor_hourly = {}

                    for prefix in ("se1", "se3", "ee"):
                        if prefix not in ar_models:
                            continue
                        # Get recent actual neighbor prices
                        recent = neighbor.get(prefix, []) if "neighbor" in dir() else []
                        last_prices = [p.get("price_eur_mwh", 0) for p in recent[-24:]] \
                            if recent else []

                        # Build forecast hours: (local_hour, is_workday)
                        forecast_hours = []
                        for i in range(n_hours):
                            h_utc = now_utc + timedelta(hours=i)
                            local_h = self._get_local_hour(h_utc)
                            if self._tz:
                                from zoneinfo import ZoneInfo
                                h_local = h_utc.replace(
                                    tzinfo=ZoneInfo("UTC")).astimezone(self._tz)
                                local_dow = h_local.weekday()
                                date_s = h_local.strftime("%Y-%m-%d")
                            else:
                                h_local = h_utc + timedelta(hours=3)
                                local_dow = h_local.weekday()
                                date_s = h_local.strftime("%Y-%m-%d")
                            is_wd = (local_dow < 5) and (date_s not in self.holidays)
                            forecast_hours.append((local_h, is_wd))

                        ar_preds = compute_ar_forecast(
                            ar_models[prefix], last_prices, forecast_hours)
                        ar_neighbor_hourly[prefix] = ar_preds

                    _LOGGER.info("AR neighbor forecasts computed for %d hours",
                                 n_hours)
                except Exception as err:
                    _LOGGER.warning("AR forecast failed: %s", err)

            # Build features for forecast window
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            feature_rows = build_forecast_features(
                start_utc=now,
                hours=min(FORECAST_HOURS, len(weather)),
                weather_data=weather,
                holidays=self.holidays,
                demand=DEMAND_DEFAULTS,
                neighbor_spreads=neighbor_spreads,
                nuclear_data=nuclear_data,
                nuclear_hourly=nuclear_hourly_data,
                ar_neighbor_hourly=ar_neighbor_hourly,
                netload_hourly=netload_hourly,
            )

            # Run model inference
            predictions = self.model.predict_batch(feature_rows)

            # Build PV forecast (length = number of predictions; all zeros when PV disabled)
            pv_kwh = self._compute_pv_forecast(weather, len(predictions))

            # Build unified forecast: spot + consumer + weather (+ PV-aware) per hour
            forecast = []
            for i, pred in enumerate(predictions):
                ts = now + timedelta(hours=i)
                local_hour = self._get_local_hour(ts)
                is_night = local_hour < 7 or local_hour >= 22
                consumer = self._spot_to_consumer_eur_kwh(pred, is_night)

                entry: dict[str, Any] = {
                    "timestamp": ts.isoformat(),
                    "spot_eur_mwh": round(pred, 2),
                    "consumer_eur_kwh": round(consumer, 4),
                    "wind": round(weather[i].get("wind_weighted", 0), 1) if i < len(weather) else None,
                    "solar": round(weather[i].get("solar_weighted", 0), 0) if i < len(weather) else None,
                    "temp": round(weather[i].get("temp_weighted", 0), 1) if i < len(weather) else None,
                }

                # PV-aware augmentation (only when capacity > 0 or external entity)
                if self._pv_enabled:
                    p_h = pv_kwh[i]
                    c_h = self._resolve_baseload(ts)
                    s_h = self._spot_to_sell_eur_kwh(pred)
                    m_h = marginal_effective_eur_kwh(
                        buy_eur_kwh=consumer,
                        sell_eur_kwh=s_h,
                        pv_kwh=p_h,
                        baseload_kwh=c_h,
                    )
                    n_h = net_household_cost_eur(
                        buy_eur_kwh=consumer,
                        sell_eur_kwh=s_h,
                        pv_kwh=p_h,
                        consumption_kwh=c_h,
                    )
                    entry["pv_production_kwh"] = round(p_h, 3)
                    entry["baseload_kwh"] = round(c_h, 3)
                    entry["effective_eur_kwh"] = round(m_h, 4)
                    entry["net_household_cost_eur"] = round(n_h, 4)
                    entry["is_export_hour"] = bool(p_h > c_h)
                    entry["sell_eur_kwh"] = round(s_h, 4)

                forecast.append(entry)

            # Run the prediction pipeline before the duration model so
            # the D(k) curves see the pipeline's spot/consumer values.
            pipeline_diagnostics: dict[str, Any] = {}
            dk_by_date: dict[str, dict] = {}
            if self._pipeline is not None and forecast:
                try:
                    pipeline_diagnostics, dk_by_date = self._apply_pipeline_pre_dk(
                        forecast)
                except Exception as e:
                    _LOGGER.warning("Prediction pipeline overwrite failed: %s", e)
                    pipeline_diagnostics = {"error": str(e)}

            # Per-day metadata (date/weekday/source + optional PV-aware
            # effective-price D(k)). The canonical price D(k) arrays are
            # injected from the pipeline below.
            duration_forecast = self._compute_duration_forecast(
                forecast, weather, ar_neighbor_hourly, nuclear_data, now,
                netload_hourly=netload_hourly,
            )
            for day in duration_forecast:
                m = dk_by_date.get(day.get("date"))
                if m is not None:
                    for k, v in m.items():
                        day[k] = v

            # Prepend actual D(k) from historical spot prices (yesterday + day before)
            try:
                historical_prices = await self.api.fetch_spot_prices_historical(days=2)
            except Exception:
                historical_prices = []
            actual_dk = self._compute_actual_duration_curves(
                historical_prices or spot_prices, now)
            if actual_dk:
                # Replace forecast entries with actual data for overlapping days,
                # then prepend any remaining actual-only days
                actual_dates = {d["date"]: d for d in actual_dk}
                merged = []
                replaced = 0
                for fd in duration_forecast:
                    if fd["date"] in actual_dates:
                        merged.append(actual_dates.pop(fd["date"]))
                        replaced += 1
                    else:
                        merged.append(fd)
                # Prepend remaining actual days (no forecast overlap)
                remaining = sorted(actual_dates.values(), key=lambda d: d["date"])
                duration_forecast = remaining + merged
                _LOGGER.info(
                    "Actual D(k): %d days total (%d replaced forecast, %d prepended): %s",
                    len(actual_dk), replaced, len(remaining),
                    [d["date"] for d in actual_dk],
                )

            # ── DtACI per-D(i) calibration layer ────────────────────
            # When enabled, run the four-zone bundle: capture today's
            # forecast, reconcile newly-actual days, attach calibrated
            # bands to forecast-mode entries, persist state.
            dtaci_diagnostics: dict[str, Any] = {}
            if self.enable_dtaci_dk:
                self._dtaci_init_bundles()
                self._dtaci_record_forecasts(duration_forecast)
                self._dtaci_reconcile_actuals(duration_forecast)
                self._dtaci_attach_bands(duration_forecast)
                self._dtaci_save()
                dtaci_diagnostics = self._dtaci_diagnostics()

            # Merge into rolling history (keeps past predictions for charts)
            for f in forecast:
                self._forecast_history[f["timestamp"]] = f

            # Prune history older than 24 hours (enough for intra-day chart context)
            cutoff = (now - timedelta(hours=24)).isoformat()
            self._forecast_history = {
                k: v for k, v in self._forecast_history.items() if k >= cutoff
            }

            # Build combined forecast from history (sorted by timestamp)
            combined_forecast = sorted(
                self._forecast_history.values(), key=lambda x: x["timestamp"]
            )

            # (Pipeline ran earlier — before _compute_duration_forecast —
            # so the same spot prices feed both the forecast rows AND the
            # D(k) duration curves.)

            # Active data sources description
            sources = ["weather"]
            if neighbor_spreads:
                sources.append("cross-border")
            if nuclear_data:
                sources.append("nuclear")

            result = {
                "current_consumer_eur_kwh": forecast[0]["consumer_eur_kwh"] if forecast else 0.0,
                "current_spot_eur_mwh": forecast[0]["spot_eur_mwh"] if forecast else 0.0,
                "forecast": combined_forecast,
                "duration_forecast": duration_forecast,
                "dtaci_diagnostics": dtaci_diagnostics,
                "data_sources_active": " + ".join(sources),
                "last_update": now.isoformat(),
                "stale": False,
                "data_age_minutes": 0,
                # PV-aware metadata (always emitted; flags downstream)
                "pv_enabled": bool(self._pv_enabled),
                "pv_capacity_kwp": self.pv_capacity_kwp if self._pv_enabled else 0.0,
                "pv_source": (
                    "external" if (self._pv_enabled and self.pv_external_entity)
                    else ("internal" if self._pv_enabled else "disabled")
                ),
                "baseload_kwh_per_hour": (
                    self.baseload_kwh_per_hour if self._pv_enabled else 0.0
                ),
                "current_effective_eur_kwh": (
                    forecast[0].get("effective_eur_kwh")
                    if (self._pv_enabled and forecast) else None
                ),
                # Diagnostics from the L1-L4 prediction pipeline
                "pipeline_diagnostics": pipeline_diagnostics,
            }

            # Cache successful result and restore normal interval
            self._last_successful_data = result
            self._last_successful_time = now
            self.update_interval = timedelta(seconds=UPDATE_INTERVAL_WEATHER)
            _LOGGER.info(
                "Update completed: %d forecast hours, sources: %s",
                len(forecast), " + ".join(sources),
            )
            return result

        except ApiClientError as err:
            return self._return_cached_or_fail(err)
        except Exception as err:
            _LOGGER.exception("Unexpected error during update")
            return self._return_cached_or_fail(err)

    def _apply_pipeline_pre_dk(
        self,
        forecast: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        """Run the L1+L2+L3+L4+floor prediction pipeline plus fan-chart
        sampling and write the result into every row of ``forecast``.
        Both ``spot_eur_mwh`` and ``consumer_eur_kwh`` are set from the
        pipeline output; per-row ``P5_eur_mwh`` … ``P95_eur_mwh``
        percentiles are added alongside.

        Returns:
            (diagnostics_dict, dk_by_date) where dk_by_date maps ISO
            date string to a dict with the canonical 24-level D(k)
            arrays — ``dk_cheap_eur_mwh``, ``dk_peak_eur_mwh``,
            ``dk_cheap_eur_kwh``, ``dk_peak_eur_kwh`` — for the caller
            to inject onto duration_forecast entries.
        """
        import numpy as np

        if not forecast or self._pipeline is None:
            return {}, {}

        # Build the input arrays the pipeline expects
        timestamps = np.array(
            [_iso_to_naive_ts(f["timestamp"]) for f in forecast],
            dtype="datetime64[ns]",
        )
        wind  = np.array([float(f.get("wind")  or 0.0) for f in forecast])
        solar = np.array([float(f.get("solar") or 0.0) for f in forecast])
        temp  = np.array([float(f.get("temp")  or 0.0) for f in forecast])

        # Y_fi_lag168 cold-start prior (rolling history not yet 7 days deep)
        lag168 = np.zeros(len(forecast), dtype=float)

        out = self._pipeline.compute_forecast(
            timestamps=timestamps,
            wind=wind, solar=solar, temp=temp,
            recent_fi_residuals={"lag168": lag168},
            enable_fan_chart=True,
        )
        pipeline_mean = out["mean_eur_mwh"]

        # Overwrite each forecast row with the pipeline's spot, consumer,
        # and fan-chart percentiles. Group consumer prices by local date
        # so we can build the per-day D(k) arrays in one pass.
        by_date_consumer: dict[str, list[float]] = {}
        for i, f in enumerate(forecast):
            spot = float(pipeline_mean[i])
            f["spot_eur_mwh"] = round(spot, 4)
            consumer = f.get("consumer_eur_kwh")
            try:
                ts = _iso_to_naive_ts(f["timestamp"]).replace(tzinfo=timezone.utc)
                local_h = self._get_local_hour(ts)
                is_night = local_h < 7 or local_h >= 22
                consumer = self._spot_to_consumer_eur_kwh(spot, is_night)
                f["consumer_eur_kwh"] = round(consumer, 4)
                local_date = self._local_date_str(ts)
            except Exception:
                local_date = None
            for q in ("P5", "P25", "P50", "P75", "P95"):
                f[f"{q}_eur_mwh"] = float(out[f"{q}_eur_mwh"][i])
            if local_date is not None and consumer is not None:
                by_date_consumer.setdefault(local_date, []).append(float(consumer))

        # Spot 24-level D(k) directly from the pipeline's hourly means.
        spot_curves = self._pipeline.compute_duration_curves(pipeline_mean, timestamps)
        dk_by_date: dict[str, dict[str, list[float]]] = {}
        for d in spot_curves:
            if d.get("hours_in_day") != 24:
                continue
            dk_by_date[d["date"]] = {
                "dk_cheap_eur_mwh": [round(v, 2) for v in d["dk_cheap_eur_mwh"]],
                "dk_peak_eur_mwh":  [round(v, 2) for v in d["dk_peak_eur_mwh"]],
            }

        # Consumer 24-level D(k) — sort the per-hour consumer prices for
        # each complete day and take cumulative means in both directions.
        for date_str, prices in by_date_consumer.items():
            if len(prices) != 24:
                continue
            entry = dk_by_date.setdefault(date_str, {})
            asc = sorted(prices)
            desc = sorted(prices, reverse=True)
            cheap = []
            peak = []
            s_c = 0.0
            s_p = 0.0
            for i in range(24):
                s_c += asc[i]
                s_p += desc[i]
                cheap.append(round(s_c / (i + 1), 4))
                peak.append(round(s_p / (i + 1), 4))
            entry["dk_cheap_eur_kwh"] = cheap
            entry["dk_peak_eur_kwh"] = peak

        # Persist calibrator state every cycle
        try:
            self._pipeline.save_state()
        except Exception as e:
            _LOGGER.debug("pipeline save_state non-critical: %s", e)

        diagnostics = {
            "pipeline_bias_eur_mwh": out.get("bias_eur_mwh", 0.0),
            "pipeline_ar1_phi": self._pipeline._ar1_phi,
            "pipeline_n_features": int(self._pipeline._ridge_coef.size),
            "pipeline_floor_eur_mwh": -5.0,
        }
        return diagnostics, dk_by_date

    def _compute_duration_forecast(
        self,
        forecast: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        ar_neighbor_hourly: dict[str, list[float]] | None,
        nuclear_data: dict[str, float] | None,
        now: datetime,
        netload_hourly: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the per-day skeleton (date / weekday / source / optional
        PV-aware D(k)). The canonical 24-level price D(k) arrays
        (`dk_cheap_eur_mwh`, `dk_peak_eur_mwh`, `dk_cheap_eur_kwh`,
        `dk_peak_eur_kwh`) are injected by the caller from
        `_apply_pipeline_pre_dk`.
        """
        if not self.model.duration_model:
            return []

        import math
        dur_model = self.model.duration_model
        hdd_threshold = DEMAND_DEFAULTS.get("hdd_threshold", 17.0)

        # Segment hour mapping from model config
        seg_hours: dict[str, list[int]] = {}
        for seg_name, seg_cfg in dur_model.segments.items():
            seg_hours[seg_name] = seg_cfg.get("hours", [])

        # v2.2: build a UTC-hour-keyed lookup for net-load forecasts so
        # we can compute per-segment net_load_mean / net_load_squared_mean
        # below. Falls back to {} when Fingrid forecasts are unavailable
        # (segment_features will use 0.0 for the new keys, matching the
        # training-side fallback in train_duration_model).
        netload_lookup: dict[str, dict[str, float]] = {}
        if netload_hourly:
            for series_name, entries in netload_hourly.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    ts = entry.get("timestamp")
                    if not ts:
                        continue
                    ts_key = ts[:13]   # "YYYY-MM-DDTHH"
                    if ts_key not in netload_lookup:
                        netload_lookup[ts_key] = {}
                    netload_lookup[ts_key][series_name] = float(
                        entry.get("value_mw", 0.0))

        # Group forecast hours by local date
        by_date: dict[str, list[dict]] = {}
        for i, entry in enumerate(forecast):
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if self._tz:
                    from zoneinfo import ZoneInfo
                    aware = ts.replace(tzinfo=ZoneInfo("UTC")) if ts.tzinfo is None else ts
                    local = aware.astimezone(self._tz)
                else:
                    local = ts + timedelta(hours=3)
            except Exception:
                continue

            date_str = local.strftime("%Y-%m-%d")
            if date_str not in by_date:
                by_date[date_str] = []

            # UTC-hour key for net_load lookup (no minutes/seconds)
            utc_hour_key = (ts.replace(tzinfo=timezone.utc)
                            if ts.tzinfo is None else ts).strftime(
                "%Y-%m-%dT%H")
            by_date[date_str].append({
                "local_hour": local.hour,
                "dow": local.weekday(),
                "forecast_idx": i,
                "utc_hour_key": utc_hour_key,
                "wind": weather[i].get("wind_weighted", 0) if i < len(weather) else 3.0,
                "solar": weather[i].get("solar_weighted", 0) if i < len(weather) else 0.0,
                "temp": weather[i].get("temp_weighted", 0) if i < len(weather) else 5.0,
            })

        result: list[dict[str, Any]] = []

        for date_str in sorted(by_date.keys()):
            day_hours = by_date[date_str]
            hour_lookup = {h["local_hour"]: h for h in day_hours}

            # Require all segments to have complete hours → guarantees 24 D(k) levels
            all_complete = True
            for hours_list in seg_hours.values():
                if any(h not in hour_lookup for h in hours_list):
                    all_complete = False
                    break
            if not all_complete:
                continue

            dow = day_hours[0]["dow"]
            is_holiday = date_str in self.holidays
            is_wd = 1.0 if (dow < 5 and not is_holiday) else 0.0
            mo = int(date_str.split("-")[1])

            # Build per-segment features (all segments guaranteed complete)
            segment_features: dict[str, dict[str, float]] = {}

            for seg_name, hours_list in seg_hours.items():
                seg_hrs = [hour_lookup[h] for h in hours_list]

                wind_mean = sum(h["wind"] for h in seg_hrs) / len(seg_hrs)
                solar_mean = sum(h["solar"] for h in seg_hrs) / len(seg_hrs)
                temp_mean = sum(h["temp"] for h in seg_hrs) / len(seg_hrs)
                hdd_mean = max(0.0, hdd_threshold - temp_mean)

                # AR neighbor means for this segment
                def _ar_seg_mean(prefix: str, fallback: float = 40.0) -> float:
                    if not ar_neighbor_hourly or prefix not in ar_neighbor_hourly:
                        return fallback
                    idxs = [h["forecast_idx"] for h in seg_hrs]
                    vals = [ar_neighbor_hourly[prefix][j]
                            for j in idxs
                            if j < len(ar_neighbor_hourly[prefix])]
                    return sum(vals) / len(vals) if vals else fallback

                nuclear_deficit = 0.05  # default when no Fingrid data
                if nuclear_data and "nuclear_mw" in nuclear_data:
                    nuclear_deficit = max(0.0, 1.0 - nuclear_data["nuclear_mw"] / 4372.0)

                # v2.2: per-segment net-load aggregates. Compute the
                # GW-scale net_load over the segment's hours; both
                # net_load_mean and net_load_squared_mean default to 0.0
                # if Fingrid forecasts aren't available for those hours
                # — matches the training-time fallback so old
                # model_coefs without these features still load.
                NETLOAD_CENTER_GW = 6.0
                seg_netloads: list[float] = []
                if netload_lookup:
                    nuc_mw_const = (
                        nuclear_data["nuclear_mw"] * 4372.0
                        if nuclear_data and "nuclear_mw" in nuclear_data
                        else 3500.0
                    )
                    for h in seg_hrs:
                        nl_data = netload_lookup.get(h["utc_hour_key"])
                        if not nl_data:
                            continue
                        cons_mw = nl_data.get("consumption_mw", 0.0)
                        wind_mw = nl_data.get("wind_forecast_mw", 0.0)
                        solar_mw = nl_data.get("solar_forecast_mw", 0.0)
                        seg_netloads.append(
                            (cons_mw - wind_mw - solar_mw - nuc_mw_const)
                            / 1000.0
                        )
                if seg_netloads:
                    nl_mean = sum(seg_netloads) / len(seg_netloads)
                    nl_sq_mean = sum(
                        (n - NETLOAD_CENTER_GW) ** 2 for n in seg_netloads
                    ) / len(seg_netloads)
                else:
                    nl_mean = 0.0
                    nl_sq_mean = 0.0

                segment_features[seg_name] = {
                    "wind_mean": wind_mean,
                    "solar_mean": solar_mean,
                    "hdd_mean": hdd_mean,
                    "se3_mean": _ar_seg_mean("se3"),
                    "se1_mean": _ar_seg_mean("se1"),
                    "nuclear_deficit": nuclear_deficit,
                    "is_workday": is_wd,
                    "month_sin": math.sin(2 * math.pi * mo / 12),
                    "month_cos": math.cos(2 * math.pi * mo / 12),
                    "wind_log_scarcity": math.log1p(max(0.0, 8.0 - wind_mean)),
                    # v2.2: net-load aggregates (0.0 if Fingrid offline)
                    "net_load_mean": nl_mean,
                    "net_load_squared_mean": nl_sq_mean,
                }

            if not segment_features:
                continue

            # Run duration model
            day_result = dur_model.predict_day(segment_features)
            dk_spot = day_result.get("duration_curve", [])
            if len(dk_spot) != 24:
                _LOGGER.warning(
                    "D(k) for %s has %d levels (expected 24), skipping",
                    date_str, len(dk_spot),
                )
                continue

            # Convert to consumer EUR/kWh with per-segment tariff:
            # Extract sorted prices from each segment, convert using
            # the segment's correct day/night rate, then merge and
            # recompute consumer D(k).
            segment_curves = day_result.get("segment_curves", {})
            night_segments = {"night"}  # Segments using night tariff
            consumer_sorted_prices: list[float] = []
            spot_hourly_prices: list[float] = []
            for seg_name, curve in segment_curves.items():
                is_night = seg_name in night_segments
                for i in range(len(curve)):
                    if i == 0:
                        p = curve[0]
                    else:
                        # v2.1.1: removed the `max(0, p)` floor here
                        # (matching the same fix in DurationModel) so
                        # negative spot forecasts surface in
                        # the spot D(k) array. The consumer-side
                        # conversion via `_spot_to_consumer_eur_kwh`
                        # still floors at the fixed-overhead level.
                        p = (i + 1) * curve[i] - i * curve[i - 1]
                    spot_hourly_prices.append(p)
                    consumer_sorted_prices.append(
                        self._spot_to_consumer_eur_kwh(p, is_night))
            consumer_sorted_prices.sort()
            running_sum = 0.0
            dk_consumer: list[float] = []
            for i, cp in enumerate(consumer_sorted_prices):
                running_sum += cp
                dk_consumer.append(round(running_sum / (i + 1), 4))

            if len(dk_consumer) != 24:
                _LOGGER.warning(
                    "Consumer D(k) for %s has %d levels (expected 24), skipping",
                    date_str, len(dk_consumer),
                )
                continue

            # Phase A: dual cheap/peak D(k) curves (length 12 each).
            # `dk_cheap[k-1]` = mean of cheapest k hours, monotone non-decreasing.
            # `dk_peak[k-1]`  = mean of priciest k hours, monotone non-increasing.
            # Phase A.3: prefer the duration model's own dk_cheap_12 /
            # dk_peak_12 when emitted (dual-trained model). They reflect
            # per-direction PAVA on independent Ridge fits, which is more
            # accurate than deriving the peak end by sorting cheap-end
            # forecasts. Fall back to sort-based reconstruction otherwise.
            model_cheap = day_result.get("dk_cheap_12") or []
            model_peak = day_result.get("dk_peak_12") or []
            try:
                if (len(model_cheap) == 12 and len(model_peak) == 12
                        and day_result.get("schema") == "dual"):
                    # Spot-end from the dual model. Convert to consumer
                    # by applying a *single* tariff to each price tier
                    # is not strictly possible (tier ↔ hour mapping is
                    # lost), so we keep the model's spot output for the
                    # `dk_*_spot_eur_mwh` attributes and use the merged-
                    # hourly sort path for the consumer-tariff curves.
                    dk_cheap_spot = list(model_cheap)
                    dk_peak_spot = list(model_peak)
                    dk_cheap_cons, dk_peak_cons = compute_dk_cheap_peak(
                        consumer_sorted_prices)
                else:
                    dk_cheap_cons, dk_peak_cons = compute_dk_cheap_peak(
                        consumer_sorted_prices)
                    dk_cheap_spot, dk_peak_spot = compute_dk_cheap_peak(
                        spot_hourly_prices)
            except ValueError as exc:
                _LOGGER.warning(
                    "Cheap/peak D(k) computation failed for %s: %s",
                    date_str, exc,
                )
                continue

            day_entry: dict[str, Any] = {
                "date": date_str,
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
                "source": "forecast",
                # Canonical 24-level D(k) arrays (`dk_cheap_eur_mwh`,
                # `dk_peak_eur_mwh`, `dk_cheap_eur_kwh`,
                # `dk_peak_eur_kwh`) are injected by the caller from the
                # pipeline output to keep all per-day prices aligned with
                # the per-hour forecast rows.
            }

            # ── Phase 1 PV-aware D(k) cheap/peak ────────────────────
            # Computed directly from the 24 hourly `effective_eur_kwh`
            # (marginal cost) values for this date in the hourly forecast.
            # This bypasses the duration model intermediate — order
            # statistics over m_h are mathematically clean (Theorem in
            # TECHNICAL_GUIDE) and bounded in [s_h, b_h] per hour.
            if self._pv_enabled:
                day_effectives: list[float] = []
                for h in day_hours:
                    idx = h["forecast_idx"]
                    if idx < len(forecast):
                        m_val = forecast[idx].get("effective_eur_kwh")
                        if m_val is not None:
                            day_effectives.append(float(m_val))
                if len(day_effectives) == 24:
                    asc = sorted(day_effectives)
                    desc = sorted(day_effectives, reverse=True)
                    cheap_pv: list[float] = []
                    peak_pv: list[float] = []
                    s_c = 0.0
                    s_p = 0.0
                    for i in range(24):
                        s_c += asc[i]
                        s_p += desc[i]
                        cheap_pv.append(round(s_c / (i + 1), 4))
                        peak_pv.append(round(s_p / (i + 1), 4))
                    day_entry["dk_cheap_pv_eur_kwh"] = cheap_pv
                    day_entry["dk_peak_pv_eur_kwh"] = peak_pv

            result.append(day_entry)

        _LOGGER.info("Duration forecast computed: %d days", len(result))
        return result

    def _compute_actual_duration_curves(
        self,
        spot_prices: list[dict[str, Any]],
        now: datetime,
    ) -> list[dict[str, Any]]:
        """Compute D(k) from actual spot prices for completed past days.

        Returns up to 2 days of actual D(k) curves (yesterday + day before).
        Each entry matches the duration_forecast format with an extra
        'source': 'actual' field.
        """
        if not spot_prices:
            return []

        # Group spot prices by local date
        by_date: dict[str, dict[int, float]] = {}
        for entry in spot_prices:
            ts_str = entry.get("timestamp", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if self._tz:
                    from zoneinfo import ZoneInfo
                    aware = ts.replace(tzinfo=ZoneInfo("UTC")) if ts.tzinfo is None else ts
                    local = aware.astimezone(self._tz)
                else:
                    local = ts + timedelta(hours=3)
                date_str = local.strftime("%Y-%m-%d")
                if date_str not in by_date:
                    by_date[date_str] = {}
                by_date[date_str][local.hour] = entry.get("price_eur_mwh", 0.0)
            except Exception:
                continue

        # Include today + completed past days (Nordpool day-ahead prices
        # are published by 14:00 the day before, so today is always known)
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if self._tz:
            from zoneinfo import ZoneInfo
            tomorrow_str = (now.replace(
                tzinfo=ZoneInfo("UTC")).astimezone(self._tz) + timedelta(days=1)
            ).strftime("%Y-%m-%d")

        result: list[dict[str, Any]] = []
        for date_str in sorted(by_date.keys()):
            if date_str >= tomorrow_str:
                continue
            hours_map = by_date[date_str]
            if len(hours_map) < 24:
                continue

            spot_prices_24 = [hours_map[h] for h in range(24)]
            consumer_prices: list[float] = [
                self._spot_to_consumer_eur_kwh(hours_map[h], h < 7 or h >= 22)
                for h in range(24)
            ]

            spot_asc  = sorted(spot_prices_24)
            spot_desc = sorted(spot_prices_24, reverse=True)
            cons_asc  = sorted(consumer_prices)
            cons_desc = sorted(consumer_prices, reverse=True)

            dk_cheap_mwh: list[float] = []
            dk_peak_mwh:  list[float] = []
            dk_cheap_kwh: list[float] = []
            dk_peak_kwh:  list[float] = []
            s_sa = s_sd = s_ca = s_cd = 0.0
            for i in range(24):
                s_sa += spot_asc[i];  dk_cheap_mwh.append(round(s_sa / (i + 1), 2))
                s_sd += spot_desc[i]; dk_peak_mwh.append(round(s_sd / (i + 1), 2))
                s_ca += cons_asc[i];  dk_cheap_kwh.append(round(s_ca / (i + 1), 4))
                s_cd += cons_desc[i]; dk_peak_kwh.append(round(s_cd / (i + 1), 4))

            dow = datetime.strptime(date_str, "%Y-%m-%d").weekday()
            result.append({
                "date": date_str,
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
                "source": "actual",
                "dk_cheap_eur_mwh": dk_cheap_mwh,
                "dk_peak_eur_mwh":  dk_peak_mwh,
                "dk_cheap_eur_kwh": dk_cheap_kwh,
                "dk_peak_eur_kwh":  dk_peak_kwh,
            })

        _LOGGER.info("Actual D(k) computed: %d days", len(result))
        return result
