"""DataUpdateCoordinator for Spot Price Predictor."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
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
    CONF_PV_MEASURED_POWER_ENTITY,
    CONF_PV_EXPORT_GRID_FEE,
    CONF_BASELOAD_KWH_PER_HOUR,
    CONF_BASELOAD_DAY_FACTOR,
    CONF_BASELOAD_NIGHT_FACTOR,
    CONF_ANNUAL_CONSUMPTION_KWH,
    CONF_CONSUMPTION_ENTITY,
    CONF_CONSUMPTION_PROFILE_ENTITY,
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
    PV_NOWCAST_REFRESH_K_DELTA,
    PV_NOWCAST_REFRESH_MIN_GAP_SECONDS,
    FORECAST_HOURS,
    DEFAULT_TIMEZONE,
)

from .dk_utils import compute_dk_cheap_peak
from .features import build_forecast_features
from .consumption_profile_loader import load_profile_from_entity_attrs
from .holidays import build_holiday_set
from .model import SpotPriceModel
from .pv_aware_cvar import (
    compute_pv_aware_cvar_for_day,
    price_rel_std_for_lead,
)
from . import price_forecast_verifier as _pfv
from . import pv_nowcast
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
        # v2.12.0 — real-time measured PV power sensor (W) for the
        # intraday nowcast correction. Empty → nowcast disabled.
        self.pv_measured_entity = entry.data.get(
            CONF_PV_MEASURED_POWER_ENTITY, "") or ""
        # Smoothed clear-sky index (persists across cycles); last value
        # actually baked into a published forecast; and the last-fired
        # nowcast-triggered refresh time (rate limiting).
        self._pv_nowcast_k: float | None = None
        self._pv_nowcast_k_applied: float | None = None
        self._pv_nowcast_last_refresh_utc: datetime | None = None
        self._pv_nowcast_diag: dict[str, Any] = {}
        self._pv_forecast_snapshot: tuple[datetime, list[float]] | None = None
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
        self.consumption_profile_entity = (entry.data.get(
            CONF_CONSUMPTION_PROFILE_ENTITY, "") or "")

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
        # Rolling hourly RAW pipeline forecasts awaiting reconciliation
        # against realised spot prices; feeds the per-hour bias corrector
        # and the fan-chart calibrator. Key: UTC hour "YYYY-MM-DDTHH:00";
        # value: pre-correction pipeline mean (EUR/MWh). Only hours beyond
        # the known-price horizon at forecast time are recorded; each is
        # fed once and removed. Pruned to 14 days.
        self._bias_forecast_history: dict[str, float] = {}

        # v2.14.0 — learned per-lead-time price-forecast uncertainty.
        # Replaces the static `price_rel_std_for_lead` heuristic that
        # feeds the PV-aware CVaR's buy-price perturbation with a
        # site-specific profile learned from forecast-vs-realized
        # relative error. Lazy-loaded on first duration-forecast build;
        # state file lives under
        # `<config_dir>/.storage/spot_price_predictor_price_verifier/`.
        self._price_verifier: _pfv.PriceForecastVerifier | None = None
        self._price_verifier_path: Path | None = None
        # Realized 24h consumer (buy) price per local date, retained from
        # `_compute_actual_duration_curves` so the verifier can reconcile
        # a forecast against the day's cleared price once it is known.
        self._actual_consumer_prices: dict[str, list[float]] = {}

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

    @staticmethod
    def _align_neighbour_prices(
        forecast_ts_iso: list[str],
        neighbor: dict[str, list[dict[str, Any]]],
    ) -> dict[str, "np.ndarray"]:
        """Align neighbour-zone hourly prices to the forecast timestamps.

        Returns a `{zone: ndarray(len(forecast_ts_iso))}` dict. Missing
        zones, missing hours, or unparseable timestamps are filled with
        NaN; callers (the Pipeline) treat NaN as "no signal" and
        contribute zero to the L2 term.
        """
        import numpy as np

        # Build lookup tables zone → {hour-ISO → price}. Keep the
        # timestamp as a naive UTC ISO at the hour granularity so it
        # matches whatever Sahkotin / Elpriset / Elering return.
        lookup: dict[str, dict[str, float]] = {}
        for zone, entries in (neighbor or {}).items():
            if not isinstance(entries, list):
                continue
            zone_map: dict[str, float] = {}
            for e in entries:
                ts = e.get("timestamp") if isinstance(e, dict) else None
                p  = e.get("price_eur_mwh") if isinstance(e, dict) else None
                if ts is None or p is None:
                    continue
                key = str(ts).split("+")[0].split("Z")[0]
                key = key.replace("T", " ")[:13]  # YYYY-MM-DD HH
                try:
                    zone_map[key] = float(p)
                except (TypeError, ValueError):
                    continue
            if zone_map:
                lookup[zone] = zone_map

        # Materialise aligned arrays.
        out: dict[str, np.ndarray] = {}
        n = len(forecast_ts_iso)
        for zone in ("se1", "se3", "ee"):
            arr = np.full(n, np.nan, dtype=float)
            zmap = lookup.get(zone)
            if not zmap:
                out[zone] = arr
                continue
            for i, ts_iso in enumerate(forecast_ts_iso):
                key = str(ts_iso).split("+")[0].split("Z")[0]
                key = key.replace("T", " ")[:13]
                if key in zmap:
                    arr[i] = zmap[key]
            out[zone] = arr
        return out

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

    def _align_weather_to_now(
        self, weather: list[dict[str, Any]], now: datetime,
    ) -> list[dict[str, Any]]:
        """Re-base the Open-Meteo weather list so index 0 == ``now``.

        Open-Meteo returns the hourly grid starting at 00:00 UTC of the
        current day, so positional indexing against a ``now``-anchored
        forecast clock shifts every weather/solar value later by
        ``now.hour`` hours. Each row now carries a ``timestamp`` (added in
        api_client v2.11.5); we drop the already-elapsed rows so the first
        remaining row is the current hour.

        Falls back to a positional ``now.hour`` slice if timestamps are
        absent (older api_client), and to the input unchanged if neither
        is possible.
        """
        if not weather:
            return weather

        if all("timestamp" in w for w in weather):
            dated: list[tuple[datetime, dict[str, Any]]] = []
            for w in weather:
                try:
                    ts = _iso_to_naive_ts(w["timestamp"]).replace(
                        tzinfo=timezone.utc)
                except Exception:
                    continue
                dated.append((ts, w))
            if dated:
                dated.sort(key=lambda t: t[0])
                aligned = [w for ts, w in dated if ts >= now]
                if aligned:
                    return aligned

        # Positional fallback: assume the array starts at 00:00 UTC today.
        offset = now.hour
        return weather[offset:] if 0 < offset < len(weather) else weather

    def _read_external_pv_forecast(self) -> dict[datetime, float] | list[float] | None:
        """Read up to 168 h of PV forecast from a configured HA entity.

        Source-agnostic: auto-detects four common attribute conventions
        published by HA PV-forecast integrations and templates. Returns
        either a ``{utc_hour -> kWh}`` dict (when the source carries
        timestamps — preferred, lets the caller align by time) or a
        positional ``list`` of hourly kWh (when it does not), or None if no
        convention matches — coordinator falls back to the internal estimator.

        Timestamp alignment (v2.11.5): per-entry timestamps / dict keys are
        now honoured instead of being discarded. Naive timestamps (e.g.
        Forecast.Solar publishes local time without an offset) are
        interpreted in the configured local timezone when available, else
        UTC, then floored to the hour. This prevents the forecast from
        being shifted when the source series starts at 00:00 rather than
        the current hour.

        Supported attribute conventions (checked in order):

        1. ``forecast`` — list[dict] with hourly entries; value keys:
           ``pv_kwh``, ``kwh``, ``energy``, ``value``; timestamp keys:
           ``period_start``, ``datetime``, ``timestamp``, ``time``,
           ``start``, … Unit kWh.
        2. ``wh_hours`` — dict {ISO timestamp -> Wh}, /1000 → kWh.
        3. ``watts`` — dict {ISO timestamp -> W}; at 1-hour granularity
           1 W ≈ 0.001 kWh.
        4. ``irradiance`` — list[number] of pre-multiplied PV power
           (positional, no timestamps). Unit auto-detected by magnitude:
           any value > 50 → assume W (divide by 1000); otherwise kWh.

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

            def _parse_ts(s: Any) -> datetime | None:
                """Parse an external timestamp to a UTC hour. Naive strings
                are interpreted in the configured local tz when available
                (Forecast.Solar etc. publish local time), else UTC."""
                try:
                    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                except Exception:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self._tz or timezone.utc)
                return dt.astimezone(timezone.utc).replace(
                    minute=0, second=0, microsecond=0)

            def _scaled(v: Any, divisor: float) -> float | None:
                try:
                    return _clamp(float(v) / divisor)
                except (TypeError, ValueError):
                    return None

            # 1) forecast = list of dicts in kWh (honour per-entry timestamp)
            forecast_attr = attrs.get("forecast")
            if isinstance(forecast_attr, list) and forecast_attr:
                ts_keys = ("period_start", "datetime", "date_time",
                           "timestamp", "time", "start", "from", "hour")
                values: list[float] = []
                dated: dict[datetime, float] = {}
                for entry in forecast_attr:
                    ts = None
                    if isinstance(entry, dict):
                        v = (entry.get("pv_kwh")
                             or entry.get("kwh")
                             or entry.get("energy")
                             or entry.get("value")
                             or 0.0)
                        for k in ts_keys:
                            if k in entry:
                                ts = _parse_ts(entry[k])
                                if ts is not None:
                                    break
                    else:
                        v = entry
                    cv = _clamp(v)
                    values.append(cv)
                    if ts is not None:
                        dated[ts] = cv
                # Prefer timestamp alignment when most entries carry a time.
                if len(dated) >= max(1, len(values) // 2):
                    return dated
                if values:
                    return values

            # 2) wh_hours = dict {ISO ts: Wh}
            wh_hours = attrs.get("wh_hours")
            if isinstance(wh_hours, dict) and wh_hours:
                dated = {}
                for k, v in wh_hours.items():
                    ts = _parse_ts(k)
                    sv = _scaled(v, 1000.0)
                    if ts is not None and sv is not None:
                        dated[ts] = sv
                if dated:
                    return dated
                items = sorted(wh_hours.items(), key=lambda kv: kv[0])
                return [_clamp(float(v) / 1000.0) for _, v in items]

            # 3) watts = dict {ISO ts: W}; at 1-hour granularity 1 W ≈ 1 Wh
            watts = attrs.get("watts")
            if isinstance(watts, dict) and watts:
                dated = {}
                for k, v in watts.items():
                    ts = _parse_ts(k)
                    sv = _scaled(v, 1000.0)
                    if ts is not None and sv is not None:
                        dated[ts] = sv
                if dated:
                    return dated
                items = sorted(watts.items(), key=lambda kv: kv[0])
                return [_clamp(float(v) / 1000.0) for _, v in items]

            # 4) irradiance = list[number]: pre-multiplied PV POWER (W or
            #    kWh), auto-detected by magnitude. When a parallel time axis
            #    (`iso_time` / `time`) is present we MUST align by timestamp:
            #    consuming the list positionally against the forecast clock
            #    misplaces production (a source whose list starts at local
            #    midnight drops midday values into the night). The companion
            #    time axis (e.g. meteo_7day_forecast_total.iso_time) is naive
            #    LOCAL time, which `_parse_ts` interprets in the configured
            #    zone before flooring to the UTC hour.
            irr_attr = attrs.get("irradiance")
            if isinstance(irr_attr, list) and irr_attr:
                # Keep index alignment with the time axis: coerce bad entries
                # to 0 rather than dropping them.
                numeric: list[float] = []
                for v in irr_attr:
                    try:
                        numeric.append(float(v))
                    except (TypeError, ValueError):
                        numeric.append(0.0)
                if not any(numeric):
                    return None
                # If the largest magnitude exceeds 50, assume Watts → kWh.
                divisor = 1000.0 if max(numeric) > 50.0 else 1.0
                scaled = [_clamp(v / divisor) for v in numeric]
                time_axis = attrs.get("iso_time") or attrs.get("time")
                if isinstance(time_axis, list) and time_axis:
                    dated_irr: dict[datetime, float] = {}
                    for ts_raw, val in zip(time_axis, scaled):
                        ts = _parse_ts(ts_raw)
                        if ts is not None:
                            dated_irr[ts] = val
                    if len(dated_irr) >= max(1, len(scaled) // 2):
                        return dated_irr
                # No usable time axis → positional (caller still gates by
                # irradiance, so a misaligned source can't put PV at night).
                return scaled

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
        start_utc: datetime | None = None,
    ) -> list[float]:
        """Build per-hour PV production forecast (kWh).

        Returns a list of length `n_hours`, where index ``i`` corresponds
        to ``start_utc + i`` hours. External-entity output is aligned by
        timestamp when available (dict form) — index ``i`` is looked up at
        ``start_utc + i`` — otherwise truncated/extended positionally. When
        PV is disabled, returns all zeros.
        """
        if not self._pv_enabled:
            return [0.0] * n_hours

        if start_utc is None:
            start_utc = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0)

        # Try external entity first
        external = self._read_external_pv_forecast()
        if external:
            # Physical sanity gate against the model's OWN irradiance.
            # An external PV forecast can be misaligned in time (wrong tz,
            # positional fallback, or simply spurious night entries), which
            # places production in the middle of the night where the sun is
            # down. The Open-Meteo irradiance is aligned to the same forecast
            # clock (weather[i] == start_utc + i, same index as the lookup
            # below), so it is ground truth for "is the sun up": when
            # irradiance is effectively zero, PV MUST be zero regardless of
            # what the external source claims. Daytime values pass through.
            sun_down_w_m2 = 5.0  # below this the sun is effectively down
            have_irradiance = bool(weather)
            def _sun_is_up(i: int) -> bool:
                # No irradiance data to judge against → don't gate (trust the
                # external source rather than zeroing all PV).
                if not have_irradiance or i >= len(weather):
                    return True
                solar = weather[i].get("solar_weighted", 0.0)
                return float(solar or 0.0) > sun_down_w_m2
            if isinstance(external, dict):
                # Timestamp-aligned: pick the value for each forecast hour.
                out = [
                    (float(external.get(start_utc + timedelta(hours=i), 0.0))
                     if _sun_is_up(i) else 0.0)
                    for i in range(n_hours)
                ]
            else:
                out = list(external[:n_hours])
                while len(out) < n_hours:
                    out.append(0.0)
                out = [v if _sun_is_up(i) else 0.0 for i, v in enumerate(out)]
            return self._apply_pv_nowcast(out, start_utc)

        # Internal estimator from Open-Meteo solar irradiance
        out = []
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
        return self._apply_pv_nowcast(out, start_utc)

    def _read_measured_pv_kw(self) -> float | None:
        """Read the measured-PV sensor as a production rate (kW ≈ kWh/h).

        Accepts watts (the common ``power`` sensor) and converts to kW.
        Returns ``None`` when no sensor is configured, the state is
        missing/unavailable, or the value is non-numeric — the caller
        then skips the nowcast for this cycle. Auto-scales: a value that
        looks like watts (≥ 100) is divided by 1000; a small value is
        assumed already in kW.
        """
        if not self.pv_measured_entity or self.hass is None:
            return None
        state = self.hass.states.get(self.pv_measured_entity)
        if state is None or state.state in (None, "", "unknown", "unavailable"):
            return None
        try:
            val = float(state.state)
        except (TypeError, ValueError):
            return None
        if val < 0.0:
            return None
        unit = ""
        try:
            unit = str(state.attributes.get("unit_of_measurement", "")).lower()
        except Exception:
            unit = ""
        if unit in ("w", "watt", "watts"):
            return val / 1000.0
        if unit in ("kw", "kilowatt", "kilowatts"):
            return val
        # No/unknown unit → magnitude heuristic (residential PV rarely
        # exceeds ~30 kW; a reading ≥ 100 is almost certainly watts).
        return val / 1000.0 if val >= 100.0 else val

    def _apply_pv_nowcast(
        self, pv_forecast: list[float], start_utc: datetime,
    ) -> list[float]:
        """Correct TODAY's remaining forecast with the measured clear-sky
        index. No-op (returns the input) when no measured sensor is
        configured or no valid index can be formed this cycle.

        The correction scales only forecast hours whose LOCAL date equals
        today's — today's sky says nothing about tomorrow. Diagnostics
        (``k``, applied flag, realized fraction, confidence) are stashed
        on ``self._pv_nowcast_diag`` for the daily_forecast builder.
        """
        self._pv_nowcast_diag = {}
        if not self.pv_measured_entity or not pv_forecast:
            return pv_forecast

        # Snapshot the RAW (pre-correction) forecast + its start so the
        # fast nowcast tick can form a correct clear-sky index against
        # the right hour later in the 6-hour window.
        self._pv_forecast_snapshot = (start_utc, list(pv_forecast))

        measured_kw = self._read_measured_pv_kw()
        forecast_now = pv_forecast[0] if pv_forecast else 0.0
        k_raw = pv_nowcast.clear_sky_index(measured_kw, forecast_now)
        self._pv_nowcast_k = pv_nowcast.smooth_index(self._pv_nowcast_k, k_raw)

        now_utc = datetime.now(timezone.utc)
        today_local = self._local_date_str(now_utc)
        # Offsets from now (hours) and today-mask per forecast index.
        hours_from_now: list[float] = []
        today_mask: list[bool] = []
        for i in range(len(pv_forecast)):
            hour_utc = start_utc + timedelta(hours=i)
            hours_from_now.append(
                (hour_utc - now_utc).total_seconds() / 3600.0
            )
            today_mask.append(self._local_date_str(hour_utc) == today_local)

        k = self._pv_nowcast_k
        applied = k is not None and abs(k - 1.0) >= 1e-9
        if applied:
            corrected: list[float] = []
            for pv, h, is_today in zip(pv_forecast, hours_from_now, today_mask):
                # Only lift/suppress today's still-future hours.
                if is_today and h > 0.0:
                    corrected.append(max(0.0, float(pv) * float(k)))
                else:
                    corrected.append(float(pv))
            out = corrected
        else:
            out = pv_forecast
        # Record the index baked into this published forecast so the
        # fast tick only triggers a refresh once the live index has
        # drifted materially away from what is currently published.
        self._pv_nowcast_k_applied = k

        # Realized PV-energy fraction over today's hours only.
        today_pv = [pv for pv, t in zip(pv_forecast, today_mask) if t]
        today_hours = [h for h, t in zip(hours_from_now, today_mask) if t]
        realized = pv_nowcast.realized_pv_fraction(today_pv, today_hours)
        self._pv_nowcast_diag = {
            "pv_nowcast_k": round(float(k), 4) if k is not None else None,
            "pv_nowcast_applied": bool(applied),
            "pv_realized_fraction": round(realized, 4),
            "pv_nowcast_confidence": pv_nowcast.nowcast_confidence(
                realized, measurement_live=measured_kw is not None,
            ),
        }
        return out

    @callback
    def _async_pv_nowcast_tick(self, now: datetime) -> None:
        """Fast PV-nowcast poll (every ``PV_NOWCAST_POLL_SECONDS``).

        Updates the smoothed clear-sky index from the measured sensor and
        requests a full refresh when it has drifted materially from the
        published forecast — the intraday reactivity the 6-hour weather
        cycle cannot provide. Cheap: one sensor read + arithmetic; the
        expensive recompute only happens on the gated refresh.
        """
        if not self.pv_measured_entity or self._pv_forecast_snapshot is None:
            return
        start_utc, raw_fc = self._pv_forecast_snapshot
        now_utc = datetime.now(timezone.utc)
        idx = int((now_utc - start_utc).total_seconds() // 3600)
        if idx < 0 or idx >= len(raw_fc):
            return
        measured_kw = self._read_measured_pv_kw()
        k_raw = pv_nowcast.clear_sky_index(measured_kw, raw_fc[idx])
        self._pv_nowcast_k = pv_nowcast.smooth_index(self._pv_nowcast_k, k_raw)

        gap = (
            None if self._pv_nowcast_last_refresh_utc is None
            else (now_utc - self._pv_nowcast_last_refresh_utc).total_seconds()
        )
        if pv_nowcast.should_trigger_refresh(
            self._pv_nowcast_k,
            self._pv_nowcast_k_applied,
            gap,
            k_delta=PV_NOWCAST_REFRESH_K_DELTA,
            min_gap_seconds=PV_NOWCAST_REFRESH_MIN_GAP_SECONDS,
        ):
            self._pv_nowcast_last_refresh_utc = now_utc
            _LOGGER.info(
                "PV nowcast index drifted (k=%.2f vs applied %s); "
                "requesting refresh",
                self._pv_nowcast_k, self._pv_nowcast_k_applied,
            )
            self.hass.async_create_task(self.async_request_refresh())

    def _pv_dk_by_local_date(
        self, forecast: list[dict[str, Any]],
    ) -> dict[str, dict[str, list[float]]]:
        """Reconstruct PV-aware D(k) per local date from rolling history.

        The fresh forecast window starts at ``now``, so the current local
        day is only partially present and is dropped by the 24-hour gate
        in ``_compute_duration_forecast`` — leaving PV-aware D(k) starting
        a day later than the grid D(k) (which back-fills today from actual
        spot). Here we union the rolling ``_forecast_history`` (which holds
        today's already-elapsed hours, each carrying ``effective_eur_kwh``
        and PV) with the current ``forecast`` so any local date with a
        complete 24-hour ``effective_eur_kwh`` series gets a PV-aware
        D(k). The caller injects these onto whichever entry (forecast or
        actual) represents that date.

        Returns ``{date_str: {"dk_cheap_pv_eur_kwh": [...24...],
        "dk_peak_pv_eur_kwh": [...24...]}}``. Empty when PV is disabled.
        """
        if not self._pv_enabled:
            return {}

        # Union history + current forecast, deduped by timestamp (current
        # forecast wins — it carries the freshest pipeline-corrected price).
        rows: dict[str, dict[str, Any]] = {}
        for r in self._forecast_history.values():
            ts = r.get("timestamp")
            if ts:
                rows[ts] = r
        for r in forecast:
            ts = r.get("timestamp")
            if ts:
                rows[ts] = r

        by_date: dict[str, list[float]] = {}
        for r in rows.values():
            m = r.get("effective_eur_kwh")
            if m is None:
                continue
            try:
                ts_utc = _iso_to_naive_ts(r["timestamp"]).replace(
                    tzinfo=timezone.utc)
                date_str = self._local_date_str(ts_utc)
            except Exception:
                continue
            by_date.setdefault(date_str, []).append(float(m))

        out: dict[str, dict[str, list[float]]] = {}
        for date_str, effs in by_date.items():
            if len(effs) != 24:
                continue
            asc = sorted(effs)
            desc = sorted(effs, reverse=True)
            cheap: list[float] = []
            peak: list[float] = []
            s_c = 0.0
            s_p = 0.0
            for i in range(24):
                s_c += asc[i]
                s_p += desc[i]
                cheap.append(round(s_c / (i + 1), 4))
                peak.append(round(s_p / (i + 1), 4))
            out[date_str] = {
                "dk_cheap_pv_eur_kwh": cheap,
                "dk_peak_pv_eur_kwh": peak,
            }
        return out

    # ── DtACI per-D(i) calibration layer ──────────────────────────

    def _dtaci_init_bundles(self) -> None:
        """Lazy-init the DtACI bundle(s) — FI only (see DTACI_ZONES).

        Each bundle is loaded from `<config_dir>/.storage/<DOMAIN>_dtaci/
        dtaci_dk_<zone>.json` if present, otherwise cold-started. Idempotent —
        safe to call every cycle. Stale state files for zones no longer in
        DTACI_ZONES (e.g. the removed SE1/SE3/EE neighbour bundles) are
        cleaned up once on init.
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
            # Remove stale state files for de-scoped zones (SE1/SE3/EE).
            for stale in base.glob("dtaci_dk_*.json"):
                zone = stale.stem.replace("dtaci_dk_", "")
                if zone not in DTACI_ZONES:
                    try:
                        stale.unlink()
                        _LOGGER.info(
                            "DtACI: removed stale neighbour-zone state %s",
                            stale.name)
                    except OSError:
                        pass
            _LOGGER.info(
                "DtACI: initialised %d zone bundle(s) in %s",
                len(self._dtaci_bundles), base,
            )
        except Exception as err:
            _LOGGER.exception("DtACI: bundle init failed: %s", err)

    @staticmethod
    def _bias_hour_key(ts: datetime) -> str:
        """Canonical UTC hour key for the bias reconciliation ledger."""
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")

    def _bias_record_forecasts(
        self,
        timestamps,          # np.ndarray datetime64 (naive UTC)
        raw_mean,            # np.ndarray pre-correction pipeline means
        known_until: datetime | None,
    ) -> None:
        """Record raw hourly pipeline forecasts for hours whose spot price
        was NOT yet published at forecast time, so a later cycle can pair
        them with the realised price and feed the per-hour bias corrector.

        Re-recording an hour on a subsequent cycle overwrites it — the
        freshest pre-auction forecast is the one the correction applies
        to in production. Pruned to the last 14 days.
        """
        known_key = (self._bias_hour_key(known_until)
                     if known_until is not None else "")
        secs = timestamps.astype("datetime64[s]")
        for t, p in zip(secs, raw_mean):
            key = str(t)[:13] + ":00"          # YYYY-MM-DDTHH:00 (UTC)
            if key > known_key:
                self._bias_forecast_history[key] = float(p)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)
                  ).strftime("%Y-%m-%dT%H:00")
        for old in [k for k in self._bias_forecast_history if k < cutoff]:
            del self._bias_forecast_history[old]

    def _bias_reconcile_actuals(
        self, spot_prices: list[dict[str, Any]],
    ) -> int:
        """Pair realised hourly spot prices with previously recorded raw
        pipeline forecasts and feed them to the pipeline calibrators
        (per-hour bias EMA + DtACI fan-chart). Each hour is fed once.
        Returns the number of pairs ingested.
        """
        if (self._pipeline is None or not spot_prices
                or not self._bias_forecast_history):
            return 0
        import numpy as np
        preds: list[float] = []
        acts: list[float] = []
        keys: list[str] = []
        for entry in spot_prices:
            ts_str = entry.get("timestamp") or ""
            price = entry.get("price_eur_mwh")
            if not ts_str or price is None:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            key = self._bias_hour_key(ts)
            pred = self._bias_forecast_history.get(key)
            if pred is None:
                continue
            preds.append(float(pred))
            acts.append(float(price))
            keys.append(key)
        if not preds:
            return 0
        try:
            self._pipeline.update_with_actuals(
                np.asarray(preds), np.asarray(acts),
                timestamps=np.array([np.datetime64(k + ":00") for k in keys]),
            )
        except Exception as exc:
            _LOGGER.warning("bias corrector: reconcile failed: %s", exc)
            return 0
        for k in keys:
            self._bias_forecast_history.pop(k, None)
        _LOGGER.info("bias corrector: reconciled %d realised hour(s)",
                     len(keys))
        return len(keys)

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

    # ── Learned lead-time price-uncertainty layer ─────────────────

    def _price_verifier_init(self) -> None:
        """Lazy-load the price-forecast verifier. Idempotent; best-effort.

        On any failure the verifier stays None and the CVaR falls back
        to the static `price_rel_std_for_lead` heuristic (v2.13.0
        behaviour), so this never blocks a forecast build.
        """
        if self._price_verifier is not None:
            return
        try:
            base = (Path(self.hass.config.path()) / ".storage"
                    / f"{DOMAIN}_price_verifier")
            base.mkdir(parents=True, exist_ok=True)
            path = base / "price_rel_std.json"
            self._price_verifier_path = path
            self._price_verifier = _pfv.load_or_create(str(path))
            _LOGGER.info("Price verifier: loaded from %s", path)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Price verifier: init failed (%s); using "
                            "static lead-time prior", err)
            self._price_verifier = None

    def _price_verifier_reconcile(
        self, duration_forecast: list[dict[str, Any]],
    ) -> int:
        """Feed newly-cleared days' realized buy curves to the learner.

        For each `source == "actual"` entry whose realized 24h consumer
        price we retained, reconcile against any stored forecasts.
        Returns the number of (lead) samples ingested this cycle.
        """
        if self._price_verifier is None:
            return 0
        n = 0
        for d in duration_forecast:
            if d.get("source") != "actual":
                continue
            date_str = d.get("date")
            realized = self._actual_consumer_prices.get(date_str or "")
            if not realized or len(realized) < 24:
                continue
            n += self._price_verifier.reconcile(date_str, realized)
        if n:
            _LOGGER.info("Price verifier: reconciled %d new lead sample(s)", n)
        # Bound the retained realized-price buffer to the reconciliation
        # window (actuals only ever span the last couple of days).
        if len(self._actual_consumer_prices) > 8:
            for old in sorted(self._actual_consumer_prices)[:-8]:
                del self._actual_consumer_prices[old]
        return n

    def _price_verifier_save(self) -> None:
        """Persist verifier state atomically. Best-effort."""
        if self._price_verifier is None or self._price_verifier_path is None:
            return
        try:
            _pfv.save(str(self._price_verifier_path), self._price_verifier)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Price verifier: save failed: %s", err)

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
            # Open-Meteo returns the hourly grid from 00:00 UTC today, not
            # from `now`. Re-base it so index 0 == `now` before any
            # positional consumer (features, PV, duration) reads it.
            weather = self._align_weather_to_now(weather, now)
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
            pv_kwh = self._compute_pv_forecast(weather, len(predictions), now)

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
                # Feed realised prices to the per-hour bias corrector and
                # fan-chart calibrator BEFORE forecasting, so this cycle's
                # forecast benefits from the freshest correction state.
                try:
                    self._bias_reconcile_actuals(spot_prices)
                except Exception as e:
                    _LOGGER.debug("bias reconcile skipped: %s", e)
                try:
                    pipeline_diagnostics, dk_by_date = self._apply_pipeline_pre_dk(
                        forecast, neighbor=neighbor, spot_prices=spot_prices)
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

            # PV-aware D(k) horizon fix: the fresh forecast starts at `now`
            # so the current day is partial and was dropped by the 24-hour
            # gate, leaving PV D(k) one day behind the grid D(k) (which
            # back-fills today from actuals). Reconstruct today's full-day
            # PV-aware D(k) from the rolling forecast history and inject it
            # onto any entry (incl. the actual `today`) that lacks it.
            if self._pv_enabled:
                pv_dk = self._pv_dk_by_local_date(forecast)
                for d in duration_forecast:
                    extra = pv_dk.get(d.get("date"))
                    if extra and "dk_cheap_pv_eur_kwh" not in d:
                        d["dk_cheap_pv_eur_kwh"] = extra["dk_cheap_pv_eur_kwh"]
                        d["dk_peak_pv_eur_kwh"] = extra["dk_peak_pv_eur_kwh"]

            # ── DtACI per-D(i) calibration layer ────────────────────
            # When enabled, run the FI bundle: capture today's forecast,
            # reconcile newly-actual days, attach calibrated bands to
            # forecast-mode entries, persist state.
            dtaci_diagnostics: dict[str, Any] = {}
            if self.enable_dtaci_dk:
                self._dtaci_init_bundles()
                self._dtaci_record_forecasts(duration_forecast)
                self._dtaci_reconcile_actuals(duration_forecast)
                self._dtaci_attach_bands(duration_forecast)
                self._dtaci_save()
                dtaci_diagnostics = self._dtaci_diagnostics()

            # ── Learned lead-time price-uncertainty layer ───────────
            # Reconcile any newly-cleared day against the forecasts the
            # verifier stored for it (recorded during the CVaR build in
            # `_compute_duration_forecast`), then persist. The learned
            # profile is consumed on the *next* forecast build.
            if self._price_verifier is not None:
                self._price_verifier_reconcile(duration_forecast)
                self._price_verifier_save()

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
                # Model's current capacity-weighted effective wind (120 m hub
                # height, FI wind regions) — surfaced as its own sensor so
                # downstream consumers don't re-fetch Open-Meteo.
                "current_wind": forecast[0].get("wind") if forecast else None,
                "forecast": combined_forecast,
                "duration_forecast": duration_forecast,
                "dtaci_diagnostics": dtaci_diagnostics,
                "price_verifier_diagnostics": (
                    self._price_verifier.diagnostics()
                    if self._price_verifier is not None else {}
                ),
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
        neighbor: dict[str, list[dict[str, Any]]] | None = None,
        spot_prices: list[dict[str, Any]] | None = None,
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

        # Build neighbour-price arrays aligned with the forecast timestamps.
        # `neighbor` is the dict returned by api.fetch_neighbor_prices —
        # zones are SE1, SE3, EE; each value is a list of
        # {timestamp, price_eur_mwh} entries. Missing or short series
        # become zero columns inside Pipeline._build_features (graceful
        # fallback to the v2.8.x no-cross-border behaviour).
        recent_neighbour_prices: dict[str, np.ndarray] | None = None
        if neighbor:
            try:
                forecast_ts_iso = [f["timestamp"] for f in forecast]
                recent_neighbour_prices = self._align_neighbour_prices(
                    forecast_ts_iso, neighbor)
            except Exception as e:
                _LOGGER.debug(
                    "neighbour-price alignment failed (%s); zero fallback", e,
                )

        out = self._pipeline.compute_forecast(
            timestamps=timestamps,
            wind=wind, solar=solar, temp=temp,
            recent_fi_residuals={"lag168": lag168},
            recent_neighbour_prices=recent_neighbour_prices,
            enable_fan_chart=True,
        )
        pipeline_mean = out["mean_eur_mwh"]

        # Record the RAW (pre-correction) forecasts for hours beyond the
        # known-price horizon; a later cycle reconciles them against the
        # published price and feeds the per-hour bias corrector.
        try:
            known_until: datetime | None = None
            for entry in spot_prices or []:
                ts_str = entry.get("timestamp") or ""
                try:
                    kts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if kts.tzinfo is None:
                    kts = kts.replace(tzinfo=timezone.utc)
                if known_until is None or kts > known_until:
                    known_until = kts
            self._bias_record_forecasts(
                timestamps, out["mean_uncorrected_eur_mwh"], known_until)
        except Exception as e:
            _LOGGER.debug("bias forecast recording skipped: %s", e)

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
                # Recompute PV-aware fields against the pipeline-corrected
                # price. Pass 1 (pre-pipeline) computed these from the raw
                # model spot; now that `spot`/`consumer` are overwritten the
                # stale values would be internally inconsistent — e.g. at
                # night (pv=0) `effective_eur_kwh` must equal
                # `consumer_eur_kwh`, which only holds after this recompute.
                if self._pv_enabled and "effective_eur_kwh" in f:
                    p_h = float(f.get("pv_production_kwh", 0.0))
                    c_h = float(f.get("baseload_kwh", 0.0))
                    s_h = self._spot_to_sell_eur_kwh(spot)
                    f["sell_eur_kwh"] = round(s_h, 4)
                    f["effective_eur_kwh"] = round(
                        marginal_effective_eur_kwh(
                            buy_eur_kwh=consumer,
                            sell_eur_kwh=s_h,
                            pv_kwh=p_h,
                            baseload_kwh=c_h,
                        ),
                        4,
                    )
                    f["net_household_cost_eur"] = round(
                        net_household_cost_eur(
                            buy_eur_kwh=consumer,
                            sell_eur_kwh=s_h,
                            pv_kwh=p_h,
                            consumption_kwh=c_h,
                        ),
                        4,
                    )
                    f["is_export_hour"] = bool(p_h > c_h)
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

        # Lazy-load the learned lead-time price-uncertainty profile.
        self._price_verifier_init()

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

                # ── PV-aware CVaR (Phase D step 3b/c) ──────────────────
                # Computed from the day's 24 hourly buy / sell / PV /
                # consumption values via a parametric scenario sampler
                # (pv_aware_cvar.compute_pv_aware_cvar_for_day). Adds
                # mean / CVaR_95 / fan-chart quantiles + PV self-
                # consumption bookkeeping. Strictly additive — does
                # not modify existing day_entry fields.
                #
                # Consumption source — in priority order:
                #   1. CONF_CONSUMPTION_PROFILE_ENTITY  (external EMA module
                #      publishes shape + monthly factor → derive 24h)
                #   2. forecast row's baseload_kwh field (existing
                #      coordinator path: annual_kwh + optional smoothing)
                try:
                    day_buys: list[float] = []
                    day_sells: list[float] = []
                    day_pvs: list[float] = []
                    day_cons_fallback: list[float] = []
                    day_local_ts: list[datetime] = []
                    for h in day_hours:
                        idx = h["forecast_idx"]
                        if idx >= len(forecast):
                            day_buys = []
                            break
                        f_row = forecast[idx]
                        day_buys.append(float(f_row.get("consumer_eur_kwh", 0.0)))
                        day_sells.append(float(f_row.get("sell_eur_kwh", 0.0)))
                        day_pvs.append(float(f_row.get("pv_production_kwh", 0.0)))
                        day_cons_fallback.append(
                            float(f_row.get("baseload_kwh", 0.0)))
                        day_local_ts.append(
                            datetime.strptime(date_str, "%Y-%m-%d").replace(
                                hour=h["local_hour"]
                            )
                        )

                    # Try external EMA profile entity if configured.
                    profile_attrs: dict | None = None
                    profile_used = "coordinator_baseload"
                    if (self.consumption_profile_entity
                            and self.hass is not None):
                        state = self.hass.states.get(
                            self.consumption_profile_entity)
                        if state is not None and state.attributes:
                            profile_attrs = dict(state.attributes)

                    if profile_attrs is not None:
                        profile = load_profile_from_entity_attrs(
                            profile_attrs,
                            fallback_annual_kwh=self.annual_consumption_kwh,
                        )
                        day_cons = profile.consumption_for_timestamps(
                            day_local_ts).tolist()
                        profile_used = profile.data_provenance
                    else:
                        day_cons = day_cons_fallback

                    if len(day_buys) == 24 and len(day_cons) == 24:
                        import numpy as _np
                        # v2.13.0 — lead-time price uncertainty. Days 0-1
                        # are cleared day-ahead prices (rel_std 0); days
                        # 2+ are the ML forecast (smooth mean) and get a
                        # growing price perturbation so their CVaR tail
                        # does not collapse at the cleared→forecast
                        # boundary.
                        try:
                            from datetime import datetime as _dt2
                            today_local = self._local_date_str(
                                datetime.now(timezone.utc))
                            days_ahead = (
                                _dt2.strptime(date_str, "%Y-%m-%d").date()
                                - _dt2.strptime(today_local, "%Y-%m-%d").date()
                            ).days
                        except Exception:
                            days_ahead = 0
                        # v2.14.0 — record this forecast so it can be
                        # scored once the day clears, and use the learned
                        # per-lead rel_std (falls back to the static
                        # heuristic during warm-up / if unavailable).
                        if self._price_verifier is not None:
                            self._price_verifier.record_forecast(
                                date_str, days_ahead, day_buys)
                            _price_rel_std = (
                                self._price_verifier.rel_std_for_lead(
                                    days_ahead))
                        else:
                            _price_rel_std = price_rel_std_for_lead(days_ahead)
                        cvar = compute_pv_aware_cvar_for_day(
                            _np.array(day_buys),
                            _np.array(day_sells),
                            _np.array(day_pvs),
                            _np.array(day_cons),
                            price_rel_std=_price_rel_std,
                        )
                        # Tail-risk number + PV bookkeeping + provenance.
                        day_entry["pv_aware_cvar95_eur_kwh"] = round(
                            cvar["cvar95_eur_kwh"], 4)
                        day_entry["pv_aware_self_consumed_kwh"] = round(
                            cvar["pv_self_consumed_kwh"], 2)
                        day_entry["pv_aware_exported_kwh"] = round(
                            cvar["pv_exported_kwh"], 2)
                        day_entry["pv_aware_data_provenance"] = profile_used
                        # v2.11.9: also publish the kernel's expected value
                        # and fan-chart quantiles (already computed) so
                        # dashboards can show expected-vs-worst-case risk.
                        day_entry["pv_aware_mean_eur_kwh"] = round(
                            cvar["mean_eur_kwh"], 4)
                        day_entry["pv_aware_p5_eur_kwh"] = round(
                            cvar["p5_eur_kwh"], 4)
                        day_entry["pv_aware_p95_eur_kwh"] = round(
                            cvar["p95_eur_kwh"], 4)
                        # v2.14.0 — publish the (learned, lead-time)
                        # price-forecast uncertainty applied to this
                        # day's CVaR so downstream planners (ENP) can
                        # size their own scenario dispersion from it.
                        day_entry["price_rel_std"] = round(
                            float(cvar.get("price_rel_std", 0.0)), 4)
                        # Deterministic NO-PV baseline (consumption-weighted
                        # consumer price; no self-consumption, no export) so
                        # dashboards can show the with-vs-without-PV saving.
                        # With pv=0 the CVaR kernel is degenerate (no spread),
                        # so this equals the grid mean — computed directly.
                        _cons_sum = float(sum(day_cons))
                        if _cons_sum > 0:
                            _grid_cost = float(sum(
                                c * b for c, b in zip(day_cons, day_buys)
                            )) / _cons_sum
                            day_entry["grid_cost_eur_kwh"] = round(_grid_cost, 4)
                except Exception as exc:  # noqa: BLE001
                    _LOGGER.debug(
                        "PV-aware CVaR skipped for %s: %s", date_str, exc,
                    )

            # v2.12.0 — attach intraday PV-nowcast diagnostics to TODAY's
            # entry only. Downstream planners read these to (a) know the
            # day-0 effective-price curve has been corrected with measured
            # PV, and (b) size the residual uncertainty band via the
            # realized-fraction / confidence signal. Present only when a
            # measured-PV sensor is configured (diag dict non-empty).
            if self._pv_nowcast_diag:
                try:
                    today_local = self._local_date_str(
                        datetime.now(timezone.utc))
                    if date_str == today_local:
                        day_entry.update(self._pv_nowcast_diag)
                except Exception:  # noqa: BLE001
                    pass

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
            # Retain the realized (cleared) consumer buy curve so the
            # price-forecast verifier can reconcile it against the
            # forecasts it stored for this date at each lead time. Same
            # `consumer_eur_kwh` basis as the forecast `day_buys`.
            self._actual_consumer_prices[date_str] = list(consumer_prices)

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
