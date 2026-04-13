"""DataUpdateCoordinator for Spot Price Predictor."""

from __future__ import annotations

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
    CONF_ENABLE_NEIGHBOR_PRICES,
    CONF_OPERATOR,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    DEFAULT_SELLER_MARGIN,
    OPERATORS,
    DEFAULT_VAT_MULTIPLIER,
    DEFAULT_ENERGY_TAX,
    DEMAND_DEFAULTS,
    UPDATE_INTERVAL_WEATHER,
    FORECAST_HOURS,
    DEFAULT_TIMEZONE,
)

from .features import build_forecast_features
from .holidays import build_holiday_set
from .model import SpotPriceModel

_LOGGER = logging.getLogger(__name__)

RETRY_INTERVAL_SECONDS = 900  # 15 minutes after failure


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

    def _spot_to_consumer_eur_kwh(self, spot_eur_mwh: float, is_night: bool) -> float:
        """Convert spot EUR/MWh to consumer EUR/kWh using configured tariffs."""
        transfer = self.night_rate if is_night else self.day_rate
        spot_kwh = max(0.0, spot_eur_mwh) / 1000.0
        return (spot_kwh + self.seller_margin + transfer + self.energy_tax) \
            * self.vat_multiplier

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
            if self.enable_neighbor_prices:
                try:
                    neighbor = await self.api.fetch_neighbor_prices()
                    neighbor_spreads = self.api.compute_rolling_spreads(spot_prices, neighbor)
                except Exception as err:
                    _LOGGER.warning("Cross-border data fetch failed: %s", err)

            # Fingrid nuclear data
            nuclear_data: dict[str, float] | None = None
            if self.has_fingrid:
                try:
                    nuclear_data = await self.api.fetch_fingrid_data()
                except Exception as err:
                    _LOGGER.warning("Fingrid nuclear data fetch failed: %s", err)

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
            )

            # Run model inference
            predictions = self.model.predict_batch(feature_rows)

            # Build unified forecast: spot + consumer + weather per hour
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
                forecast.append(entry)

            # D(k) duration curve forecast (7-day daily curves)
            duration_forecast = self._compute_duration_forecast(
                forecast, weather, ar_neighbor_hourly, nuclear_data, now,
            )

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
                "data_sources_active": " + ".join(sources),
                "last_update": now.isoformat(),
                "stale": False,
                "data_age_minutes": 0,
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

    def _compute_duration_forecast(
        self,
        forecast: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        ar_neighbor_hourly: dict[str, list[float]] | None,
        nuclear_data: dict[str, float] | None,
        now: datetime,
    ) -> list[dict[str, Any]]:
        """Compute 7-day D(k) duration curve forecast.

        Returns list of daily entries (only complete 24h days):
        [{"date": "2026-04-13", "weekday": "Mon",
          "dk_consumer_eur_kwh": [24 floats], "dk_spot_eur_mwh": [24 floats]}, ...]
        dk_consumer_eur_kwh[k-1] = D(k) consumer price in EUR/kWh, k=1..24
        dk_spot_eur_mwh[k-1] = D(k) spot price in EUR/MWh, k=1..24
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

            by_date[date_str].append({
                "local_hour": local.hour,
                "dow": local.weekday(),
                "forecast_idx": i,
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
            for seg_name, curve in segment_curves.items():
                is_night = seg_name in night_segments
                for i in range(len(curve)):
                    if i == 0:
                        p = curve[0]
                    else:
                        p = (i + 1) * curve[i] - i * curve[i - 1]
                        p = max(0.0, p)
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

            day_entry: dict[str, Any] = {
                "date": date_str,
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
                "dk_consumer_eur_kwh": dk_consumer,
                "dk_spot_eur_mwh": [round(v, 2) for v in dk_spot],
            }
            result.append(day_entry)

        _LOGGER.info("Duration forecast computed: %d days", len(result))
        return result
