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
    CONF_ENABLE_TIER2,
    CONF_OPERATOR,
    CONF_CUSTOM_DAY_RATE,
    CONF_CUSTOM_NIGHT_RATE,
    CONF_CUSTOM_VAT,
    CONF_CUSTOM_ENERGY_TAX,
    CONF_SELLER_MARGIN,
    DEFAULT_SELLER_MARGIN,
    CONF_SEARCH_START_HOURS,
    CONF_SEARCH_DURATION_HOURS,
    DEFAULT_SEARCH_START_HOURS,
    DEFAULT_SEARCH_DURATION_HOURS,
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
    """Coordinator that fetches data and runs model inference."""

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

        # Operator tariff config
        operator_id = entry.data.get(CONF_OPERATOR, "elenia")
        if operator_id == "custom":
            self.day_rate = entry.data.get(CONF_CUSTOM_DAY_RATE, 0.05)
            self.night_rate = entry.data.get(CONF_CUSTOM_NIGHT_RATE, 0.04)
            self.vat_multiplier = entry.data.get(CONF_CUSTOM_VAT, DEFAULT_VAT_MULTIPLIER)
            self.energy_tax = entry.data.get(CONF_CUSTOM_ENERGY_TAX, DEFAULT_ENERGY_TAX)
        else:
            op = OPERATORS.get(operator_id, OPERATORS["elenia"])
            self.day_rate = op["day_rate"]
            self.night_rate = op["night_rate"]
            self.vat_multiplier = DEFAULT_VAT_MULTIPLIER
            self.energy_tax = DEFAULT_ENERGY_TAX

        self.seller_margin = entry.data.get(CONF_SELLER_MARGIN, DEFAULT_SELLER_MARGIN)

        self.enable_tier2 = entry.data.get(CONF_ENABLE_TIER2, False)
        self.has_fingrid = bool(fingrid_key)
        self.search_start_hours = entry.data.get(
            CONF_SEARCH_START_HOURS, DEFAULT_SEARCH_START_HOURS
        )
        self.search_duration_hours = entry.data.get(
            CONF_SEARCH_DURATION_HOURS, DEFAULT_SEARCH_DURATION_HOURS
        )

        # Build holiday set
        now = datetime.now(timezone.utc)
        self.holidays = build_holiday_set(now.year - 1, now.year + 2)

        # Cache last successful result so sensors stay available during API failures
        self._last_successful_data: dict[str, Any] | None = None
        self._last_successful_time: datetime | None = None

        # Rolling forecast history: keeps past predictions so charts
        # can show data from the beginning of the day, not just from
        # the last refresh time. Key = ISO timestamp, value = forecast entry.
        self._forecast_history: dict[str, dict] = {}
        self._consumer_history: dict[str, dict] = {}

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

            # Tier 2: cross-border prices
            tier2_spreads: dict[str, float] | None = None
            if self.enable_tier2:
                try:
                    neighbor = await self.api.fetch_neighbor_prices()
                    tier2_spreads = self.api.compute_rolling_spreads(spot_prices, neighbor)
                except Exception as err:
                    _LOGGER.warning("Tier 2 data fetch failed: %s", err)

            # Tier 3: Fingrid data
            tier3_data: dict[str, float] | None = None
            if self.has_fingrid:
                try:
                    tier3_data = await self.api.fetch_fingrid_data()
                except Exception as err:
                    _LOGGER.warning("Tier 3 data fetch failed: %s", err)

            # Nuclear outage schedule (Nord Pool UMM, public, no key required)
            tier3_hourly: dict[str, list[float]] | None = None
            if tier3_data and "nuclear_mw" in tier3_data:
                try:
                    outage_schedule = await self.api.fetch_nuclear_outage_schedule()
                    if outage_schedule:
                        now_utc = datetime.now(timezone.utc).replace(
                            minute=0, second=0, microsecond=0)
                        nuclear_hourly = self.api.compute_hourly_nuclear_mw(
                            current_nuclear_mw=tier3_data["nuclear_mw"],
                            outage_schedule=outage_schedule,
                            start_utc=now_utc,
                            hours=min(FORECAST_HOURS, len(weather)),
                        )
                        tier3_hourly = {"nuclear_mw": nuclear_hourly}
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
            if ar_models and self.enable_tier2:
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
                        try:
                            from zoneinfo import ZoneInfo
                            tz = ZoneInfo(DEFAULT_TIMEZONE)
                        except Exception:
                            tz = None
                        forecast_hours = []
                        for i in range(n_hours):
                            h_utc = now_utc + timedelta(hours=i)
                            if tz:
                                h_local = h_utc.astimezone(tz)
                            else:
                                h_local = h_utc + timedelta(hours=3)
                            local_h = h_local.hour
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
                tier2_spreads=tier2_spreads,
                tier3_data=tier3_data,
                tier3_hourly=tier3_hourly,
                ar_neighbor_hourly=ar_neighbor_hourly,
            )

            # Run model inference
            predictions = self.model.predict_batch(feature_rows)

            # Build forecast list with timestamps and weather context
            forecast = []
            for i, pred in enumerate(predictions):
                ts = now + timedelta(hours=i)
                entry = {
                    "timestamp": ts.isoformat(),
                    "price_eur_mwh": round(pred, 2),
                }
                # Include weather data for dashboard charts
                if i < len(weather):
                    entry["wind_weighted"] = round(weather[i].get("wind_weighted", 0), 1)
                    entry["solar_weighted"] = round(weather[i].get("solar_weighted", 0), 0)
                    entry["temp_weighted"] = round(weather[i].get("temp_weighted", 0), 1)
                forecast.append(entry)

            # Compute consumer prices (EUR/kWh) with tariff
            consumer_forecast = []
            for entry_item in forecast:
                ts = datetime.fromisoformat(entry_item["timestamp"])
                try:
                    from zoneinfo import ZoneInfo
                    local_hour = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(DEFAULT_TIMEZONE)).hour
                except Exception:
                    local_hour = (ts + timedelta(hours=3)).hour
                is_night = local_hour < 7 or local_hour >= 22
                transfer = self.night_rate if is_night else self.day_rate
                spot_kwh = entry_item["price_eur_mwh"] / 1000.0
                consumer_price = (spot_kwh + self.seller_margin + transfer + self.energy_tax) * self.vat_multiplier
                consumer_forecast.append({
                    "timestamp": entry_item["timestamp"],
                    "price_eur_kwh": round(consumer_price, 5),
                })

            # Cheapest hours calculation (user-configurable search window)
            cheapest_hours = self._find_cheapest_hours(
                forecast, now,
                start_offset_hours=self.search_start_hours,
                duration_hours=self.search_duration_hours,
            )
            # Add consumer price equivalents (c/kWh) using configured tariffs
            self._enrich_cheapest_with_consumer(cheapest_hours, consumer_forecast)

            # D(k) duration curve forecast (7-day daily curves)
            duration_forecast = self._compute_duration_forecast(
                forecast, weather, ar_neighbor_hourly, tier3_data, now,
            )

            # Merge into rolling history (keeps past predictions for charts)
            # New predictions overwrite older ones for the same timestamp
            for f in forecast:
                self._forecast_history[f["timestamp"]] = f
            for c in consumer_forecast:
                self._consumer_history[c["timestamp"]] = c

            # Prune history older than 7 days
            cutoff = (now - timedelta(days=7)).isoformat()
            self._forecast_history = {
                k: v for k, v in self._forecast_history.items() if k >= cutoff
            }
            self._consumer_history = {
                k: v for k, v in self._consumer_history.items() if k >= cutoff
            }

            # Build combined forecast from history (sorted by timestamp)
            combined_forecast = sorted(
                self._forecast_history.values(), key=lambda x: x["timestamp"]
            )
            combined_consumer = sorted(
                self._consumer_history.values(), key=lambda x: x["timestamp"]
            )

            # Current hour values
            current_spot = forecast[0]["price_eur_mwh"] if forecast else 0.0
            current_consumer = consumer_forecast[0]["price_eur_kwh"] if consumer_forecast else 0.0

            # Tiers active description
            tiers = ["Tier 1 (weather)"]
            if tier2_spreads:
                tiers.append("Tier 2 (cross-border)")
            if tier3_data:
                tiers.append("Tier 3 (Fingrid)")

            result = {
                "spot_price": current_spot,
                "spot_forecast": combined_forecast,
                "consumer_price": current_consumer,
                "consumer_forecast": combined_consumer,
                "cheapest_hours": cheapest_hours,
                "duration_forecast": duration_forecast,
                "tiers_active": " + ".join(tiers),
                "last_update": now.isoformat(),
                "stale": False,
                "data_age_minutes": 0,
            }

            # Cache successful result and restore normal interval
            self._last_successful_data = result
            self._last_successful_time = now
            self.update_interval = timedelta(seconds=UPDATE_INTERVAL_WEATHER)
            _LOGGER.info(
                "Update completed: %d forecast hours, tiers: %s",
                len(forecast), " + ".join(tiers),
            )
            return result

        except ApiClientError as err:
            return self._return_cached_or_fail(err)
        except Exception as err:
            _LOGGER.exception("Unexpected error during update")
            return self._return_cached_or_fail(err)

    @staticmethod
    def _format_offset(hours: int) -> str:
        """Format hours as 'Nd Nh' string."""
        d, h = divmod(hours, 24)
        return f"{d}d {h}h"

    @staticmethod
    def _find_cheapest_hours(
        forecast: list[dict[str, Any]],
        now: datetime,
        start_offset_hours: int = 24,
        duration_hours: int = 48,
    ) -> dict[str, Any]:
        """Find cheapest consecutive hour blocks in a configurable window.

        Args:
            forecast: Full forecast list with timestamp + price_eur_mwh.
            now: Current UTC time.
            start_offset_hours: Hours from now to start of search window.
                Default 24 = tomorrow midnight (approximately).
            duration_hours: Length of search window in hours.
                Default 48 = two days.

        Returns start timestamps and average prices for blocks of
        1, 2, 3, 4, 6, and 8 consecutive hours, plus a list of all
        hours with below-average price (useful for flexible loads).
        """
        window_start = now + timedelta(hours=start_offset_hours)
        window_end = window_start + timedelta(hours=duration_hours)

        # Filter forecast to the search window
        upcoming = []
        for f in forecast:
            try:
                ts = datetime.fromisoformat(f["timestamp"])
                if window_start <= ts < window_end:
                    upcoming.append(f)
            except (ValueError, TypeError):
                continue

        result: dict[str, Any] = {
            "search_start": window_start.isoformat(),
            "search_end": window_end.isoformat(),
            "search_window": (
                f"start {SpotPriceCoordinator._format_offset(start_offset_hours)}"
                f" + duration {SpotPriceCoordinator._format_offset(duration_hours)}"
            ),
            "hours_in_window": len(upcoming),
        }

        if not upcoming:
            return result

        prices = [f["price_eur_mwh"] for f in upcoming]
        avg_price = sum(prices) / len(prices)
        result["avg_price_in_window"] = round(avg_price, 2)

        # Find cheapest N consecutive hours
        def cheapest_block(n: int) -> tuple[str | None, float | None]:
            if len(upcoming) < n:
                return None, None
            best_avg = float("inf")
            best_start = None
            for i in range(len(upcoming) - n + 1):
                block_avg = sum(prices[i:i + n]) / n
                if block_avg < best_avg:
                    best_avg = block_avg
                    best_start = upcoming[i]["timestamp"]
            return best_start, round(best_avg, 2) if best_start else None

        for n in (1, 2, 3, 4, 6, 8):
            start, avg = cheapest_block(n)
            result[f"cheapest_{n}h_start"] = start
            key = "cheapest_1h_price" if n == 1 else f"cheapest_{n}h_avg_price"
            result[key] = avg

        # All hours with price below window average
        result["hours_below_avg"] = [
            f["timestamp"] for f in upcoming
            if f["price_eur_mwh"] < avg_price
        ]

        return result

    def _enrich_cheapest_with_consumer(
        self,
        cheapest: dict[str, Any],
        consumer_forecast: list[dict[str, Any]],
    ) -> None:
        """Add consumer c/kWh prices to cheapest hours attributes.

        Uses the pre-computed consumer forecast (which already includes
        the configured tariffs, VAT, energy tax, seller margin) to look up
        average consumer prices for each cheapest block. This avoids
        hardcoding any rates in dashboard templates.
        """
        # Build timestamp → consumer price lookup
        cons_by_ts: dict[str, float] = {
            c["timestamp"]: c["price_eur_kwh"] for c in consumer_forecast
        }
        if not cons_by_ts:
            return

        # For each block size, compute avg consumer price from the block hours
        for n in (1, 2, 3, 4, 6, 8):
            start_key = f"cheapest_{n}h_start"
            start_ts = cheapest.get(start_key)
            if not start_ts:
                continue

            # Find the block hours in consumer forecast
            try:
                block_start = datetime.fromisoformat(start_ts)
            except (ValueError, TypeError):
                continue

            block_prices = []
            for hour_offset in range(n):
                ts = (block_start + timedelta(hours=hour_offset)).isoformat()
                if ts in cons_by_ts:
                    block_prices.append(cons_by_ts[ts])

            if block_prices:
                avg_cons = sum(block_prices) / len(block_prices)
                cons_key = (
                    "cheapest_1h_consumer_price"
                    if n == 1
                    else f"cheapest_{n}h_avg_consumer_price"
                )
                cheapest[cons_key] = round(avg_cons * 100, 2)  # c/kWh

        # Window average in consumer c/kWh
        window_start = cheapest.get("search_start")
        window_end = cheapest.get("search_end")
        if window_start and window_end:
            window_prices = []
            for c in consumer_forecast:
                if window_start <= c["timestamp"] < window_end:
                    window_prices.append(c["price_eur_kwh"])
            if window_prices:
                cheapest["avg_consumer_in_window"] = round(
                    sum(window_prices) / len(window_prices) * 100, 2
                )

    def _spot_to_consumer_ckwh(self, spot_eur_mwh: float, is_night: bool) -> float:
        """Convert spot EUR/MWh to consumer c/kWh using configured tariffs."""
        transfer = self.night_rate if is_night else self.day_rate
        spot_kwh = max(0.0, spot_eur_mwh) / 1000.0
        return (spot_kwh + self.seller_margin + transfer + self.energy_tax) \
            * self.vat_multiplier * 100

    def _compute_duration_forecast(
        self,
        forecast: list[dict[str, Any]],
        weather: list[dict[str, Any]],
        ar_neighbor_hourly: dict[str, list[float]] | None,
        tier3_data: dict[str, float] | None,
        now: datetime,
    ) -> list[dict[str, Any]]:
        """Compute 7-day D(k) duration curve forecast.

        Returns list of daily entries:
        [{"date": "2026-04-12", "d1": 8.2, "d4": 9.5, "d8": 10.1, "d24": 12.3,
          "dk_consumer": [24 floats in c/kWh], "dk_spot": [24 floats in EUR/MWh]}, ...]
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

        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(DEFAULT_TIMEZONE)
        except Exception:
            tz = None

        # Group forecast hours by local date
        by_date: dict[str, list[dict]] = {}
        for i, entry in enumerate(forecast):
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if tz:
                    local = ts.astimezone(tz) if ts.tzinfo else ts.replace(
                        tzinfo=ZoneInfo("UTC")).astimezone(tz)
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
            if len(day_hours) < 20:
                continue

            hour_lookup = {h["local_hour"]: h for h in day_hours}
            dow = day_hours[0]["dow"]
            is_holiday = date_str in self.holidays
            is_wd = 1.0 if (dow < 5 and not is_holiday) else 0.0
            mo = int(date_str.split("-")[1])

            # Build per-segment features
            segment_features: dict[str, dict[str, float]] = {}

            for seg_name, hours_list in seg_hours.items():
                seg_hrs = [hour_lookup[h] for h in hours_list if h in hour_lookup]
                if len(seg_hrs) < 2:
                    continue

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
                if tier3_data and "nuclear_mw" in tier3_data:
                    nuclear_deficit = max(0.0, 1.0 - tier3_data["nuclear_mw"] / 4372.0)

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
            if not dk_spot:
                continue

            # Convert to consumer c/kWh (use average of day/night rate for D(k))
            # D(k) represents cheapest k hours which span mixed day/night periods
            dk_consumer = [
                round(self._spot_to_consumer_ckwh(v, False), 2) for v in dk_spot
            ]

            n = len(dk_spot)
            day_entry: dict[str, Any] = {
                "date": date_str,
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
                "d1": dk_consumer[0] if n > 0 else None,
                "d4": dk_consumer[3] if n > 3 else None,
                "d8": dk_consumer[7] if n > 7 else None,
                "d24": dk_consumer[min(23, n - 1)],
                "dk_consumer": dk_consumer,
                "dk_spot": [round(v, 2) for v in dk_spot],
            }
            result.append(day_entry)

        _LOGGER.info("Duration forecast computed: %d days", len(result))
        return result
