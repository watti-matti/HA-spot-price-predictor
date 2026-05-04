"""Pure Python feature engineering for inference. No numpy/pandas.

v2.0.0: up to 17 sign-validated features with AR neighbor prices.
Log-linear prediction: price = exp(sum(coef * feature) + intercept) - offset
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .holidays import build_holiday_set
from .const import DEMAND_DEFAULTS, DEFAULT_TIMEZONE

_LOGGER = logging.getLogger(__name__)

PI2 = 2.0 * math.pi


def daylight_hours(doy: int, lat_deg: float = 62.0) -> float:
    """Approximate astronomical day length for day-of-year at given latitude."""
    lat = math.radians(lat_deg)
    decl = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81))))
    cos_ha = -math.tan(lat) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    return 2.0 * math.degrees(math.acos(cos_ha)) / 15.0


def _gauss_bump(hour: float, centre: float, sigma: float) -> float:
    return math.exp(-0.5 * ((hour - centre) / sigma) ** 2)


def compute_features_for_hour(
    utc_dt: datetime,
    wind_weighted: float,
    solar_weighted: float,
    temp_weighted: float,
    holidays: set[str],
    demand: dict[str, Any] | None = None,
    neighbor_spreads: dict[str, float] | None = None,
    nuclear_data: dict[str, float] | None = None,
    ar_neighbor_prices: dict[str, float] | None = None,
    netload_data: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute all 17 features for a single forecast hour.

    Args:
        utc_dt: UTC datetime for this hour.
        wind_weighted: Capacity-weighted wind speed (m/s).
        solar_weighted: Capacity-weighted solar irradiance (W/m2).
        temp_weighted: Capacity-weighted temperature (C).
        holidays: Set of ISO date strings for holidays.
        demand: Demand config overrides.
        neighbor_spreads: Dict of rolling spread values per neighbor (se3).
        nuclear_data: Dict with nuclear_mw (normalized 0-1).
        ar_neighbor_prices: Dict with ar_se1, ar_se3, ar_ee (EUR/MWh / 100).

    Returns:
        Feature dict ready for model.predict_single().
    """
    d = demand or DEMAND_DEFAULTS
    try:
        from zoneinfo import ZoneInfo
        local_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(DEFAULT_TIMEZONE))
    except Exception:
        local_dt = utc_dt + timedelta(hours=3)
    local_h = local_dt.hour
    dow = local_dt.weekday()
    mo = local_dt.month
    doy = local_dt.timetuple().tm_yday
    date_str = local_dt.strftime("%Y-%m-%d")

    latitude = d.get("latitude", 62.0)

    # Time cycles
    hour_sin = math.sin(PI2 * local_h / 24.0)
    hour_cos = math.cos(PI2 * local_h / 24.0)
    month_sin = math.sin(PI2 * mo / 12.0)
    month_cos = math.cos(PI2 * mo / 12.0)

    is_hol = 1.0 if date_str in holidays else 0.0
    is_workday = 1.0 if (dow < 5 and is_hol == 0.0) else 0.0

    # Demand peaks (used for wind_calm interactions)
    am_center = d.get("peak_am_center", 9)
    am_sigma = d.get("peak_am_sigma", 1.8)
    pm_center = d.get("peak_pm_center", 19)
    pm_sigma = d.get("peak_pm_sigma", 2.0)

    raw_am = _gauss_bump(local_h, am_center, am_sigma)
    raw_pm = _gauss_bump(local_h, pm_center, pm_sigma)

    double_peak_am = raw_am * is_workday
    double_peak_pm = raw_pm * is_workday

    # Thermal demand
    hdd_threshold = d.get("hdd_threshold", 17.0)
    t = temp_weighted
    w = wind_weighted
    s = solar_weighted
    hdd = max(0.0, hdd_threshold - t)
    hdd_sq = hdd ** 2

    # Wind nonlinear features
    wind_log_scarcity = math.log1p(max(0.0, 8.0 - w))
    wind_calm = max(0.0, 6.0 - w)
    wind_calm_x_peak_am = wind_calm * double_peak_am
    wind_calm_x_peak_pm = wind_calm * double_peak_pm

    # Scarcity indicator (for nuclear_x_scarcity)
    low_wind = max(0.0, 5.0 - w)
    peak_demand_full = max(raw_am, raw_pm)
    scarcity_indicator = low_wind * hdd * peak_demand_full

    feat: dict[str, float] = {
        "wind_speed_weighted": w,
        "solar_irradiance_weighted": s,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "is_holiday": is_hol,
        "hdd_sq": hdd_sq,
        "wind_log_scarcity": wind_log_scarcity,
        "wind_calm_x_peak_am": wind_calm_x_peak_am,
        "wind_calm_x_peak_pm": wind_calm_x_peak_pm,
    }

    # Cross-border: AR neighbor prices
    if ar_neighbor_prices:
        feat["ar_se1"] = ar_neighbor_prices.get("ar_se1", 0.0)
        feat["ar_se3"] = ar_neighbor_prices.get("ar_se3", 0.0)
        feat["ar_ee"] = ar_neighbor_prices.get("ar_ee", 0.0)
    else:
        feat["ar_se1"] = 0.0
        feat["ar_se3"] = 0.0
        feat["ar_ee"] = 0.0

    # Cross-border: export potential SE3
    if neighbor_spreads:
        spread = neighbor_spreads.get("se3", 0.0)
        feat["export_potential_se3"] = max(0.0, -spread)
    else:
        feat["export_potential_se3"] = 0.0

    # Nuclear: nuclear features
    nuc = 0.0
    if nuclear_data:
        nuc = nuclear_data.get("nuclear_mw", 0.0)
    nuclear_deficit = max(0.0, 1.0 - nuc)
    feat["nuclear_deficit"] = nuclear_deficit
    feat["nuclear_x_scarcity"] = nuclear_deficit * scarcity_indicator

    # v2.2: net-load features. The Fingrid forecasts are passed as
    # raw MW for this specific hour; we centre the squared term around
    # the long-run mean of 6.0 GW (matches what the training pipeline
    # used). Defaults to 0.0 (neutral) when forecasts are unavailable
    # so missing-Fingrid installs continue to work with the v2.1
    # 17-feature contribution.
    NETLOAD_CENTER_GW = 6.0     # long-run mean from training data
    if netload_data:
        cons_mw = float(netload_data.get("consumption_mw", 0.0))
        wind_mw = float(netload_data.get("wind_forecast_mw", 0.0))
        solar_mw = float(netload_data.get("solar_forecast_mw", 0.0))
        # Re-derive nuclear MW from the normalized fraction in nuclear_data.
        nuclear_mw_value = (nuc * 4372.0) if nuc > 0 else 3500.0
        net_load_mw = cons_mw - wind_mw - solar_mw - nuclear_mw_value
        net_load_gw = net_load_mw / 1000.0
        feat["net_load_gw"] = net_load_gw
        feat["net_load_squared"] = (net_load_gw - NETLOAD_CENTER_GW) ** 2
        feat["net_load_x_workday"] = net_load_gw * is_workday
        feat["net_load_x_scarcity"] = net_load_gw * scarcity_indicator
    else:
        feat["net_load_gw"] = 0.0
        feat["net_load_squared"] = 0.0
        feat["net_load_x_workday"] = 0.0
        feat["net_load_x_scarcity"] = 0.0

    return feat


def compute_ar_forecast(
    ar_model: dict,
    last_prices: list[float],
    forecast_hours: list[tuple[int, bool]],
) -> list[float]:
    """Compute AR neighbor price forecast for multiple hours.

    Args:
        ar_model: Dict with profile_wd, profile_we, ar_coefs.
        last_prices: Last 24+ actual prices (most recent last).
        forecast_hours: List of (local_hour, is_workday) tuples.

    Returns:
        List of predicted prices (EUR/MWh / 100 for feature normalization).
    """
    profile_wd = ar_model["profile_wd"]
    profile_we = ar_model["profile_we"]
    ar_c = ar_model["ar_coefs"]

    # Initialize deviation from last 2 known prices
    if len(last_prices) >= 2:
        # Approximate profile for last hour
        last_h = forecast_hours[0][0] if forecast_hours else 12
        last_wd = forecast_hours[0][1] if forecast_hours else True
        last_profile = profile_wd[last_h] if last_wd else profile_we[last_h]
        dev_t1 = last_prices[-1] - last_profile
        dev_t2 = last_prices[-2] - last_profile
    else:
        dev_t1 = dev_t2 = 0.0

    preds = []
    for h, wd in forecast_hours:
        profile = profile_wd[h] if wd else profile_we[h]
        dev_new = ar_c[0] * dev_t1 + ar_c[1] * dev_t2
        pred = max(0.0, profile + dev_new)
        preds.append(pred / 100.0)  # normalize
        dev_t2 = dev_t1
        dev_t1 = dev_new

    return preds


def build_forecast_features(
    start_utc: datetime,
    hours: int,
    weather_data: list[dict[str, float]],
    holidays: set[str],
    demand: dict[str, Any] | None = None,
    neighbor_spreads: dict[str, float] | None = None,
    nuclear_data: dict[str, float] | None = None,
    nuclear_hourly: dict[str, list[float]] | None = None,
    ar_neighbor_hourly: dict[str, list[float]] | None = None,
    netload_hourly: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, float]]:
    """Build feature dicts for each hour of the forecast window.

    Args:
        start_utc: UTC datetime for hour 0.
        hours: Number of hours to forecast.
        weather_data: List of dicts with wind_weighted, solar_weighted, temp_weighted.
        holidays: Set of ISO date strings.
        demand: Demand config overrides.
        neighbor_spreads: Constant spread values per neighbor.
        nuclear_data: Constant grid data values.
        nuclear_hourly: Per-hour overrides for nuclear features.
        ar_neighbor_hourly: Per-hour AR neighbor prices {se1: [...], se3: [...], ee: [...]}.

    Returns:
        List of feature dicts, one per hour.
    """
    # v2.2: build a timestamp-keyed lookup for net-load forecasts so we
    # can pull per-hour values into compute_features_for_hour.
    netload_lookup: dict[str, dict[str, float]] = {}
    if netload_hourly:
        for series_name, entries in netload_hourly.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                ts = entry.get("timestamp")
                if not ts:
                    continue
                # Normalise to "YYYY-MM-DDTHH" hour key
                ts_key = ts[:13]
                if ts_key not in netload_lookup:
                    netload_lookup[ts_key] = {}
                netload_lookup[ts_key][series_name] = float(
                    entry.get("value_mw", 0.0))

    rows = []
    for i in range(min(hours, len(weather_data))):
        utc_dt = start_utc + timedelta(hours=i)
        wd = weather_data[i]

        # Per-hour nuclear data
        hour_nuclear = dict(nuclear_data) if nuclear_data else None
        if nuclear_hourly and hour_nuclear is not None:
            for key, values in nuclear_hourly.items():
                if i < len(values):
                    hour_nuclear[key] = values[i]

        # Per-hour AR neighbor prices
        hour_ar = None
        if ar_neighbor_hourly:
            hour_ar = {}
            for key, values in ar_neighbor_hourly.items():
                if i < len(values):
                    hour_ar[f"ar_{key}"] = values[i]

        # Per-hour net-load forecast inputs
        hour_netload = None
        if netload_lookup:
            ts_key = utc_dt.strftime("%Y-%m-%dT%H")
            hour_netload = netload_lookup.get(ts_key)

        feat = compute_features_for_hour(
            utc_dt=utc_dt,
            wind_weighted=wd.get("wind_weighted", 0.0),
            solar_weighted=wd.get("solar_weighted", 0.0),
            temp_weighted=wd.get("temp_weighted", 0.0),
            holidays=holidays,
            demand=demand,
            neighbor_spreads=neighbor_spreads,
            nuclear_data=hour_nuclear,
            ar_neighbor_prices=hour_ar,
            netload_data=hour_netload,
        )
        rows.append(feat)
    return rows
