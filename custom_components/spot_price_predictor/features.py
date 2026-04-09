"""Pure Python feature engineering for inference. No numpy/pandas.

Only the 14 features validated via greedy forward selection with sign
constraints are computed. Each feature has an economically correct
coefficient sign in the trained model.
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
    tier2_spreads: dict[str, float] | None = None,
    tier3_data: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute all 14 features for a single forecast hour.

    Args:
        utc_dt: UTC datetime for this hour.
        wind_weighted: Capacity-weighted wind speed (m/s).
        solar_weighted: Capacity-weighted solar irradiance (W/m2).
        temp_weighted: Capacity-weighted temperature (C).
        holidays: Set of ISO date strings for holidays.
        demand: Demand config overrides.
        tier2_spreads: Dict of rolling spread values per neighbor (se3, ee).
        tier3_data: Dict with nuclear_mw (normalized 0-1).

    Returns:
        Feature dict ready for model.predict_single().
    """
    d = demand or DEMAND_DEFAULTS
    try:
        from zoneinfo import ZoneInfo
        local_dt = utc_dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(DEFAULT_TIMEZONE))
    except Exception:
        local_dt = utc_dt + timedelta(hours=3)  # Fallback EEST
    local_h = local_dt.hour
    dow = local_dt.weekday()  # 0=Mon
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

    # Demand peaks
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

    # Daylight deficit (for solar_x_deficit interaction)
    dl = daylight_hours(doy, latitude)
    deficit = max(0.0, 12.0 - dl)
    solar_x_deficit = s * deficit

    # Wind drought penalty: low wind on workdays
    wind_drought = max(0.0, 4.0 - w) ** 2
    wind_drought_penalty = wind_drought * is_workday

    # Nonlinear wind features
    wind_log_scarcity = math.log1p(max(0.0, 8.0 - w))
    wind_calm = max(0.0, 6.0 - w)
    wind_calm_x_peak_am = wind_calm * double_peak_am
    wind_calm_x_peak_pm = wind_calm * double_peak_pm

    # Scarcity indicator (intermediate for nuclear_x_scarcity)
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
        "double_peak_am": double_peak_am,
        "double_peak_pm": double_peak_pm,
        "is_holiday": is_hol,
        "hdd_sq": hdd_sq,
        "wind_drought_penalty": wind_drought_penalty,
        "solar_x_deficit": solar_x_deficit,
        "wind_log_scarcity": wind_log_scarcity,
        "wind_calm_x_peak_am": wind_calm_x_peak_am,
        "wind_calm_x_peak_pm": wind_calm_x_peak_pm,
    }

    # Tier 2: export potential (SE3, EE only)
    if tier2_spreads:
        for prefix in ("se3", "ee"):
            spread = tier2_spreads.get(prefix, 0.0)
            feat[f"export_potential_{prefix}"] = max(0.0, -spread)
    else:
        for prefix in ("se3", "ee"):
            feat[f"export_potential_{prefix}"] = 0.0

    # Tier 3: nuclear x scarcity interaction
    nuc = 0.0
    if tier3_data:
        nuc = tier3_data.get("nuclear_mw", 0.0)
    nuclear_deficit = max(0.0, 1.0 - nuc)
    feat["nuclear_x_scarcity"] = nuclear_deficit * scarcity_indicator

    return feat


def build_forecast_features(
    start_utc: datetime,
    hours: int,
    weather_data: list[dict[str, float]],
    holidays: set[str],
    demand: dict[str, Any] | None = None,
    tier2_spreads: dict[str, float] | None = None,
    tier3_data: dict[str, float] | None = None,
    tier3_hourly: dict[str, list[float]] | None = None,
) -> list[dict[str, float]]:
    """Build feature dicts for each hour of the forecast window.

    Args:
        start_utc: UTC datetime for hour 0.
        hours: Number of hours to forecast.
        weather_data: List of dicts with wind_weighted, solar_weighted, temp_weighted
                      per hour (must have at least `hours` entries).
        holidays: Set of ISO date strings.
        demand: Demand config overrides.
        tier2_spreads: Constant spread values per neighbor (simplified for forecast).
        tier3_data: Constant grid data values (simplified for forecast).
        tier3_hourly: Per-hour overrides for tier3 features (e.g. nuclear_mw
                      from outage schedule). Keys map to lists of per-hour values.

    Returns:
        List of feature dicts, one per hour.
    """
    rows = []
    for i in range(min(hours, len(weather_data))):
        utc_dt = start_utc + timedelta(hours=i)
        wd = weather_data[i]

        # Build per-hour tier3 data: start from constant, override with hourly
        hour_tier3 = dict(tier3_data) if tier3_data else None
        if tier3_hourly and hour_tier3 is not None:
            for key, values in tier3_hourly.items():
                if i < len(values):
                    hour_tier3[key] = values[i]

        feat = compute_features_for_hour(
            utc_dt=utc_dt,
            wind_weighted=wd.get("wind_weighted", 0.0),
            solar_weighted=wd.get("solar_weighted", 0.0),
            temp_weighted=wd.get("temp_weighted", 0.0),
            holidays=holidays,
            demand=demand,
            tier2_spreads=tier2_spreads,
            tier3_data=hour_tier3,
        )
        rows.append(feat)
    return rows
