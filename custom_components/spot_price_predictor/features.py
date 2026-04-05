"""Pure Python feature engineering for inference. No numpy/pandas."""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .holidays import build_holiday_set
from .const import DEMAND_DEFAULTS, DEFAULT_TIMEZONE, FINGRID_MAX_VALUES

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
    """Compute all features for a single forecast hour.

    Args:
        utc_dt: UTC datetime for this hour.
        wind_weighted: Capacity-weighted wind speed (m/s).
        solar_weighted: Capacity-weighted solar irradiance (W/m2).
        temp_weighted: Capacity-weighted temperature (C).
        holidays: Set of ISO date strings for holidays.
        demand: Demand config overrides.
        tier2_spreads: Dict of rolling spread values per neighbor (se1, se3, ee).
        tier3_data: Dict with nuclear_mw, flow_fi_se1, etc. (normalized 0-1).

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

    is_weekend = 1.0 if dow >= 5 else 0.0
    is_hol = 1.0 if date_str in holidays else 0.0
    is_workday = 1.0 if (dow < 5 and is_hol == 0.0) else 0.0
    is_nonwork = 1.0 - is_workday

    # Demand peaks
    am_center = d.get("peak_am_center", 9)
    am_sigma = d.get("peak_am_sigma", 1.8)
    pm_center = d.get("peak_pm_center", 19)
    pm_sigma = d.get("peak_pm_sigma", 2.0)

    raw_am = _gauss_bump(local_h, am_center, am_sigma)
    raw_pm = _gauss_bump(local_h, pm_center, pm_sigma)

    double_peak_am = raw_am * is_workday
    double_peak_pm = raw_pm * is_workday
    peak_am_weekend = raw_am * is_nonwork
    peak_pm_weekend = raw_pm * is_nonwork

    # Sauna hour
    sauna_hours = d.get("sauna_hours", [20, 21])
    sauna_days = d.get("sauna_days", [4, 5])
    sauna_hour = 1.0 if (dow in sauna_days and local_h in sauna_hours) else 0.0

    # Monday ramp
    ramp_hours = d.get("monday_ramp_hours", [6, 7, 8, 9])
    monday_ramp = 1.0 if (dow == 0 and local_h in ramp_hours) else 0.0

    # Thermal demand
    hdd_threshold = d.get("hdd_threshold", 17.0)
    t = temp_weighted
    w = wind_weighted
    s = solar_weighted
    hdd = max(0.0, hdd_threshold - t)
    hdd_sq = hdd ** 2

    # Daylight deficit
    dl = daylight_hours(doy, latitude)
    deficit = max(0.0, 12.0 - dl)

    # Cross-terms
    wind_x_hdd = w * hdd
    solar_x_deficit = s * deficit
    temp_x_hdd = t * hdd

    # Physics supply
    rated_speed = d.get("wind_rated_speed", 14)
    rho_rel = 288.15 / (273.15 + t) if (273.15 + t) != 0 else 1.0
    w_capped = min(w, float(rated_speed))
    wind_power_density = rho_rel * (w_capped / rated_speed) ** 3

    t_cell = t + 0.03 * s
    pv_efficiency = 1.0 - 0.004 * (t_cell - 25.0)
    solar_power_temp = s * pv_efficiency

    renewable_surplus = max(0.0, w - 6.0) * max(0.0, s - 100.0) / 100.0

    # Scarcity
    low_wind = max(0.0, 5.0 - w)
    peak_demand_full = max(raw_am, raw_pm)
    scarcity_indicator = low_wind * hdd * peak_demand_full

    wind_drought = max(0.0, 4.0 - w) ** 2
    wind_drought_penalty = wind_drought * is_workday

    cold_morning_stress = max(0.0, hdd - 20.0) * raw_am * is_workday

    cold_calm_dark = (
        max(0.0, hdd - 15.0)
        * max(0.0, 6.0 - w)
        * max(0.0, deficit - 4.0) / 10.0
    )

    feat: dict[str, float] = {
        "wind_speed_weighted": w,
        "solar_irradiance_weighted": s,
        "temperature_weighted": t,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "double_peak_am": double_peak_am,
        "double_peak_pm": double_peak_pm,
        "peak_am_weekend": peak_am_weekend,
        "peak_pm_weekend": peak_pm_weekend,
        "sauna_hour": sauna_hour,
        "monday_ramp": monday_ramp,
        "is_holiday": is_hol,
        "is_weekend": is_weekend,
        "hdd": hdd,
        "hdd_sq": hdd_sq,
        "daylight_deficit": deficit,
        "wind_x_hdd": wind_x_hdd,
        "solar_x_deficit": solar_x_deficit,
        "temp_x_hdd": temp_x_hdd,
        "wind_power_density": wind_power_density,
        "solar_power_temp": solar_power_temp,
        "renewable_surplus": renewable_surplus,
        "scarcity_indicator": scarcity_indicator,
        "wind_drought_penalty": wind_drought_penalty,
        "cold_morning_stress": cold_morning_stress,
        "cold_calm_dark": cold_calm_dark,
    }

    # Tier 2: cross-border spreads
    if tier2_spreads:
        for prefix in ("se1", "se3", "ee"):
            spread = tier2_spreads.get(prefix, 0.0)
            feat[f"import_potential_{prefix}"] = max(0.0, spread)
            feat[f"export_potential_{prefix}"] = max(0.0, -spread)
    else:
        for prefix in ("se1", "se3", "ee"):
            feat[f"import_potential_{prefix}"] = 0.0
            feat[f"export_potential_{prefix}"] = 0.0

    # Tier 3: grid data (already normalized)
    if tier3_data:
        for key in ("nuclear_mw", "flow_fi_se1", "flow_fi_se3", "flow_fi_ee"):
            feat[key] = tier3_data.get(key, 0.0)
    else:
        for key in ("nuclear_mw", "flow_fi_se1", "flow_fi_se3", "flow_fi_ee"):
            feat[key] = 0.0

    return feat


def build_forecast_features(
    start_utc: datetime,
    hours: int,
    weather_data: list[dict[str, float]],
    holidays: set[str],
    demand: dict[str, Any] | None = None,
    tier2_spreads: dict[str, float] | None = None,
    tier3_data: dict[str, float] | None = None,
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

    Returns:
        List of feature dicts, one per hour.
    """
    rows = []
    for i in range(min(hours, len(weather_data))):
        utc_dt = start_utc + timedelta(hours=i)
        wd = weather_data[i]
        feat = compute_features_for_hour(
            utc_dt=utc_dt,
            wind_weighted=wd.get("wind_weighted", 0.0),
            solar_weighted=wd.get("solar_weighted", 0.0),
            temp_weighted=wd.get("temp_weighted", 0.0),
            holidays=holidays,
            demand=demand,
            tier2_spreads=tier2_spreads,
            tier3_data=tier3_data,
        )
        rows.append(feat)
    return rows
