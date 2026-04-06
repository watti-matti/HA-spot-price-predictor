"""
Dynamic feature engineering driven by region config.

Builds features in three tiers based on available data:
  Tier 1 (28 features): Weather + demand patterns (always available)
  Tier 2 (+6 features): Cross-border import/export potential
  Tier 3 (+4 features): Grid infrastructure (nuclear, capacity)

The feature set adapts automatically to whatever data sources are present.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from src.holidays import build_holiday_set

logger = logging.getLogger(__name__)


def daylight_hours(doy: int, lat_deg: float = 62.0) -> float:
    """Approximate astronomical day length for day-of-year at given latitude."""
    lat = math.radians(lat_deg)
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    cos_ha = -math.tan(lat) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    return 2 * math.degrees(math.acos(cos_ha)) / 15.0


def _gauss_bump(hour: float, centre: float, sigma: float) -> float:
    """Gaussian bump function centered at `centre` with width `sigma`."""
    return math.exp(-0.5 * ((hour - centre) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Tier 1: Base features (28)
# ---------------------------------------------------------------------------

TIER1_FEATURES = [
    # Supply
    "wind_speed_weighted", "solar_irradiance_weighted", "temperature_weighted",
    # Time cycles
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Demand patterns
    "double_peak_am", "double_peak_pm",
    "peak_am_weekend", "peak_pm_weekend",
    "sauna_hour", "monday_ramp",
    "is_holiday", "is_weekend",
    # Thermal demand
    "hdd", "hdd_sq", "daylight_deficit",
    "wind_x_hdd", "solar_x_deficit", "temp_x_hdd",
    # Physics supply
    "wind_power_density", "solar_power_temp", "renewable_surplus",
    # Scarcity
    "scarcity_indicator", "wind_drought_penalty",
    "cold_morning_stress", "cold_calm_dark",
]


def _build_tier1(
    df: pd.DataFrame,
    config: dict[str, Any],
    holidays: set[str],
) -> pd.DataFrame:
    """Build Tier 1 features from weather + price data."""
    demand = config.get("demand", {})
    region = config.get("region", {})
    latitude = region.get("latitude", 62.0)

    pi = math.pi
    idx = df.index  # UTC DatetimeIndex

    # Convert UTC to local time using region timezone (handles DST)
    region = config.get("region", {})
    tz_name = region.get("timezone", "Europe/Helsinki")
    try:
        from zoneinfo import ZoneInfo
        local_dt = idx.tz_convert(ZoneInfo(tz_name))
    except Exception:
        local_dt = idx + pd.Timedelta(hours=3)  # Fallback EEST
    local_h = local_dt.hour.to_numpy()
    dow = local_dt.dayofweek.to_numpy()  # 0=Mon
    mo = local_dt.month.to_numpy()
    doy = local_dt.dayofyear.to_numpy()
    date_str = local_dt.strftime("%Y-%m-%d")

    # Time cycles
    df["hour_sin"] = np.sin(2 * pi * local_h / 24)
    df["hour_cos"] = np.cos(2 * pi * local_h / 24)
    df["month_sin"] = np.sin(2 * pi * mo / 12)
    df["month_cos"] = np.cos(2 * pi * mo / 12)
    df["is_weekend"] = (dow >= 5).astype(float)

    # Holiday detection
    is_hol_arr = np.array([1.0 if d in holidays else 0.0 for d in date_str])
    df["is_holiday"] = is_hol_arr

    # Workday flag (weekday AND not holiday)
    is_workday = (dow < 5).astype(float) * (1.0 - is_hol_arr)
    is_nonwork = 1.0 - is_workday

    # Demand peak profiles
    am_center = demand.get("peak_am_center", 9)
    am_sigma = demand.get("peak_am_sigma", 1.8)
    pm_center = demand.get("peak_pm_center", 19)
    pm_sigma = demand.get("peak_pm_sigma", 2.0)

    raw_am = np.array([_gauss_bump(h, am_center, am_sigma) for h in local_h])
    raw_pm = np.array([_gauss_bump(h, pm_center, pm_sigma) for h in local_h])

    df["double_peak_am"] = raw_am * is_workday
    df["double_peak_pm"] = raw_pm * is_workday
    df["peak_am_weekend"] = raw_am * is_nonwork
    df["peak_pm_weekend"] = raw_pm * is_nonwork

    # Sauna hour
    sauna_hours = demand.get("sauna_hours", [20, 21])
    sauna_days_str = demand.get("sauna_days", ["friday", "saturday"])
    sauna_dow = []
    day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
               "friday": 4, "saturday": 5, "sunday": 6}
    for d in sauna_days_str:
        if d.lower() in day_map:
            sauna_dow.append(day_map[d.lower()])
    df["sauna_hour"] = (
        np.isin(dow, sauna_dow) & np.isin(local_h, sauna_hours)
    ).astype(float)

    # First-workday ramp: fires on the first workday after any non-work day
    # (replaces Monday-only ramp to capture post-holiday cold-starts)
    ramp_hours = demand.get("monday_ramp_hours", [6, 7, 8, 9])
    dates_unique = pd.Series(local_dt.date).values
    is_workday_daily = {}
    for d_val in np.unique(dates_unique):
        d_str = str(d_val)
        d_dow = d_val.weekday()
        is_workday_daily[d_val] = (d_dow < 5) and (d_str not in holidays)

    first_workday = np.zeros(len(df), dtype=float)
    for i, d_val in enumerate(dates_unique):
        if is_workday_daily.get(d_val, False):
            prev_day = d_val - pd.Timedelta(days=1).to_pytimedelta()
            if not is_workday_daily.get(prev_day, True):
                first_workday[i] = 1.0

    df["monday_ramp"] = (
        (first_workday == 1.0) & np.isin(local_h, ramp_hours)
    ).astype(float)

    # Thermal demand
    hdd_threshold = demand.get("hdd_threshold", 17.0)
    t = df["temperature_weighted"].to_numpy()
    hdd = np.maximum(0.0, hdd_threshold - t)
    df["hdd"] = hdd
    df["hdd_sq"] = hdd ** 2

    # Daylight deficit
    dl = np.array([daylight_hours(int(d), latitude) for d in doy])
    deficit = np.maximum(0.0, 12.0 - dl)
    df["daylight_deficit"] = deficit

    # Cross-terms
    w = df["wind_speed_weighted"].to_numpy()
    s = df["solar_irradiance_weighted"].to_numpy()
    df["wind_x_hdd"] = w * hdd
    df["solar_x_deficit"] = s * deficit
    df["temp_x_hdd"] = t * hdd

    # Physics-corrected supply
    rated_speed = demand.get("wind_rated_speed", 14)
    rho_rel = 288.15 / (273.15 + t)
    w_capped = np.minimum(w, float(rated_speed))
    df["wind_power_density"] = rho_rel * (w_capped / rated_speed) ** 3

    t_cell = t + 0.03 * s
    pv_efficiency = 1.0 - 0.004 * (t_cell - 25.0)
    df["solar_power_temp"] = s * pv_efficiency

    df["renewable_surplus"] = (
        np.maximum(0.0, w - 6.0) * np.maximum(0.0, s - 100.0) / 100.0
    )

    # Scarcity indicators
    low_wind = np.maximum(0.0, 5.0 - w)
    peak_demand_full = np.maximum(raw_am, raw_pm)
    df["scarcity_indicator"] = low_wind * hdd * peak_demand_full

    wind_drought = np.maximum(0.0, 4.0 - w) ** 2
    df["wind_drought_penalty"] = wind_drought * is_workday

    df["cold_morning_stress"] = np.maximum(0.0, hdd - 20.0) * raw_am * is_workday

    df["cold_calm_dark"] = (
        np.maximum(0.0, hdd - 15.0)
        * np.maximum(0.0, 6.0 - w)
        * np.maximum(0.0, deficit - 4.0) / 10.0
    )

    return df


# ---------------------------------------------------------------------------
# Tier 2: Cross-border trade features (+6)
# ---------------------------------------------------------------------------

TIER2_PREFIXES = ["se1", "se3", "ee"]


def _build_tier2(
    df: pd.DataFrame,
    fi_prices: pd.Series,
    neighbor_prices: dict[str, pd.Series],
    window_days: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Build import/export potential features from price spreads.

    Returns:
        Updated DataFrame and list of new feature column names.
    """
    new_features: list[str] = []
    window_hours = window_days * 24

    for prefix, series in neighbor_prices.items():
        if series is None or len(series) == 0:
            continue

        # Align on common index
        spread_name = f"spread_7d_fi_{prefix}"
        aligned_fi = fi_prices.reindex(series.index)
        spread = aligned_fi - series
        spread_rolling = spread.rolling(window=window_hours, min_periods=24).mean()

        # Map back to main DataFrame index
        spread_aligned = spread_rolling.reindex(df.index).ffill()

        import_name = f"import_potential_{prefix}"
        export_name = f"export_potential_{prefix}"

        df[import_name] = np.maximum(0.0, spread_aligned.values)
        df[export_name] = np.maximum(0.0, -spread_aligned.values)

        # Fill any remaining NaN with 0
        df[import_name] = df[import_name].fillna(0.0)
        df[export_name] = df[export_name].fillna(0.0)

        new_features.extend([import_name, export_name])
        logger.info("  Tier 2: added %s, %s", import_name, export_name)

    return df, new_features


# ---------------------------------------------------------------------------
# Tier 3: Grid infrastructure features (+0-4)
# ---------------------------------------------------------------------------

def _build_tier3(
    df: pd.DataFrame,
    grid_data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Add grid infrastructure features (nuclear, capacity) to DataFrame.

    Returns:
        Updated DataFrame and list of new feature column names.
    """
    new_features: list[str] = []

    for feature_name, series in grid_data.items():
        if series is None or len(series) == 0:
            continue

        # Align to main DataFrame index
        aligned = series.reindex(df.index).ffill().bfill()
        df[feature_name] = aligned.fillna(0.0)
        new_features.append(feature_name)
        logger.info("  Tier 3: added %s (%d non-null)", feature_name, aligned.notna().sum())

    return df, new_features


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_features(
    prices: pd.Series,
    weather: pd.DataFrame,
    config: dict[str, Any],
    neighbor_prices: dict[str, pd.Series] | None = None,
    grid_data: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the complete feature matrix from all available data sources.

    Args:
        prices: FI spot prices (EUR/MWh), UTC index.
        weather: Weather DataFrame (wind, solar, temp weighted columns).
        config: Region config dict.
        neighbor_prices: Optional dict of neighbor zone price series (Tier 2).
        grid_data: Optional dict of grid infrastructure series (Tier 3).

    Returns:
        Tuple of (DataFrame with all features + price, list of feature column names).
    """
    training = config.get("training", {})
    price_clip = training.get("price_clip_max", 500)
    spread_window = config.get("spread_window_days", 7)

    # Build holiday set
    start_year = prices.index.min().year - 1
    end_year = prices.index.max().year + 2
    holidays = build_holiday_set(config, start_year, end_year)
    logger.info("Holiday set: %d dates (%d-%d)", len(holidays), start_year, end_year - 1)

    # Merge price + weather
    df = prices.to_frame("price_eur_mwh").join(weather, how="inner").dropna()
    logger.info("Merged price+weather: %d rows", len(df))

    # Tier 1: Base features
    df = _build_tier1(df, config, holidays)
    feature_cols = list(TIER1_FEATURES)
    logger.info("Tier 1: %d features", len(feature_cols))

    # Tier 2: Cross-border trade features
    if neighbor_prices:
        df, tier2_features = _build_tier2(df, prices, neighbor_prices, spread_window)
        feature_cols.extend(tier2_features)
        logger.info("Tier 2: +%d features (total: %d)", len(tier2_features), len(feature_cols))
    else:
        logger.info("Tier 2: skipped (no neighbor prices)")

    # Tier 3: Grid infrastructure features
    if grid_data:
        df, tier3_features = _build_tier3(df, grid_data)
        feature_cols.extend(tier3_features)
        logger.info("Tier 3: +%d features (total: %d)", len(tier3_features), len(feature_cols))
    else:
        logger.info("Tier 3: skipped (no grid data)")

    # Clipped price for training target
    df["price_clipped"] = df["price_eur_mwh"].clip(upper=price_clip)

    # Drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < before:
        logger.info("Dropped %d rows with NaN features", before - len(df))

    logger.info("Final feature matrix: %d rows x %d features", len(df), len(feature_cols))
    return df, feature_cols
