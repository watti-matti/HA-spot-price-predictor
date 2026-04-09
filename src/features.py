"""
Dynamic feature engineering driven by region config.

Builds features in three tiers based on available data:
  Tier 1 (11 features): Weather + demand + wind nonlinear (always available)
  Tier 2 (+4 features): AR neighbor prices (SE1, SE3, EE) + export potential
  Tier 3 (+2 features): Nuclear deficit + nuclear x scarcity interaction

Model v6.0: log-linear Ridge regression on 17 sign-validated features.
AR(2) models predict cross-border neighbor prices using workday/weekend
hourly profiles + damped autoregressive deviation.
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
# Tier 1: Base features (11)
# ---------------------------------------------------------------------------

TIER1_FEATURES = [
    # Supply (more supply -> lower price)
    "wind_speed_weighted", "solar_irradiance_weighted",
    # Time cycles
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Calendar
    "is_holiday",
    # Thermal demand
    "hdd_sq",
    # Wind nonlinear (log-scarcity + calm x peak interactions)
    "wind_log_scarcity",
    "wind_calm_x_peak_am",
    "wind_calm_x_peak_pm",
]


def _build_tier1(
    df: pd.DataFrame,
    config: dict[str, Any],
    holidays: set[str],
) -> pd.DataFrame:
    """Build Tier 1 features from weather + demand data."""
    demand = config.get("demand", {})
    region = config.get("region", {})
    latitude = region.get("latitude", 62.0)

    pi = math.pi
    idx = df.index

    tz_name = region.get("timezone", "Europe/Helsinki")
    try:
        from zoneinfo import ZoneInfo
        local_dt = idx.tz_convert(ZoneInfo(tz_name))
    except Exception:
        local_dt = idx + pd.Timedelta(hours=3)
    local_h = local_dt.hour.to_numpy()
    dow = local_dt.dayofweek.to_numpy()
    mo = local_dt.month.to_numpy()
    doy = local_dt.dayofyear.to_numpy()
    date_str = local_dt.strftime("%Y-%m-%d")

    # Time cycles
    df["hour_sin"] = np.sin(2 * pi * local_h / 24)
    df["hour_cos"] = np.cos(2 * pi * local_h / 24)
    df["month_sin"] = np.sin(2 * pi * mo / 12)
    df["month_cos"] = np.cos(2 * pi * mo / 12)

    # Holiday detection
    is_hol_arr = np.array([1.0 if d in holidays else 0.0 for d in date_str])
    df["is_holiday"] = is_hol_arr

    # Workday flag (weekday AND not holiday)
    is_workday = (dow < 5).astype(float) * (1.0 - is_hol_arr)

    # Demand peak profiles (used for wind_calm interactions)
    am_center = demand.get("peak_am_center", 9)
    am_sigma = demand.get("peak_am_sigma", 1.8)
    pm_center = demand.get("peak_pm_center", 19)
    pm_sigma = demand.get("peak_pm_sigma", 2.0)

    raw_am = np.array([_gauss_bump(h, am_center, am_sigma) for h in local_h])
    raw_pm = np.array([_gauss_bump(h, pm_center, pm_sigma) for h in local_h])

    double_peak_am = raw_am * is_workday
    double_peak_pm = raw_pm * is_workday

    # Thermal demand
    hdd_threshold = demand.get("hdd_threshold", 17.0)
    t = df["temperature_weighted"].to_numpy()
    hdd = np.maximum(0.0, hdd_threshold - t)
    df["hdd_sq"] = hdd ** 2

    # Wind features
    w = df["wind_speed_weighted"].to_numpy()
    df["wind_log_scarcity"] = np.log1p(np.maximum(0.0, 8.0 - w))
    wind_calm = np.maximum(0.0, 6.0 - w)
    df["wind_calm_x_peak_am"] = wind_calm * double_peak_am
    df["wind_calm_x_peak_pm"] = wind_calm * double_peak_pm

    # Scarcity indicator (intermediate for nuclear_x_scarcity in Tier 3)
    low_wind = np.maximum(0.0, 5.0 - w)
    peak_demand_full = np.maximum(raw_am, raw_pm)
    df["_scarcity_indicator"] = low_wind * hdd * peak_demand_full
    df["_is_workday"] = is_workday
    df["_hour"] = local_h

    return df


# ---------------------------------------------------------------------------
# Tier 2: AR neighbor prices + export potential (+4)
# ---------------------------------------------------------------------------

def build_ar_models(
    neighbor_prices: dict[str, pd.Series],
    df_index: pd.DatetimeIndex,
    holidays: set[str],
    split_frac: float = 0.85,
) -> dict[str, dict]:
    """Fit AR(2) models on deviation from hourly daytype profiles.

    For each neighbor (SE1, SE3, EE):
    1. Build hourly price profiles for workdays and weekends
    2. Compute deviation from profile
    3. Fit AR(2) on training portion of deviation
    4. Apply damping if max root > 0.95 for stability

    Returns dict mapping prefix -> {profile_wd, profile_we, ar_coefs}.
    """
    try:
        from zoneinfo import ZoneInfo
        local = df_index.tz_convert(ZoneInfo("Europe/Helsinki"))
    except Exception:
        local = df_index + pd.Timedelta(hours=3)

    hour_arr = local.hour.values
    dow_arr = local.dayofweek.values
    date_str_arr = local.strftime("%Y-%m-%d")
    is_workday_arr = np.array(
        [(d < 5) and (ds not in holidays)
         for d, ds in zip(dow_arr, date_str_arr)], dtype=int)

    ar_models = {}

    for prefix, series in neighbor_prices.items():
        if series is None or len(series) == 0:
            continue
        raw = series.reindex(df_index).ffill()
        vals = raw.values

        # Hourly profiles per daytype
        profile_wd = np.zeros(24)
        profile_we = np.zeros(24)
        for h in range(24):
            wd_mask = (hour_arr == h) & (is_workday_arr == 1)
            we_mask = (hour_arr == h) & (is_workday_arr == 0)
            if wd_mask.sum() > 0:
                profile_wd[h] = float(np.nanmean(vals[wd_mask]))
            if we_mask.sum() > 0:
                profile_we[h] = float(np.nanmean(vals[we_mask]))

        # Deviation from profile
        profile_vals = np.array([
            profile_wd[h] if wd else profile_we[h]
            for h, wd in zip(hour_arr, is_workday_arr)])
        deviation = vals - profile_vals

        # Fit AR(2) on training portion
        n = len(deviation)
        split = int(n * split_frac)
        X_ar = np.column_stack([deviation[1:n - 1], deviation[:n - 2]])
        y_ar = deviation[2:]
        Xtr, ytr = X_ar[:split], y_ar[:split]
        ar_coefs, _, _, _ = np.linalg.lstsq(Xtr, ytr, rcond=None)

        # Damp if max root > 0.95 for 170h stability
        roots = np.roots([1, -ar_coefs[0], -ar_coefs[1]])
        max_root = float(np.max(np.abs(roots)))
        if max_root > 0.95:
            ar_coefs = ar_coefs * (0.95 / max_root)

        ar_models[prefix] = {
            "profile_wd": profile_wd.tolist(),
            "profile_we": profile_we.tolist(),
            "ar_coefs": ar_coefs.tolist(),
        }
        logger.info("  AR(%s): coefs=[%.4f, %.4f] max_root=%.4f",
                     prefix.upper(), ar_coefs[0], ar_coefs[1], max_root)

    return ar_models


def _build_tier2(
    df: pd.DataFrame,
    fi_prices: pd.Series,
    neighbor_prices: dict[str, pd.Series],
    holidays: set[str],
    window_days: int = 7,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Build AR neighbor price features + export potential.

    Returns:
        Updated DataFrame, list of new feature column names, AR model params.
    """
    new_features: list[str] = []
    window_hours = window_days * 24

    # Fit AR models
    ar_models = build_ar_models(neighbor_prices, df.index, holidays)

    hour_arr = df["_hour"].values if "_hour" in df.columns else None
    is_workday_arr = df["_is_workday"].values if "_is_workday" in df.columns else None

    # AR neighbor price features
    for prefix in ["se1", "se3", "ee"]:
        if prefix not in ar_models or prefix not in neighbor_prices:
            continue
        m = ar_models[prefix]
        raw = neighbor_prices[prefix].reindex(df.index).ffill()
        vals = raw.values
        profile_wd = np.array(m["profile_wd"])
        profile_we = np.array(m["profile_we"])
        ar_c = np.array(m["ar_coefs"])

        # Profile at each hour
        profile = np.array([
            profile_wd[int(h)] if wd else profile_we[int(h)]
            for h, wd in zip(hour_arr, is_workday_arr)])
        deviation = vals - profile

        # One-step-ahead AR prediction
        ar_pred = np.zeros(len(vals))
        for i in range(2, len(vals)):
            ar_pred[i] = ar_c[0] * deviation[i - 1] + ar_c[1] * deviation[i - 2]

        feature_name = f"ar_{prefix}"
        df[feature_name] = (profile + ar_pred) / 100  # normalize
        new_features.append(feature_name)
        logger.info("  Tier 2: added %s (AR neighbor price)", feature_name)

    # Export potential SE3 (spread-based, kept from previous model)
    se3_series = neighbor_prices.get("se3")
    if se3_series is not None and len(se3_series) > 0:
        aligned_fi = fi_prices.reindex(se3_series.index)
        spread = aligned_fi - se3_series
        spread_rolling = spread.rolling(window=window_hours, min_periods=24).mean()
        spread_aligned = spread_rolling.reindex(df.index).ffill()
        df["export_potential_se3"] = np.maximum(0.0, -spread_aligned.values)
        df["export_potential_se3"] = df["export_potential_se3"].fillna(0.0)
        new_features.append("export_potential_se3")
        logger.info("  Tier 2: added export_potential_se3")

    return df, new_features, ar_models


# ---------------------------------------------------------------------------
# Tier 3: Nuclear deficit + scarcity interaction (+0-2)
# ---------------------------------------------------------------------------

def _build_tier3(
    df: pd.DataFrame,
    grid_data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Add nuclear deficit and nuclear x scarcity features.

    Returns:
        Updated DataFrame and list of new feature column names.
    """
    new_features: list[str] = []

    nuc_series = grid_data.get("nuclear_mw")
    if nuc_series is not None and len(nuc_series) > 0:
        aligned = nuc_series.reindex(df.index).ffill().bfill()
        nuclear_mw = aligned.fillna(0.0)
        nuclear_deficit = np.maximum(0.0, 1.0 - nuclear_mw)

        # Standalone nuclear deficit
        df["nuclear_deficit"] = nuclear_deficit
        new_features.append("nuclear_deficit")
        logger.info("  Tier 3: added nuclear_deficit (max=%.3f)", nuclear_deficit.max())

        # Nuclear deficit x scarcity interaction
        if "_scarcity_indicator" in df.columns:
            df["nuclear_x_scarcity"] = nuclear_deficit * df["_scarcity_indicator"]
            new_features.append("nuclear_x_scarcity")
            logger.info("  Tier 3: added nuclear_x_scarcity")

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
) -> tuple[pd.DataFrame, list[str], dict | None]:
    """Build the complete feature matrix from all available data sources.

    Args:
        prices: FI spot prices (EUR/MWh), UTC index.
        weather: Weather DataFrame (wind, solar, temp weighted columns).
        config: Region config dict.
        neighbor_prices: Optional dict of neighbor zone price series (Tier 2).
        grid_data: Optional dict of grid infrastructure series (Tier 3).

    Returns:
        Tuple of (DataFrame, feature column names, AR model params or None).
    """
    training = config.get("training", {})
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

    # Tier 2: AR neighbor prices + export potential
    ar_models = None
    if neighbor_prices:
        df, tier2_features, ar_models = _build_tier2(
            df, prices, neighbor_prices, holidays, spread_window)
        feature_cols.extend(tier2_features)
        logger.info("Tier 2: +%d features (total: %d)",
                     len(tier2_features), len(feature_cols))
    else:
        logger.info("Tier 2: skipped (no neighbor prices)")

    # Tier 3: Nuclear features
    if grid_data:
        df, tier3_features = _build_tier3(df, grid_data)
        feature_cols.extend(tier3_features)
        logger.info("Tier 3: +%d features (total: %d)",
                     len(tier3_features), len(feature_cols))
    else:
        logger.info("Tier 3: skipped (no grid data)")

    # Remove internal intermediate columns
    for col in list(df.columns):
        if col.startswith("_"):
            df = df.drop(columns=[col])

    # Drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < before:
        logger.info("Dropped %d rows with NaN features", before - len(df))

    logger.info("Final feature matrix: %d rows x %d features", len(df), len(feature_cols))
    return df, feature_cols, ar_models
