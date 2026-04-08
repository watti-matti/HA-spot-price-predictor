"""
Dynamic feature engineering driven by region config.

Builds features in three tiers based on available data:
  Tier 1 (10 features): Weather + demand patterns (always available)
  Tier 2 (+2 features): Cross-border export potential (SE3, EE)
  Tier 3 (+2 features): Nuclear outage interaction (scarcity)

Features are selected via greedy forward selection with sign constraints:
only features with economically correct coefficient signs are included.
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
    # Supply (sign-validated: more supply -> lower price)
    "wind_speed_weighted", "solar_irradiance_weighted",
    # Time cycles
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Demand patterns (sign-validated: more demand -> higher price)
    "double_peak_am", "double_peak_pm",
    "is_holiday",
    # Thermal demand (sign-validated)
    "hdd_sq",
    # Scarcity (sign-validated: more scarcity -> higher price)
    "wind_drought_penalty",
    # Interaction
    "solar_x_deficit",
]


def _build_tier1(
    df: pd.DataFrame,
    config: dict[str, Any],
    holidays: set[str],
) -> pd.DataFrame:
    """Build Tier 1 features from weather + price data.

    Only features validated via greedy forward selection with sign constraints
    are included. Each feature has an economically correct coefficient sign.
    """
    demand = config.get("demand", {})
    region = config.get("region", {})
    latitude = region.get("latitude", 62.0)

    pi = math.pi
    idx = df.index  # UTC DatetimeIndex

    # Convert UTC to local time using region timezone (handles DST)
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

    # Holiday detection
    is_hol_arr = np.array([1.0 if d in holidays else 0.0 for d in date_str])
    df["is_holiday"] = is_hol_arr

    # Workday flag (weekday AND not holiday)
    is_workday = (dow < 5).astype(float) * (1.0 - is_hol_arr)

    # Demand peak profiles
    am_center = demand.get("peak_am_center", 9)
    am_sigma = demand.get("peak_am_sigma", 1.8)
    pm_center = demand.get("peak_pm_center", 19)
    pm_sigma = demand.get("peak_pm_sigma", 2.0)

    raw_am = np.array([_gauss_bump(h, am_center, am_sigma) for h in local_h])
    raw_pm = np.array([_gauss_bump(h, pm_center, pm_sigma) for h in local_h])

    df["double_peak_am"] = raw_am * is_workday
    df["double_peak_pm"] = raw_pm * is_workday

    # Thermal demand
    hdd_threshold = demand.get("hdd_threshold", 17.0)
    t = df["temperature_weighted"].to_numpy()
    hdd = np.maximum(0.0, hdd_threshold - t)
    df["hdd_sq"] = hdd ** 2

    # Daylight deficit (intermediate for solar_x_deficit interaction)
    dl = np.array([daylight_hours(int(d), latitude) for d in doy])
    deficit = np.maximum(0.0, 12.0 - dl)

    # Solar x deficit interaction
    s = df["solar_irradiance_weighted"].to_numpy()
    df["solar_x_deficit"] = s * deficit

    # Wind drought penalty: low wind on workdays
    w = df["wind_speed_weighted"].to_numpy()
    wind_drought = np.maximum(0.0, 4.0 - w) ** 2
    df["wind_drought_penalty"] = wind_drought * is_workday

    # Scarcity indicator (intermediate for nuclear_x_scarcity in Tier 3)
    low_wind = np.maximum(0.0, 5.0 - w)
    peak_demand_full = np.maximum(raw_am, raw_pm)
    df["_scarcity_indicator"] = low_wind * hdd * peak_demand_full

    return df


# ---------------------------------------------------------------------------
# Tier 2: Cross-border export potential features (+2)
# ---------------------------------------------------------------------------

# Only export potential features survived sign-constrained selection.
# Import potential features had wrong signs (high import potential is a symptom
# of high FI prices, not a cause of lower prices in the model's timeframe).
TIER2_EXPORT_PREFIXES = ["se3", "ee"]


def _build_tier2(
    df: pd.DataFrame,
    fi_prices: pd.Series,
    neighbor_prices: dict[str, pd.Series],
    window_days: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Build export potential features from price spreads.

    Returns:
        Updated DataFrame and list of new feature column names.
    """
    new_features: list[str] = []
    window_hours = window_days * 24

    for prefix, series in neighbor_prices.items():
        if series is None or len(series) == 0:
            continue
        if prefix not in TIER2_EXPORT_PREFIXES:
            continue

        # Align on common index
        aligned_fi = fi_prices.reindex(series.index)
        spread = aligned_fi - series
        spread_rolling = spread.rolling(window=window_hours, min_periods=24).mean()

        # Map back to main DataFrame index
        spread_aligned = spread_rolling.reindex(df.index).ffill()

        export_name = f"export_potential_{prefix}"
        df[export_name] = np.maximum(0.0, -spread_aligned.values)
        df[export_name] = df[export_name].fillna(0.0)

        new_features.append(export_name)
        logger.info("  Tier 2: added %s", export_name)

    return df, new_features


# ---------------------------------------------------------------------------
# Tier 3: Nuclear x scarcity interaction (+0-1)
# ---------------------------------------------------------------------------

def _build_tier3(
    df: pd.DataFrame,
    grid_data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Add nuclear x scarcity interaction feature.

    Only nuclear_x_scarcity survived sign-constrained feature selection.
    Raw nuclear_mw and nuclear_deficit had sign instability due to
    collinearity, but their interaction with scarcity is robust.

    Returns:
        Updated DataFrame and list of new feature column names.
    """
    new_features: list[str] = []

    # Load nuclear production data (needed for interaction)
    nuc_series = grid_data.get("nuclear_mw")
    if nuc_series is not None and len(nuc_series) > 0:
        aligned = nuc_series.reindex(df.index).ffill().bfill()
        nuclear_mw = aligned.fillna(0.0)
        nuclear_deficit = np.maximum(0.0, 1.0 - nuclear_mw)

        # Nuclear deficit x scarcity: fires when nuclear is down AND
        # weather conditions are stressed (low wind + cold + peak demand)
        if "_scarcity_indicator" in df.columns:
            df["nuclear_x_scarcity"] = nuclear_deficit * df["_scarcity_indicator"]
            new_features.append("nuclear_x_scarcity")
            logger.info("  Tier 3: added nuclear_x_scarcity (max=%.3f)",
                        df["nuclear_x_scarcity"].max())

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

    # Remove internal intermediate columns
    for col in list(df.columns):
        if col.startswith("_"):
            df = df.drop(columns=[col])

    # Clipped price for training target
    df["price_clipped"] = df["price_eur_mwh"].clip(upper=price_clip)

    # Drop rows with any NaN in feature columns
    before = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < before:
        logger.info("Dropped %d rows with NaN features", before - len(df))

    logger.info("Final feature matrix: %d rows x %d features", len(df), len(feature_cols))
    return df, feature_cols
