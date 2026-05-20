"""Generic loader for Fingrid Datahub hourly-consumption CSV exports.

The Datahub CSV format (as of 2024+):

  - Semicolon-delimited
  - Comma decimal separator (Finnish locale)
  - UTF-8 with BOM
  - Columns: Mittauspisteen tunnus, Tuotteen tyyppi, Resoluutio,
    Yksikkötyyppi, Lukeman tyyppi, Alkuaika, Määrä, Laatu
  - Resoluutio is one of "PT1H" (hourly) or "PT15M" (15-minute);
    a single file can contain both (e.g. older periods were
    hourly-billed, recent periods are 15-min).
  - Alkuaika is ISO-8601 UTC with a trailing 'Z'
  - Määrä is the kWh of the period
  - Laatu in {"OK", "EST"}; "EST" means estimated reading. Default
    behaviour keeps both.

This module is **generic** — no user-specific defaults baked in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Column name aliases (the Finnish characters round-trip oddly through
# some readers; we normalise to ASCII names internally).
_RENAME_MAP = {
    "Mittauspisteen tunnus": "metering_point",
    "Tuotteen tyyppi":       "product_type",
    "Resoluutio":            "resolution",
    "Lukeman tyyppi":        "reading_type",
    "Alkuaika":              "ts_iso",
    "Laatu":                 "quality",
}


@dataclass
class FingridSeries:
    """Hourly kWh series loaded from a Datahub CSV export."""

    timestamps: pd.DatetimeIndex   # UTC, hourly buckets
    kwh:        np.ndarray         # kWh consumed/exported in that hour
    source:     str                # original file basename
    span_days:  float


def load_hourly(
    path: str | Path,
    *,
    include_estimated: bool = True,
) -> FingridSeries:
    """Load a Datahub CSV and return one kWh value per UTC hour.

    Quarter-hourly rows are summed into their hour bucket; hourly
    rows are passed through. Quality-flag handling is configurable
    (EST rows kept by default since the alternative is gaps).
    """
    p = Path(path)
    df = pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig")
    # Reader sometimes mangles the Yksikkötyyppi / Määrä column
    # names because of the BOM or codec. Find by position.
    if "Maara" in df.columns:
        amount_col = "Maara"
    elif "Määrä" in df.columns:
        amount_col = "Määrä"
    else:
        # Penultimate column is always the amount.
        amount_col = df.columns[-2]

    df = df.rename(columns=_RENAME_MAP).rename(columns={amount_col: "kwh"})

    if not include_estimated:
        df = df[df["quality"] == "OK"]

    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True)

    # Aggregate to hourly buckets. PT15M rows sum within an hour;
    # PT1H rows already represent an hour.
    df["hour_bucket"] = df["ts"].dt.floor("h")
    agg = df.groupby("hour_bucket")["kwh"].sum()
    agg.index.name = None
    agg = agg.sort_index()

    span_days = (agg.index[-1] - agg.index[0]).total_seconds() / 86400.0
    return FingridSeries(
        timestamps=agg.index,
        kwh=agg.values.astype(float),
        source=p.name,
        span_days=span_days,
    )


def total_household_demand(
    grid_import: FingridSeries,
    grid_export: FingridSeries | None,
    pv_total_kwh: pd.Series | None,
    pv_install_date_utc: pd.Timestamp | None = None,
) -> FingridSeries:
    """Reconstruct total household demand from metered grid flows.

    pre-PV-install:
        total_demand = grid_import

    post-PV-install:
        total_demand = grid_import + (pv_total - grid_export)
                     = grid_import + pv_self_consumed

    Parameters
    ----------
    grid_import : FingridSeries
        ``home_consumption.csv`` — BN01 reading from grid meter.
    grid_export : FingridSeries or None
        ``PV_production.csv`` if available; if None, post-PV demand
        is approximated as grid_import + pv_total (assumes nothing
        exported, only valid if user really self-consumes everything).
    pv_total_kwh : pandas.Series indexed by UTC hourly timestamp, or
        None. Caller's estimate of total PV produced (e.g. from
        irradiance via pv_estimate). Required when grid_export is
        present so we can compute pv_self_consumed.
    pv_install_date_utc : pandas.Timestamp or None
        Date PV came online. Inferred from grid_export.timestamps[0]
        if None and grid_export is provided.
    """
    grid_import_df = pd.DataFrame({"grid_import": grid_import.kwh},
                                    index=grid_import.timestamps)

    if grid_export is None:
        return FingridSeries(
            timestamps=grid_import_df.index,
            kwh=grid_import_df["grid_import"].values,
            source=grid_import.source,
            span_days=grid_import.span_days,
        )

    # Join grid_import and grid_export on common hourly index.
    export_df = pd.DataFrame({"grid_export": grid_export.kwh},
                              index=grid_export.timestamps)
    joined = grid_import_df.join(export_df, how="left").fillna(
        {"grid_export": 0.0}
    )

    if pv_install_date_utc is None and len(grid_export.timestamps) > 0:
        pv_install_date_utc = grid_export.timestamps[0]

    if pv_total_kwh is None:
        # Approximation: assume zero export, total = grid_import alone.
        # In practice this is wrong post-PV-install.
        joined["total"] = joined["grid_import"]
    else:
        pv_total_df = pd.DataFrame({"pv_total": pv_total_kwh.values},
                                     index=pv_total_kwh.index)
        joined = joined.join(pv_total_df, how="left").fillna(
            {"pv_total": 0.0}
        )
        pv_self = np.maximum(
            joined["pv_total"].values - joined["grid_export"].values,
            0.0,
        )
        # Pre-install: total = grid_import; post-install: + self-consumed.
        post = np.asarray(joined.index >= pv_install_date_utc)
        joined["total"] = joined["grid_import"].values + np.where(
            post, pv_self, 0.0
        )

    span_days = (joined.index[-1] - joined.index[0]).total_seconds() / 86400.0
    return FingridSeries(
        timestamps=joined.index,
        kwh=joined["total"].values,
        source=(grid_import.source + " + " + grid_export.source),
        span_days=span_days,
    )


def hour_of_week_shape_from_series(
    timestamps: pd.DatetimeIndex,
    kwh: np.ndarray,
    local_tz: str = "Europe/Helsinki",
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Same contract as `lib_ha_db.hour_of_week_shape`, but takes raw arrays.

    Returns ``(shape[7×24], mean_kwh_per_hour, sigma[7×24], n_obs)``.
    Shape is normalised to overall mean = 1.0.
    """
    local = timestamps.tz_convert(local_tz)
    weekday = local.weekday.values
    hour    = local.hour.values

    sum_ = np.zeros((7, 24), dtype=float)
    sq_  = np.zeros((7, 24), dtype=float)
    cnt  = np.zeros((7, 24), dtype=int)

    valid = np.isfinite(kwh)
    for wd, h, v in zip(weekday[valid], hour[valid], kwh[valid]):
        sum_[wd, h] += v
        sq_[wd, h]  += v * v
        cnt[wd, h]  += 1

    mean_cell = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    var_cell = np.where(
        cnt > 1,
        (sq_ - cnt * mean_cell ** 2) / np.maximum(cnt - 1, 1),
        0.0,
    )
    sigma_cell = np.sqrt(np.maximum(var_cell, 0.0))

    overall_mean = float(np.nanmean(mean_cell))
    shape = mean_cell / overall_mean if overall_mean > 0 else mean_cell

    return shape, overall_mean, sigma_cell, int(valid.sum())


def monthly_factor_from_series(
    timestamps: pd.DatetimeIndex,
    kwh: np.ndarray,
    local_tz: str = "Europe/Helsinki",
) -> tuple[np.ndarray, np.ndarray]:
    """12-element monthly multiplier vector + observation counts."""
    local = timestamps.tz_convert(local_tz)
    month = local.month.values

    sum_ = np.zeros(12)
    cnt = np.zeros(12, dtype=int)
    valid = np.isfinite(kwh)
    for m, v in zip(month[valid], kwh[valid]):
        sum_[m - 1] += v
        cnt[m - 1] += 1

    mean_per_month = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    overall = float(np.nanmean(mean_per_month))
    factor = mean_per_month / overall if overall > 0 else mean_per_month
    return factor, cnt
