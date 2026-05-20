"""Read-only HA recorder DB access for the PV-aware CVaR study.

Generic — the caller supplies the DB path and entity IDs. Never
embed user-specific defaults here. Outputs of this library that
land on disk must respect the privacy contract documented at
`docs/household_profile_schema.md`.

Usage::

    from studies.lib_ha_db import (
        connect_readonly, hourly_kwh_series, hour_of_week_shape,
    )
    con = connect_readonly("/path/to/home-assistant_v2.db")
    series = hourly_kwh_series(con, "sensor.net_power_use")
    shape, mean_kw, n_obs = hour_of_week_shape(series)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class HourlyKwhSeries:
    """Hourly mean-power observations from the recorder, as kWh."""

    timestamps: np.ndarray   # UTC datetime objects
    kwh:        np.ndarray   # energy that hour, kWh
    unit:       str          # original unit on the sensor ("W" or "kW")
    sensor_id:  str


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open a HA recorder DB read-only and forbid any write attempt."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(p)
    uri = f"file:{p}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def list_consumption_candidates(con: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return ``[(statistic_id, unit), ...]`` for likely consumption sensors."""
    patterns = (
        "%power%", "%energy%", "%consumption%", "%load%", "%grid%",
        "%pv%", "%solar%", "%kulutus%", "%charger%",
    )
    rows: list[tuple[str, str]] = []
    cur = con.cursor()
    for pat in patterns:
        cur.execute(
            "SELECT statistic_id, unit_of_measurement FROM statistics_meta "
            "WHERE statistic_id LIKE ? ORDER BY statistic_id",
            (pat,),
        )
        rows.extend(cur.fetchall())
    # Deduplicate keeping order.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for sid, unit in rows:
        if sid not in seen:
            seen.add(sid)
            out.append((sid, unit or ""))
    return out


def hourly_kwh_series(
    con: sqlite3.Connection,
    sensor_id: str,
) -> HourlyKwhSeries:
    """Read the hourly long-term statistics for a power sensor.

    HA's `statistics` table stores one row per hour with `mean`,
    `min`, `max`, `state` columns. For a power sensor (W or kW), the
    `mean` is the hourly-averaged power; we multiply by 1 hour and
    unit-correct to get hourly kWh.
    """
    cur = con.cursor()
    cur.execute(
        "SELECT id, unit_of_measurement FROM statistics_meta "
        "WHERE statistic_id=?",
        (sensor_id,),
    )
    r = cur.fetchone()
    if not r:
        raise KeyError(f"sensor {sensor_id!r} not in statistics_meta")
    metadata_id, unit = r
    unit = unit or ""

    cur.execute(
        "SELECT start_ts, mean FROM statistics WHERE metadata_id=? "
        "ORDER BY start_ts",
        (metadata_id,),
    )
    rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no statistics rows for {sensor_id!r}")

    ts = np.array(
        [datetime.fromtimestamp(r[0], tz=timezone.utc) for r in rows],
        dtype=object,
    )
    mean_power = np.array([r[1] for r in rows], dtype=float)

    # Power → energy: 1 hour of mean power, in kWh.
    if unit.lower() == "w":
        kwh = mean_power / 1000.0
    elif unit.lower() == "kw":
        kwh = mean_power.copy()
    else:
        raise ValueError(
            f"sensor {sensor_id!r} has unit {unit!r}; expected W or kW"
        )

    return HourlyKwhSeries(timestamps=ts, kwh=kwh, unit=unit,
                            sensor_id=sensor_id)


def hour_of_week_shape(
    series: HourlyKwhSeries,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Aggregate the series into a normalised hour-of-week shape.

    Returns
    -------
    shape : ndarray of shape ``(7, 24)``
        Mean-normalised consumption (overall mean = 1.0) by
        ``[weekday, hour_of_day]``. Local-time aware (uses the
        timestamp's local hour assuming the DB is stored in UTC and
        the local zone is captured by the caller; for FI this is
        UTC+2/+3).
    mean_kwh : float
        Overall hourly mean kWh across the observation window.
    sigma : ndarray of shape ``(7, 24)``
        Standard deviation per cell, kWh.
    n_obs : int
        Number of valid (non-NaN) hourly observations.
    """
    # Convert UTC timestamps to local. HA stats are UTC-stamped. We
    # use Europe/Helsinki because every sensor we ingest here is from
    # the FI bidding zone. Caller can pre-shift if a different zone
    # is needed.
    try:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo("Europe/Helsinki")
    except Exception:
        local_tz = timezone.utc  # fallback: aggregate in UTC

    sum_ = np.zeros((7, 24), dtype=float)
    cnt = np.zeros((7, 24), dtype=int)
    sq_ = np.zeros((7, 24), dtype=float)

    n_valid = 0
    for ts, kwh in zip(series.timestamps, series.kwh):
        if not np.isfinite(kwh):
            continue
        local = ts.astimezone(local_tz)
        wd = local.weekday()
        h = local.hour
        sum_[wd, h] += kwh
        sq_[wd, h] += kwh * kwh
        cnt[wd, h] += 1
        n_valid += 1

    mean_cell = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    var_cell = np.where(
        cnt > 1,
        (sq_ - cnt * mean_cell ** 2) / np.maximum(cnt - 1, 1),
        0.0,
    )
    sigma_cell = np.sqrt(np.maximum(var_cell, 0.0))

    overall_mean = float(np.nanmean(mean_cell))
    shape = mean_cell / overall_mean if overall_mean > 0 else mean_cell

    return shape, overall_mean, sigma_cell, n_valid


def monthly_consumption_factor(
    series: HourlyKwhSeries,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-month mean consumption multiplier (12 numbers).

    Returns ``(factor, n_obs_per_month)``. ``factor[m]`` is the
    month-mean kWh divided by the overall-mean kWh. Cells with no
    observations carry ``NaN``.
    """
    try:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo("Europe/Helsinki")
    except Exception:
        local_tz = timezone.utc

    sum_ = np.zeros(12)
    cnt = np.zeros(12, dtype=int)
    for ts, kwh in zip(series.timestamps, series.kwh):
        if not np.isfinite(kwh):
            continue
        local = ts.astimezone(local_tz)
        m = local.month - 1
        sum_[m] += kwh
        cnt[m] += 1

    mean_per_month = np.where(cnt > 0, sum_ / np.maximum(cnt, 1), np.nan)
    overall = float(np.nanmean(mean_per_month))
    factor = mean_per_month / overall if overall > 0 else mean_per_month
    return factor, cnt
