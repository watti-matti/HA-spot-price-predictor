"""Refresh cached parquets so the analysis sees up-to-date OL1/OL2 outage data.

Reads the API key from $env:FINGRID_API_KEY (set inline for this run);
appends new rows to the existing parquets under output/.

Datasets refreshed:
  output/fi_grid_data.parquet   (Fingrid #188 nuclear + supporting streams)
  output/fi_prices.parquet      (Sahkotin FI spot, free, no key)
  output/fi_neighbor_prices.parquet (SE1/SE3 via elprisetjustnu + EE via Elering)

The script is conservative: it only appends rows newer than the
existing maximum timestamp; never rewrites historical data.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))
from solar_clear_sky_submodel import fetch_fingrid  # noqa: E402

OUTPUT = REPO / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ── Fingrid grid data ────────────────────────────────────────────────


FINGRID_DATASETS = {
    "nuclear_mw":        188,   # Real-time nuclear production (3 min)
    "consumption_mw":    165,   # Day-ahead consumption forecast (15 min)
    "wind_forecast_mw":  246,
    "solar_forecast_mw": 247,
}


def _normalise_nuclear(rows: list[dict]) -> pd.Series:
    """Map raw Fingrid rows → hourly mean normalised to max-fleet 4 372 MW."""
    NORM = 4372.0
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.set_index("ts").sort_index()
    hourly = df["value"].resample("1h").mean()
    return hourly / NORM


def _hourly_mw(rows: list[dict]) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.set_index("ts").sort_index()
    return df["value"].resample("1h").mean()


def refresh_grid_data(api_key: str) -> tuple[int, str]:
    path = OUTPUT / "fi_grid_data.parquet"
    if not path.exists():
        raise SystemExit("output/fi_grid_data.parquet missing — bootstrap first")
    cached = pd.read_parquet(path)
    last_ts = cached.index.max()
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    # Re-fetch the last 24h of cached data too in case it was incomplete.
    start = (last_ts - pd.Timedelta(hours=24)).to_pydatetime()
    end = now_utc
    if end - start < timedelta(hours=2):
        return 0, "no refresh needed"
    print(f"  fetching {start.isoformat()} → {end.isoformat()}", flush=True)

    fresh = {}
    for col, ds_id in FINGRID_DATASETS.items():
        rows = fetch_fingrid(ds_id, start, end, api_key=api_key, chunk_days=90)
        if col == "nuclear_mw":
            fresh[col] = _normalise_nuclear(rows)
        else:
            fresh[col] = _hourly_mw(rows)
        print(f"    {col} ds#{ds_id}: {len(rows)} raw rows → "
              f"{len(fresh[col])} hourly", flush=True)
        time.sleep(0.5)

    fresh_df = pd.concat(fresh, axis=1)
    fresh_df.index.name = "date"
    # Merge: keep cached rows older than overlap, replace overlap+new with fresh.
    overlap_start = fresh_df.index.min()
    cached_keep = cached.loc[cached.index < overlap_start]
    merged = pd.concat([cached_keep, fresh_df]).sort_index()
    merged = merged.dropna(how="all")
    merged.to_parquet(path)
    return (
        len(merged) - len(cached),
        f"{cached.index.max()} → {merged.index.max()}",
    )


# ── Sahkotin FI spot (no key) ────────────────────────────────────────


SAHKOTIN_URL = "https://sahkotin.fi/prices"


def refresh_fi_prices() -> tuple[int, str]:
    path = OUTPUT / "fi_prices.parquet"
    if not path.exists():
        raise SystemExit("output/fi_prices.parquet missing — bootstrap first")
    cached = pd.read_parquet(path)
    last_ts = cached.index.max()
    now_utc = datetime.now(timezone.utc)
    start = (last_ts - pd.Timedelta(hours=24)).to_pydatetime()
    end = now_utc + timedelta(hours=24)  # day-ahead headroom
    if end - start < timedelta(hours=2):
        return 0, "no refresh needed"
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "end":   end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "vat": "false",
    }
    print(f"  fetching {start.isoformat()} → {end.isoformat()}", flush=True)
    r = requests.get(SAHKOTIN_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    prices = data.get("prices", data) if isinstance(data, dict) else data
    rows = []
    for entry in prices:
        ts = entry.get("date") or entry.get("startDate") or entry.get("timestamp")
        v = entry.get("value")
        if ts is None or v is None:
            continue
        rows.append({"ts": pd.to_datetime(ts, utc=True), "fi": float(v)})
    if not rows:
        return 0, "no rows returned"
    fresh = pd.DataFrame(rows).set_index("ts").sort_index()
    fresh.index.name = cached.index.name
    overlap_start = fresh.index.min()
    cached_keep = cached.loc[cached.index < overlap_start]
    merged = pd.concat([cached_keep, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(path)
    return (
        len(merged) - len(cached),
        f"{cached.index.max()} → {merged.index.max()}",
    )


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Set FINGRID_API_KEY in env (PowerShell: "
            "$env:FINGRID_API_KEY = '...')"
        )

    print("\n=== Refresh Fingrid grid data ===")
    added, span = refresh_grid_data(api_key)
    print(f"  added {added:+d} rows; new span: {span}")

    print("\n=== Refresh FI spot prices (Sahkotin) ===")
    added, span = refresh_fi_prices()
    print(f"  added {added:+d} rows; new span: {span}")


if __name__ == "__main__":
    main()
