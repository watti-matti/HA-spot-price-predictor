"""Extract a privacy-respecting household profile from an HA recorder DB.

Reads a user-supplied DB path and a sensor mapping (CLI or
auto-detection), produces a JSON profile with shape-only statistics
per the schema in `docs/household_profile_schema.md`. No raw
timestamps, no absolute energy totals.

Usage::

    python studies/extract_household_profile.py \\
        --db /path/to/home-assistant_v2.db \\
        --total-consumption-sensor sensor.net_power_use \\
        --pv-sensor sensor.pv_all_power \\
        --out studies/_private/household_profile.json

The `--out` default lives under `studies/_private/`, which is
gitignored. Public defaults must NOT be derived from any individual
extraction; the synthetic fallback at
`studies/sim_household_default_profile.json` is the only profile
shipped with the public release.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))

from lib_ha_db import (  # noqa: E402
    connect_readonly, hourly_kwh_series, hour_of_week_shape,
    monthly_consumption_factor, list_consumption_candidates,
)


# ── Privacy validation ───────────────────────────────────────────────


# A full timestamp like "2026-05-20T18:30:00" is a leak (gives the
# exact hour of an event). A date-only string "2026-05-20" is fine
# under extraction_metadata where we explicitly record the window
# bounds and extraction date.
_FULL_TIMESTAMP_RX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _validate_profile_no_leaks(profile: dict[str, Any]) -> None:
    """Refuse to write a profile that contains raw timestamps or
    suspiciously large absolute totals.
    """
    def _walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            if _FULL_TIMESTAMP_RX.match(node):
                raise ValueError(
                    f"profile field {path!r} contains a full timestamp "
                    f"({node!r}); only date-only strings (YYYY-MM-DD) "
                    f"are allowed and only under extraction_metadata"
                )

    _walk(profile)

    # Sanity check on absolute totals — annual consumption should be
    # a plausible household range, not >200 MWh.
    annual = profile.get("derived_annual_kwh_estimate")
    if annual is not None and annual > 200_000:
        raise ValueError(
            f"derived_annual_kwh_estimate {annual} exceeds 200 MWh; "
            f"likely a unit error or a raw kWh leak"
        )


# ── Extraction ───────────────────────────────────────────────────────


def extract(
    db_path: Path,
    total_consumption_sensor: str,
    pv_sensor: str | None,
    climate_zone: str,
) -> dict[str, Any]:
    con = connect_readonly(db_path)

    total_series = hourly_kwh_series(con, total_consumption_sensor)
    shape, mean_kwh, sigma, n_obs = hour_of_week_shape(total_series)
    monthly_factor, monthly_n = monthly_consumption_factor(total_series)

    pv_summary: dict[str, Any] | None = None
    if pv_sensor:
        try:
            pv_series = hourly_kwh_series(con, pv_sensor)
            pv_shape, pv_mean, _, pv_n = hour_of_week_shape(pv_series)
            pv_summary = {
                "shape_hour_weekday": _round_2d(pv_shape, decimals=3),
                "mean_kwh_per_hour": round(pv_mean, 4),
                "n_obs_hours": int(pv_n),
            }
        except (KeyError, ValueError):
            pv_summary = None

    # Hourly observation count — for the report only, no raw values.
    extraction_metadata = {
        "schema_version": "1.0",
        "source": "ha_recorder",
        "extraction_window_days": float(
            (total_series.timestamps[-1] - total_series.timestamps[0])
            .total_seconds() / 86400
        ),
        "extraction_window_n_hours": int(len(total_series.timestamps)),
        "window_iso_date_only": (
            total_series.timestamps[0].date().isoformat()
            + " to "
            + total_series.timestamps[-1].date().isoformat()
        ),
        "climate_zone": climate_zone,
        "total_consumption_sensor": total_consumption_sensor,
        "pv_sensor": pv_sensor,
        "extracted_at_iso_date": datetime.now(timezone.utc).date().isoformat(),
        "warning_seasonal_coverage": (
            "Spring observations only" if extraction_metadata_spring(total_series)
            else None
        ),
    }

    # Annual estimate from hourly mean × 8760. Documented as
    # extrapolation, not measurement.
    annual_estimate_kwh = round(mean_kwh * 8760.0, 1)

    profile: dict[str, Any] = {
        "extraction_metadata": extraction_metadata,
        "baseload": {
            "mean_kwh_per_hour": round(mean_kwh, 4),
            "shape_hour_weekday": _round_2d(shape, decimals=3),
            "sigma_hour_weekday_kwh": _round_2d(sigma, decimals=4),
            "monthly_factor": _round_1d(monthly_factor, decimals=3),
            "monthly_obs_count": [int(x) for x in monthly_n.tolist()],
        },
        "derived_annual_kwh_estimate": annual_estimate_kwh,
        "pv_summary": pv_summary,
    }

    _validate_profile_no_leaks(profile)
    return profile


def extraction_metadata_spring(series) -> bool:
    months = {ts.month for ts in series.timestamps}
    return months.issubset({3, 4, 5, 6})


def _round_2d(arr: np.ndarray, decimals: int) -> list[list[float]]:
    return [
        [round(float(v), decimals) if np.isfinite(v) else None
         for v in row]
        for row in arr
    ]


def _round_1d(arr: np.ndarray, decimals: int) -> list[float | None]:
    return [
        round(float(v), decimals) if np.isfinite(v) else None
        for v in arr
    ]


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--total-consumption-sensor", required=True)
    ap.add_argument("--pv-sensor", default=None)
    ap.add_argument("--climate-zone", default="FI_south")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "studies" / "_private" / "household_profile.json",
    )
    ap.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print likely consumption / PV sensors and exit",
    )
    args = ap.parse_args()

    if args.list_candidates:
        con = connect_readonly(args.db)
        for sid, unit in list_consumption_candidates(con):
            print(f"  {sid:55s} unit={unit!r}")
        return 0

    profile = extract(
        args.db,
        args.total_consumption_sensor,
        args.pv_sensor,
        args.climate_zone,
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if "_private" not in out.parts:
        print(
            f"WARNING: output {out} is not under studies/_private/. The "
            f"privacy contract requires household profiles to live in the "
            f"gitignored directory. Proceeding anyway because you asked.",
            file=sys.stderr,
        )
    with out.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)

    meta = profile["extraction_metadata"]
    base = profile["baseload"]
    print(f"Wrote {out}")
    print(f"  window:      {meta['window_iso_date_only']} "
          f"({meta['extraction_window_days']:.1f} days, "
          f"{meta['extraction_window_n_hours']} hourly observations)")
    print(f"  mean kWh/h:  {base['mean_kwh_per_hour']:.4f}  "
          f"(~{base['mean_kwh_per_hour'] * 24:.1f} kWh/day, "
          f"~{profile['derived_annual_kwh_estimate']:.0f} kWh/year extrapolated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
