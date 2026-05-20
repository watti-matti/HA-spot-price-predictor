"""Build a household profile from Fingrid Datahub CSV exports.

Complements `extract_household_profile.py` (which reads HA recorder
DB). The Datahub CSVs give multi-year hourly history at the cost of
sensor-level granularity: we get *grid import* and *grid export*
totals, not per-load breakdown.

Strategy
--------
1.  Load `home_consumption.csv` (grid import, full window).
2.  Load `PV_production.csv` (grid export — really PV surplus to grid)
    if present, and infer PV install date from its start.
3.  For the pre-install window, total household demand equals
    grid import. Use this period to compute the **monthly seasonal
    factor** across all 12 months (the missing piece in a
    spring-only HA-DB extraction).
4.  Optionally combine with the post-install period by adding back
    estimated self-consumed PV from cached irradiance via
    `pv_estimate.estimate_pv_kwh_per_hour`. The post-install shape
    is the **EMHASS-optimised** consumption signature; the
    pre-install shape is the **natural** signature.

Outputs
-------
The script writes a profile JSON to `studies/_private/` per the
schema in `docs/household_profile_schema.md`. By default, two
profiles are written:

  household_profile_pre_pv.json  — natural (Dec 2022 - early Sep 2023)
  household_profile_post_pv.json — EMHASS-shaped (Sep 2023 onwards)

Plus, the monthly_factor in the canonical
`household_profile.json` is overwritten with the pre-PV data
(since it covers all 12 months), while the shape_hour_weekday is
kept from the (post-EMHASS) HA-DB extraction if present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studies"))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from lib_fingrid_csv import (  # noqa: E402
    load_hourly, total_household_demand,
    hour_of_week_shape_from_series, monthly_factor_from_series,
)
from pv_estimate import estimate_pv_kwh_per_hour  # noqa: E402


PV_KWP = 8.91
PV_TILT_DEG = 45.0
PV_AZIMUTH_DEG = 160.0
PV_EFFICIENCY = 0.85


# ── PV estimate from cached irradiance for the post-install window ──


def estimate_pv_total_kwh_per_hour(
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    weather = pd.read_parquet(REPO / "output" / "fi_weather.parquet")
    weather = weather.reindex(timestamps).ffill(limit=2)
    irr = weather["solar_irradiance_weighted"].values
    pv = np.array([
        estimate_pv_kwh_per_hour(
            float(g) if np.isfinite(g) else 0.0,
            capacity_kwp=PV_KWP, tilt_deg=PV_TILT_DEG,
            azimuth_deg=PV_AZIMUTH_DEG, efficiency=PV_EFFICIENCY,
        )
        for g in irr
    ])
    return pd.Series(pv, index=timestamps)


# ── Profile assembly ─────────────────────────────────────────────────


def assemble_profile(
    series_timestamps: pd.DatetimeIndex,
    series_kwh: np.ndarray,
    source_label: str,
) -> dict:
    shape, mean_kwh, sigma, n_obs = hour_of_week_shape_from_series(
        series_timestamps, series_kwh
    )
    monthly_factor, monthly_n = monthly_factor_from_series(
        series_timestamps, series_kwh
    )
    span_days = (series_timestamps[-1] - series_timestamps[0]).total_seconds() / 86400
    return {
        "extraction_metadata": {
            "schema_version": "1.0",
            "source":         source_label,
            "extraction_window_days":   round(span_days, 2),
            "extraction_window_n_hours": int(n_obs),
            "window_iso_date_only": (
                series_timestamps[0].date().isoformat()
                + " to "
                + series_timestamps[-1].date().isoformat()
            ),
            "climate_zone": "FI_south",
            "months_observed": sorted(
                {int(m) for m in series_timestamps.tz_convert(
                    "Europe/Helsinki").month}
            ),
        },
        "baseload": {
            "mean_kwh_per_hour": round(float(mean_kwh), 4),
            "shape_hour_weekday": [
                [round(float(v), 3) if np.isfinite(v) else None for v in row]
                for row in shape
            ],
            "sigma_hour_weekday_kwh": [
                [round(float(v), 4) if np.isfinite(v) else None for v in row]
                for row in sigma
            ],
            "monthly_factor": [
                round(float(v), 3) if np.isfinite(v) else None
                for v in monthly_factor
            ],
            "monthly_obs_count": [int(x) for x in monthly_n.tolist()],
        },
        "derived_annual_kwh_estimate":
            round(float(mean_kwh) * 8760, 1),
    }


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fingrid-dir", required=True, type=Path,
                     help="Directory containing home_consumption.csv etc.")
    ap.add_argument("--out-pre-pv", type=Path,
                     default=REPO / "studies" / "_private"
                                  / "household_profile_pre_pv.json")
    ap.add_argument("--out-post-pv", type=Path,
                     default=REPO / "studies" / "_private"
                                  / "household_profile_post_pv.json")
    ap.add_argument("--out-combined", type=Path,
                     default=REPO / "studies" / "_private"
                                  / "household_profile_combined.json")
    args = ap.parse_args()

    grid_import = load_hourly(args.fingrid_dir / "home_consumption.csv")
    grid_export_path = args.fingrid_dir / "PV_production.csv"
    grid_export = (load_hourly(grid_export_path)
                    if grid_export_path.exists() else None)

    if grid_export is not None:
        pv_install_date = grid_export.timestamps[0]
        print(f"PV install inferred: {pv_install_date}")
    else:
        pv_install_date = None
        print("No PV export file found; treating all data as pre-PV.")

    print(f"Grid import: {grid_import.span_days:.1f} days, "
          f"{len(grid_import.timestamps)} hourly buckets, "
          f"total {grid_import.kwh.sum():.0f} kWh")

    # PRE-PV PROFILE -----------------------------------------------------
    if pv_install_date is not None:
        mask = grid_import.timestamps < pv_install_date
        pre_ts = grid_import.timestamps[mask]
        pre_kwh = grid_import.kwh[mask]
        pre_profile = assemble_profile(
            pre_ts, pre_kwh,
            source_label="fingrid_datahub_pre_pv_install",
        )
        args.out_pre_pv.parent.mkdir(parents=True, exist_ok=True)
        args.out_pre_pv.write_text(json.dumps(pre_profile, indent=2),
                                     encoding="utf-8")
        print(f"  pre-PV profile:  "
              f"{pre_profile['extraction_metadata']['window_iso_date_only']}, "
              f"mean {pre_profile['baseload']['mean_kwh_per_hour']:.3f} kWh/h, "
              f"months observed: "
              f"{pre_profile['extraction_metadata']['months_observed']}")
    else:
        pre_profile = None

    # POST-PV PROFILE (reconstructed total demand) ----------------------
    if grid_export is not None:
        post_mask = grid_import.timestamps >= pv_install_date
        post_ts = grid_import.timestamps[post_mask]
        # PV reconstruction needs irradiance — only available where the
        # cached weather parquet covers.
        pv_total = estimate_pv_total_kwh_per_hour(post_ts)
        total_series = total_household_demand(
            grid_import, grid_export, pv_total, pv_install_date,
        )
        post_total_kwh = pd.Series(
            total_series.kwh, index=total_series.timestamps
        ).reindex(post_ts).fillna(0).values
        post_profile = assemble_profile(
            post_ts, post_total_kwh,
            source_label=(
                "fingrid_datahub_post_pv_install"
                " + irradiance-reconstructed PV self-consumption"
            ),
        )
        args.out_post_pv.write_text(json.dumps(post_profile, indent=2),
                                      encoding="utf-8")
        print(f"  post-PV profile: "
              f"{post_profile['extraction_metadata']['window_iso_date_only']}, "
              f"mean {post_profile['baseload']['mean_kwh_per_hour']:.3f} kWh/h, "
              f"months observed: "
              f"{post_profile['extraction_metadata']['months_observed']}")
    else:
        post_profile = None

    # COMBINED PROFILE: monthly_factor from pre-PV (12-month coverage), -
    # shape_hour_weekday from post-PV (EMHASS-optimised signature).
    if pre_profile and post_profile:
        combined = {
            "extraction_metadata": {
                **post_profile["extraction_metadata"],
                "source": "combined: pre-PV monthly_factor + post-PV shape",
                "monthly_factor_source": pre_profile["extraction_metadata"]["source"],
                "shape_source":          post_profile["extraction_metadata"]["source"],
            },
            "baseload": {
                "mean_kwh_per_hour":   post_profile["baseload"]["mean_kwh_per_hour"],
                "shape_hour_weekday":  post_profile["baseload"]["shape_hour_weekday"],
                "sigma_hour_weekday_kwh": post_profile["baseload"]["sigma_hour_weekday_kwh"],
                "monthly_factor":      pre_profile["baseload"]["monthly_factor"],
                "monthly_obs_count":   pre_profile["baseload"]["monthly_obs_count"],
            },
            "derived_annual_kwh_estimate": post_profile["derived_annual_kwh_estimate"],
        }
        args.out_combined.write_text(json.dumps(combined, indent=2),
                                       encoding="utf-8")
        print(f"  combined profile -> {args.out_combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
