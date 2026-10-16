"""Build one consolidated 4-year hourly dataset for local modelling.

Why
---
The inputs are spread across five parquets with three different index
resolutions and two different gap conventions, and one of them
(`fi_weather_sites`) exists only because it was rescued out of a
gitignored cache. Every study re-derives the join, and `build_dataframe`
inner-joins then `dropna()`s, which silently discarded 207 analysis
hours before the step-fill was added.

This writes a single aligned table so modelling starts from one artifact
instead of reassembling it.

CAUSAL ADMISSIBILITY — read before using any column as a price feature
----------------------------------------------------------------------
Nord Pool clears on EXPECTATIONS. Bidders submit curves built from
forecasts of weather, demand and generation, and the price is the
outcome of those expectations. A REALISED value is therefore not a
better version of the right input — it is an input from the wrong side
of the event, and regressing price on it is anti-causal.

Only a minority of the available Fingrid series are ex-ante:

  ex-ante (usable as features)   246 wind fc, 247 solar fc, 165 cons fc,
                                 267/268 capacities, 112/115 EE DA
                                 capacity (the last two ENDED 2026-10-02)
  ex-post (diagnostics only)     188 nuclear, 75 wind actual, 124 cons
                                 actual, 55/57/60/61 flows, 38/39/44/45
                                 intraday capacity ("free capacity AFTER
                                 the day-ahead market" — post-auction by
                                 definition)

Every column carries a `causality` tag in the sidecar metadata.

Leak-freeness and market-relevance are DIFFERENT properties: a realised
value lagged 168 h is perfectly observable at the forecast origin, yet
still is not what anyone bid on for the target hour. That is why the
lagged neighbour prices correlate only 0.39, and why the lagged
post-auction capacity series fitted +1.397 where physics requires a
negative sign.

One measured exception worth knowing. For SLOW variables the lagged
realised value is a good proxy under normal conditions and fails mainly
during disturbances. Nuclear persists at corr 0.774 over 168 h, and a
week-old value predicts the current one to within 3.9 % of level in
normal hours (corr 0.900) against 16.3 % when disturbed (corr 0.523),
disturbed being 29.5 % of hours. Cross-border capacity does NOT share
that property as published: free capacity and realised flow are both
fast (0.385 and 0.396 at 168 h), so the slow physical capacity beneath
them is simply not observable now that the day-ahead capacity series
(24/25/26/27) have been discontinued.

Run:  python studies/build_modelling_dataset.py
      python studies/build_modelling_dataset.py --years 4
      python studies/build_modelling_dataset.py --start 2022-10-17
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "studies", REPO / "custom_components" / "spot_price_predictor"):
    sys.path.insert(0, str(p))

STORE = REPO / "data_store"
OUT = STORE / "modelling_dataset_4y.parquet"
META = STORE / "modelling_dataset_4y.meta.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EX_ANTE = {
    "wind_forecast_mw", "solar_forecast_mw", "consumption_mw",
    "solar_capacity_mw", "wind_capacity_mw",
    "ntc_ee_fi_da_mw", "ntc_fi_ee_da_mw",
}
EX_POST = {
    "nuclear_mw", "wind_actual_mw", "consumption_actual_mw",
    "flow_fi_se1_mw", "flow_fi_se3_mw", "flow_fi_ee_mw", "flow_fi_no_mw",
    "ntc_se1_fi_mw", "ntc_fi_se1_mw", "ntc_se3_fi_mw", "ntc_fi_se3_mw",
}
# Step series: a reporting gap carries no information, so carry forward.
STEP_FILL = ("solar_capacity_mw", "wind_capacity_mw")
# Short-gap interpolation only; longest observed runs are 15 h and 24 h.
INTERP_FILL = ("nuclear_mw", "consumption_mw", "consumption_actual_mw",
               "wind_actual_mw")
INTERP_LIMIT = 3

SOURCES = ("fi_prices", "fi_neighbor_prices", "fi_weather",
           "fi_weather_sites", "fi_grid_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--start", default=None,
                    help="explicit UTC start; overrides --years")
    a = ap.parse_args()

    frames = {}
    for name in SOURCES:
        f = STORE / f"{name}.parquet"
        if not f.exists():
            print(f"ERROR: {f.relative_to(REPO)} missing.\n"
                  "       Run `python -m src.data_store update` with "
                  "FINGRID_API_KEY set, then\n"
                  "       `python studies/consolidate_weather_store.py`.")
            return 2
        frames[name] = pd.read_parquet(f)

    # Neighbour prices went 15-minute on 2025-09-30, so take the hourly
    # MEAN rather than a sample: the :45 quarter differs from :00 by
    # sd 17-23 EUR/MWh, which is pure noise against a trainer that joins
    # on :00 (the v2.18.0 coordinator fix).
    nb = frames["fi_neighbor_prices"].resample("1h").mean()

    end = min(frames["fi_prices"].index.max(), nb.index.max(),
              frames["fi_weather"].index.max(),
              frames["fi_grid_data"].index.max())
    start = (pd.Timestamp(a.start, tz="UTC") if a.start
             else end - pd.Timedelta(days=365.25 * a.years))
    # Every source begins at its own first publication. Starting the
    # spine before that produces a block of all-NaN leading rows (25 h,
    # measured) that looks like missing data but is just a window edge.
    first = max(frames["fi_prices"].index.min(), nb.index.min(),
                frames["fi_weather"].index.min(),
                frames["fi_grid_data"].index.min())
    start = max(start, first)
    idx = pd.date_range(start.ceil("h"), end.floor("h"), freq="1h", tz="UTC")
    print(f"  spine {idx[0]} .. {idx[-1]}  ({len(idx):,} hourly rows)")

    out = pd.DataFrame(index=idx)
    meta: dict[str, dict] = {}

    def add(col, s, causality, desc):
        # A bare ndarray would get a RangeIndex here and reindex to
        # all-NaN against the DatetimeIndex spine, silently. Attach the
        # spine explicitly when the caller passes positional data.
        if not isinstance(s, pd.Series):
            s = pd.Series(np.asarray(s), index=idx)
        out[col] = s.reindex(idx)
        if not out[col].notna().any():
            raise ValueError(
                f"column {col!r} is entirely NaN after alignment — check "
                "the source index, do not ship an empty column")
        meta[col] = {"causality": causality, "description": desc}

    add("fi_price_eur_mwh", frames["fi_prices"].iloc[:, 0], "target",
        "FI day-ahead spot price, EUR/MWh")

    for z in ("se1", "se3", "ee"):
        add(f"{z}_price_eur_mwh", nb[z], "ex-post",
            f"{z.upper()} cleared price — clears in the SAME auction as "
            "FI, so same-hour use leaks the target; admissible only lagged")

    for c in frames["fi_weather"].columns:
        add(c, frames["fi_weather"][c], "ex-ante",
            "Open-Meteo forecast-as-issued, capacity-weighted aggregate")

    for c in frames["fi_weather_sites"].columns:
        var, site = c.split("__", 1)
        add(c, frames["fi_weather_sites"][c], "ex-ante",
            f"Open-Meteo {var} at {site}, forecast-as-issued")

    grid = frames["fi_grid_data"]
    for c in grid.columns:
        s = grid[c]
        if c in STEP_FILL:
            s = s.ffill().bfill()
        elif c in INTERP_FILL:
            s = s.interpolate(limit=INTERP_LIMIT)
        add(c, s,
            "ex-ante" if c in EX_ANTE else
            "ex-post" if c in EX_POST else "unclassified",
            "Fingrid series — see the module docstring for causality")

    # Calendar on the LOCAL calendar: the v2.18.0 fix, because a UTC
    # weekday was wrong for 2.93 % of hours against a +3.29 EUR/MWh
    # coefficient.
    from holidays import build_holiday_set
    loc = idx.tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                       for d in loc])
    add("is_holiday", pd.Series(is_hol, index=idx), "ex-ante",
        "Finnish public holiday, local calendar")
    add("is_workday", pd.Series((loc.weekday < 5) * (1.0 - is_hol), index=idx),
        "ex-ante", "weekday and not a public holiday, local calendar")
    add("local_hour", pd.Series(loc.hour, index=idx), "ex-ante",
        "hour of day, Europe/Helsinki")
    add("local_dow", pd.Series(loc.weekday, index=idx), "ex-ante",
        "day of week, Europe/Helsinki, 0=Mon")

    # Derived physics: reuse the shipped transforms rather than
    # re-deriving them, so the dataset matches what the model sees.
    try:
        from exp_extra_features import sigmoid_turbine_rho, solar_effective
        add("sigmoid_wind_rho",
            sigmoid_turbine_rho(out["wind_speed_weighted"].values,
                                out["temperature_weighted"].values),
            "ex-ante",
            "sigmoid turbine curve x air density, shipped transform")
        add("solar_effective",
            solar_effective(out["solar_irradiance_weighted"].values,
                            out["temperature_weighted"].values),
            "ex-ante",
            "irradiance x cell-temperature derate, shipped transform")
    except Exception as e:
        print(f"  note: derived physics skipped ({type(e).__name__}: {e})")

    # A DISCONTINUED series is the most dangerous thing in here: it
    # looks like a normal ex-ante column but is NaN for every future
    # hour, so a model trained on it silently loses that input in
    # production. Fingrid ended the FI<->EE day-ahead capacity series on
    # 2026-10-02 ("no updates after 29.9.2026"), and the FI<->SE
    # day-ahead series on 2024-10-29. Demote any column whose last valid
    # value is more than 7 days before the end of the spine.
    STALE_DAYS = 7
    for c in list(out.columns):
        lv = out[c].last_valid_index()
        if lv is None or meta[c]["causality"] != "ex-ante":
            continue
        if (out.index[-1] - lv) > pd.Timedelta(days=STALE_DAYS):
            meta[c]["causality"] = "ex-ante-ended"
            meta[c]["ends_at"] = str(lv)
            meta[c]["description"] += (
                f" — DISCONTINUED after {lv.date()}; excluded from "
                "feature_columns because it is NaN for all later hours")
            print(f"  DEMOTED (series ended {lv.date()}): {c}")

    # The default modelling feature set: ex-ante, still publishing.
    feature_columns = [c for c, v in meta.items()
                       if v["causality"] == "ex-ante"]
    target_column = next(c for c, v in meta.items()
                         if v["causality"] == "target")

    cov = out.notna().mean()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    counts = {k: sum(1 for v in meta.values() if v["causality"] == k)
              for k in ("target", "ex-ante", "ex-post", "unclassified")}
    META.write_text(json.dumps({
        "rows": int(len(out)), "columns": int(len(out.columns)),
        "start": str(out.index[0]), "end": str(out.index[-1]),
        "causality_counts": counts,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "complete_rows_on_features": int(
            out[feature_columns + [target_column]].notna().all(axis=1).sum()),
        "coverage": {c: round(float(cov[c]), 6) for c in out.columns},
        "columns_meta": meta,
    }, indent=2), encoding="utf-8")

    print(f"  wrote {OUT.relative_to(REPO)}  "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"        {len(out):,} rows x {len(out.columns)} columns")
    print(f"  causality: {counts}")
    ft = feature_columns + [target_column]
    ok = int(out[ft].notna().all(axis=1).sum())
    print(f"  modelling set: {len(feature_columns)} features + target, "
          f"{ok:,}/{len(out):,} complete rows ({100*ok/len(out):.2f}%)")
    thin = cov[cov < 0.99].sort_values()
    if len(thin):
        print("  columns under 99 % coverage:")
        for c, v in thin.items():
            print(f"    {c:46s} {100 * v:6.2f}%")
    else:
        print("  every column at or above 99 % coverage")
    print(f"  metadata -> {META.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
