"""Does retraining on fresh data help, and where on the timeline?

Compares the SHIPPED frozen artifact against a walk-forward refit of the
same nine-feature structure, scored on identical rows in a single run —
which is the only valid way to compare two models here, because the
harness eval window ends wherever the data store ends (see
docs/TRAINING_BASELINE.md).

  FROZEN   `backtest_harness.build_production_prediction` — the deployed
           pipeline, replayed exactly (frozen L1, frozen L2 coefficients
           intercept-first, neighbours lagged 168 h, `Y_fi_lag168`
           zeroed, physics deseasonalised with the artifact's own
           `physics_seasonal`, softplus floor). Pinned to
           `Pipeline.compute_forecast` at <1e-6 EUR/MWh by
           tests/test_harness_production_parity.py.
  REFIT    same structure, L1 and L2 refit each month on data strictly
           before that month. This is the "always freshly retrained"
           upper bound, not a prediction of what one retrain today
           yields.

Two structural changes motivate the timeline view, and both are
DERIVED from the data rather than assumed — see `--aurora` / `--pv-ramp`
for the defaults and how to override them:

  Aurora Line (FI<->SE1 transfer capacity).  corr(FI, SE1) steps from
  0.343 in 2025-10 to 0.885 in 2025-11, and mean |FI-SE1| falls 36.26
  -> 14.41 -> 7.04. Winter 2026 re-widens (congestion returns under
  load) before the regime locks in from 2026-05 at corr 0.94-0.98 and
  spread 1.5-6.7 EUR/MWh.

  PV build-out.  Installed capacity +126.5 MW (2026-05), +186.2
  (2026-06), +103.2 (2026-07), reaching 2073.8 MW.

CONFOUND, stated plainly: the sustained Aurora regime and the PV ramp
both begin in 2026-05. Any error change from that month cannot be
attributed to one or the other from this data alone. The monthly table
prints both drivers beside the errors so the reader can see the overlap
rather than be told a single-cause story.

The model has NO transmission or capacity variable, so neither change is
representable in it. That is the point of looking: if refitting recovers
the Aurora months, the coefficients were absorbing the old regime; if it
does not, the model cannot express the change at all and needs a
structural term.

Run:
  python studies/exp_fresh_data_timeline.py
  python studies/exp_fresh_data_timeline.py --aurora 2026-05-01
  python studies/exp_fresh_data_timeline.py --eval-start 2025-01-01
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

import seasonal_decomposition as sd                     # noqa: E402
from holidays import build_holiday_set                  # noqa: E402
from exp_extra_features import build_dataframe, SEASONAL_ARTIFACT  # noqa: E402
from honest_horizon_study import sfit                   # noqa: E402
from backtest_harness import build_production_prediction, _snapshot_id  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
# Derived from data_store: see the module docstring.
AURORA_DEFAULT = "2025-11-01"
PV_RAMP_DEFAULT = "2026-05-01"
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-start", default="2025-07-01",
                    help="first month scored (default 2025-07-01)")
    ap.add_argument("--aurora", default=AURORA_DEFAULT,
                    help=f"Aurora Line step date (default {AURORA_DEFAULT}, "
                         "derived from the corr(FI,SE1) step)")
    ap.add_argument("--pv-ramp", default=PV_RAMP_DEFAULT,
                    help=f"PV build-out start (default {PV_RAMP_DEFAULT}, "
                         "derived from solar_capacity_mw jumps)")
    a = ap.parse_args()
    eval_start = pd.Timestamp(a.eval_start, tz="UTC")
    aurora = pd.Timestamp(a.aurora, tz="UTC")
    pv_ramp = pd.Timestamp(a.pv_ramp, tz="UTC")

    df = build_dataframe()
    idx = df.index
    n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values

    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                       for d in loc])
    is_wd = (loc.weekday < 5).astype(float) * (1.0 - is_hol)

    fi = df["fi"].values
    wind_rho = df["sigmoid_wind_rho"].values
    solar_eff = df["solar_effective"].values

    print("Building FROZEN (deployed artifact replay)…", flush=True)
    frozen = build_production_prediction(df)

    print("Building REFIT (monthly walk-forward)…", flush=True)
    refit = np.full(n, np.nan)
    months = pd.date_range(eval_start, idx[-1], freq="MS", tz="UTC")
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        blk = np.asarray((idx >= m0) & (idx < m1))
        if tr.sum() < 8000 or blk.sum() == 0:
            continue

        comp_fi = sd.fit_components(
            fi[tr], ts[tr],
            depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                        if k in SEASONAL_ARTIFACT["components"]["fi"]))
        seas = sd.compute_seasonal_part(ts, comp_fi)
        Yfi = fi - seas
        lag_fi = np.nan_to_num(pd.Series(Yfi, index=idx).shift(168).values)

        Yz, Zlag = {}, {}
        for z in ("temp", "se1", "se3", "ee"):
            shipped = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(
                df[z].values[tr], ts[tr],
                depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                            if k in shipped))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        for z in ("se1", "se3", "ee"):
            Zlag[z] = np.nan_to_num(
                pd.Series(Yz[z], index=idx).shift(168).values)

        pw = sd.fit_components(wind_rho[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        ps = sd.fit_components(solar_eff[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        Yw = wind_rho - sd.compute_seasonal_part(ts, pw)
        Ys = solar_eff - sd.compute_seasonal_part(ts, ps)

        X = np.column_stack([np.ones(n), lag_fi, is_wd, Yw, Ys, Yz["temp"],
                             Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol])
        ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
        coef = sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0})
        refit[blk] = np.maximum(seas[blk] + X[blk] @ coef, -5.0)

    # Score on IDENTICAL rows only.
    ev = (np.asarray(idx >= eval_start) & np.isfinite(frozen)
          & np.isfinite(refit) & np.isfinite(fi))
    print(f"  scored {ev.sum():,} hours on identical rows\n")

    e_fr, e_re = frozen[ev] - fi[ev], refit[ev] - fi[ev]
    act = fi[ev]
    mo = pd.DatetimeIndex(idx[ev]).tz_convert("Europe/Helsinki").strftime("%Y-%m")

    # Drivers, per month, so cause sits beside effect.
    se1 = df["se1"].values[ev]
    cap_s = (pd.Series(df["solar_capacity_mw"].values, index=idx)
             .ffill().bfill().values[ev])

    out: dict = {"snapshot_id": _snapshot_id(), "scored_hours": int(ev.sum()),
                 "eval_start": str(eval_start), "aurora": str(aurora),
                 "pv_ramp": str(pv_ramp), "monthly": {}, "events": {}}

    print("=" * 92)
    print("FROZEN (deployed) vs REFIT (walk-forward) — identical rows, one run")
    print(f"snapshot {out['snapshot_id']}   Aurora {a.aurora}   "
          f"PV ramp {a.pv_ramp}")
    print("=" * 92)
    print(f"\n  {'month':8s} {'n':>5s} {'price':>7s} | {'FROZEN':>7s} "
          f"{'bias':>7s} | {'REFIT':>7s} {'bias':>7s} | {'ΔMAE':>7s} "
          f"{'ΔMAE%':>7s} | {'|FI-SE1|':>8s} {'PVcap':>7s}")
    for m in sorted(set(mo)):
        k = mo == m
        if k.sum() < 100:
            continue
        f_mae, r_mae = np.abs(e_fr[k]).mean(), np.abs(e_re[k]).mean()
        rec = {"n": int(k.sum()), "mean_price": float(act[k].mean()),
               "frozen_mae": float(f_mae), "frozen_bias": float(e_fr[k].mean()),
               "refit_mae": float(r_mae), "refit_bias": float(e_re[k].mean()),
               "delta_mae": float(r_mae - f_mae),
               "delta_mae_pct": float(100.0 * (f_mae - r_mae) / f_mae),
               "abs_fi_se1": float(np.abs(act[k] - se1[k]).mean()),
               "pv_capacity_mw": float(cap_s[k].mean())}
        out["monthly"][m] = rec
        mark = ""
        if m == str(aurora)[:7]:
            mark = " <- Aurora"
        if m == str(pv_ramp)[:7]:
            mark = " <- PV ramp"
        print(f"  {m:8s} {rec['n']:5d} {rec['mean_price']:7.1f} | "
              f"{f_mae:7.2f} {rec['frozen_bias']:+7.2f} | "
              f"{r_mae:7.2f} {rec['refit_bias']:+7.2f} | "
              f"{rec['delta_mae']:+7.2f} {rec['delta_mae_pct']:+7.1f} | "
              f"{rec['abs_fi_se1']:8.2f} {rec['pv_capacity_mw']:7.0f}{mark}")

    def seg(mask: np.ndarray, label: str) -> dict:
        if mask.sum() < 100:
            return {}
        f_mae, r_mae = np.abs(e_fr[mask]).mean(), np.abs(e_re[mask]).mean()
        rec = {"n": int(mask.sum()), "mean_price": float(act[mask].mean()),
               "frozen_mae": float(f_mae),
               "frozen_bias": float(e_fr[mask].mean()),
               "refit_mae": float(r_mae),
               "refit_bias": float(e_re[mask].mean()),
               "delta_mae_pct": float(100.0 * (f_mae - r_mae) / f_mae)}
        out["events"][label] = rec
        print(f"  {label:26s} {rec['n']:6d} {rec['mean_price']:7.1f} | "
              f"{f_mae:7.2f} {rec['frozen_bias']:+7.2f} | "
              f"{r_mae:7.2f} {rec['refit_bias']:+7.2f} | "
              f"{rec['delta_mae_pct']:+6.1f}%")
        return rec

    t = pd.DatetimeIndex(idx[ev])
    print(f"\n  {'segment':26s} {'n':>6s} {'price':>7s} | {'FROZEN':>7s} "
          f"{'bias':>7s} | {'REFIT':>7s} {'bias':>7s} | {'refit':>7s}")
    seg(np.asarray(t < aurora), "pre-Aurora")
    seg(np.asarray((t >= aurora) & (t < pv_ramp)), "Aurora, pre-PV-ramp")
    seg(np.asarray(t >= pv_ramp), "post-PV-ramp (both)")
    hp = act >= np.percentile(act, 90)
    seg(hp, "top price decile (all)")
    seg(hp & np.asarray(t >= pv_ramp), "top decile, post-PV-ramp")

    print("\n  CONFOUND: the sustained Aurora regime and the PV ramp both")
    print("  begin 2026-05, so 'post-PV-ramp' is also 'Aurora settled'.")
    print("  Neither effect is separable from this data alone, and the")
    print("  model has no transmission or capacity variable to express")
    print("  either. Read the |FI-SE1| and PVcap columns together with the")
    print("  errors rather than attributing a single cause.")
    print("\n  REFIT is the always-freshly-retrained upper bound, not the")
    print("  gain from one retrain today. Compare only within this run —")
    print("  see docs/TRAINING_BASELINE.md.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "exp_fresh_data_timeline.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
