"""Does cross-border transfer capacity, as a daily signal, improve the forecast?

The gap this closes
-------------------
The model has no transmission variable, so the Aurora Line (SE1-FI) is
unrepresentable. The capacity change is real and in the API: the
reconstructed SE1->FI available capacity steps from a ~640-1065 MW
monthly median (2025-06..10) to ~1670-1880 MW from 2025-12, and a
monthly walk-forward refit does NOT recover the affected months
(2025-11 -11.0%, 2026-01 -10.1%, 2026-02 -6.6%). Missing input, not a
fitting problem.

RECONSTRUCTING AVAILABLE CAPACITY
---------------------------------
The published intraday series is "free capacity after day-ahead
market" — headroom, with a median of 0.0 in most months because the
auction consumes the whole link. Adding back the realised flow recovers
what capacity actually WAS:

    available(SE1->FI) = max(0, -flow_fi_se1_mw) + ntc_se1_fi_mw

The day-ahead capacity series that would have given this directly
(24/25/26/27) were discontinued after 2024-10-29.

DAILY, NOT HOURLY, AND WHY THAT IS BETTER
-----------------------------------------
Capacity changes on commissioning and maintenance events, not by the
hour, so hourly resolution buys nothing and costs robustness: the
hourly reconstruction carries implausible 3500-3700 MW peaks (intraday
re-trading, and direction flips inside the aggregation window) that a
daily MEDIAN removes while preserving both the step and the outage
plateau. Verified on the 2026-04-07..21 service break: fifteen
consecutive days at 330-604 MW against ~1700-2000 either side, clean
onset and recovery.

This also rules out the rolling-maximum idea: a trailing max would have
reported ~2000 MW through that entire fortnight, erasing a 75 %
derating lasting two weeks — and that window is exactly when FI
decouples from SE1.

ORIGIN-ANCHORED, WHICH IS WHAT MAKES IT LEAK-FREE
-------------------------------------------------
A TARGET-relative lag cannot work here. At origin T forecasting T+k for
k up to 170 h, a feature lagged 24 h from the target sits at T+k-24,
which is in the future for any k > 24. That is the v2.16 neighbour-price
defect in a new costume. The existing features use a 168 h target lag
precisely so they stay at or before the origin.

For a slowly-varying quantity the natural form is different and
stronger: take the last COMPLETE day before the origin and hold it
across the whole horizon. That is what a forecaster actually knows, it
is leak-free by construction at every lead, and for a step function it
loses almost nothing — unlike prices, where a 168 h lag collapses
correlation from 0.785 to 0.39.

Variants
--------
  BASE        shipped nine features
  CAP_LVL     + available capacity in GW (level: captures the step)
  CAP_DERATE  + capacity / trailing-60d median (1.0 normal, <1 outage:
              captures service breaks specifically)
  CAP_BOTH    + both terms
  CAP_SE3     + SE3 level too, to test whether the gain is specific to
              the link that actually changed

No sign constraint is imposed. SE1 is usually cheaper than FI (net flow
averages -670 MW, i.e. import), so more import capacity should lower the
FI price and the coefficient should come out negative — but that is a
prediction to check, not an assumption to enforce.

Run:  python studies/exp_crossborder_capacity.py
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
from exp_extra_features import (build_dataframe, load_grid_extra,  # noqa: E402
                                SEASONAL_ARTIFACT)
from honest_horizon_study import sfit                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")
HORIZON = 170
ORIGIN_HOUR = 6
DERATE_WINDOW_D = 60
VARIANTS = ("BASE", "CAP_LVL", "CAP_DERATE", "CAP_BOTH", "CAP_SE3")
AURORA = pd.Timestamp("2025-12-01", tz="UTC")     # capacity-series step
OUTAGE = (pd.Timestamp("2026-04-07", tz="UTC"),
          pd.Timestamp("2026-04-22", tz="UTC"))  # verified service break


def daily_capacity(idx: pd.DatetimeIndex, flow: pd.Series, free: pd.Series
                   ) -> tuple[pd.Series, pd.Series]:
    """Daily-median available capacity, and its derating ratio."""
    imp = (-flow).clip(lower=0)
    rec = (imp + free)
    day = rec.resample("1D").median()
    # Trailing normal level, shifted so a day cannot define its own
    # baseline. min_periods keeps early history usable.
    normal = day.shift(1).rolling(DERATE_WINDOW_D, min_periods=14).median()
    derate = (day / normal).clip(0.0, 2.0)
    return day, derate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-start", default="2025-07-01")
    a = ap.parse_args()
    eval_start = pd.Timestamp(a.eval_start, tz="UTC")

    df = build_dataframe()
    idx = df.index
    n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    pos = {t: i for i, t in enumerate(idx)}

    xb = load_grid_extra(["flow_fi_se1_mw", "ntc_se1_fi_mw",
                          "flow_fi_se3_mw", "ntc_se3_fi_mw"], idx)
    cap1_d, der1_d = daily_capacity(idx, xb["flow_fi_se1_mw"],
                                    xb["ntc_se1_fi_mw"])
    cap3_d, _ = daily_capacity(idx, xb["flow_fi_se3_mw"],
                               xb["ntc_se3_fi_mw"])
    print(f"  SE1 daily capacity: median {cap1_d.median():.0f} MW, "
          f"{cap1_d.notna().sum()} days")
    print(f"  pre-Aurora {cap1_d[cap1_d.index < AURORA].median():.0f} -> "
          f"post {cap1_d[cap1_d.index >= AURORA].median():.0f} MW")
    ow = cap1_d[(cap1_d.index >= OUTAGE[0]) & (cap1_d.index < OUTAGE[1])]
    print(f"  2026-04 outage window median {ow.median():.0f} MW "
          f"({len(ow)} days), derate {der1_d[ow.index].median():.2f}")

    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                       for d in loc])
    is_wd = (loc.weekday < 5).astype(float) * (1.0 - is_hol)
    fi = df["fi"].values
    wind_rho = df["sigmoid_wind_rho"].values
    solar_eff = df["solar_effective"].values

    # Origin-anchored lookup: value of the last COMPLETE day before t.
    def as_of(day_series: pd.Series, default: float) -> np.ndarray:
        s = day_series.reindex(
            pd.date_range(day_series.index.min(), idx[-1].normalize(),
                          freq="1D", tz="UTC")).ffill()
        prev_day = (pd.DatetimeIndex(idx).normalize()
                    - pd.Timedelta(days=1))
        return s.reindex(prev_day).fillna(default).values

    cap1 = as_of(cap1_d, float(cap1_d.median())) / 1e3      # GW
    der1 = as_of(der1_d, 1.0)
    cap3 = as_of(cap3_d, float(cap3_d.median())) / 1e3

    rows: list[pd.DataFrame] = []
    coefs: dict[str, list[float]] = {k: [] for k in VARIANTS}
    months = pd.date_range(eval_start, idx[-1], freq="MS", tz="UTC")
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        if tr.sum() < 8000:
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

        base = [np.ones(n), lag_fi, is_wd, Yw, Ys, Yz["temp"],
                Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol]
        variants = {
            "BASE":       base,
            "CAP_LVL":    base + [cap1],
            "CAP_DERATE": base + [der1],
            "CAP_BOTH":   base + [cap1, der1],
            "CAP_SE3":    base + [cap1, der1, cap3],
        }
        fitted = {}
        for k, cols in variants.items():
            X = np.column_stack(cols)
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            c = sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0})
            fitted[k] = (c, X)
            if len(c) > 10:
                coefs[k].append([float(v) for v in c[10:]])

        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D",
                               tz="UTC"):
            o = o.replace(hour=ORIGIN_HOUR)
            if o not in pos:
                continue
            i0 = pos[o]
            sl = np.arange(i0 + 1, i0 + 1 + HORIZON)
            if sl[-1] >= n:
                continue
            # Origin-anchored: hold the origin-day capacity across the
            # whole horizon, which is what is actually known at T.
            for k, (c, X) in fitted.items():
                Xi = X[sl].copy()
                for j in range(10, Xi.shape[1]):
                    Xi[:, j] = X[i0, j]
                pr = np.maximum(seas[sl] + Xi @ c, -5.0)
                if not np.isfinite(pr).all():
                    continue
                rows.append(pd.DataFrame({
                    "v": k, "err": pr - fi[sl], "act": fi[sl],
                    "t": idx[sl],
                    "month": pd.DatetimeIndex(idx[sl]).tz_convert(
                        "Europe/Helsinki").strftime("%Y-%m")}))

    r = pd.concat(rows, ignore_index=True)
    base_mae = r[r.v == "BASE"].err.abs().mean()
    p90 = r[r.v == "BASE"].act.quantile(0.90)
    out: dict = {"scored_hours": int((r.v == "BASE").sum()),
                 "aurora": str(AURORA), "variants": {}, "coefficients": {}}

    print("\n" + "=" * 86)
    print("Cross-border capacity as a daily, origin-anchored feature")
    print(f"eval {r.month.min()} .. {r.month.max()}   "
          f"{int((r.v=='BASE').sum()):,} scored hours per variant")
    print("=" * 86)
    print(f"\n  {'variant':11s} {'MAE':>7s} {'bias':>7s} {'pre-Aur':>8s} "
          f"{'post-Aur':>9s} {'outage':>8s} {'p90+':>8s} {'vs BASE':>8s}")
    t = pd.DatetimeIndex(r["t"])
    for k in VARIANTS:
        m = r.v == k
        g, tt = r[m], t[m]
        pre = g.err[np.asarray(tt < AURORA)].abs().mean()
        post = g.err[np.asarray(tt >= AURORA)].abs().mean()
        og = g.err[np.asarray((tt >= OUTAGE[0]) & (tt < OUTAGE[1]))]
        mae = g.err.abs().mean()
        rec = {"mae": float(mae), "bias": float(g.err.mean()),
               "pre_aurora_mae": float(pre), "post_aurora_mae": float(post),
               "outage_mae": float(og.abs().mean()) if len(og) else None,
               "high_price_mae": float(g.err[g.act >= p90].abs().mean()),
               "vs_base_pct": float(100.0 * (base_mae - mae) / base_mae)}
        out["variants"][k] = rec
        om = f"{rec['outage_mae']:8.2f}" if rec["outage_mae"] else "     n/a"
        print(f"  {k:11s} {mae:7.2f} {rec['bias']:+7.2f} {pre:8.2f} "
              f"{post:9.2f} {om} {rec['high_price_mae']:8.2f} "
              f"{rec['vs_base_pct']:+7.2f}%")

    print("\n  fitted coefficients (mean over monthly refits):")
    names = {"CAP_LVL": ["cap_se1_gw"], "CAP_DERATE": ["derate_se1"],
             "CAP_BOTH": ["cap_se1_gw", "derate_se1"],
             "CAP_SE3": ["cap_se1_gw", "derate_se1", "cap_se3_gw"]}
    for k in ("CAP_LVL", "CAP_DERATE", "CAP_BOTH", "CAP_SE3"):
        if not coefs[k]:
            continue
        arr = np.array(coefs[k])
        out["coefficients"][k] = {
            nm: float(arr[:, i].mean()) for i, nm in enumerate(names[k])}
        txt = "  ".join(f"{nm} {arr[:, i].mean():+8.3f}"
                        for i, nm in enumerate(names[k]))
        print(f"    {k:11s} {txt}")
    print("\n  Expected sign: SE1 is normally cheaper than FI (net flow")
    print("  -670 MW = import), so more import capacity should LOWER the FI")
    print("  price — cap_se1_gw negative. A positive fit would mean the term")
    print("  is proxying something else and should not be shipped.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "exp_crossborder_capacity.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
