"""Should the wind/PV price features be scaled by installed capacity?

The problem
-----------
The L2 ridge sees `Y_sigmoid_wind_rho` and `Y_solar_effective`, which
are *weather* proxies — roughly capacity factors, bounded and
dimensionless. Price responds to delivered MW, and

    MW  ~=  installed_capacity  x  capacity_factor(weather)

so with capacity left out, the fitted coefficient has to absorb the
fleet size. The fleet is not stationary: PV went 1575 -> 2074 MW
between 2026-04 and 2026-09 as industrial plants commissioned, +32 % in
five months.

A coefficient fitted across 2023-2026 is therefore an average over
fleet sizes, and it is stale the moment the fleet grows again. That
shows up as the model's price sensitivity drifting away from the
market's: dP/dPV decayed -48.9 -> -28.6 EUR/MWh per GW across years
while the market's stayed near -41.

The fix under test: multiply each proxy by its installed capacity
BEFORE deseasonalising, so the regressor is in GW and the coefficient
is EUR/MWh per GW — a quantity that should be stationary, because the
capacity factor is (wind 0.740/0.769/0.720/0.715; PV
0.000787/0.000847/0.000840/0.000817 by year).

Two things are measured, and the second matters more
----------------------------------------------------
1. MAE/bias, overall and by segment. Expected to be small.
2. COEFFICIENT STABILITY across years. This is the actual argument:
   capacity scaling is drift immunity, not an accuracy trick. If the
   per-GW coefficient is stable under scaling and drifting without it,
   the change pays every time the fleet grows — including for the
   data-centre load growth that has not happened yet.

Variants
--------
  BASE       shipped nine features
  CAP_PV     solar proxy  x solar capacity (GW)
  CAP_WIND   wind proxy   x wind capacity (GW)
  CAP_BOTH   both

Protocol identical to exp_fingrid_dayahead_channel: origin 06:00 UTC
daily, 170 h horizon, L1 and L2 refit monthly on data strictly before
the scored month, neighbours lagged 168 h, sign constraints on the
zero-marginal-cost columns, softplus floor.

Run:  python studies/exp_capacity_scaling.py
"""
from __future__ import annotations

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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")
HORIZON = 170
ORIGIN_HOUR = 6
VARIANTS = ("BASE", "CAP_PV", "CAP_WIND", "CAP_BOTH", "CAP_NORM")

PHYS_DEPTH = ("P_hour", "P_week")
PHYS_SMOOTH = {"P_week": 7}


def _deseason(x: np.ndarray, ts: np.ndarray, tr: np.ndarray) -> np.ndarray:
    comp = sd.fit_components(x[tr], ts[tr], depth=PHYS_DEPTH,
                             smooth=PHYS_SMOOTH)
    return x - sd.compute_seasonal_part(ts, comp)


def main() -> None:
    df = build_dataframe()
    idx = df.index
    n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    pos = {t: i for i, t in enumerate(idx)}

    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                       for d in loc])
    is_wd = (loc.weekday < 5).astype(float) * (1.0 - is_hol)

    fi = df["fi"].values
    wind_rho = df["sigmoid_wind_rho"].values
    solar_eff = df["solar_effective"].values

    # Installed capacity in GW. Step series with occasional reporting
    # gaps; carry them across rather than interpolating a step.
    cap_s = (pd.Series(df["solar_capacity_mw"].values, index=idx)
             .ffill().bfill().values / 1e3)
    cap_w = (pd.Series(df["wind_capacity_mw"].values, index=idx)
             .ffill().bfill().values / 1e3)
    print(f"capacity GW: solar {cap_s.min():.2f}->{cap_s.max():.2f}   "
          f"wind {cap_w.min():.2f}->{cap_w.max():.2f}")

    # Capacity-scaled proxies: GW of delivered output, not a factor.
    pv_gw = cap_s * solar_eff
    wd_gw = cap_w * wind_rho

    # CAP_NORM isolates the question that actually matters. Multiplying
    # by capacity in GW inflates the wind column ~9x (5.2 -> 9.55 GW)
    # while ridge alpha stays at 1.0, so that column gets far less
    # effective regularisation than BASE's — which on its own shrinks
    # the fitted coefficient by about the same factor and makes the
    # drift comparison meaningless. Dividing by the mean capacity keeps
    # the column on BASE's scale, so the ONLY difference left is that
    # the regressor now tracks the fleet over time. That is the
    # hypothesis; the rest was an artifact of my parameterisation.
    pv_norm = pv_gw / cap_s.mean()
    wd_norm = wd_gw / cap_w.mean()

    rows: list[pd.DataFrame] = []
    coef_rows: list[dict] = []
    months = pd.date_range(EVAL_START, idx[-1], freq="MS", tz="UTC")
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

        Yz = {}
        for z in ("temp", "se1", "se3", "ee"):
            shipped = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(
                df[z].values[tr], ts[tr],
                depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                            if k in shipped))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        Zlag = {z: np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
                for z in ("se1", "se3", "ee")}

        Yw_base = _deseason(wind_rho, ts, tr)
        Ys_base = _deseason(solar_eff, ts, tr)
        Yw_cap = _deseason(wd_gw, ts, tr)
        Ys_cap = _deseason(pv_gw, ts, tr)

        cols = {
            "BASE":     (Yw_base, Ys_base),
            "CAP_PV":   (Yw_base, Ys_cap),
            "CAP_WIND": (Yw_cap,  Ys_base),
            "CAP_BOTH": (Yw_cap,  Ys_cap),
            "CAP_NORM": (_deseason(wd_norm, ts, tr),
                         _deseason(pv_norm, ts, tr)),
        }

        fitted = {}
        for k, (Yw, Ys) in cols.items():
            X = np.column_stack([np.ones(n), lag_fi, is_wd, Yw, Ys,
                                 Yz["temp"], Zlag["se1"], Zlag["se3"],
                                 Zlag["ee"], is_hol])
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            # indices 3, 4 are the zero-marginal-cost columns
            fitted[k] = (sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0}), X)

        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D",
                               tz="UTC"):
            o = o.replace(hour=ORIGIN_HOUR)
            if o not in pos:
                continue
            i0 = pos[o]
            sl = np.arange(i0 + 1, i0 + 1 + HORIZON)
            if sl[-1] >= n:
                continue
            for k, (c, X) in fitted.items():
                pr = np.maximum(seas[sl] + X[sl] @ c, -5.0)
                if not np.isfinite(pr).all():
                    continue
                rows.append(pd.DataFrame({
                    "v": k, "err": pr - fi[sl], "act": fi[sl],
                    "month": pd.DatetimeIndex(idx[sl]).tz_convert(
                        "Europe/Helsinki").strftime("%Y-%m"),
                }))

    # ── per-YEAR refits, to see whether the coefficient drifts ────────
    for year in sorted({d.year for d in loc}):
        yr = np.asarray((loc.year == year))
        if yr.sum() < 3000:
            continue
        comp_fi = sd.fit_components(fi[yr], ts[yr],
                                    depth=("P_hour", "P_day", "P_week"))
        Yfi_y = fi - sd.compute_seasonal_part(ts, comp_fi)
        lag_y = np.nan_to_num(pd.Series(Yfi_y, index=idx).shift(168).values)
        Yzy = {}
        for z in ("temp", "se1", "se3", "ee"):
            c = sd.fit_components(df[z].values[yr], ts[yr],
                                  depth=("P_hour", "P_day", "P_week"))
            Yzy[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        Zly = {z: np.nan_to_num(pd.Series(Yzy[z], index=idx).shift(168).values)
               for z in ("se1", "se3", "ee")}
        for k, (wsrc, ssrc) in (("BASE", (wind_rho, solar_eff)),
                                ("CAP_BOTH", (wd_gw, pv_gw)),
                                ("CAP_NORM", (wd_norm, pv_norm))):
            Yw = _deseason(wsrc, ts, yr)
            Ys = _deseason(ssrc, ts, yr)
            X = np.column_stack([np.ones(n), lag_y, is_wd, Yw, Ys,
                                 Yzy["temp"], Zly["se1"], Zly["se3"],
                                 Zly["ee"], is_hol])
            ok = yr & np.isfinite(X).all(1) & np.isfinite(Yfi_y)
            c = sfit(X[ok], Yfi_y[ok], 1.0, {3: 0.0, 4: 0.0})
            coef_rows.append({"year": int(year), "variant": k,
                              "wind": float(c[3]), "solar": float(c[4]),
                              "mean_solar_cap_gw": float(cap_s[yr].mean()),
                              "mean_wind_cap_gw": float(cap_w[yr].mean())})

    r = pd.concat(rows, ignore_index=True)
    p90 = r[r.v == "BASE"].act.quantile(0.90)
    base_mae = r[r.v == "BASE"].err.abs().mean()
    out: dict = {"eval": [r.month.min(), r.month.max()],
                 "scored_hours": int((r.v == "BASE").sum()),
                 "p90_price": float(p90), "variants": {},
                 "per_year_coefficients": coef_rows}

    print("=" * 78)
    print("Capacity scaling — origin 06:00 UTC, 170 h horizon")
    print(f"eval {r.month.min()} .. {r.month.max()}   "
          f"{int((r.v=='BASE').sum()):,} scored hours per variant")
    print("=" * 78)
    print(f"\n  {'variant':10s} {'MAE':>7s} {'bias':>7s} {'summer':>7s} "
          f"{'winter':>7s} {'p90+':>8s} {'vs BASE':>8s}")
    for k in VARIANTS:
        g = r[r.v == k]
        mo = g.month.str[-2:].astype(int)
        summer = g[mo.isin([6, 7, 8])].err.abs().mean()
        winter = g[mo.isin([12, 1, 2])].err.abs().mean()
        high = g[g.act >= p90].err.abs().mean()
        mae = g.err.abs().mean()
        rec = {"mae": float(mae), "bias": float(g.err.mean()),
               "summer_mae": float(summer), "winter_mae": float(winter),
               "high_price_mae": float(high),
               "vs_base_pct": float(100.0 * (base_mae - mae) / base_mae)}
        out["variants"][k] = rec
        print(f"  {k:10s} {mae:7.2f} {rec['bias']:7.2f} {summer:7.2f} "
              f"{winter:7.2f} {high:8.2f} {rec['vs_base_pct']:+7.2f}%")

    print(f"\n  High-price rows are actual >= {p90:.1f} EUR/MWh (top decile).")

    print("\n  COEFFICIENT DRIFT — the actual case for the change.")
    print("  BASE units are EUR/MWh per unit of a dimensionless proxy, so")
    print("  they must grow with the fleet. CAP_BOTH is EUR/MWh per GW and")
    print("  should not.\n")
    cf = pd.DataFrame(coef_rows)
    for k in ("BASE", "CAP_BOTH", "CAP_NORM"):
        sub = cf[cf.variant == k]
        print(f"  {k}")
        print(f"    {'year':6s} {'wind coef':>11s} {'solar coef':>11s} "
              f"{'PV cap GW':>10s}")
        for _, row in sub.iterrows():
            print(f"    {int(row.year):<6d} {row.wind:11.3f} "
                  f"{row.solar:11.3f} {row.mean_solar_cap_gw:10.2f}")
        for col in ("wind", "solar"):
            v = sub[col].values
            if len(v) > 1 and abs(v.mean()) > 1e-12:
                print(f"    {col} spread: min {v.min():.3f} max {v.max():.3f} "
                      f"  rel. range {abs(v.max()-v.min())/abs(v.mean()):.2f}")
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "exp_capacity_scaling.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {RESULTS / 'exp_capacity_scaling.json'}")


if __name__ == "__main__":
    main()
