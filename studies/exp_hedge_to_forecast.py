"""Do the hedge-analysis winners actually improve the FORECAST?

The comprehensive hedge analysis raised the admissible ceiling to
R^2 0.537 and ranked the candidates. That is CONTEMPORANEOUS,
IN-SAMPLE explanatory power on deseasonalised price. It is not forecast
skill, and three things break the translation:

1. HORIZON. The best candidate, residual load squared (33.8 %), is built
   from Fingrid 165/246/247 — all DAY-AHEAD ONLY. Over a 170 h horizon
   they cover roughly the first 36 h. The hedge regression silently
   assumed they were known for every hour.

2. REDUNDANCY WITH WHAT EXISTS. `fi_lag168_7dmean` adds +9.52 pp of R^2,
   but the shipped model already tracks the level through its seasonal
   layer and a 3-day per-hour bias EMA. Incremental R^2 against a bare
   deseasonalised target overstates what it adds to a model that
   already has those.

3. PRIOR NEGATIVE RESULTS. docs/BACKLOG.md R5 measured same-hour net
   load at +18.92 but LAGGED net load at +1.58, falling to -0.56 once a
   lagged price was present. Residual load is the same quantity. The
   hedge analysis cannot see that because it never lags anything.

So each winner is rebuilt the way it would actually have to ship, and
scored out-of-sample on the production protocol: origin 06:00 UTC,
170 h horizon, L1 and L2 refit monthly on data strictly before the
scored month, identical rows across variants, per-lead-day reported.

Variants
--------
  BASE          shipped nine features
  RL_D1         + residual load (level and squared) from the Fingrid
                day-ahead forecasts, faded past the publication boundary
                with the 48 h half-life leadtime_fill_study measured
  RL_PROXY      + residual load rebuilt from FORECASTABLE components:
                wind and solar MW projected from the weather proxies
                (fitted on train only), consumption from a seasonal +
                lag-168 offset model. Available for the full horizon.
  FI_7D         + origin-anchored 7-day mean of own deseasonalised
                price, held across the horizon (leak-free: uses only
                hours at or before the origin)
  COMBO         RL_PROXY + FI_7D

Run:  python studies/exp_hedge_to_forecast.py
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
from lag_utils import lag_by_hours                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")
HORIZON = 170
ORIGIN_HOUR = 6
DA_HOURS = 42            # Fingrid day-ahead reach
FADE_HL = 48.0           # measured in leadtime_fill_study
VARIANTS = ("BASE", "RL_D1", "RL_PROXY", "FI_7D", "COMBO")


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
    cf = df["consumption_mw"].values.astype(float)
    wf = df["wind_forecast_mw"].values.astype(float)
    sf = df["solar_forecast_mw"].values.astype(float)

    rows: list[pd.DataFrame] = []
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
        lag_fi = np.nan_to_num(lag_by_hours(Yfi, idx, 168))

        Yz, Zlag = {}, {}
        for z in ("temp", "se1", "se3", "ee"):
            shipped = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(
                df[z].values[tr], ts[tr],
                depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                            if k in shipped))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        for z in ("se1", "se3", "ee"):
            Zlag[z] = np.nan_to_num(lag_by_hours(Yz[z], idx, 168))

        pw = sd.fit_components(wind_rho[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        ps = sd.fit_components(solar_eff[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        Yw = wind_rho - sd.compute_seasonal_part(ts, pw)
        Ys = solar_eff - sd.compute_seasonal_part(ts, ps)

        # --- residual load, two constructions -----------------------
        # (a) straight from the Fingrid day-ahead forecasts
        rl_da = cf - wf - sf
        crl = sd.fit_components(np.nan_to_num(rl_da, nan=np.nanmean(rl_da)),
                                ts, depth=("P_hour", "P_day", "P_week"))
        Yrl_da = (rl_da - sd.compute_seasonal_part(ts, crl)) / 1e3   # GW

        # (b) rebuilt from forecastable components, fitted on TRAIN only
        def proj(target, proxy):
            ok = tr & np.isfinite(target) & np.isfinite(proxy)
            A = np.column_stack([np.ones(ok.sum()), proxy[ok]])
            b = np.linalg.lstsq(A, target[ok], rcond=None)[0]
            return b[0] + b[1] * proxy
        wind_mw = proj(wf, Yw)
        solar_mw = proj(sf, Ys)
        # consumption: seasonal + lag-168 level offset (3.6 % MAPE measured)
        ccons = sd.fit_components(np.nan_to_num(cf, nan=np.nanmean(cf)), ts,
                                  depth=("P_hour", "P_day", "P_week"))
        cons_seas = sd.compute_seasonal_part(ts, ccons)
        cons_res = cf - cons_seas
        cons_mw = cons_seas + np.nan_to_num(lag_by_hours(cons_res, idx, 168))
        rl_px = cons_mw - wind_mw - solar_mw
        Yrl_px = (rl_px - sd.compute_seasonal_part(ts, crl)) / 1e3

        base = [np.ones(n), lag_fi, is_wd, Yw, Ys, Yz["temp"],
                Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol]
        variants = {
            "BASE":     base,
            "RL_D1":    base + [Yrl_da, Yrl_da ** 2],
            "RL_PROXY": base + [Yrl_px, Yrl_px ** 2],
            "FI_7D":    base + [np.zeros(n)],      # filled per origin
            "COMBO":    base + [Yrl_px, Yrl_px ** 2, np.zeros(n)],
        }
        # 7-day trailing mean of own deseasonalised price, as of each hour
        fi7 = pd.Series(Yfi, index=idx).rolling(168, min_periods=24).mean().values

        fitted = {}
        for k, cols in variants.items():
            X = np.column_stack(cols)
            if k in ("FI_7D", "COMBO"):
                X[:, -1] = np.nan_to_num(fi7)      # train on the realised mean
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            # Residual load must raise the price: scarcity moves you up a
            # convex supply stack. Constrain both the level and the
            # squared term to >= 0 so a refit cannot invert the physics,
            # which is how the cross-border capacity feature failed.
            low = {}
            if k == "RL_D1":
                low = {10: 0.0, 11: 0.0}
            elif k in ("RL_PROXY", "COMBO"):
                low = {10: 0.0, 11: 0.0}
            fitted[k] = (sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0},
                              lower=low), X)

        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D",
                               tz="UTC"):
            o = o.replace(hour=ORIGIN_HOUR)
            if o not in pos:
                continue
            i0 = pos[o]
            sl = np.arange(i0 + 1, i0 + 1 + HORIZON)
            if sl[-1] >= n:
                continue
            lead = np.arange(1, HORIZON + 1)
            fade = np.where(lead <= DA_HOURS, 1.0,
                            0.5 ** ((lead - DA_HOURS) / FADE_HL))

            for k, (c, X) in fitted.items():
                Xi = X[sl].copy()
                if k == "RL_D1":
                    # Fingrid DA reaches ~42 h; fade beyond rather than
                    # pretend it is known for the whole horizon.
                    Xi[:, 10] *= fade
                    Xi[:, 11] *= fade
                if k in ("FI_7D", "COMBO"):
                    # origin-anchored: the mean as of the origin, held
                    # across the horizon. Leak-free at every lead.
                    Xi[:, -1] = np.nan_to_num(fi7[i0])
                pr = np.maximum(seas[sl] + Xi @ c, -5.0)
                if not np.isfinite(pr).all():
                    continue
                rows.append(pd.DataFrame({
                    "v": k, "err": pr - fi[sl], "act": fi[sl],
                    "lead_d": np.minimum((lead - 1) // 24 + 1, 8),
                    "month": pd.DatetimeIndex(idx[sl]).tz_convert(
                        "Europe/Helsinki").strftime("%Y-%m")}))

    r = pd.concat(rows, ignore_index=True)
    base_mae = r[r.v == "BASE"].err.abs().mean()
    p90 = r[r.v == "BASE"].act.quantile(0.90)
    out: dict = {"scored_hours": int((r.v == "BASE").sum()), "variants": {}}

    print("=" * 80)
    print("HEDGE WINNERS, SCORED AS FORECASTS (origin 06:00 UTC, 170 h)")
    print(f"eval {r.month.min()} .. {r.month.max()}   "
          f"{int((r.v=='BASE').sum()):,} scored hours per variant")
    print("=" * 80)
    print(f"\n  {'variant':10s} {'MAE':>7s} {'bias':>7s} {'p90+':>8s} "
          f"{'vs BASE':>9s}")
    for k in VARIANTS:
        g = r[r.v == k]
        mae = g.err.abs().mean()
        rec = {"mae": float(mae), "bias": float(g.err.mean()),
               "high_price_mae": float(g.err[g.act >= p90].abs().mean()),
               "vs_base_pct": float(100 * (base_mae - mae) / base_mae)}
        out["variants"][k] = rec
        print(f"  {k:10s} {mae:7.2f} {rec['bias']:+7.2f} "
              f"{rec['high_price_mae']:8.2f} {rec['vs_base_pct']:+8.2f}%")

    print(f"\n  by lead day (MAE):")
    print(f"  {'variant':10s}" + "".join(f"{f'D+{d}':>8s}" for d in range(1, 9)))
    for k in VARIANTS:
        g = r[r.v == k]
        v = g.groupby("lead_d").err.apply(lambda s: s.abs().mean())
        print(f"  {k:10s}" + "".join(f"{v.get(d, np.nan):8.2f}"
                                     for d in range(1, 9)))
    out["by_lead"] = {k: {int(d): float(v) for d, v in
                          r[r.v == k].groupby("lead_d").err.apply(
                              lambda s: s.abs().mean()).items()}
                      for k in VARIANTS}

    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS / "exp_hedge_to_forecast.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
