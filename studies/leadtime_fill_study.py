"""Lead-time-resolved study of the day-ahead data boundary.

Question (user, 2026-08-01): the neighbour-price feed (SE1/SE3/EE) and the
Fingrid net-load feed are published day-ahead only. Beyond that boundary the
pipeline has no data. Does assuming "zero" (i.e. the climatology) for the
unknown tail introduce a systematic BIAS, and does it create an artificial
DISCONTINUITY between +2 d and +3 d?

A single aggregate number cannot answer this, so every statistic below is
resolved per lead-time day (+1 d … +7 d).

Method
------
* Forecast origins: every day at 12:00 UTC over the eval window (mimics the
  production cycle running before/after the day-ahead auction).
* At each origin the neighbour prices are treated as KNOWN through the end of
  tomorrow local time (the day-ahead auction horizon) and UNKNOWN after.
* Four strategies fill the unknown tail of the deseasonalised series Y_se:
    oracle   - the true value (upper bound; not achievable in production)
    zero     - Y = 0, i.e. fall back to the zone climatology  [CURRENT]
    persist  - Y = last observed Y (flat)
    ar1      - Y = phi^k * last observed Y, k = hours past the boundary
               (an exponential crossfade from persistence to climatology)
* Everything else is held fixed: shipped v2.16 coefficients, shipped L1,
  actual weather, true lag168. Only the fill strategy varies, so the deltas
  isolate the boundary effect.

Reports mean bias, MAE and RMSE per lead day, overall and split by season,
plus the +2 d -> +3 d step for each strategy.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO/"studies", REPO/"custom_components"/"spot_price_predictor"):
    sys.path.insert(0, str(p))
import seasonal_decomposition as sd                      # noqa: E402
from exp_extra_features import build_dataframe, SEASONAL_ARTIFACT  # noqa: E402
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HORIZON = 170
ZONES = ("se1", "se3", "ee")
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
RESULTS = REPO/"studies"/"results"; RESULTS.mkdir(parents=True, exist_ok=True)

def main() -> None:
    df = build_dataframe(); idx = df.index; ts = pd.DatetimeIndex(idx, tz="UTC").values
    art = json.loads((REPO/"custom_components"/"spot_price_predictor"/"data"
                      /"spike_model_default.json").read_text())
    coef = dict(zip(art["ridge_features"], art["ridge_coef"][1:])); b0 = art["ridge_coef"][0]
    seas = sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"]["fi"])
    Yfi = df["fi"].values - seas
    lag = pd.Series(Yfi, index=idx).shift(168).values
    Yz = {z: df[z].values - sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"][z])
          for z in ZONES}
    phi = {z: float(np.dot(Yz[z][:-1], Yz[z][1:]) / np.dot(Yz[z][:-1], Yz[z][:-1])) for z in ZONES}
    Ytemp = df["temp"].values - sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"]["temp"])
    def dc(v):
        s = pd.Series(v, index=idx); return (s - s.groupby(idx.floor("D")).transform("mean")).values
    # Everything except the neighbour block (that part never changes).
    fixed = (b0 + coef["Y_fi_lag168"]*np.nan_to_num(lag)
             + coef["is_workday"]*df["is_workday"].values
             + coef["Y_sigmoid_wind_rho"]*dc(df["sigmoid_wind_rho"].values)
             + coef["Y_solar_effective"]*dc(df["solar_effective"].values)
             + coef["Y_temp"]*Ytemp)
    pos = {t: i for i, t in enumerate(idx)}
    y = df["fi"].values
    rows = []
    origins = pd.date_range(max(EVAL_START, idx[0]+pd.Timedelta(hours=200)),
                            idx[-1]-pd.Timedelta(hours=HORIZON), freq="D", tz="UTC")
    origins = [o.replace(hour=12) for o in origins]
    for o in origins:
        if o not in pos: continue
        i0 = pos[o]
        sl = np.arange(i0+1, i0+1+HORIZON)
        if sl[-1] >= len(idx): continue
        if not np.isfinite(fixed[sl]).all(): continue
        # day-ahead boundary: known through the end of tomorrow, local time
        loc_o = o.tz_convert("Europe/Helsinki")
        boundary = (loc_o.normalize() + pd.Timedelta(days=2)).tz_convert("UTC")  # midnight after tomorrow
        known = idx[sl] < boundary
        nb = {}
        for name in ("oracle", "zero", "persist", "ar1"):
            tot = np.zeros(HORIZON)
            for z in ZONES:
                v = Yz[z][sl].copy()
                if not np.isfinite(v).all(): v = np.nan_to_num(v)
                last = v[known][-1] if known.any() else 0.0
                k = np.arange(HORIZON) - (known.sum() - 1)
                if name == "zero":     fill = np.zeros(HORIZON)
                elif name == "persist":fill = np.full(HORIZON, last)
                elif name == "ar1":    fill = (phi[z] ** np.maximum(k, 0)) * last
                else:                  fill = v
                tot += coef[f"Y_{z}"] * np.where(known, v, fill)
            nb[name] = tot
        lead_day = ((idx[sl] - o) / pd.Timedelta(hours=24)).values
        for name, contrib in nb.items():
            pred = np.maximum(0.0, seas[sl] + fixed[sl] + contrib)
            rows.append(pd.DataFrame({"strategy": name, "lead": np.ceil(lead_day).astype(int),
                                      "err": pred - y[sl],
                                      "month": pd.DatetimeIndex(idx[sl]).tz_convert("Europe/Helsinki").month}))
    r = pd.concat(rows, ignore_index=True)
    r = r[(r.lead >= 1) & (r.lead <= 7)]
    out = {"n_origins": len(origins), "phi": phi, "by_lead": {}}
    def table(sub, title):
        print(f"\n=== {title} ===")
        print(f"{'lead':>5s} | " + " | ".join(f"{s:>21s}" for s in ("zero [CURRENT]","ar1 (crossfade)","oracle")))
        print(f"{'':5s} | " + " | ".join(f"{'bias':>7s}{'MAE':>7s}{'RMSE':>7s}" for _ in range(3)))
        for L in range(1, 8):
            cells = []
            for s in ("zero", "ar1", "oracle"):
                e = sub[(sub.lead == L) & (sub.strategy == s)].err
                if len(e) == 0: cells.append(f"{'':21s}"); continue
                cells.append(f"{e.mean():+7.1f}{e.abs().mean():7.1f}{np.sqrt((e**2).mean()):7.1f}")
            print(f"  +{L}d | " + " | ".join(cells))
    table(r, "ALL SEASONS — error by lead time (EUR/MWh)")
    table(r[r.month.isin([5,6,7,8])], "SUMMER (May-Aug)")
    table(r[r.month.isin([11,12,1,2])], "WINTER (Nov-Feb)")
    print("\n=== the +2d -> +3d step (bias change across the data boundary) ===")
    for lbl, sub in (("all", r), ("summer", r[r.month.isin([5,6,7,8])]), ("winter", r[r.month.isin([11,12,1,2])])):
        line = []
        for s in ("zero", "ar1", "persist", "oracle"):
            b2 = sub[(sub.lead==2)&(sub.strategy==s)].err.mean()
            b3 = sub[(sub.lead==3)&(sub.strategy==s)].err.mean()
            line.append(f"{s}: {b2:+.1f}->{b3:+.1f} (step {b3-b2:+.1f})")
        print(f"  {lbl:7s} " + " | ".join(line))
    for L in range(1,8):
        out["by_lead"][L] = {s: {"bias": float(r[(r.lead==L)&(r.strategy==s)].err.mean()),
                                 "mae": float(r[(r.lead==L)&(r.strategy==s)].err.abs().mean())}
                             for s in ("zero","ar1","persist","oracle")}
    (RESULTS/"leadtime_fill_study.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nphi: " + ", ".join(f"{z}={phi[z]:+.3f}" for z in ZONES))
    print(f"origins={len(origins)}   wrote studies/results/leadtime_fill_study.json")

if __name__ == "__main__":
    main()
