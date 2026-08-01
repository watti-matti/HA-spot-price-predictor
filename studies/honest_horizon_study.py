"""Leak-free evaluation of the forecasting task the model actually serves.

Principle (user, 2026-08-01): "training can never use data that is not
achievable for the actual model." Every feature below is available for
EVERY forecast hour at the moment the forecast is made, and the score
covers only hours whose FI price has not yet been published.

Information set
---------------
Forecast origin: 06:00 UTC daily (09:00 Finnish, BEFORE the ~13:00 CET
day-ahead auction), so prices for D+1..D+7 are all genuinely unknown.

  available for the whole 170 h horizon
    * weather forecast (Open-Meteo, 8 days): wind, irradiance, temperature
    * calendar: hour, weekday, workday
    * FI price lagged 168 h  - for lead <= 7 d this is always in the past
    * neighbour prices lagged 168 h - same argument
    * net load PREDICTED from weather + calendar (see below)

  NOT available, and therefore excluded
    * same-hour neighbour prices - SE1/SE3/EE clear in the SAME day-ahead
      auction as FI, so observing them implies the target is published too
      (corr(FI(t), SE3(t)) = +0.82). This is the D0 leak.
    * same-hour Fingrid net load - published day-ahead only (~36 h), so it
      is replaced by a weather+calendar model over the full horizon.

Variants
--------
  A_production   the shipped v2.16 feature set, evaluated the way it
                 actually behaves in production: same-hour neighbour terms
                 are zero because they cannot be known for these hours.
  B_leakfree     same core, neighbour terms replaced by their 168 h lags.
  C_leakfree_nl  B + net load predicted from weather + calendar.

All variants are refit walk-forward (monthly) on data strictly before the
evaluated month, keep the zero-marginal-cost sign constraints, and share
the same L1 seasonal refit. Only the feature set differs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO/"studies", REPO/"custom_components"/"spot_price_predictor"):
    sys.path.insert(0, str(p))
import seasonal_decomposition as sd                                  # noqa: E402
from exp_extra_features import build_dataframe                       # noqa: E402
from backtest_harness import _fit_l1_like_shipped, PHYS_DEPTH, PHYS_SMOOTH  # noqa: E402
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

HORIZON = 170
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
RESULTS = REPO/"studies"/"results"; RESULTS.mkdir(parents=True, exist_ok=True)


def sfit(X, y, alpha=1.0, upper=None):
    """Ridge with upper bounds (intercept un-penalised)."""
    p = X.shape[1]
    A = np.vstack([X, np.sqrt(alpha)*np.eye(p)]); A[len(X), 0] = 0.0
    b = np.concatenate([y, np.zeros(p)])
    lo = np.full(p, -np.inf); hi = np.full(p, np.inf)
    for i, u in (upper or {}).items(): hi[i] = u
    return lsq_linear(A, b, bounds=(lo, hi), method="trf").x


def main() -> None:
    df = build_dataframe(); idx = df.index; n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    g = pd.read_parquet(REPO/"data_store"/"fi_grid_data.parquet")
    nl_raw = (g.consumption_mw - g.wind_forecast_mw.fillna(0)
              - g.solar_forecast_mw.fillna(0)).reindex(idx).interpolate(limit=6).values
    fi = df["fi"].values; iswd = df["is_workday"].values
    wr = df["sigmoid_wind_rho"].values; se = df["solar_effective"].values
    pos = {t: i for i, t in enumerate(idx)}
    VAR = ("A_production", "B_leakfree", "C_leakfree_nl")
    rows = []; nl_r2 = []

    for m0 in pd.date_range(EVAL_START, idx[-1], freq="MS", tz="UTC"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray(idx < m0)
        if tr.sum() < 8000: continue
        seas = sd.compute_seasonal_part(ts, _fit_l1_like_shipped("fi", fi[tr], ts[tr]))
        Yfi = fi - seas
        lag = np.nan_to_num(pd.Series(Yfi, index=idx).shift(168).values)
        Yz = {z: df[z].values - sd.compute_seasonal_part(
                  ts, _fit_l1_like_shipped(z, df[z].values[tr], ts[tr]))
              for z in ("temp", "se1", "se3", "ee")}
        Yw = wr - sd.compute_seasonal_part(
            ts, sd.fit_components(wr[tr], ts[tr], depth=PHYS_DEPTH, smooth=PHYS_SMOOTH))
        Ys = se - sd.compute_seasonal_part(
            ts, sd.fit_components(se[tr], ts[tr], depth=PHYS_DEPTH, smooth=PHYS_SMOOTH))
        nblag = [np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
                 for z in ("se1", "se3", "ee")]

        # ── net load predicted from weather + calendar (full horizon) ──
        okn = tr & np.isfinite(nl_raw)
        nlc = sd.fit_components(nl_raw[okn], ts[okn], depth=("P_hour", "P_day", "P_week"))
        Ynl = np.nan_to_num(nl_raw - sd.compute_seasonal_part(ts, nlc)) / 1000.0
        Xnl = np.column_stack([np.ones(n), Yz["temp"], Yw, Ys, Yz["temp"]*iswd, iswd])
        okf = tr & np.isfinite(Xnl).all(1) & np.isfinite(Ynl)
        cnl = sfit(Xnl[okf], Ynl[okf], alpha=1.0)
        Ynl_hat = Xnl @ cnl                       # available for every hour
        nl_r2.append(1 - np.var(Ynl[okf]-Ynl_hat[okf])/np.var(Ynl[okf]))

        core = [np.ones(n), lag, iswd, Yw, Ys, Yz["temp"]]
        feats = {
            "A_production":  core + [np.nan_to_num(Yz[z]) for z in ("se1","se3","ee")],
            "B_leakfree":    core + nblag,
            "C_leakfree_nl": core + nblag + [Ynl_hat],
        }
        coefs = {}
        for k, cols in feats.items():
            X = np.column_stack(cols)
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            coefs[k] = (sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0}), X)

        # ── forecast origins inside this month ──
        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D", tz="UTC"):
            o = o.replace(hour=6)
            if o not in pos: continue
            i0 = pos[o]; sl = np.arange(i0+1, i0+1+HORIZON)
            if sl[-1] >= n: continue
            lead = np.ceil(((idx[sl] - o) / pd.Timedelta(hours=24)).values).astype(int)
            mon = pd.DatetimeIndex(idx[sl]).tz_convert("Europe/Helsinki").month
            for k, (c, X) in coefs.items():
                Xi = X[sl].copy()
                if k == "A_production":
                    Xi[:, 6:9] = 0.0     # same-hour neighbour prices are unknowable here
                pred = np.maximum(0.0, seas[sl] + Xi @ c)
                if not np.isfinite(pred).all(): continue
                rows.append(pd.DataFrame({"v": k, "lead": lead, "err": pred - fi[sl],
                                          "month": mon}))
    r = pd.concat(rows, ignore_index=True); r = r[(r.lead >= 1) & (r.lead <= 7)]
    print(f"net-load model R^2 on its own residual: {np.mean(nl_r2):.3f}\n")
    def tbl(sub, title):
        print(f"=== {title} ===")
        print(f"{'lead':>5s} | " + " | ".join(f"{v:>18s}" for v in VAR))
        print(f"{'':5s} | " + " | ".join(f"{'MAE':>9s}{'bias':>9s}" for _ in VAR))
        for L in list(range(1, 8)) + ["all"]:
            s = sub if L == "all" else sub[sub.lead == L]
            cells = []
            for v in VAR:
                e = s[s.v == v].err
                cells.append(f"{e.abs().mean():9.2f}{e.mean():+9.2f}" if len(e) else " "*18)
            tag = "  all" if L == "all" else f"  +{L}d"
            print(f"{tag:>5s} | " + " | ".join(cells))
        print()
    tbl(r, "HONEST TASK — all seasons (EUR/MWh)")
    tbl(r[r.month.isin([5,6,7,8])], "SUMMER (May-Aug)")
    tbl(r[r.month.isin([11,12,1,2])], "WINTER (Nov-Feb)")
    out = {"net_load_r2": float(np.mean(nl_r2)), "by_lead": {}}
    for L in range(1, 8):
        out["by_lead"][L] = {v: {"mae": float(r[(r.lead==L)&(r.v==v)].err.abs().mean()),
                                 "bias": float(r[(r.lead==L)&(r.v==v)].err.mean())} for v in VAR}
    out["overall"] = {v: {"mae": float(r[r.v==v].err.abs().mean()),
                          "bias": float(r[r.v==v].err.mean())} for v in VAR}
    (RESULTS/"honest_horizon_study.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote studies/results/honest_horizon_study.json")


if __name__ == "__main__":
    main()
