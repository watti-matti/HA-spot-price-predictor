"""Statistical analysis of model performance, and what explains the price.

Two parts.

PART A — model performance, properly
------------------------------------
Beyond MAE: a Theil MSE decomposition (bias / variance / covariance),
the error distribution's shape, error autocorrelation, heteroskedasticity
against the price level, and a calibration regression.

PART B — variable attribution, framed as hedging
------------------------------------------------
For each candidate variable, the hedge regression

    deseasonalised_price = alpha + beta * x + e

gives beta as the HEDGE RATIO in EUR/MWh per unit of x, and 1 - Var(e)/
Var(price) as the fraction of price variance that position removes. That
is the economically meaningful quantity: how much price risk does
exposure to x actually hedge?

TWO METHODOLOGICAL POINTS THAT CHANGE THE ANSWERS
-------------------------------------------------
1. HAC STANDARD ERRORS. Hourly prices are strongly autocorrelated
   (FI deseasonalised AR(1) phi = 0.939, half-life 11 h). Ordinary OLS
   standard errors assume independent observations and are therefore far
   too small, inflating every t-statistic. All inference here uses
   Newey-West HAC errors with a 168-hour lag window, so "significant"
   means significant after accounting for the serial dependence.

2. DESEASONALISE FIRST. Raw price and raw consumption correlate at
   R^2 = 0.296, but deseasonalised on both sides it is 0.143 — the rest
   was shared seasonality, each variable tracking the calendar rather
   than each other. Attribution on raw series credits the calendar to
   whichever fundamental happens to share its shape. Everything below is
   computed on L1-deseasonalised series.

ONLY EX-ANTE VARIABLES ARE INCLUDED. Nord Pool clears on expectations,
so a realised value is an input from the wrong side of the event.
`modelling_dataset_4y.meta.json` tags each column; this uses
`feature_columns` only.

Run:  python studies/exp_statistical_analysis.py
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

RESULTS = REPO / "studies" / "results"
STORE = REPO / "data_store"
HAC_LAGS = 168          # one week of hourly dependence

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def hac_ols(y: np.ndarray, X: np.ndarray, names: list[str]):
    """OLS with Newey-West HAC standard errors."""
    import statsmodels.api as sm
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    res = sm.OLS(y[ok], X[ok]).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return res, int(ok.sum())


def theil(pred: np.ndarray, act: np.ndarray) -> dict:
    """Theil's MSE decomposition: bias^2 + variance + covariance.

    Separates a level error (fixable by a constant) from a dispersion
    error (the forecast varying too little or too much) from the
    irreducible imperfect-correlation part.
    """
    e = pred - act
    mse = float((e ** 2).mean())
    bias2 = float(e.mean() ** 2)
    var_part = float((pred.std() - act.std()) ** 2)
    r = float(np.corrcoef(pred, act)[0, 1])
    cov_part = float(2 * (1 - r) * pred.std() * act.std())
    return {"mse": mse, "rmse": mse ** 0.5,
            "bias_sq": bias2, "variance": var_part, "covariance": cov_part,
            "bias_frac": bias2 / mse, "variance_frac": var_part / mse,
            "covariance_frac": cov_part / mse, "corr": r}


def main() -> None:
    import statsmodels.api as sm
    import seasonal_decomposition as sd
    from exp_extra_features import build_dataframe
    from backtest_harness import build_production_prediction

    out: dict = {}

    # ── Part A: model performance ───────────────────────────────────
    df = build_dataframe()
    pred = build_production_prediction(df)
    act = df["fi"].values
    ev = (np.asarray(df.index >= "2025-07-01")
          & np.isfinite(pred) & np.isfinite(act))
    p, a = pred[ev], act[ev]
    e = p - a
    n = len(a)

    print("=" * 76)
    print("PART A — model performance")
    print("=" * 76)
    print(f"  n = {n:,}   {df.index[ev][0].date()} .. {df.index[ev][-1].date()}")

    from scipy import stats as st
    desc = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean()), "sd": float(e.std()),
            "skew": float(st.skew(e)), "excess_kurtosis": float(st.kurtosis(e)),
            "q05": float(np.percentile(e, 5)), "q50": float(np.percentile(e, 50)),
            "q95": float(np.percentile(e, 95))}
    print(f"\n  error distribution")
    print(f"    MAE {desc['mae']:.3f}   RMSE {desc['rmse']:.3f}   "
          f"bias {desc['bias']:+.3f}   sd {desc['sd']:.3f}")
    print(f"    skew {desc['skew']:+.3f}   excess kurtosis "
          f"{desc['excess_kurtosis']:+.3f}")
    print(f"    quantiles  5% {desc['q05']:+.1f}   50% {desc['q50']:+.1f}"
          f"   95% {desc['q95']:+.1f}")
    print(f"    => the median error is {desc['q50']:+.1f} while the mean is "
          f"{desc['bias']:+.1f}: the distribution is")
    print(f"       {'right' if desc['skew'] > 0 else 'left'}-skewed, so MAE and "
          f"bias describe different hours.")

    th = theil(p, a)
    print(f"\n  Theil MSE decomposition (MSE {th['mse']:.1f}, "
          f"RMSE {th['rmse']:.2f})")
    print(f"    bias^2      {th['bias_sq']:9.2f}  {100*th['bias_frac']:5.2f}%  "
          "level error, removable by a constant")
    print(f"    variance    {th['variance']:9.2f}  "
          f"{100*th['variance_frac']:5.2f}%  dispersion error "
          f"(sd pred {p.std():.1f} vs actual {a.std():.1f})")
    print(f"    covariance  {th['covariance']:9.2f}  "
          f"{100*th['covariance_frac']:5.2f}%  imperfect correlation "
          f"(r = {th['corr']:.3f})")

    # calibration with HAC errors
    Xc = sm.add_constant(p)
    rc, nc = hac_ols(a, Xc, ["const", "pred"])
    print(f"\n  calibration regression  actual = a + b * predicted  (HAC, "
          f"{HAC_LAGS} lags)")
    print(f"    intercept {rc.params[0]:+8.3f}  (HAC se {rc.bse[0]:.3f})")
    print(f"    slope     {rc.params[1]:+8.3f}  (HAC se {rc.bse[1]:.3f})   "
          f"R^2 {rc.rsquared:.4f}")
    lo, hi = rc.conf_int()[1]
    print(f"    slope 95% CI [{lo:.3f}, {hi:.3f}]  "
          f"{'includes' if lo <= 1 <= hi else 'EXCLUDES'} 1.0")

    # error autocorrelation — are errors independent?
    es = pd.Series(e)
    ac = {k: float(es.autocorr(k)) for k in (1, 24, 168)}
    print(f"\n  error autocorrelation   lag1 {ac[1]:.3f}   lag24 {ac[24]:.3f}"
          f"   lag168 {ac[168]:.3f}")
    print(f"    => errors are {'strongly' if ac[1] > 0.5 else 'mildly'} "
          "serially dependent, so naive OLS inference on them would be")
    print("       anti-conservative. This is why HAC is used throughout.")

    # heteroskedasticity against the level
    q = pd.qcut(pd.Series(a), 5, labels=False, duplicates="drop")
    print(f"\n  heteroskedasticity by actual-price quintile")
    print(f"    {'quintile':10s} {'mean price':>11s} {'error sd':>9s} "
          f"{'bias':>8s} {'MAE':>8s}")
    het = []
    for k in range(int(q.max()) + 1):
        m = (q == k).values
        het.append({"quintile": k + 1, "mean_price": float(a[m].mean()),
                    "error_sd": float(e[m].std()), "bias": float(e[m].mean()),
                    "mae": float(np.abs(e[m]).mean())})
        print(f"    {k+1:<10d} {a[m].mean():11.1f} {e[m].std():9.2f} "
              f"{e[m].mean():+8.2f} {np.abs(e[m]).mean():8.2f}")
    out["performance"] = {"descriptive": desc, "theil": th,
                          "calibration": {"intercept": float(rc.params[0]),
                                          "slope": float(rc.params[1]),
                                          "slope_hac_se": float(rc.bse[1]),
                                          "slope_ci": [float(lo), float(hi)],
                                          "r2": float(rc.rsquared)},
                          "error_autocorr": ac, "heteroskedasticity": het,
                          "n": n}

    # ── Part B: hedge analysis on the ex-ante variables ─────────────
    print("\n" + "=" * 76)
    print("PART B — hedge analysis: what explains the price")
    print("=" * 76)
    meta = json.loads((STORE / "modelling_dataset_4y.meta.json").read_text())
    ds = pd.read_parquet(STORE / "modelling_dataset_4y.parquet")
    feats = [c for c in meta["feature_columns"]
             if not c.startswith(("cloud_cover__", "global_tilted",
                                  "temperature_2m__", "wind_speed_120m__"))]
    tgt = meta["target_column"]

    ts = pd.DatetimeIndex(ds.index, tz="UTC").values
    comp = sd.fit_components(ds[tgt].ffill().values, ts,
                            depth=("P_hour", "P_day", "P_week"))
    y = ds[tgt].values - sd.compute_seasonal_part(ts, comp)
    print(f"  target deseasonalised: sd {ds[tgt].std():.1f} -> {np.nanstd(y):.1f} "
          f"EUR/MWh  ({100*(1-np.nanstd(y)/ds[tgt].std()):.1f}% of variance is "
          "calendar)")
    print(f"  {len(feats)} ex-ante candidates (per-site weather excluded as "
          "collinear with its own aggregate)\n")

    print(f"  {'variable':24s} {'hedge beta':>12s} {'HAC t':>8s} "
          f"{'R^2':>7s} {'var removed':>12s}")
    rows = []
    for c in feats:
        x = ds[c].values.astype(float)
        if np.nanstd(x) < 1e-12:
            continue
        xd = x - sd.compute_seasonal_part(
            ts, sd.fit_components(np.nan_to_num(x, nan=float(np.nanmean(x))),
                                  ts, depth=("P_hour", "P_week"),
                                  smooth={"P_week": 7}))
        X = sm.add_constant(xd)
        try:
            r, nn = hac_ols(y, X, ["const", c])
        except Exception:
            continue
        beta, t = float(r.params[1]), float(r.tvalues[1])
        rows.append({"variable": c, "beta": beta, "hac_t": t,
                     "r2": float(r.rsquared), "n": nn})
        print(f"  {c:24s} {beta:+12.4f} {t:+8.2f} {r.rsquared:7.4f} "
              f"{100*r.rsquared:11.2f}%")
    rows.sort(key=lambda d: -d["r2"])
    out["hedge_univariate"] = rows

    print(f"\n  Strongest single hedge: {rows[0]['variable']} at "
          f"R^2 {rows[0]['r2']:.4f} "
          f"({100*rows[0]['r2']:.1f}% of deseasonalised price variance).")

    # joint model + incremental contribution
    keep = [d["variable"] for d in rows if abs(d["hac_t"]) > 2.0][:10]
    Xj = np.column_stack([ds[c].values.astype(float) for c in keep])
    Xj = np.column_stack([np.ones(len(Xj)), Xj])
    rj, nj = hac_ols(y, Xj, ["const"] + keep)
    print(f"\n  joint model on the {len(keep)} HAC-significant variables: "
          f"R^2 {rj.rsquared:.4f}  (n={nj:,})")
    print(f"  sum of univariate R^2 = "
          f"{sum(d['r2'] for d in rows if d['variable'] in keep):.4f} "
          "— the excess over the joint R^2 is shared/collinear variance\n")
    print(f"  {'variable':24s} {'joint beta':>12s} {'HAC t':>8s} "
          f"{'incremental R^2':>16s}")
    inc = []
    for i, c in enumerate(keep):
        cols = [j for j in range(Xj.shape[1]) if j != i + 1]
        rr, _ = hac_ols(y, Xj[:, cols], ["const"])
        d_r2 = rj.rsquared - rr.rsquared
        inc.append({"variable": c, "joint_beta": float(rj.params[i + 1]),
                    "hac_t": float(rj.tvalues[i + 1]),
                    "incremental_r2": float(d_r2)})
        print(f"  {c:24s} {rj.params[i+1]:+12.4f} {rj.tvalues[i+1]:+8.2f} "
              f"{d_r2:16.4f}")
    out["hedge_joint"] = {"r2": float(rj.rsquared), "n": nj,
                          "variables": inc}
    out["unexplained"] = float(1 - rj.rsquared)
    print(f"\n  UNEXPLAINED: {100*(1-rj.rsquared):.1f}% of deseasonalised price "
          "variance is not spanned by any")
    print("  ex-ante fundamental available here.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS / "exp_statistical_analysis.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
