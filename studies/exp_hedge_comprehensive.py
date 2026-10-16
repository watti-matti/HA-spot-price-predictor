"""Comprehensive hedge analysis: every candidate, not a chosen subset.

The earlier pass tested 14 variables and excluded the 28 per-site weather
series on an ASSUMPTION of collinearity with their own weighted
aggregate. That assumption was never tested, and no derived, lagged,
interacted or non-linear candidate was tried at all. This builds the
widest defensible candidate set and reports all of it.

STRUCTURE — three blocks, kept strictly apart
---------------------------------------------
ADMISSIBLE   ex-ante, available at gate closure. Anything here is a
             legitimate model feature. Calendar, weather forecasts and
             their transforms, Fingrid day-ahead forecasts, installed
             capacities, and price lags at >= 168 h (which are
             observable for the whole 170 h horizon).

ORACLE       contemporaneous or realised. NOT usable as features — the
             market cleared before these were known. Reported only to
             measure the CEILING: if a perfect sub-model could predict
             the quantity, this is what it would be worth. The gap
             between admissible and oracle is what better sub-modelling
             could buy, and is the only honest way to size that work.

DERIVED      combinations the raw columns cannot express on their own:
             residual load, reserve margin, scarcity interactions, and
             piecewise transforms where the physics is threshold-like.

METHOD
------
Everything on L1-deseasonalised series (raw attribution credits the
calendar to whichever fundamental shares its shape: raw price vs raw
consumption is R^2 0.296, deseasonalised 0.143).

Inference uses Newey-West HAC with a 168 h lag window throughout.
Hourly price errors autocorrelate at 0.920 at lag 1, so ordinary OLS
standard errors are far too small and would make almost everything
look significant.

Univariate R^2 values sum to far more than any joint model, because the
candidates share variance. So selection is GREEDY FORWARD by
incremental R^2, requiring |HAC t| > 2 to enter — that reports what each
variable adds on top of what is already chosen, which is the decision-
relevant number.

Run:  python studies/exp_hedge_comprehensive.py
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
HAC_LAGS = 168
LAG = 168
MAX_STEPS = 14

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    import statsmodels.api as sm
    import seasonal_decomposition as sd
    from lag_utils import lag_by_hours

    ds = pd.read_parquet(STORE / "modelling_dataset_4y.parquet")
    meta = json.loads((STORE / "modelling_dataset_4y.meta.json").read_text())
    ts = pd.DatetimeIndex(ds.index, tz="UTC").values
    idx = ds.index

    def deseas(x, depth=("P_hour", "P_week"), smooth=None):
        x = np.asarray(x, dtype=float)
        f = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(
            np.nanmean(x)) else 0.0)
        comp = sd.fit_components(f, ts, depth=depth,
                                 smooth=smooth or {"P_week": 7})
        return x - sd.compute_seasonal_part(ts, comp)

    y = deseas(ds["fi_price_eur_mwh"].ffill().values,
               depth=("P_hour", "P_day", "P_week"), smooth={})

    cand: dict[str, tuple[str, np.ndarray]] = {}

    def put(name, block, v):
        v = np.asarray(v, dtype=float)
        if np.nanstd(v) > 1e-12:
            cand[name] = (block, v)

    # ── ADMISSIBLE ──────────────────────────────────────────────────
    for c in ("is_holiday", "is_workday", "local_hour", "local_dow"):
        put(c, "calendar", ds[c].values)
    for c in ("wind_speed_weighted", "solar_irradiance_weighted",
              "temperature_weighted", "sigmoid_wind_rho", "solar_effective"):
        put(c, "weather agg", deseas(ds[c].values))
    # per-site: the assumption under test
    for c in meta["feature_columns"]:
        if c.startswith(("cloud_cover__", "global_tilted", "temperature_2m__",
                         "wind_speed_120m__")):
            put(c, "weather site", deseas(ds[c].values))
    for c in ("wind_forecast_mw", "solar_forecast_mw", "consumption_mw",
              "solar_capacity_mw", "wind_capacity_mw"):
        put(c, "fingrid ex-ante", deseas(ds[c].values))

    T = ds["temperature_weighted"].values.astype(float)
    W = ds["wind_speed_weighted"].values.astype(float)
    put("hdd_10", "transform", deseas(np.maximum(10.0 - T, 0.0)))
    put("cdd_10", "transform", deseas(np.maximum(T - 10.0, 0.0)))
    put("temp_sq", "transform", deseas(T ** 2))
    put("wind_sq", "transform", deseas(W ** 2))
    put("wind_calm", "transform", deseas(np.maximum(6.0 - W, 0.0)))
    put("wind_log_scarcity", "transform",
        deseas(np.log1p(np.maximum(8.0 - W, 0.0))))

    wf = ds["wind_forecast_mw"].values.astype(float)
    sf = ds["solar_forecast_mw"].values.astype(float)
    cf = ds["consumption_mw"].values.astype(float)
    wcap = ds["wind_capacity_mw"].ffill().values.astype(float)
    # residual load from EX-ANTE forecasts only
    rl = cf - wf - sf
    put("residual_load_ex_ante", "derived", deseas(rl))
    put("residual_load_sq", "derived", deseas(rl ** 2 / 1e6))
    put("wind_penetration", "derived", deseas(wf / np.maximum(cf, 1.0)))
    put("wind_cap_factor", "derived", deseas(wf / np.maximum(wcap, 1.0)))
    put("rl_x_workday", "derived", deseas(rl) * ds["is_workday"].values)
    put("wind_x_temp", "derived", deseas(wf) * deseas(T))
    put("rl_per_wind_cap", "derived", deseas(rl / np.maximum(wcap, 1.0)))

    # price lags at >= 168 h: observable across the whole horizon
    fi = ds["fi_price_eur_mwh"].ffill().values
    yfi = deseas(fi, depth=("P_hour", "P_day", "P_week"), smooth={})
    put("fi_lag168", "price lag", lag_by_hours(yfi, idx, LAG))
    put("fi_lag336", "price lag", lag_by_hours(yfi, idx, 336))
    put("fi_lag168_7dmean", "price lag",
        pd.Series(lag_by_hours(yfi, idx, LAG), index=idx)
        .rolling(168, min_periods=24).mean().values)
    for z in ("se1", "se3", "ee"):
        yz = deseas(ds[f"{z}_price_eur_mwh"].ffill().values,
                    depth=("P_hour", "P_day", "P_week"), smooth={})
        put(f"{z}_lag168", "price lag", lag_by_hours(yz, idx, LAG))

    # ── ORACLE (ceiling only) ───────────────────────────────────────
    for z in ("se1", "se3", "ee"):
        put(f"ORACLE_{z}_same_hour", "oracle price",
            deseas(ds[f"{z}_price_eur_mwh"].ffill().values,
                   depth=("P_hour", "P_day", "P_week"), smooth={}))
    for c in ("nuclear_mw", "wind_actual_mw", "consumption_actual_mw",
              "flow_fi_se1_mw", "flow_fi_se3_mw", "flow_fi_ee_mw",
              "flow_fi_no_mw"):
        if c in ds.columns:
            put(f"ORACLE_{c}", "oracle realised", deseas(ds[c].values))
    wa = ds.get("wind_actual_mw")
    ca = ds.get("consumption_actual_mw")
    if wa is not None and ca is not None:
        put("ORACLE_residual_load_realised", "oracle realised",
            deseas(ca.values - wa.values - sf))

    print("=" * 84)
    print("COMPREHENSIVE HEDGE ANALYSIS")
    print("=" * 84)
    nb = {}
    for n, (b, _) in cand.items():
        nb[b] = nb.get(b, 0) + 1
    print(f"  {len(cand)} candidates across {len(nb)} blocks: " +
          ", ".join(f"{k} {v}" for k, v in sorted(nb.items())))

    def fit(cols):
        X = sm.add_constant(np.column_stack(cols))
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < 2000:
            return None, 0
        return sm.OLS(y[ok], X[ok]).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}), int(ok.sum())

    rows = []
    for n, (b, v) in cand.items():
        r, nn = fit([v])
        if r is None:
            continue
        rows.append({"variable": n, "block": b, "beta": float(r.params[1]),
                     "hac_t": float(r.tvalues[1]), "r2": float(r.rsquared),
                     "n": nn})
    rows.sort(key=lambda d: -d["r2"])

    print(f"\n  TOP 25 UNIVARIATE HEDGES (of {len(rows)})")
    print(f"  {'#':>3s} {'variable':34s} {'block':16s} {'beta':>12s} "
          f"{'HAC t':>8s} {'var removed':>12s}")
    for i, d in enumerate(rows[:25], 1):
        print(f"  {i:3d} {d['variable']:34s} {d['block']:16s} "
              f"{d['beta']:+12.4f} {d['hac_t']:+8.2f} {100*d['r2']:11.2f}%")

    print(f"\n  best per block:")
    seen = {}
    for d in rows:
        if d["block"] not in seen:
            seen[d["block"]] = d
    for b, d in sorted(seen.items(), key=lambda kv: -kv[1]["r2"]):
        print(f"    {b:16s} {d['variable']:34s} {100*d['r2']:6.2f}%  "
              f"t {d['hac_t']:+7.2f}")

    # ── greedy forward selection, admissible only ───────────────────
    adm = [d for d in rows if not d["variable"].startswith("ORACLE_")]
    print(f"\n  GREEDY FORWARD SELECTION — ADMISSIBLE ONLY "
          f"({len(adm)} candidates)")
    print(f"  {'step':>4s} {'variable':34s} {'block':16s} {'R^2':>8s} "
          f"{'+ pp':>7s} {'HAC t':>8s}")
    chosen, cols, prev = [], [], 0.0
    pool = {d["variable"] for d in adm}
    for step in range(1, MAX_STEPS + 1):
        best = None
        for nm in pool:
            r, _ = fit(cols + [cand[nm][1]])
            if r is None:
                continue
            if abs(r.tvalues[-1]) < 2.0:
                continue
            if best is None or r.rsquared > best[1]:
                best = (nm, r.rsquared, float(r.tvalues[-1]))
        if best is None or best[1] - prev < 0.0015:
            break
        nm, r2, t = best
        chosen.append({"variable": nm, "block": cand[nm][0], "r2": float(r2),
                       "delta_pp": float(100 * (r2 - prev)), "hac_t": t})
        cols.append(cand[nm][1]); pool.discard(nm)
        print(f"  {step:4d} {nm:34s} {cand[nm][0]:16s} {r2:8.4f} "
              f"{100*(r2-prev):+7.2f} {t:+8.2f}")
        prev = r2
    adm_r2 = prev

    # ── oracle ceiling ──────────────────────────────────────────────
    print(f"\n  ORACLE CEILING — add contemporaneous/realised on top")
    print(f"  (NOT shippable; measures what perfect sub-models would buy)")
    oracle_names = [d["variable"] for d in rows
                    if d["variable"].startswith("ORACLE_")]
    ocols, oprev = list(cols), adm_r2
    opicked = []
    for step in range(1, 7):
        best = None
        for nm in oracle_names:
            if nm in [o["variable"] for o in opicked]:
                continue
            r, _ = fit(ocols + [cand[nm][1]])
            if r is None or abs(r.tvalues[-1]) < 2.0:
                continue
            if best is None or r.rsquared > best[1]:
                best = (nm, r.rsquared, float(r.tvalues[-1]))
        if best is None or best[1] - oprev < 0.0015:
            break
        nm, r2, t = best
        opicked.append({"variable": nm, "r2": float(r2),
                        "delta_pp": float(100 * (r2 - oprev))})
        ocols.append(cand[nm][1])
        print(f"  {step:4d} {nm:34s} {'oracle':16s} {r2:8.4f} "
              f"{100*(r2-oprev):+7.2f} {t:+8.2f}")
        oprev = r2

    print("\n" + "=" * 84)
    print(f"  ADMISSIBLE ceiling   R^2 {adm_r2:.4f}  "
          f"({100*adm_r2:.1f}% of deseasonalised price variance)")
    print(f"  ORACLE ceiling       R^2 {oprev:.4f}  ({100*oprev:.1f}%)")
    print(f"  perfect sub-models would add {100*(oprev-adm_r2):.1f} pp")
    print(f"  IRREDUCIBLE          {100*(1-oprev):.1f}% unexplained even with "
          "oracle knowledge")
    print("=" * 84)

    out = {"n_candidates": len(cand), "univariate": rows,
           "forward_admissible": chosen, "admissible_r2": float(adm_r2),
           "oracle_added": opicked, "oracle_r2": float(oprev)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS / "exp_hedge_comprehensive.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {f.relative_to(REPO)}")


if __name__ == "__main__":
    main()
