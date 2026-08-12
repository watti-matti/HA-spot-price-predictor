"""Phase-0 frozen backtest harness — the single source of truth for model
experiments (per the 2026-07 systematic plan).

Regime
------
* DAY-AHEAD, walk-forward: for each calendar month in the eval window the
  candidate is refit on all data strictly before that month, then predicts
  the month day-by-day. This mirrors a quarterly/monthly-retrained
  production model without look-ahead.
* No one-step AR(1) term — at day-ahead horizons (12-36 h) the AR
  contribution has decayed to ~nothing, and including it makes every
  variant look near-oracle (measured 2026-07: one-step MAE ~11 vs
  day-ahead ~19 on identical models).
* Physics features (sigmoid_wind_rho / solar_effective) are centred
  per-day (24 h batch local mean) for the production-faithful configs —
  this is exactly what Pipeline._deseasonalize_physics does when the
  artifact carries no physics_seasonal block.
* Weather inputs are historical actuals (weather-oracle) and neighbour
  prices are same-hour (auction-simultaneous). Both conventions are
  optimistic in absolute terms but identical across configs, so the
  DELTAS between configs are honest.
* Point forecasts are floored at 0 EUR/MWh, matching the convention used
  by every study script in this repo.

Configs
-------
  DEPLOYED    shipped L1 components + shipped ridge_coef
              (data/spike_model_default.json, train window ends 2024-11),
              per-day physics centring. Production as-is.
  FRESH       L1 + L2 refit at each month start (walk-forward). Trainer
              deseasonalises physics seasonally, inference centres
              per-day — the SAME mismatch production has today, so the
              only change vs DEPLOYED is data freshness.
  FRESH_CONS  as FRESH, but inference subtracts the trainer's stored
              physics seasonal part (train/inference consistent). The
              delta vs FRESH isolates the consistency fix.

Output: studies/results/backtest_harness.{md,json}, pinned to the
data_store snapshot_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studies"))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import seasonal_decomposition as sd  # noqa: E402
from exp_extra_features import build_dataframe, SEASONAL_ARTIFACT  # noqa: E402
from v2510_layer3_ar_wind import fit_ridge  # noqa: E402
from holidays import build_holiday_set  # noqa: E402
import price_floor as _pf  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO / "custom_components" / "spot_price_predictor" / "data"

EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
# Feature order of the SHIPPED artifact (without the intercept, which is
# first in `ridge_coef`). Asserted against the artifact in
# build_predictions so a retrain that changes the set fails loudly here
# rather than silently mis-aligning the design matrix — which is exactly
# what happened between v2.17.0 and v2.18.0, when this list still named
# the same-hour neighbour features and the harness could not run at all.
FEATS = ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
         "Y_solar_effective", "Y_temp",
         "Y_se1_lag168", "Y_se3_lag168", "Y_ee_lag168", "is_holiday"]
NEIGHBOUR_LAG_HOURS = 168
PRICE_FLOOR = -5.0
# Physics seasonal-fit config — matches exp_extra_features.build_dataframe.
PHYS_DEPTH = ("P_hour", "P_week")
PHYS_SMOOTH = {"P_week": 7}


def _snapshot_id() -> str:
    mf = REPO / "data_store" / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text()).get("snapshot_id", "unknown")
    return "no-data-store"


def _day_center(values: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """Per-24h-batch local-mean centring — what the production Pipeline
    does to physics features when the artifact has no physics_seasonal."""
    s = pd.Series(values, index=index)
    return (s - s.groupby(index.floor("D")).transform("mean")).values


def _fit_l1_like_shipped(name: str, x: np.ndarray, ts: np.ndarray) -> dict:
    """Refit an L1 component set with the SAME depth the shipped artifact
    uses for this input, so freshness is the only change."""
    shipped = SEASONAL_ARTIFACT["components"][name]
    depth = tuple(k for k in ("P_hour", "P_day", "P_week") if k in shipped)
    return sd.fit_components(x, ts, depth=depth)


def build_production_prediction(df: pd.DataFrame) -> np.ndarray:
    """Exactly what the deployed pipeline computes, hour by hour.

    This is the harness's reference config and the reason WP0 existed:
    the FRESH configs below refit L1 every month, while production runs
    the FROZEN artifact. Refitting averages the week-bin noise away;
    production applies one bin deterministically. That difference is why
    offline work concluded weekday *under*-prediction while the field saw
    *over*-prediction (docs/BACKLOG.md D6).

    Faithful to `Pipeline.compute_forecast` in every respect that moves
    the number:
      * frozen L1 from seasonal_components_default.json
      * frozen L2 coefficients, intercept FIRST
      * neighbours lagged NEIGHBOUR_LAG_HOURS, deseasonalised at the hour
        they were observed in
      * physics terms deseasonalised with the artifact's `physics_seasonal`
        block, not per-day centred
      * `is_workday` on the LOCAL calendar, minus public holidays
      * `Y_fi_lag168` zeroed, as the coordinator hard-codes it
      * softplus floor at PRICE_FLOOR, not a clamp at zero

    Not modelled: the per-hour bias corrector (it needs a realised-price
    feedback loop, and its value is measured separately by
    studies/bias_corrector_warmup_study.py) and the L3 AR(1) term (the
    harness is a day-ahead regime by construction).
    """
    idx = df.index
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    n = len(df)
    spike = json.loads((DATA_DIR / "spike_model_default.json").read_text())
    coef = dict(zip(["intercept"] + list(spike["ridge_features"]),
                    np.asarray(spike["ridge_coef"], dtype=float)))
    phys = spike.get("physics_seasonal") or {}

    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                       for d in loc])
    is_wd = (loc.weekday < 5).astype(float) * (1.0 - is_hol)

    def _des_phys(name: str, vals: np.ndarray) -> np.ndarray:
        comp = phys.get(name)
        return (vals - sd.compute_seasonal_part(ts, comp) if comp
                else vals - float(np.nanmean(vals)))

    Y_wr = _des_phys("sigmoid_wind_rho", df["sigmoid_wind_rho"].values)
    Y_se = _des_phys("solar_effective", df["solar_effective"].values)
    Y_temp = df["temp"].values - sd.compute_seasonal_part(
        ts, SEASONAL_ARTIFACT["components"]["temp"])

    Yz = {}
    for z in ("se1", "se3", "ee"):
        raw = df[z].values.astype(float)
        raw = np.where(np.isfinite(raw), raw, np.nanmean(raw))
        y = raw - sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"][z])
        Yz[z] = (pd.Series(y, index=idx)
                 .shift(NEIGHBOUR_LAG_HOURS).fillna(0.0).values)

    seas = sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"]["fi"])
    ridge = (coef["intercept"]
             + coef["Y_fi_lag168"] * 0.0            # coordinator zeroes it
             + coef["is_workday"] * is_wd
             + coef["Y_sigmoid_wind_rho"] * Y_wr
             + coef["Y_solar_effective"] * Y_se
             + coef["Y_temp"] * Y_temp
             + coef["Y_se1_lag168"] * Yz["se1"]
             + coef["Y_se3_lag168"] * Yz["se3"]
             + coef["Y_ee_lag168"] * Yz["ee"]
             + coef["is_holiday"] * is_hol)
    # The real softplus floor, not a hard clamp — near the floor the two
    # differ by log(2), and a parity test against Pipeline must see the
    # same curve.
    return _pf.apply_floor(seas + ridge, floor=PRICE_FLOOR)


def build_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {config: prediction array aligned to df.index} covering the
    eval window (NaN outside it)."""
    idx = df.index
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    n = len(df)
    fi = df["fi"].values
    is_workday = df["is_workday"].values
    wind_rho = df["sigmoid_wind_rho"].values   # raw physics, pre-deseason
    solar_eff = df["solar_effective"].values
    # Calendar flags on the LOCAL clock, matching the trainer and the
    # runtime. `df["is_workday"]` from build_dataframe is UTC-based.
    _loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    _hol = build_holiday_set(2022, 2028)
    is_holiday = np.array([1.0 if d.strftime("%Y-%m-%d") in _hol else 0.0
                           for d in _loc])
    is_workday_local = (_loc.weekday < 5).astype(float) * (1.0 - is_holiday)

    preds = {k: np.full(n, np.nan) for k in ("PRODUCTION", "FRESH", "FRESH_CONS")}

    # ── PRODUCTION: exactly what the deployed pipeline computes ──
    spike = json.loads((DATA_DIR / "spike_model_default.json").read_text())
    assert list(spike["ridge_features"]) == FEATS, (
        f"shipped artifact declares {spike['ridge_features']}, harness "
        f"expects {FEATS}. Update FEATS and build_production_prediction "
        f"together — a silent mismatch mis-aligns the design matrix."
    )
    ev = np.asarray(idx >= EVAL_START)
    preds["PRODUCTION"][ev] = build_production_prediction(df)[ev]

    # ── FRESH / FRESH_CONS: walk-forward monthly refit ──
    month_starts = pd.date_range(EVAL_START, idx[-1], freq="MS", tz="UTC")
    for m0 in month_starts:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray(idx < m0)
        blk = np.asarray((idx >= m0) & (idx < m1))
        if not blk.any():
            continue
        # L1 refits (same component depths as shipped).
        comp_fi = _fit_l1_like_shipped("fi", fi[tr], ts[tr])
        seas_fi = sd.compute_seasonal_part(ts, comp_fi)
        Yfi = fi - seas_fi
        lag = pd.Series(Yfi, index=idx).shift(168).values
        Yz = {}
        for z in ("temp", "se1", "se3", "ee"):
            comp = _fit_l1_like_shipped(z, df[z].values[tr], ts[tr])
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, comp)
        # Neighbours enter LAGGED — they clear in the same day-ahead
        # auction as FI, so a same-hour value is unknowable at forecast
        # time (the v2.16 leak).
        Zlag = {z: pd.Series(Yz[z], index=idx)
                .shift(NEIGHBOUR_LAG_HOURS).fillna(0.0).values
                for z in ("se1", "se3", "ee")}
        # Physics: trainer-style seasonal deseason (fit on train).
        pcomp = {name: sd.fit_components(vals[tr], ts[tr], depth=PHYS_DEPTH,
                                         smooth=PHYS_SMOOTH)
                 for name, vals in (("wind", wind_rho), ("solar", solar_eff))}
        Yw_seas = wind_rho - sd.compute_seasonal_part(ts, pcomp["wind"])
        Ys_seas = solar_eff - sd.compute_seasonal_part(ts, pcomp["solar"])
        # Ridge fit — trainer convention (seasonal physics deseason).
        X_tr = np.column_stack([
            np.ones(n), lag, is_workday_local, Yw_seas, Ys_seas,
            Yz["temp"], Zlag["se1"], Zlag["se3"], Zlag["ee"], is_holiday])
        ok = tr & np.isfinite(X_tr).all(axis=1) & np.isfinite(Yfi)
        c = fit_ridge(X_tr[ok], Yfi[ok], alpha=1.0)
        # Inference on the month block.
        # FRESH: production-faithful per-day centring (legacy mismatch).
        Xb_legacy = np.column_stack([
            np.ones(n), lag, is_workday_local,
            _day_center(wind_rho, idx), _day_center(solar_eff, idx),
            Yz["temp"], Zlag["se1"], Zlag["se3"], Zlag["ee"],
            is_holiday])[blk]
        preds["FRESH"][blk] = np.maximum(PRICE_FLOOR, seas_fi[blk] + Xb_legacy @ c)
        # FRESH_CONS: consistent stored-seasonal deseason.
        Xb_cons = X_tr[blk]
        preds["FRESH_CONS"][blk] = np.maximum(PRICE_FLOOR, seas_fi[blk] + Xb_cons @ c)
    return preds


def score(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    idx = df.index
    y = df["fi"].values
    ev = np.asarray(idx >= EVAL_START) & np.isfinite(preds["PRODUCTION"])
    for p in preds.values():
        ev &= np.isfinite(p)
    mo = idx.month.values[ev]
    hr = idx.hour.values[ev]
    ye = y[ev]
    p95 = np.percentile(ye, 95)
    segments = {
        "ALL":               np.ones(ev.sum(), dtype=bool),
        "WINTER Dec-Feb":    np.isin(mo, [12, 1, 2]),
        "SUMMER May-Jul":    np.isin(mo, [5, 6, 7]),
        "midday 8-12 UTC":   (hr >= 8) & (hr < 12),
        "evening 15-19 UTC": (hr >= 15) & (hr < 19),
        "tail p95 price":    ye >= p95,
    }
    out = {"n_eval": int(ev.sum()),
           "eval_window": [str(idx[ev][0]), str(idx[ev][-1])],
           "price_p95": float(p95), "segments": {}, "monthly": {}}
    # Per-month WEEKDAY bias. A single aggregate hides the level errors
    # this harness exists to catch: a month at +18 and a month at -20
    # average to nothing. Weekday-only because that is where the
    # reported symptom lives.
    ei = idx[ev]
    loc = pd.DatetimeIndex(ei).tz_convert("Europe/Helsinki")
    wd = np.asarray(loc.weekday < 5)
    for key in sorted({f"{d.year}-{d.month:02d}" for d in loc}):
        m = np.array([f"{d.year}-{d.month:02d}" == key for d in loc]) & wd
        if m.sum() < 100:
            continue
        row = {"n": int(m.sum()), "mean_price": float(ye[m].mean())}
        for cfg, p in preds.items():
            e = p[ev][m] - ye[m]
            row[cfg] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean())}
        out["monthly"][key] = row
    for sname, m in segments.items():
        seg = {"n": int(m.sum()), "mean_price": float(ye[m].mean())}
        for cfg, p in preds.items():
            e = p[ev][m] - ye[m]
            seg[cfg] = {"bias": float(e.mean()),
                        "mae": float(np.abs(e).mean())}
        out["segments"][sname] = seg
    return out


def write_md(res: dict, out: Path) -> None:
    cfgs = ["PRODUCTION", "FRESH", "FRESH_CONS"]
    lines = [
        "# Backtest harness — frozen day-ahead walk-forward",
        "",
        f"Data snapshot: `{res['snapshot_id']}` | eval "
        f"{res['eval_window'][0][:10]} → {res['eval_window'][1][:10]} "
        f"({res['n_eval']:,} h)",
        "",
        "Regime: day-ahead (no one-step AR). PRODUCTION is the frozen "
        "shipped artifact evaluated exactly as `Pipeline.compute_forecast` "
        "does — frozen L1, frozen coefficients, neighbours lagged 168 h, "
        "physics deseasonalised with the artifact's `physics_seasonal` "
        "block, local-calendar workday and holiday flags, `Y_fi_lag168` "
        "zeroed, softplus floor. Parity is pinned by "
        "tests/test_harness_production_parity.py. FRESH configs refit L1 "
        "and the ridge monthly and are for model experiments only — do NOT "
        "read them as production behaviour (backlog D6). Weather inputs are "
        "identical across configs, so the deltas are the honest signal.",
        "",
        "| segment | n | mean € | " + " | ".join(
            f"{c} MAE (bias)" for c in cfgs) + " |",
        "|---|---:|---:|" + "---:|" * len(cfgs),
    ]
    for sname, seg in res["segments"].items():
        cells = " | ".join(
            f"{seg[c]['mae']:.2f} ({seg[c]['bias']:+.1f})" for c in cfgs)
        lines.append(f"| {sname} | {seg['n']:,} | "
                     f"{seg['mean_price']:.0f} | {cells} |")
    if res.get("monthly"):
        lines += ["", "## Weekday bias by month (PRODUCTION)", "",
                  "| month | n | mean € | bias | MAE |", "|---|---:|---:|---:|---:|"]
        for k, r in res["monthly"].items():
            lines.append(f"| {k} | {r['n']:,} | {r['mean_price']:.0f} | "
                         f"{r['PRODUCTION']['bias']:+.1f} | "
                         f"{r['PRODUCTION']['mae']:.1f} |")
    lines += [
        "",
        "Isolated deltas: FRESH − DEPLOYED = value of retraining on fresh "
        "data; FRESH_CONS − FRESH = value of the physics-deseasonalisation "
        "consistency fix.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Building dataframe…", flush=True)
    df = build_dataframe()
    print(f"  rows = {len(df):,}  span = {df.index[0].date()} → "
          f"{df.index[-1].date()}", flush=True)
    preds = build_predictions(df)
    res = score(df, preds)
    res["snapshot_id"] = _snapshot_id()
    (RESULTS_DIR / "backtest_harness.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    write_md(res, RESULTS_DIR / "backtest_harness.md")
    print(f"\nsnapshot = {res['snapshot_id']}   n_eval = {res['n_eval']:,}")
    print(f"{'segment':18s} | {'PRODUCTION':>16s} | {'FRESH':>16s} | "
          f"{'FRESH_CONS':>16s}")
    for sname, seg in res["segments"].items():
        cells = " | ".join(
            f"{seg[c]['mae']:6.2f} ({seg[c]['bias']:+6.1f})"
            for c in ("PRODUCTION", "FRESH", "FRESH_CONS"))
        print(f"{sname:18s} | {cells}")
    if res.get("monthly"):
        print()
        hdr = ("month", "n", "mean EUR", "bias", "MAE")
        print("%-9s %6s %9s %8s %7s   (PRODUCTION, weekday only)" % hdr)
        for k, r in res["monthly"].items():
            print("%-9s %6d %9.1f %+8.2f %7.2f" % (
                k, r["n"], r["mean_price"],
                r["PRODUCTION"]["bias"], r["PRODUCTION"]["mae"]))
    print("\nWrote studies/results/backtest_harness.{md,json}")


if __name__ == "__main__":
    main()
