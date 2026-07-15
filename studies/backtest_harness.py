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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO / "custom_components" / "spot_price_predictor" / "data"

EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
FEATS = ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
         "Y_solar_effective", "Y_temp", "Y_se1", "Y_se3", "Y_ee"]
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

    preds = {k: np.full(n, np.nan) for k in ("DEPLOYED", "FRESH", "FRESH_CONS")}

    # ── DEPLOYED: shipped L1 + shipped ridge coefs, per-day centring ──
    spike = json.loads((DATA_DIR / "spike_model_default.json").read_text())
    coef = np.asarray(spike["ridge_coef"], dtype=float)   # [intercept, 8]
    assert spike["ridge_features"] == FEATS, "artifact feature order changed"
    seas_fi_dep = sd.compute_seasonal_part(ts, SEASONAL_ARTIFACT["components"]["fi"])
    Yfi_dep = fi - seas_fi_dep
    lag_dep = pd.Series(Yfi_dep, index=idx).shift(168).values
    Yz_dep = {z: df[z].values - sd.compute_seasonal_part(
        ts, SEASONAL_ARTIFACT["components"][z]) for z in ("temp", "se1", "se3", "ee")}
    X_dep = np.column_stack([
        np.ones(n), lag_dep, is_workday,
        _day_center(wind_rho, idx), _day_center(solar_eff, idx),
        Yz_dep["temp"], Yz_dep["se1"], Yz_dep["se3"], Yz_dep["ee"]])
    pred_dep = np.maximum(0.0, seas_fi_dep + X_dep @ coef)
    ev = np.asarray(idx >= EVAL_START)
    preds["DEPLOYED"][ev] = pred_dep[ev]

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
        # Physics: trainer-style seasonal deseason (fit on train).
        pcomp = {name: sd.fit_components(vals[tr], ts[tr], depth=PHYS_DEPTH,
                                         smooth=PHYS_SMOOTH)
                 for name, vals in (("wind", wind_rho), ("solar", solar_eff))}
        Yw_seas = wind_rho - sd.compute_seasonal_part(ts, pcomp["wind"])
        Ys_seas = solar_eff - sd.compute_seasonal_part(ts, pcomp["solar"])
        # Ridge fit — trainer convention (seasonal physics deseason).
        X_tr = np.column_stack([
            np.ones(n), lag, is_workday, Yw_seas, Ys_seas,
            Yz["temp"], Yz["se1"], Yz["se3"], Yz["ee"]])
        ok = tr & np.isfinite(X_tr).all(axis=1) & np.isfinite(Yfi)
        c = fit_ridge(X_tr[ok], Yfi[ok], alpha=1.0)
        # Inference on the month block.
        # FRESH: production-faithful per-day centring (legacy mismatch).
        Xb_legacy = np.column_stack([
            np.ones(n), lag, is_workday,
            _day_center(wind_rho, idx), _day_center(solar_eff, idx),
            Yz["temp"], Yz["se1"], Yz["se3"], Yz["ee"]])[blk]
        preds["FRESH"][blk] = np.maximum(0.0, seas_fi[blk] + Xb_legacy @ c)
        # FRESH_CONS: consistent stored-seasonal deseason.
        Xb_cons = X_tr[blk]
        preds["FRESH_CONS"][blk] = np.maximum(0.0, seas_fi[blk] + Xb_cons @ c)
    return preds


def score(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    idx = df.index
    y = df["fi"].values
    ev = np.asarray(idx >= EVAL_START) & np.isfinite(preds["DEPLOYED"])
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
           "price_p95": float(p95), "segments": {}}
    for sname, m in segments.items():
        seg = {"n": int(m.sum()), "mean_price": float(ye[m].mean())}
        for cfg, p in preds.items():
            e = p[ev][m] - ye[m]
            seg[cfg] = {"bias": float(e.mean()),
                        "mae": float(np.abs(e).mean())}
        out["segments"][sname] = seg
    return out


def write_md(res: dict, out: Path) -> None:
    cfgs = ["DEPLOYED", "FRESH", "FRESH_CONS"]
    lines = [
        "# Backtest harness — frozen day-ahead walk-forward",
        "",
        f"Data snapshot: `{res['snapshot_id']}` | eval "
        f"{res['eval_window'][0][:10]} → {res['eval_window'][1][:10]} "
        f"({res['n_eval']:,} h)",
        "",
        "Regime: day-ahead (no one-step AR), monthly walk-forward refit for "
        "FRESH configs, per-day physics centring for production-faithful "
        "configs, weather-oracle inputs (identical across configs — deltas "
        "are the honest signal).",
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
    print(f"{'segment':18s} | {'DEPLOYED':>16s} | {'FRESH':>16s} | "
          f"{'FRESH_CONS':>16s}")
    for sname, seg in res["segments"].items():
        cells = " | ".join(
            f"{seg[c]['mae']:6.2f} ({seg[c]['bias']:+6.1f})"
            for c in ("DEPLOYED", "FRESH", "FRESH_CONS"))
        print(f"{sname:18s} | {cells}")
    print("\nWrote studies/results/backtest_harness.{md,json}")


if __name__ == "__main__":
    main()
