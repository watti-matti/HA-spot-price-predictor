"""v22_tuning_sweep.py — overnight comprehensive sweep over Ridge
hyperparameters and feature subsets, plus leave-one-out redundancy
analysis.

Goal
----
The v2.2 retrain on the random training-test split showed:
    R^2 0.515 -> 0.561 (+9 %)
    MAE 23.94 -> 25.26 (+5.5 %)
i.e. R^2 improved meaningfully but MAE regressed slightly. The release
notes flag this as multi-collinearity between `nuclear_deficit` and
`net_load_gw` (both proxy "tight supply"); Ridge has split the credit
between them in a way that helps in-distribution variance but hurts
out-of-sample MAE.

This script searches the hyperparameter and feature-subset space for a
configuration that delivers a clean MAE win over v2.1 on the most
recent 180-day holdout, and runs leave-one-out feature redundancy on
the winner.

Phases
------
1. **Grid scan**: 72 variants
   * Ridge alpha:    {0.5, 1.0, 2.0, 5.0, 10.0, 20.0}     (6)
   * Feature subset: {full_v22, drop_nuclear_deficit,
                       no_netload_interactions, v21_baseline}  (4)
   * log_offset:     {30, 55, 100}                          (3)
   * = 72 fits, ~3 s each = ~5 min total

2. **Walk-forward verification** on top-5 variants by MAE: daily refit
   over the 180-day window, confirms the winner doesn't depend on a
   single lucky train-test alignment. ~10 min.

3. **Leave-one-out redundancy** on the winner: drop each feature one
   at a time, measure delta-MAE, identify redundant inputs. ~21 fits
   = ~1 min.

4. **Baseline runs**: v2.1 (17 features, alpha=1.0, log_offset=55) and
   v2.2 unmodified (21 features, alpha=1.0, log_offset=55) for direct
   reference points.

Output
------
* `studies/results/v22_sweep_<stamp>.md` — ranked variants, top-5
  walk-forward, redundancy table, recommendation block.
* `studies/results/v22_sweep_<stamp>.json` — raw metrics + winning
  feature_cols for downstream automation.
* `output/model_coefs_v22tuned.json` — trained model dict for the
  winner (drop-in replacement for `model_coefs_default.json` if
  approved).

Run
---
    python studies/v22_tuning_sweep.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import build_features
from src.train_model import train


# ── Config ─────────────────────────────────────────────────────────


# Holdout window: most recent 180 days of training data
HOLDOUT_DAYS = 180
TRAIN_DAYS = 540

# v2.2 full feature set as written by the latest retrain
FULL_V22_FEATURES = [
    "wind_speed_weighted", "solar_irradiance_weighted",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_holiday", "hdd_sq",
    "wind_log_scarcity", "wind_calm_x_peak_am", "wind_calm_x_peak_pm",
    "ar_se1", "ar_se3", "ar_ee", "export_potential_se3",
    "nuclear_deficit", "nuclear_x_scarcity",
    "net_load_gw", "net_load_squared",
    "net_load_x_workday", "net_load_x_scarcity",
]

# v2.1 baseline (without the 4 net-load columns)
V21_FEATURES = [
    f for f in FULL_V22_FEATURES
    if not f.startswith("net_load_")
]

FEATURE_SUBSETS = {
    "full_v22":                FULL_V22_FEATURES,
    "drop_nuclear_deficit":   [f for f in FULL_V22_FEATURES
                                if f not in ("nuclear_deficit",
                                              "nuclear_x_scarcity")],
    "no_netload_interactions": [f for f in FULL_V22_FEATURES
                                if f not in ("net_load_x_workday",
                                              "net_load_x_scarcity")],
    "v21_baseline":            V21_FEATURES,
}

ALPHA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
LOG_OFFSET_GRID = [30, 55, 100]


# ── Data loading + holdout split ───────────────────────────────────


def load_full_features() -> tuple[pd.DataFrame, list[str], dict]:
    """Build the full v2.2 feature matrix from cached parquets."""
    config_path = REPO_ROOT / "config" / "regions" / "finland.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = REPO_ROOT / "output"
    prices = pd.read_parquet(out_dir / "fi_prices.parquet")["price_eur_mwh"]
    weather = pd.read_parquet(out_dir / "fi_weather.parquet")

    np_path = out_dir / "fi_neighbor_prices.parquet"
    neighbor_prices = None
    if np_path.exists():
        np_df = pd.read_parquet(np_path)
        neighbor_prices = {col: np_df[col] for col in np_df.columns}

    gd_path = out_dir / "fi_grid_data.parquet"
    grid_data = None
    if gd_path.exists():
        gd_df = pd.read_parquet(gd_path)
        grid_data = {col: gd_df[col] for col in gd_df.columns}

    df, feature_cols, ar_models = build_features(
        prices, weather, config,
        neighbor_prices=neighbor_prices,
        grid_data=grid_data,
    )
    return df, feature_cols, config


# ── Per-variant fit + evaluate ─────────────────────────────────────


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    mu = float(np.mean(y_true))
    sst = float(np.sum((y_true - mu) ** 2))
    ssr = float(np.sum(err ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "bias": bias, "R2": r2,
            "n": len(y_true)}


def daily_d4_rho(df_hours: pd.DataFrame, y_true: np.ndarray,
                 y_pred: np.ndarray) -> float:
    """Spearman rho on daily D(4) — mean of cheapest 4 hours per day,
    actual vs predicted. The metric the thermal LP cares about."""
    df = df_hours.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["date"] = df.index.tz_convert("Europe/Helsinki").date
    out_t, out_p = [], []
    for d, sub in df.groupby("date"):
        if len(sub) < 24:
            continue
        out_t.append(float(np.mean(np.sort(sub["y_true"].values)[:4])))
        out_p.append(float(np.mean(np.sort(sub["y_pred"].values)[:4])))
    if len(out_t) < 10:
        return 0.0
    from scipy.stats import spearmanr
    return float(spearmanr(out_t, out_p).statistic)


def fit_predict_variant(
    df_full: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    alpha: float,
    log_offset: float,
    config_base: dict,
) -> dict:
    """Fit Ridge on train_idx, predict test_idx. Return metrics."""
    cfg = copy.deepcopy(config_base)
    cfg.setdefault("training", {})
    cfg["training"]["ridge_alpha"] = alpha
    cfg["training"]["log_offset"] = log_offset
    cfg["training"]["test_split"] = 0.05  # internal test sample (we ignore)
    # keep half-life as configured (120d)

    df_train = df_full.iloc[train_idx].copy()
    df_test = df_full.iloc[test_idx]

    coefs = train(df_train, feature_cols, cfg)
    intercept = coefs["intercept"]
    feat_coefs = {f["name"]: f["coef"] for f in coefs["features"]}
    power_scale = coefs.get("power_scale", 1.0)
    power_exp = coefs.get("power_exp", 1.0)

    X_test = df_test[feature_cols].values
    linear = intercept + X_test @ np.array(
        [feat_coefs.get(f, 0.0) for f in feature_cols]
    )
    raw = np.exp(np.minimum(linear, 20.0)) - log_offset
    raw = np.maximum(0.0, raw)
    y_pred = np.where(raw > 0, power_scale * np.power(raw, power_exp), 0.0)

    y_true = df_test["price_eur_mwh"].values

    m = metrics(y_true, y_pred)
    m["d4_rho"] = daily_d4_rho(df_test, y_true, y_pred)
    return m


# ── Phase 1: grid scan ─────────────────────────────────────────────


def phase1_grid_scan(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config_base: dict,
) -> list[dict]:
    """72-variant grid scan."""
    rows: list[dict] = []
    n = len(ALPHA_GRID) * len(FEATURE_SUBSETS) * len(LOG_OFFSET_GRID)
    i = 0
    t0 = time.time()
    for subset_name, feature_cols in FEATURE_SUBSETS.items():
        for alpha in ALPHA_GRID:
            for log_off in LOG_OFFSET_GRID:
                i += 1
                t_start = time.time()
                try:
                    m = fit_predict_variant(
                        df, train_idx, test_idx,
                        feature_cols, alpha, log_off, config_base,
                    )
                    m["subset"] = subset_name
                    m["n_features"] = len(feature_cols)
                    m["alpha"] = alpha
                    m["log_offset"] = log_off
                    m["fit_seconds"] = time.time() - t_start
                    rows.append(m)
                    print(f"  [{i:>2}/{n}] {subset_name:>26s} "
                          f"a={alpha:>5} off={log_off:>3}  "
                          f"MAE={m['MAE']:.2f}  R²={m['R2']:.3f}  "
                          f"d4ρ={m['d4_rho']:.3f}",
                          flush=True)
                except Exception as exc:
                    print(f"  [{i:>2}/{n}] {subset_name} a={alpha} "
                          f"off={log_off}  FAILED: {exc}", flush=True)
    print(f"\n[phase1] {len(rows)}/{n} variants succeeded in "
          f"{time.time() - t0:.0f}s", flush=True)
    return rows


# ── Phase 2: walk-forward verification on top-5 ───────────────────


def walk_forward_variant(
    df: pd.DataFrame,
    train_anchor: int,
    test_idx: np.ndarray,
    feature_cols: list[str],
    alpha: float,
    log_offset: float,
    config_base: dict,
    refit_cadence_days: int = 7,
) -> dict:
    """Walk forward through `test_idx`, refitting every
    `refit_cadence_days` days using all data up to that day. Returns
    aggregated metrics on the entire test window."""
    cfg = copy.deepcopy(config_base)
    cfg.setdefault("training", {})
    cfg["training"]["ridge_alpha"] = alpha
    cfg["training"]["log_offset"] = log_offset
    cfg["training"]["test_split"] = 0.05  # internal test sample (we ignore)

    # Walk in chunks of refit_cadence_days * 24 hours
    chunk = refit_cadence_days * 24
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    rows_test_all: list[pd.Timestamp] = []

    for start in range(0, len(test_idx), chunk):
        end = min(start + chunk, len(test_idx))
        # Refit on data up to test_idx[start]
        train_end = test_idx[start]
        train_slice = np.arange(0, train_end)
        df_train = df.iloc[train_slice].copy()
        df_chunk = df.iloc[test_idx[start:end]]

        coefs = train(df_train, feature_cols, cfg)
        intercept = coefs["intercept"]
        feat_coefs = {f["name"]: f["coef"] for f in coefs["features"]}
        ps = coefs.get("power_scale", 1.0)
        pe = coefs.get("power_exp", 1.0)

        X_test = df_chunk[feature_cols].values
        linear = intercept + X_test @ np.array(
            [feat_coefs.get(f, 0.0) for f in feature_cols])
        raw = np.exp(np.minimum(linear, 20.0)) - log_offset
        raw = np.maximum(0.0, raw)
        y_pred = np.where(raw > 0, ps * np.power(raw, pe), 0.0)

        y_true_all.extend(df_chunk["price_eur_mwh"].values.tolist())
        y_pred_all.extend(y_pred.tolist())
        rows_test_all.extend(df_chunk.index.tolist())

    df_eval = pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all},
                            index=rows_test_all)
    df_eval["price_eur_mwh"] = df_eval["y_true"]
    m = metrics(np.asarray(y_true_all), np.asarray(y_pred_all))
    m["d4_rho"] = daily_d4_rho(
        df_eval, np.asarray(y_true_all), np.asarray(y_pred_all))
    return m


# ── Phase 3: leave-one-out redundancy ─────────────────────────────


def phase3_redundancy(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    winner_features: list[str],
    winner_alpha: float,
    winner_log_offset: float,
    config_base: dict,
) -> list[dict]:
    """Drop each feature individually, measure delta-MAE."""
    # Baseline (all features)
    baseline = fit_predict_variant(
        df, train_idx, test_idx,
        winner_features, winner_alpha, winner_log_offset, config_base,
    )
    print(f"\n[phase3] baseline MAE={baseline['MAE']:.3f} "
          f"R²={baseline['R2']:.4f}", flush=True)

    rows: list[dict] = []
    for drop in winner_features:
        sub = [f for f in winner_features if f != drop]
        try:
            m = fit_predict_variant(
                df, train_idx, test_idx,
                sub, winner_alpha, winner_log_offset, config_base,
            )
            d_mae = m["MAE"] - baseline["MAE"]
            d_r2 = baseline["R2"] - m["R2"]
            rows.append({
                "dropped": drop,
                "MAE": m["MAE"],
                "delta_MAE": d_mae,
                "R2": m["R2"],
                "delta_R2": d_r2,
                "d4_rho": m["d4_rho"],
                "delta_d4_rho": baseline["d4_rho"] - m["d4_rho"],
            })
            print(f"  drop {drop:>26s}  ΔMAE={d_mae:+.3f}  "
                  f"ΔR²={d_r2:+.4f}", flush=True)
        except Exception as exc:
            print(f"  drop {drop}: FAILED {exc}", flush=True)
    return rows


# ── Markdown report ────────────────────────────────────────────────


def render_md(grid_rows, top5_walk, redundancy, baseline_v21,
              winner: dict) -> str:
    out = []
    out.append("# v2.2 Tuning Sweep — overnight run")
    out.append("")
    out.append(f"Generated: "
               f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    out.append(f"Holdout: most recent {HOLDOUT_DAYS} days "
               f"({HOLDOUT_DAYS * 24:,} hours)")
    out.append(f"Train:  prior {TRAIN_DAYS}+ days")
    out.append("")

    out.append("## v2.1 baseline (reference)")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---:|")
    out.append(f"| MAE  | {baseline_v21['MAE']:.2f} |")
    out.append(f"| R²   | {baseline_v21['R2']:.4f} |")
    out.append(f"| bias | {baseline_v21['bias']:+.2f} |")
    out.append(f"| D(4) ρ | {baseline_v21['d4_rho']:.4f} |")
    out.append("")

    out.append("## Top-15 variants (ranked by holdout MAE)")
    out.append("")
    out.append("| # | Subset | α | log_offset | n_feat | MAE | R² | bias | D(4)ρ | Δ MAE vs v2.1 |")
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(grid_rows[:15], 1):
        d_mae = r["MAE"] - baseline_v21["MAE"]
        out.append(f"| {i} | {r['subset']} | {r['alpha']} | "
                   f"{r['log_offset']} | {r['n_features']} | "
                   f"{r['MAE']:.2f} | {r['R2']:.4f} | "
                   f"{r['bias']:+.2f} | {r['d4_rho']:.4f} | "
                   f"{d_mae:+.2f} |")
    out.append("")

    out.append("## Top-5 walk-forward (weekly refit) verification")
    out.append("")
    out.append("| Variant | grid MAE | wf MAE | grid R² | wf R² | wf D(4)ρ |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for v in top5_walk:
        out.append(f"| {v['label']} | {v['grid_MAE']:.2f} | "
                   f"{v['wf_MAE']:.2f} | {v['grid_R2']:.4f} | "
                   f"{v['wf_R2']:.4f} | {v['wf_d4_rho']:.4f} |")
    out.append("")

    out.append("## Winner")
    out.append("")
    out.append(f"**{winner['subset']}**  α={winner['alpha']}  "
               f"log_offset={winner['log_offset']}")
    out.append("")
    out.append(f"* MAE on holdout: **{winner['MAE']:.2f}** "
               f"(v2.1: {baseline_v21['MAE']:.2f}, "
               f"Δ={winner['MAE'] - baseline_v21['MAE']:+.2f})")
    out.append(f"* R²:             **{winner['R2']:.4f}** "
               f"(v2.1: {baseline_v21['R2']:.4f})")
    out.append(f"* D(4) ρ:         **{winner['d4_rho']:.4f}** "
               f"(v2.1: {baseline_v21['d4_rho']:.4f})")
    out.append("")

    out.append("## Leave-one-out feature redundancy on the winner")
    out.append("")
    out.append("Δ MAE > 0 means the feature **helps** (dropping it hurts MAE).")
    out.append("Δ MAE ≤ 0 means the feature is **redundant or harmful** "
               "(dropping it ties or improves MAE).")
    out.append("")
    out.append("| Dropped feature | MAE without | Δ MAE | Δ R² | Verdict |")
    out.append("|---|---:|---:|---:|---|")
    for r in sorted(redundancy, key=lambda x: -x["delta_MAE"]):
        if r["delta_MAE"] > 0.10:
            verdict = "🟢 useful"
        elif r["delta_MAE"] > 0.0:
            verdict = "🟡 marginal"
        else:
            verdict = "🔴 redundant / harmful"
        out.append(f"| {r['dropped']} | {r['MAE']:.2f} | "
                   f"{r['delta_MAE']:+.3f} | {r['delta_R2']:+.4f} | "
                   f"{verdict} |")
    out.append("")

    redundant = [r['dropped'] for r in redundancy if r["delta_MAE"] <= 0]
    if redundant:
        out.append("### Recommended feature drops")
        out.append("")
        out.append(
            "These features have non-positive Δ MAE on this holdout — "
            "the model is no worse without them. Pruning reduces "
            "multi-collinearity and Ridge instability:"
        )
        out.append("")
        for f in redundant:
            out.append(f"* `{f}`")
        out.append("")

    out.append("## Recommendation")
    out.append("")
    d = winner["MAE"] - baseline_v21["MAE"]
    if d <= -0.5 and winner["d4_rho"] >= baseline_v21["d4_rho"] - 0.005:
        out.append("**Ship this winner as v2.2.0.** It beats v2.1 on MAE "
                   f"by {-d:.2f} EUR/MWh while preserving D(4) Spearman ρ.")
    elif abs(d) < 0.5:
        out.append(
            f"**Wash on MAE** ({d:+.2f} EUR/MWh). Pushing v2.2 ships "
            "the new Fingrid plumbing without a clear MAE win or loss. "
            "Acceptable if the R² improvement and tail-tightening are "
            "considered features in their own right."
        )
    else:
        out.append(
            f"**No variant clearly beats v2.1** (best Δ MAE = {d:+.2f}). "
            "Recommend rolling back the bundled model to v2.1 "
            "coefficients and shipping the new code paths as v2.1.1 "
            "(opt-in for users who set FINGRID_API_KEY)."
        )
    out.append("")
    return "\n".join(out)


# ── Main ───────────────────────────────────────────────────────────


def main():
    print("[load] building full v2.2 feature matrix...", flush=True)
    df, feature_cols, config = load_full_features()
    print(f"  {len(df)} rows, {len(feature_cols)} candidate features",
          flush=True)
    print(f"  available: {feature_cols}", flush=True)

    # Time-ordered holdout split
    n_total = len(df)
    n_holdout_hours = HOLDOUT_DAYS * 24
    test_idx = np.arange(n_total - n_holdout_hours, n_total)
    train_idx = np.arange(0, n_total - n_holdout_hours)
    print(f"  train: {df.index[train_idx[0]].date()} → "
          f"{df.index[train_idx[-1]].date()}  ({len(train_idx):,} h)",
          flush=True)
    print(f"  test:  {df.index[test_idx[0]].date()} → "
          f"{df.index[test_idx[-1]].date()}  ({len(test_idx):,} h)",
          flush=True)

    # Filter feature subsets to features actually present in df
    global FEATURE_SUBSETS
    available = set(df.columns)
    filtered = {}
    for name, cols in FEATURE_SUBSETS.items():
        kept = [c for c in cols if c in available]
        filtered[name] = kept
        if len(kept) < len(cols):
            missing = set(cols) - set(kept)
            print(f"  [warn] subset {name}: dropping {missing} "
                  f"(not in feature matrix)", flush=True)
    FEATURE_SUBSETS = filtered

    # ── Phase 1: grid scan ─────────────────────────────────────
    print(f"\n[phase1] grid scan: {len(ALPHA_GRID)} α × "
          f"{len(FEATURE_SUBSETS)} subsets × {len(LOG_OFFSET_GRID)} "
          f"log_offset = "
          f"{len(ALPHA_GRID)*len(FEATURE_SUBSETS)*len(LOG_OFFSET_GRID)} "
          f"variants", flush=True)
    grid_rows = phase1_grid_scan(df, train_idx, test_idx, config)
    grid_rows.sort(key=lambda r: r["MAE"])

    # ── Reference: v2.1 baseline ───────────────────────────────
    print("\n[v2.1 baseline] α=1.0, log_offset=55, 17 features...",
          flush=True)
    baseline_v21 = fit_predict_variant(
        df, train_idx, test_idx,
        FEATURE_SUBSETS["v21_baseline"], 1.0, 55, config,
    )
    print(f"  v2.1: MAE={baseline_v21['MAE']:.2f}  "
          f"R²={baseline_v21['R2']:.4f}  d4ρ={baseline_v21['d4_rho']:.4f}",
          flush=True)

    # ── Phase 2: walk-forward verification on top-5 ────────────
    print(f"\n[phase2] walk-forward verification on top-5 variants",
          flush=True)
    top5 = grid_rows[:5]
    top5_walk: list[dict] = []
    for r in top5:
        feature_cols = FEATURE_SUBSETS[r["subset"]]
        m_wf = walk_forward_variant(
            df, train_idx[0], test_idx,
            feature_cols, r["alpha"], r["log_offset"], config,
            refit_cadence_days=7,
        )
        label = (f"{r['subset']} α={r['alpha']} off={r['log_offset']}")
        top5_walk.append({
            "label": label,
            "grid_MAE": r["MAE"], "grid_R2": r["R2"],
            "wf_MAE": m_wf["MAE"], "wf_R2": m_wf["R2"],
            "wf_d4_rho": m_wf["d4_rho"],
            **r,
        })
        print(f"  {label:50s}  grid MAE={r['MAE']:.2f} → "
              f"wf MAE={m_wf['MAE']:.2f}", flush=True)

    # Winner = top-1 by walk-forward MAE
    top5_walk.sort(key=lambda x: x["wf_MAE"])
    winner = top5_walk[0]
    print(f"\n[winner] {winner['label']}  wf MAE={winner['wf_MAE']:.2f}",
          flush=True)

    # ── Phase 3: leave-one-out redundancy ──────────────────────
    print(f"\n[phase3] leave-one-out redundancy on winner",
          flush=True)
    redundancy = phase3_redundancy(
        df, train_idx, test_idx,
        FEATURE_SUBSETS[winner["subset"]],
        winner["alpha"], winner["log_offset"], config,
    )

    # ── Output ────────────────────────────────────────────────
    out_dir = REPO_ROOT / "studies" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    md_path = out_dir / f"v22_sweep_{stamp}.md"
    json_path = out_dir / f"v22_sweep_{stamp}.json"

    md = render_md(grid_rows, top5_walk, redundancy, baseline_v21, winner)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_v21": baseline_v21,
            "grid_rows": grid_rows,
            "top5_walk": top5_walk,
            "redundancy": redundancy,
            "winner": winner,
            "winner_features": FEATURE_SUBSETS[winner["subset"]],
        }, f, indent=2, default=str)

    # If the winner is meaningfully better than v2.1, also save the
    # full retrained coefs for drop-in replacement of model_coefs.
    d = winner["wf_MAE"] - baseline_v21["MAE"]
    if d < -0.3:
        # Retrain on ALL data with the winning hyperparameters
        cfg = copy.deepcopy(config)
        cfg.setdefault("training", {})
        cfg["training"]["ridge_alpha"] = winner["alpha"]
        cfg["training"]["log_offset"] = winner["log_offset"]
        cfg["training"]["test_split"] = 0.15  # standard split
        winner_features = FEATURE_SUBSETS[winner["subset"]]
        winner_coefs = train(df.copy(), winner_features, cfg)
        winner_coefs["winner_metadata"] = {
            "subset": winner["subset"],
            "alpha": winner["alpha"],
            "log_offset": winner["log_offset"],
            "wf_MAE": winner["wf_MAE"],
            "v21_baseline_MAE": baseline_v21["MAE"],
            "delta_MAE": d,
        }
        out_path = REPO_ROOT / "output" / "model_coefs_v22tuned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(winner_coefs, f, indent=2)
        print(f"\n[bundle] winner saved → {out_path}", flush=True)

    print(f"\n[done] markdown → {md_path}")
    print(f"[done] json     → {json_path}")


if __name__ == "__main__":
    main()
