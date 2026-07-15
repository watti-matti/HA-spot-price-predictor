"""
Main training pipeline: fetch data, build features, train Ridge regression, export.

Usage:
    python -m src.train_model --region finland --fingrid-key YOUR_KEY [--years 4]

The pipeline adapts to available data sources:
  - Base (11 features): Always available (Sahkotin + Open-Meteo)
  - Cross-border (+4 features): If elprisetjustnu.se + Elering reachable
  - Nuclear (+2 features): If --fingrid-key provided or FINGRID_API_KEY env var set
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data_sources import fetch_prices, fetch_weather, fetch_neighbor_prices, fetch_grid_data
from src.features import build_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 512
PW_BREAKS_DEFAULT = [1.0]  # Single knee: dampens near-zero noise, no high-price ceiling


# ---------------------------------------------------------------------------
# Provenance (reproducibility metadata)
# ---------------------------------------------------------------------------

def _git_info(repo_root: Path) -> dict:
    """Best-effort git SHA + dirty flag of the training code."""
    import subprocess

    def _run(args):
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        return {"sha": _run(["rev-parse", "HEAD"]) or None,
                "dirty": bool(_run(["status", "--porcelain"]))}
    except Exception:
        return {"sha": None, "dirty": None}


def _content_hash(frame) -> str:
    """Deterministic short sha256 of a Series/DataFrame's values + index.

    Hashes the data *content*, so identical numbers give an identical hash
    regardless of parquet re-encoding — the anchor for "same data in ->
    same model out".
    """
    try:
        h = pd.util.hash_pandas_object(frame, index=True).values
        return hashlib.sha256(h.tobytes()).hexdigest()[:16]
    except Exception:
        return "?"


def _source_prov(obj) -> dict | None:
    """Row count, columns, date range, and content hash for one input.

    The content hash is the key drift detector: the weather series comes
    from Open-Meteo's *mutable* historical-forecast archive, so a changed
    hash for the same window flags that the provider revised the archive
    (e.g. after new industrial PV capacity shifts the modelled solar).
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        try:
            obj = pd.DataFrame(obj)
        except Exception:
            return {"keys": sorted(map(str, obj.keys()))}
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    idx = obj.index
    return {
        "rows": int(len(obj)),
        "columns": [str(c) for c in obj.columns],
        "start": str(idx.min()) if len(idx) else None,
        "end": str(idx.max()) if len(idx) else None,
        "content_sha256_16": _content_hash(obj),
    }


def _store_snapshot() -> str | None:
    """The data-store ``snapshot_id`` this model trained on, linking the
    model to an exact set of source data. None if the store isn't in use."""
    try:
        from src.data_store import load_manifest
        return (load_manifest() or {}).get("snapshot_id")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Time-decay weighting
# ---------------------------------------------------------------------------

def _make_time_weights(n_samples: int, half_life_days: int) -> np.ndarray:
    """Exponential decay weights: newest sample = 1.0, oldest decays toward 0.

    Weights are normalised so they sum to n_samples (preserves Ridge alpha scale).
    """
    if half_life_days <= 0:
        return np.ones(n_samples, dtype=np.float64)
    decay = np.log(2.0) / (half_life_days * 24.0)
    age = np.arange(n_samples - 1, -1, -1, dtype=np.float64)
    w = np.exp(-decay * age)
    w *= n_samples / w.sum()
    return w


# ---------------------------------------------------------------------------
# Batched Ridge regression solver
# ---------------------------------------------------------------------------

def _batched_stats(
    X: np.ndarray, weights: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted per-feature mean and std in batches."""
    P = X.shape[1]
    w_sum = 0.0
    w_xsum = np.zeros(P, dtype=np.float64)
    w_x2sum = np.zeros(P, dtype=np.float64)

    for i in range(0, len(X), batch_size):
        Xb = X[i:i + batch_size].astype(np.float64)
        wb = weights[i:i + Xb.shape[0], np.newaxis]
        w_sum += wb.sum()
        w_xsum += (wb * Xb).sum(axis=0)
        w_x2sum += (wb * Xb ** 2).sum(axis=0)

    mean = w_xsum / w_sum
    var = np.maximum(0.0, w_x2sum / w_sum - mean ** 2)
    std = np.sqrt(var)
    std[std < 1e-10] = 1.0
    return mean, std


def _solve_normal_eq(
    X: np.ndarray, y: np.ndarray, y_mean: float,
    feat_mean: np.ndarray, feat_std: np.ndarray,
    weights: np.ndarray, alpha: float, batch_size: int,
) -> np.ndarray:
    """Solve weighted Ridge regression via batched normal equations."""
    P = X.shape[1]
    XtX = np.zeros((P, P), dtype=np.float64)
    Xty = np.zeros(P, dtype=np.float64)

    for i in range(0, len(X), batch_size):
        Xb = X[i:i + batch_size].astype(np.float64)
        Xb = (Xb - feat_mean) / feat_std
        yb = y[i:i + batch_size] - y_mean
        wb = weights[i:i + Xb.shape[0]]
        Xbw = Xb * wb[:, np.newaxis]
        XtX += Xbw.T @ Xb
        Xty += Xbw.T @ yb
        del Xb, Xbw

    XtX += alpha * np.eye(P)
    return np.linalg.solve(XtX, Xty)


def _predict(
    X: np.ndarray, coefs: np.ndarray, intercept: float
) -> np.ndarray:
    """Predict with linear model."""
    return X.astype(np.float64) @ coefs + intercept


# ---------------------------------------------------------------------------
# Two-stage training
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict[str, Any],
) -> dict:
    """Train log-linear Ridge with adaptive power-stretch calibration.

    1. Fit Ridge on log(price + offset) target
    2. Fit power stretch scale * max(0, raw)^power via Nelder-Mead
    3. Final prediction: scale * max(0, exp(linear) - offset) ^ power

    The power stretch extends the prediction range to 65+ EUR/MWh
    while maintaining rank concordance. Parameters are fitted on
    training data and stored for inference.

    Returns:
        Dict of model coefficients suitable for JSON serialization.
    """
    training = config.get("training", {})
    half_life = training.get("half_life_days", 365)
    alpha = training.get("ridge_alpha", 1.0)
    test_split = training.get("test_split", 0.15)
    batch_size = training.get("batch_size", BATCH_SIZE)
    log_offset = training.get("log_offset", 55)

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df["price_eur_mwh"].values.astype(np.float64)

    # Log-transform target (clip extreme negatives to avoid log(<=0))
    y_shifted = y_raw + log_offset
    n_clipped = int(np.sum(y_shifted <= 0))
    if n_clipped > 0:
        logger.warning("  Clipping %d extreme negative prices (< -%.0f EUR/MWh)",
                        n_clipped, log_offset)
        y_shifted = np.maximum(y_shifted, 1.0)
    y = np.log(y_shifted)
    logger.info("Log-transform: offset=%d, target range [%.2f, %.2f]",
                log_offset, y.min(), y.max())

    # Time-ordered split
    split = int(len(X_raw) * (1.0 - test_split))
    X_tr, X_te = X_raw[:split], X_raw[split:]
    y_tr, y_te = y[:split], y[split:]
    y_raw_te = y_raw[split:]

    logger.info("Training: %d train, %d test, %d features",
                len(X_tr), len(X_te), len(feature_cols))

    # Time-decay weights
    weights = _make_time_weights(len(X_tr), half_life)
    logger.info("Time-decay: half-life=%dd, oldest_weight=%.4f", half_life, weights[0])

    # ── Ridge regression in log-space ────────────────────────────────
    logger.info("Log-linear Ridge on %d features...", X_tr.shape[1])

    feat_mean, feat_std = _batched_stats(X_tr, weights, batch_size)
    y_mean = float(np.average(y_tr, weights=weights))

    coefs_std = _solve_normal_eq(
        X_tr, y_tr, y_mean, feat_mean, feat_std, weights, alpha, batch_size
    )

    # Un-standardise to original feature scale
    coefs_orig = coefs_std / feat_std
    intercept = y_mean - (feat_mean / feat_std) @ coefs_std

    # Evaluate: log-linear back-transform + power stretch
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.optimize import minimize as sp_minimize

    preds_log_te = _predict(X_te, coefs_orig, intercept)
    preds_raw_te = np.maximum(0, np.exp(preds_log_te) - log_offset)

    # Fit power stretch on training predictions
    preds_log_tr = _predict(X_tr, coefs_orig, intercept)
    preds_raw_tr = np.maximum(0, np.exp(preds_log_tr) - log_offset)
    train_actual = np.maximum(y_raw[:split], 0)

    pw_exp_min = training.get("power_exp_min", 0.5)
    pw_exp_max = training.get("power_exp_max", 3.0)
    pw_maxiter = training.get("power_optim_maxiter", 1000)
    pw_tol = training.get("power_optim_tol", 0.001)

    def _obj_power(params):
        scale, power = params
        if scale <= 0 or power < pw_exp_min or power > pw_exp_max:
            return 1e6
        pred = scale * np.power(preds_raw_tr + 1e-10, power)
        return float(np.average(np.abs(train_actual - pred), weights=weights))

    opt = sp_minimize(_obj_power, [1.0, 1.0], method="Nelder-Mead",
                      options={"maxiter": pw_maxiter, "xatol": pw_tol})
    power_scale = float(opt.x[0])
    power_exp = float(opt.x[1])
    logger.info("  Power stretch: scale=%.4f, power=%.4f (Nelder-Mead)", power_scale, power_exp)

    # Apply power stretch to test predictions
    preds_te = power_scale * np.power(preds_raw_te + 1e-10, power_exp)
    mae1 = mean_absolute_error(y_raw_te, preds_te)
    r2_1 = r2_score(y_raw_te, preds_te)
    logger.info("  MAE=%.2f EUR/MWh, R2=%.4f, max_pred=%.1f, min_pred=%.2f",
                mae1, r2_1, preds_te.max(), preds_te.min())

    # ── Build output dict ─────────────────────────────────────────────
    data_sources = {"weather": True, "neighbor_prices": False, "nuclear": False}
    for name in feature_cols:
        if name.startswith(("import_potential_", "export_potential_", "ar_")):
            data_sources["neighbor_prices"] = True
        if name.startswith(("nuclear_",)):
            data_sources["nuclear"] = True

    coefs_dict = {
        "model_version": "v2.0.0",
        "model_type": "log-linear",
        "log_offset": log_offset,
        "power_scale": power_scale,
        "power_exp": power_exp,
        "intercept": float(intercept),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "data_sources": data_sources,
        "metrics": {
            "mae": float(mae1),
            "r2": float(r2_1),
            "max_prediction": float(preds_te.max()),
            "train_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
        },
        "features": [
            {"name": name, "coef": float(c)}
            for name, c in zip(feature_cols, coefs_orig)
        ],
    }

    # Cleanup
    del X_tr, X_te
    gc.collect()

    return coefs_dict


# ---------------------------------------------------------------------------
# Duration model: D(k) = avg price for cheapest k hours
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Duration model defaults (overridden by config.duration_model)
# ---------------------------------------------------------------------------
_DURATION_DEFAULTS = {
    "segments": {
        "night":   [22, 23, 0, 1, 2, 3, 4, 5, 6],  # 9h aligned with night tariff 22-07
        "morning": [7, 8, 9, 10, 11],                # 5h day tariff
        "midday":  [12, 13, 14, 15, 16, 17],          # 6h day tariff
        "evening": [18, 19, 20, 21],                   # 4h day tariff 18-22
    },
    "features": [
        "wind_mean", "solar_mean", "hdd_mean", "se3_mean", "se1_mean",
        "nuclear_deficit", "is_workday", "month_sin", "month_cos",
        "wind_log_scarcity",
        # v2.2 net-load
        "net_load_mean", "net_load_squared_mean",
    ],
    "log_offset": 55,
    "lambda": 0.990,
    "ridge_alpha": 1.0,
    "min_train_days": 180,
    "exp_cap": 20.0,
}


def _get_duration_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge duration_model config with defaults."""
    dc = config.get("duration_model", {})
    merged = dict(_DURATION_DEFAULTS)
    merged.update({k: v for k, v in dc.items() if v is not None})
    # Ensure segments is a dict of lists (YAML may load as-is)
    return merged


def train_duration_model(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict | None:
    """Train D(k) duration model: Ridge per (segment, k) with forgetting factor.

    Uses segment-level features and log(D(k) + offset) targets.
    Coefficients are exported on original feature scale (un-standardised).

    Returns:
        Duration model dict for JSON serialization, or None if insufficient data.
    """
    from zoneinfo import ZoneInfo
    from scipy.stats import spearmanr

    # Load duration config (from YAML or defaults)
    dc = _get_duration_config(config)
    DURATION_SEGMENTS = dc["segments"]
    DURATION_FEATURES = dc["features"]
    DURATION_LOG_OFFSET = dc["log_offset"]
    DURATION_LAMBDA = dc["lambda"]
    DURATION_RIDGE_ALPHA = dc["ridge_alpha"]
    DURATION_MIN_TRAIN = dc["min_train_days"]
    DURATION_EXP_CAP = dc["exp_cap"]
    DURATION_SEG_HOURS = {seg: len(hrs) for seg, hrs in DURATION_SEGMENTS.items()}

    logger.info("")
    logger.info("=" * 60)
    logger.info("DURATION MODEL (lambda=%.3f, half-life=%.0fd)",
                DURATION_LAMBDA, -np.log(2) / np.log(DURATION_LAMBDA))
    logger.info("=" * 60)

    tz = ZoneInfo(config.get("region", {}).get("timezone", "Europe/Helsinki"))
    local_idx = df.index.tz_convert(tz)
    fi = df["price_eur_mwh"].values
    hours = local_idx.hour.values
    dates = local_idx.date
    dow = local_idx.dayofweek.values
    months = local_idx.month.values

    # Raw arrays for segment features
    wind = df["wind_speed_weighted"].values
    solar = df["solar_irradiance_weighted"].values
    temp_col = "temperature_weighted"
    if temp_col not in df.columns:
        # fallback: try weather merge
        temp_col = [c for c in df.columns if "temp" in c.lower()]
        temp_col = temp_col[0] if temp_col else None
    temp = df[temp_col].values if temp_col else np.zeros(len(fi))
    hdd_threshold = config.get("demand", {}).get("hdd_threshold", 17.0)
    hdd = np.maximum(0, hdd_threshold - temp)

    # Neighbor prices — use RAW EUR/MWh from parquet (not normalised ar_*)
    # The duration prototype uses raw neighbor prices for segment mean features
    out_dir = Path(config.get("_out_dir", "output"))
    np_path = out_dir / "fi_neighbor_prices.parquet"
    if np_path.exists():
        np_df = pd.read_parquet(np_path)
        se3 = np_df["se3"].reindex(df.index).ffill().bfill().fillna(0).values
        se1 = np_df["se1"].reindex(df.index).ffill().bfill().fillna(0).values
        logger.info("  Neighbor prices loaded (raw EUR/MWh) from %s", np_path)
    else:
        se3 = np.zeros(len(fi))
        se1 = np.zeros(len(fi))
        logger.info("  No neighbor prices found, se3/se1 features = 0")

    # Nuclear deficit (already in 0-1 range in df)
    nuc_col = [c for c in df.columns if "nuclear_deficit" in c.lower()]
    nuc_deficit = df[nuc_col[0]].values if nuc_col else np.zeros(len(fi))

    # v2.2: net-load features (per-segment mean). Falls back to zeros if
    # the upstream Fingrid fetchers were unavailable.
    if "net_load_gw" in df.columns:
        net_load_gw = df["net_load_gw"].values
    else:
        net_load_gw = np.zeros(len(fi))
    if "net_load_squared" in df.columns:
        net_load_sq = df["net_load_squared"].values
    else:
        net_load_sq = np.zeros(len(fi))

    # ── Build segment records ────────────────────────────────────────
    wind_scarcity_base = config.get("features", {}).get("wind_log_scarcity_base", 8.0)
    unique_dates = sorted(set(dates))
    segment_data = {seg: [] for seg in DURATION_SEGMENTS}

    for d in unique_dates:
        d_mask = np.array([dd == d for dd in dates])
        if d_mask.sum() < 20:
            continue
        is_wd = 1.0 if dow[d_mask][0] < 5 else 0.0
        mo = months[d_mask][0]

        for seg_name, seg_hours_list in DURATION_SEGMENTS.items():
            seg_mask = d_mask & np.isin(hours, seg_hours_list)
            n_h = seg_mask.sum()
            if n_h != DURATION_SEG_HOURS[seg_name]:
                continue

            seg_prices = fi[seg_mask]
            sorted_seg = np.sort(seg_prices)
            # Phase A: cheap-end curve (mean of cheapest k hours, k=1..|seg|)
            #          and peak-end curve (mean of priciest k hours, k=1..|seg|).
            # cheap_curve is the legacy `duration_curve`.
            cheap_curve = np.cumsum(sorted_seg) / np.arange(1, len(sorted_seg) + 1)
            sorted_desc = sorted_seg[::-1]
            peak_curve = np.cumsum(sorted_desc) / np.arange(1, len(sorted_desc) + 1)

            seg_wind = wind[seg_mask]
            segment_data[seg_name].append({
                "features": {
                    "wind_mean": float(seg_wind.mean()),
                    "solar_mean": float(solar[seg_mask].mean()),
                    "hdd_mean": float(hdd[seg_mask].mean()),
                    "se3_mean": float(se3[seg_mask].mean()),
                    "se1_mean": float(se1[seg_mask].mean()),
                    "nuclear_deficit": float(nuc_deficit[seg_mask].mean()),
                    "is_workday": is_wd,
                    "month_sin": float(np.sin(2 * np.pi * mo / 12)),
                    "month_cos": float(np.cos(2 * np.pi * mo / 12)),
                    "wind_log_scarcity": float(np.log1p(
                        np.maximum(0, wind_scarcity_base - seg_wind)).mean()),
                    # v2.2: per-segment net-load aggregates. Always present
                    # in the feature dict so model_coefs.json schema is
                    # stable; values are 0.0 when Fingrid forecasts are
                    # unavailable.
                    "net_load_mean": float(net_load_gw[seg_mask].mean()),
                    "net_load_squared_mean": float(net_load_sq[seg_mask].mean()),
                },
                "date": str(d),
                # Legacy alias for cheap end (kept so any consumer reading
                # `duration_curve` continues to work)
                "duration_curve": cheap_curve.tolist(),
                "cheap_curve": cheap_curve.tolist(),
                "peak_curve": peak_curve.tolist(),
            })

    # Check data sufficiency
    min_records = DURATION_MIN_TRAIN + 90
    for seg in DURATION_SEGMENTS:
        n = len(segment_data[seg])
        logger.info("  %s: %d records (%d hours)", seg, n, DURATION_SEG_HOURS[seg])
        if n < min_records:
            logger.warning("  Insufficient data for %s (%d < %d), skipping duration model",
                           seg, n, min_records)
            return None

    # ── Pre-build matrices ───────────────────────────────────────────
    # Phase A: Train both `Y_cheap` (mean-of-cheapest-k targets) and
    # `Y_peak` (mean-of-priciest-k targets) per segment.  The features
    # X are identical for both directions; only the targets differ.
    n_features = len(DURATION_FEATURES)
    seg_matrices = {}
    for seg_name in DURATION_SEGMENTS:
        data = segment_data[seg_name]
        n = len(data)
        n_dur = DURATION_SEG_HOURS[seg_name]
        X = np.array([[d["features"][f] for f in DURATION_FEATURES] for d in data],
                      dtype=np.float64)
        Y_cheap = np.zeros((n_dur, n), dtype=np.float64)
        Y_peak = np.zeros((n_dur, n), dtype=np.float64)
        for k in range(n_dur):
            cheap_vals = np.array(
                [d["cheap_curve"][k] + DURATION_LOG_OFFSET for d in data])
            peak_vals = np.array(
                [d["peak_curve"][k] + DURATION_LOG_OFFSET for d in data])
            Y_cheap[k] = np.log(np.maximum(cheap_vals, 1.0))
            Y_peak[k] = np.log(np.maximum(peak_vals, 1.0))
        seg_matrices[seg_name] = {
            "X": X,
            "Y": Y_cheap,           # legacy alias
            "Y_cheap": Y_cheap,
            "Y_peak": Y_peak,
            "n": n,
            "n_dur": n_dur,
            "dates": [d["date"] for d in data],
        }

    # ── Train: expanding window with lambda forgetting ────────────────
    # Use all data (no test holdout) since we evaluate via rolling Spearman
    segments_out = {}
    all_preds = {}  # date -> {seg: [D(k)...]}

    for seg_name in DURATION_SEGMENTS:
        m = seg_matrices[seg_name]
        X, n, n_dur = m["X"], m["n"], m["n_dur"]
        Y_cheap = m["Y_cheap"]
        Y_peak = m["Y_peak"]
        dates_s = m["dates"]

        logger.info("  Training %s (%d days, %d levels, dual cheap+peak)...",
                    seg_name, n, n_dur)

        # Final model: train on ALL data with lambda weighting
        # Use augmented matrix [X | 1] to fit intercept without penalising it.
        # This ensures exported intercept+coefs predictions exactly match the
        # training objective. Plain centering changes the Ridge penalty and
        # produces different solutions (see test_intercept_coefs_predict_correctly).
        w = DURATION_LAMBDA ** np.arange(n - 1, -1, -1, dtype=np.float64)
        sqrt_w = np.sqrt(w)

        X_aug = np.column_stack([X, np.ones(n)])  # [n x (p+1)]
        Xw_aug = X_aug * sqrt_w[:, None]
        A = Xw_aug.T @ Xw_aug + DURATION_RIDGE_ALPHA * np.eye(n_features + 1)
        A[n_features, n_features] -= DURATION_RIDGE_ALPHA  # don't penalise intercept

        cheap_models_for_seg = []
        peak_models_for_seg = []
        for k in range(n_dur):
            # Cheap-end Ridge fit
            yw_c = Y_cheap[k] * sqrt_w
            beta_c = np.linalg.solve(A, Xw_aug.T @ yw_c)
            cheap_models_for_seg.append({
                "k": k,
                "intercept": round(float(beta_c[n_features]), 8),
                "coefs": [round(float(c), 8) for c in beta_c[:n_features]],
            })
            # Peak-end Ridge fit
            yw_p = Y_peak[k] * sqrt_w
            beta_p = np.linalg.solve(A, Xw_aug.T @ yw_p)
            peak_models_for_seg.append({
                "k": k,
                "intercept": round(float(beta_p[n_features]), 8),
                "coefs": [round(float(c), 8) for c in beta_p[:n_features]],
            })

        segments_out[seg_name] = {
            "hours": DURATION_SEGMENTS[seg_name],
            "n_levels": n_dur,
            # Phase A: dual cheap/peak models
            "cheap_models": cheap_models_for_seg,
            "peak_models": peak_models_for_seg,
            # Legacy alias (kept so old DurationModel JSON readers still work)
            "models": cheap_models_for_seg,
        }

        # Evaluate: expanding window predictions for Spearman
        # Uses same augmented matrix [X|1] approach as final model
        # Cheap-end only — preserves the legacy D(4) Spearman headline.
        for t in range(DURATION_MIN_TRAIN, n):
            wt = DURATION_LAMBDA ** np.arange(t - 1, -1, -1, dtype=np.float64)
            sqrt_wt = np.sqrt(wt)
            Xt_aug = np.column_stack([X[:t], np.ones(t)])
            Xwt_aug = Xt_aug * sqrt_wt[:, None]
            At = Xwt_aug.T @ Xwt_aug + DURATION_RIDGE_ALPHA * np.eye(n_features + 1)
            At[n_features, n_features] -= DURATION_RIDGE_ALPHA  # don't penalise intercept
            x_test_aug = np.append(X[t], 1.0)
            raw = []
            for k in range(n_dur):
                ywt = Y_cheap[k, :t] * sqrt_wt
                bt = np.linalg.solve(At, Xwt_aug.T @ ywt)
                lp = float(bt @ x_test_aug)
                raw.append(max(0.0, np.exp(min(lp, DURATION_EXP_CAP)) - DURATION_LOG_OFFSET))
            # PAVA — non-decreasing for the cheap end
            for i in range(1, len(raw)):
                if raw[i] < raw[i - 1]:
                    j = i
                    while j > 0 and raw[j] < raw[j - 1]:
                        avg = (raw[j] + raw[j - 1]) / 2
                        raw[j] = raw[j - 1] = avg
                        j -= 1

            date_key = dates_s[t]
            if date_key not in all_preds:
                all_preds[date_key] = {}
            all_preds[date_key][seg_name] = raw

    # ── Reconstruct full-day D(k) from segments ──────────────────────
    fullday_lookup = {}
    for d in unique_dates:
        d_mask = np.array([dd == d for dd in dates])
        day_prices = fi[d_mask]
        if len(day_prices) >= 23:
            sorted_p = np.sort(day_prices)
            fullday_lookup[str(d)] = (
                np.cumsum(sorted_p) / np.arange(1, len(sorted_p) + 1)
            ).tolist()

    eval_days = 0
    rho_d4_sum = 0.0
    rho_d8_sum = 0.0
    rho_d24_sum = 0.0
    pred_d4_list = []
    act_d4_list = []

    for date_key in sorted(all_preds.keys()):
        segs = all_preds[date_key]
        if len(segs) != 4 or date_key not in fullday_lookup:
            continue

        # Extract sorted prices from each segment
        pred_prices = []
        for seg_name in DURATION_SEGMENTS:
            curve = segs[seg_name]
            n_h = len(curve)
            for i in range(n_h):
                if i == 0:
                    pred_prices.append(curve[0])
                else:
                    pred_prices.append(max(0.0, (i + 1) * curve[i] - i * curve[i - 1]))
        pred_prices.sort()
        n_p = len(pred_prices)
        pred_dk = [sum(pred_prices[:k + 1]) / (k + 1) for k in range(n_p)]
        actual_dk = fullday_lookup[date_key]
        n_a = min(len(pred_dk), len(actual_dk))

        if n_a >= 24:
            pred_d4_list.append(pred_dk[3])
            act_d4_list.append(actual_dk[3])
            eval_days += 1

    # Last-year Spearman
    n_eval = len(pred_d4_list)
    if n_eval > 365:
        rho_last = float(spearmanr(pred_d4_list[-365:], act_d4_list[-365:]).statistic)
    else:
        rho_last = float(spearmanr(pred_d4_list, act_d4_list).statistic) if n_eval > 10 else 0.0

    rho_all = float(spearmanr(pred_d4_list, act_d4_list).statistic) if n_eval > 10 else 0.0

    logger.info("  Duration eval: %d days, D(4) rho_all=%.3f, rho_last365=%.3f",
                n_eval, rho_all, rho_last)

    # ── Build output ─────────────────────────────────────────────────
    duration_dict = {
        "model_version": "v1.0",
        "log_offset": DURATION_LOG_OFFSET,
        "exp_cap": DURATION_EXP_CAP,
        "lambda": DURATION_LAMBDA,
        "ridge_alpha": DURATION_RIDGE_ALPHA,
        "feature_names": DURATION_FEATURES,
        "segments": segments_out,
        "metrics": {
            "d4_rho_all": round(rho_all, 4),
            "d4_rho_last365": round(rho_last, 4),
            "eval_days": n_eval,
        },
    }

    return duration_dict


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train spot price prediction model")
    parser.add_argument("--region", default="finland",
                        help="Region config name (without .yaml)")
    parser.add_argument("--years", type=int, default=None,
                        help="Override training years from config")
    parser.add_argument("--half-life", type=int, default=None,
                        help="Override time-decay half-life (days)")
    parser.add_argument("--out-dir", default="output",
                        help="Output directory for model artifacts")
    parser.add_argument("--skip-cross-border", "--skip-tier2", action="store_true",
                        dest="skip_cross_border",
                        help="Skip cross-border neighbor prices")
    parser.add_argument("--skip-nuclear", "--skip-tier3", action="store_true",
                        dest="skip_nuclear",
                        help="Skip nuclear/grid data")
    parser.add_argument("--no-piecewise", action="store_true",
                        help="Skip Stage 2 piecewise calibration (Stage 1 only)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load data from cached parquet files instead of fetching")
    parser.add_argument("--skip-duration", action="store_true",
                        help="Skip D(k) duration model training")
    parser.add_argument("--fingrid-key", default=None,
                        help="Fingrid API key (alternative to FINGRID_API_KEY env var)")
    args = parser.parse_args()

    # Set Fingrid API key from CLI arg if provided
    if args.fingrid_key:
        os.environ["FINGRID_API_KEY"] = args.fingrid_key

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load region config
    config_path = Path("config/regions") / f"{args.region}.yaml"
    if not config_path.exists():
        logger.error("Region config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    training = config.setdefault("training", {})
    if args.years is not None:
        training["years"] = args.years
    if args.half_life is not None:
        training["half_life_days"] = args.half_life

    years = training.get("years", 4)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start_dt = end_dt - timedelta(days=365 * years)

    region_name = config.get("region", {}).get("name", args.region)
    logger.info("=" * 60)
    logger.info("Training: %s (%s -> %s, %d years)",
                region_name, start_dt.date(), end_dt.date(), years)
    logger.info("=" * 60)

    # ── Fetch or load data ──────────────────────────────────────────────
    if args.use_cache:
        # Prefer the canonical incremental data store (src/data_store.py);
        # fall back to the run's out_dir parquets for older checkouts.
        from src.data_store import STORE as _STORE
        cache_dir = _STORE if (_STORE / "fi_prices.parquet").exists() else out_dir
        logger.info("Loading cached data from %s", cache_dir)
        prices = pd.read_parquet(cache_dir / "fi_prices.parquet")["price_eur_mwh"]
        weather = pd.read_parquet(cache_dir / "fi_weather.parquet")
        logger.info("  Prices: %d rows, Weather: %d rows", len(prices), len(weather))

        neighbor_prices = None
        if not args.skip_cross_border:
            np_path = cache_dir / "fi_neighbor_prices.parquet"
            if np_path.exists():
                np_df = pd.read_parquet(np_path)
                neighbor_prices = {col: np_df[col] for col in np_df.columns}
                logger.info("  Neighbor prices: %d columns", len(neighbor_prices))

        grid_data = None
        if not args.skip_nuclear:
            gd_path = cache_dir / "fi_grid_data.parquet"
            api_key = os.environ.get("FINGRID_API_KEY", "").strip()
            # Helper: do the cached columns include the v2.2 net-load
            # series? Older caches had only `nuclear_mw`.
            REQUIRED_COLS = {
                "nuclear_mw", "consumption_mw",
                "wind_forecast_mw", "solar_forecast_mw",
            }
            need_refetch = True
            if gd_path.exists():
                gd_df = pd.read_parquet(gd_path)
                if REQUIRED_COLS.issubset(set(gd_df.columns)):
                    grid_data = {col: gd_df[col] for col in gd_df.columns}
                    logger.info("  Grid data: %d columns (cached)",
                                len(grid_data))
                    need_refetch = False
                else:
                    missing = REQUIRED_COLS - set(gd_df.columns)
                    logger.info(
                        "  Grid cache exists but missing %s; re-fetching",
                        missing,
                    )
            # Refetch if cache absent or stale
            if need_refetch and api_key:
                logger.info("  Fetching grid data for v2.2 net-load features")
                grid_data = fetch_grid_data(config, start_dt, end_dt)
                if grid_data:
                    pd.DataFrame(grid_data).to_parquet(gd_path)
                    logger.info("  Grid data: %d columns (refetched)",
                                len(grid_data))
                else:
                    grid_data = None
    else:
        prices = fetch_prices(config, start_dt, end_dt)
        # Persist prices immediately so a later weather timeout doesn't
        # force a re-fetch of prices on the next run.
        prices.to_frame().to_parquet(out_dir / "fi_prices.parquet")

        # Per-location weather cache lives under the output dir, so a
        # re-run after a transient timeout resumes (only the failed
        # location is re-fetched) instead of starting from scratch.
        weather = fetch_weather(
            config,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            cache_dir=out_dir / ".weather_cache",
        )
        weather.to_parquet(out_dir / "fi_weather.parquet")

        # Cross-border neighbor prices
        neighbor_prices = None
        if not args.skip_cross_border:
            neighbor_prices = fetch_neighbor_prices(config, start_dt, end_dt)
            if not neighbor_prices:
                logger.info("No neighbor prices available, cross-border features disabled")
                neighbor_prices = None
            else:
                pd.DataFrame(neighbor_prices).to_parquet(out_dir / "fi_neighbor_prices.parquet")

        # Nuclear/grid data
        grid_data = None
        if not args.skip_nuclear:
            grid_data = fetch_grid_data(config, start_dt, end_dt)
            if not grid_data:
                logger.info("No grid data available, nuclear features disabled")
                grid_data = None
            else:
                pd.DataFrame(grid_data).to_parquet(out_dir / "fi_grid_data.parquet")

    # ── Build features ────────────────────────────────────────────────
    df, feature_cols, ar_models = build_features(
        prices, weather, config,
        neighbor_prices=neighbor_prices,
        grid_data=grid_data,
    )
    # Capture input-data provenance (row counts, date ranges, content
    # hashes) before releasing the frames.
    data_provenance = {
        "prices":          _source_prov(prices),
        "weather":         _source_prov(weather),
        "neighbor_prices": _source_prov(neighbor_prices),
        "grid_data":       _source_prov(grid_data),
    }
    del prices, weather, neighbor_prices, grid_data
    gc.collect()

    # ── Train hourly model ────────────────────────────────────────────
    coefs = train(df, feature_cols, config)

    # Store AR model parameters for inference
    if ar_models:
        coefs["ar_models"] = ar_models

    # ── Train duration model D(k) ────────────────────────────────────
    if not getattr(args, 'skip_duration', False):
        config["_out_dir"] = str(out_dir)
        duration_coefs = train_duration_model(df, config)
        if duration_coefs:
            coefs["duration_model"] = duration_coefs
    else:
        logger.info("Skipping duration model (--skip-duration)")

    del df
    gc.collect()

    # ── Provenance: make the model self-documenting & reproducible ────
    # Every field here is deterministic for a given (data, code, config)
    # EXCEPT `trained_at_utc` (wall clock). Reproducibility is verified by
    # comparing everything else — in particular the per-source
    # `content_sha256_16` hashes, which are identical iff the fetched data
    # is identical. A changed weather hash for the same window is the
    # signal that Open-Meteo revised its historical-forecast archive.
    coefs["provenance"] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_window": {
            "start": start_dt.isoformat(),
            "end":   end_dt.isoformat(),
            "years": years,
        },
        "region": region_name,
        "git": _git_info(Path(__file__).resolve().parents[1]),
        "config": {
            "half_life_days":    training.get("half_life_days"),
            "ridge_alpha":       training.get("ridge_alpha", 1.0),
            "log_offset":        training.get("log_offset", 55),
            "used_cache":        bool(args.use_cache),
            "skip_cross_border": bool(args.skip_cross_border),
            "skip_nuclear":      bool(args.skip_nuclear),
            "skip_duration":     bool(getattr(args, "skip_duration", False)),
        },
        "data": data_provenance,
        "data_store_snapshot": _store_snapshot(),
        "env": {
            "python": sys.version.split()[0],
            "numpy":  np.__version__,
            "pandas": pd.__version__,
        },
    }

    # ── Save results ──────────────────────────────────────────────────
    coefs_path = out_dir / "model_coefs.json"
    with open(coefs_path, "w") as f:
        json.dump(coefs, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Model:      %s", coefs.get("model_type", "linear"))
    logger.info("  Features:   %d", coefs["feature_count"])
    logger.info("  Sources:    %s", coefs["data_sources"])
    logger.info("  MAE:        %.2f EUR/MWh", coefs["metrics"]["mae"])
    logger.info("  R2:         %.4f", coefs["metrics"]["r2"])
    logger.info("  Max pred:   %.1f EUR/MWh", coefs["metrics"]["max_prediction"])
    logger.info("")
    logger.info("  Saved: %s", coefs_path)

    # Feature importance
    logger.info("")
    logger.info("  FEATURE IMPORTANCE (top 15 by |coefficient|):")
    stage2_feats = [
        (f["name"], f["coef"]) for f in coefs["features"]
        if " " not in f["name"]  # Skip polynomial interaction names
    ]
    for name, c in sorted(stage2_feats, key=lambda x: abs(x[1]), reverse=True)[:15]:
        bar = "#" * min(30, int(abs(c) / 2))
        logger.info("    %-35s %+10.3f  %s", name, c, bar)


if __name__ == "__main__":
    main()
