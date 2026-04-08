"""
Main training pipeline: fetch data, build features, train Ridge regression, export.

Usage:
    python -m src.train_model --region finland [--years 4] [--half-life 365]

The pipeline adapts to available data sources:
  - Tier 1 (28 features): Always available (Sahkotin + Open-Meteo)
  - Tier 2 (+6 features): If mgrey.se + Elering reachable
  - Tier 3 (+4 features): If FINGRID_API_KEY env var is set
"""

import argparse
import gc
import json
import logging
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
PW_BREAKS_DEFAULT = [20, 40, 120]


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
    """Train the two-stage Ridge regression model.

    Returns:
        Dict of model coefficients suitable for JSON serialization.
    """
    training = config.get("training", {})
    half_life = training.get("half_life_days", 365)
    alpha = training.get("ridge_alpha", 1.0)
    test_split = training.get("test_split", 0.15)
    batch_size = training.get("batch_size", BATCH_SIZE)
    pw_breaks_cfg = training.get("piecewise_breakpoints", PW_BREAKS_DEFAULT)

    X_raw = df[feature_cols].values.astype(np.float32)
    y = df["price_clipped"].values.astype(np.float64)

    # Auto breakpoints: use percentiles of training price distribution
    if pw_breaks_cfg == "auto":
        p50 = float(np.percentile(y, 50))
        p75 = float(np.percentile(y, 75))
        p95 = float(np.percentile(y, 95))
        pw_breaks = [round(p50, 1), round(p75, 1), round(p95, 1)]
        logger.info("Auto breakpoints from percentiles (p50/p75/p95): %s EUR/MWh", pw_breaks)
    else:
        pw_breaks = list(pw_breaks_cfg)

    # Time-ordered split
    split = int(len(X_raw) * (1.0 - test_split))
    X_tr, X_te = X_raw[:split], X_raw[split:]
    y_tr, y_te = y[:split], y[split:]

    logger.info("Training: %d train, %d test, %d features",
                len(X_tr), len(X_te), len(feature_cols))

    # Time-decay weights
    weights = _make_time_weights(len(X_tr), half_life)
    logger.info("Time-decay: half-life=%dd, oldest_weight=%.4f", half_life, weights[0])

    # ── Stage 1: Base model ──────────────────────────────────────────
    logger.info("Stage 1: Ridge regression on %d features...", X_tr.shape[1])

    feat_mean, feat_std = _batched_stats(X_tr, weights, batch_size)
    y_mean = float(np.average(y_tr, weights=weights))

    coefs_std = _solve_normal_eq(
        X_tr, y_tr, y_mean, feat_mean, feat_std, weights, alpha, batch_size
    )

    # Un-standardise to original feature scale
    coefs_orig = coefs_std / feat_std
    intercept = y_mean - (feat_mean / feat_std) @ coefs_std

    # Evaluate stage 1
    from sklearn.metrics import mean_absolute_error, r2_score
    preds_te = _predict(X_te, coefs_orig, intercept)
    mae1 = mean_absolute_error(y_te, preds_te)
    r2_1 = r2_score(y_te, preds_te)
    logger.info("  Stage 1: MAE=%.2f EUR/MWh, R2=%.4f", mae1, r2_1)

    # Store stage 1 coefficients
    stage1_coefs = {
        "intercept": float(intercept),
        "features": [
            {"name": name, "coef": float(c)}
            for name, c in zip(feature_cols, coefs_orig)
        ],
    }

    # ── Stage 2: Piecewise calibration ────────────────────────────────
    logger.info("Stage 2: Piecewise calibration (breakpoints: %s)...", pw_breaks)

    # Stage 1 predictions on train and test
    s1_tr = _predict(X_tr, coefs_orig, intercept)
    s1_te = preds_te

    # Augment with piecewise ReLU features
    pw_names = ["stage1_pred"] + [f"pw_relu_{bp}" for bp in pw_breaks]
    pw_tr = np.column_stack(
        [s1_tr] + [np.maximum(0.0, s1_tr - bp) for bp in pw_breaks]
    )
    pw_te = np.column_stack(
        [s1_te] + [np.maximum(0.0, s1_te - bp) for bp in pw_breaks]
    )

    X_tr_aug = np.hstack([X_tr, pw_tr.astype(np.float32)])
    X_te_aug = np.hstack([X_te, pw_te.astype(np.float32)])
    aug_names = feature_cols + pw_names

    feat_mean2, feat_std2 = _batched_stats(X_tr_aug, weights, batch_size)
    coefs_std2 = _solve_normal_eq(
        X_tr_aug, y_tr, y_mean, feat_mean2, feat_std2, weights, alpha, batch_size
    )
    coefs_orig2 = coefs_std2 / feat_std2
    intercept2 = y_mean - (feat_mean2 / feat_std2) @ coefs_std2

    preds2_te = _predict(X_te_aug, coefs_orig2, intercept2)
    mae2 = mean_absolute_error(y_te, preds2_te)
    r2_2 = r2_score(y_te, preds2_te)
    logger.info("  Stage 2: MAE=%.2f EUR/MWh, R2=%.4f", mae2, r2_2)

    # ── Build output dict ─────────────────────────────────────────────
    # Determine active tiers
    tier_info = {"tier1": True, "tier2": False, "tier3": False}
    for name in feature_cols:
        if name.startswith(("import_potential_", "export_potential_")):
            tier_info["tier2"] = True
        if name.startswith(("nuclear_", "flow_fi_", "import_capacity_")):
            tier_info["tier3"] = True

    coefs_dict = {
        "model_version": "v5.0",
        "intercept": float(intercept2),
        "piecewise_breakpoints": pw_breaks,
        "feature_count": len(aug_names),
        "feature_names": aug_names,
        "tier_info": tier_info,
        "metrics": {
            "stage1_mae": float(mae1),
            "stage1_r2": float(r2_1),
            "stage2_mae": float(mae2),
            "stage2_r2": float(r2_2),
            "train_samples": int(len(X_tr)),
            "test_samples": int(len(X_te)),
        },
        "stage1": stage1_coefs,
        "features": [
            {"name": name, "coef": float(c)}
            for name, c in zip(aug_names, coefs_orig2)
        ],
    }

    # Cleanup
    del X_tr, X_te, X_tr_aug, X_te_aug, pw_tr, pw_te
    gc.collect()

    return coefs_dict


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
    parser.add_argument("--skip-tier2", action="store_true",
                        help="Skip Tier 2 (cross-border prices)")
    parser.add_argument("--skip-tier3", action="store_true",
                        help="Skip Tier 3 (grid data)")
    args = parser.parse_args()

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

    # ── Fetch data ────────────────────────────────────────────────────
    prices = fetch_prices(config, start_dt, end_dt)
    weather = fetch_weather(
        config,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )

    # Save data artifacts
    prices.to_frame().to_parquet(out_dir / "fi_prices.parquet")
    weather.to_parquet(out_dir / "fi_weather.parquet")

    # Tier 2: Cross-border prices
    neighbor_prices = None
    if not args.skip_tier2:
        neighbor_prices = fetch_neighbor_prices(config, start_dt, end_dt)
        if not neighbor_prices:
            logger.info("No neighbor prices available, Tier 2 disabled")
            neighbor_prices = None
        else:
            # Cache for evaluation
            pd.DataFrame(neighbor_prices).to_parquet(out_dir / "fi_neighbor_prices.parquet")

    # Tier 3: Grid data
    grid_data = None
    if not args.skip_tier3:
        grid_data = fetch_grid_data(config, start_dt, end_dt)
        if not grid_data:
            logger.info("No grid data available, Tier 3 disabled")
            grid_data = None
        else:
            pd.DataFrame(grid_data).to_parquet(out_dir / "fi_grid_data.parquet")

    # ── Build features ────────────────────────────────────────────────
    df, feature_cols = build_features(
        prices, weather, config,
        neighbor_prices=neighbor_prices,
        grid_data=grid_data,
    )
    del prices, weather, neighbor_prices, grid_data
    gc.collect()

    # ── Train model ───────────────────────────────────────────────────
    coefs = train(df, feature_cols, config)
    del df
    gc.collect()

    # ── Save results ──────────────────────────────────────────────────
    coefs_path = out_dir / "model_coefs.json"
    with open(coefs_path, "w") as f:
        json.dump(coefs, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Features:   %d", coefs["feature_count"])
    logger.info("  Tiers:      %s", coefs["tier_info"])
    logger.info("  Stage 1:    MAE=%.2f, R2=%.4f",
                coefs["metrics"]["stage1_mae"], coefs["metrics"]["stage1_r2"])
    logger.info("  Stage 2:    MAE=%.2f, R2=%.4f",
                coefs["metrics"]["stage2_mae"], coefs["metrics"]["stage2_r2"])
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
