"""
Model evaluation: rebuild predictions from saved artifacts, compute metrics,
and generate an interactive HTML dashboard (Chart.js, dark theme).

Usage:
    python -m src.evaluate --region finland [--out-dir output]

Produces output/evaluation_report.html with:
  - Overall metrics (MAE, RMSE, R2, MAPE)
  - Interactive time series with slider: actual vs predicted + residuals
  - Scatter plot: actual vs predicted with diagonal
  - Residual histogram with gradient coloring
  - Hourly MAE bar chart (blue gradient)
  - Monthly MAE bar chart (pink gradient)
  - Tier contribution comparison table
  - Feature importance horizontal bar chart (color-coded by tier)
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
from src.features import build_features, TIER1_FEATURES
from src.train_model import _make_time_weights, _batched_stats, _solve_normal_eq, _predict

logger = logging.getLogger(__name__)


def _predict_model(
    X: np.ndarray,
    coefs: dict,
) -> np.ndarray:
    """Predict using saved model coefficients (log-linear or legacy two-stage)."""
    model_type = coefs.get("model_type", "linear")

    if model_type == "log-linear":
        # Log-linear: exp(X @ coefs + intercept) - offset
        feature_coefs = np.array([f["coef"] for f in coefs["features"]], dtype=np.float64)
        intercept = coefs["intercept"]
        log_offset = coefs.get("log_offset", 55)
        log_pred = X @ feature_coefs + intercept
        return np.exp(log_pred) - log_offset

    # Legacy: two-stage piecewise model
    pw_breaks = coefs.get("piecewise_breakpoints", [])
    if "stage1" in coefs and pw_breaks:
        s1 = coefs["stage1"]
        s1_coefs = np.array([f["coef"] for f in s1["features"]], dtype=np.float64)
        s1_pred = X @ s1_coefs + s1["intercept"]
        pw_feats = np.column_stack(
            [s1_pred] + [np.maximum(0.0, s1_pred - bp) for bp in pw_breaks]
        )
        X_aug = np.hstack([X, pw_feats])
        s2_coefs = np.array([f["coef"] for f in coefs["features"]], dtype=np.float64)
        return X_aug @ s2_coefs + coefs["intercept"]

    # Simple linear
    feature_coefs = np.array([f["coef"] for f in coefs["features"]], dtype=np.float64)
    return X @ feature_coefs + coefs["intercept"]


def _train_tier_variant(
    df: pd.DataFrame,
    feature_subset: list[str],
    config: dict,
) -> dict:
    """Quick train a model variant for tier comparison. Returns metrics dict."""
    from sklearn.metrics import mean_absolute_error, r2_score

    training = config.get("training", {})
    half_life = training.get("half_life_days", 365)
    alpha = training.get("ridge_alpha", 1.0)
    test_split = training.get("test_split", 0.15)

    X = df[feature_subset].values.astype(np.float32)
    y = df["price_clipped"].values.astype(np.float64)

    split = int(len(X) * (1.0 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    weights = _make_time_weights(len(X_tr), half_life)
    feat_mean, feat_std = _batched_stats(X_tr, weights, 512)
    y_mean = float(np.average(y_tr, weights=weights))
    coefs_std = _solve_normal_eq(X_tr, y_tr, y_mean, feat_mean, feat_std, weights, alpha, 512)
    coefs_orig = coefs_std / feat_std
    intercept = y_mean - (feat_mean / feat_std) @ coefs_std

    preds = _predict(X_te, coefs_orig, intercept)
    return {
        "mae": float(mean_absolute_error(y_te, preds)),
        "r2": float(r2_score(y_te, preds)),
        "n_features": len(feature_subset),
    }


def run_evaluation(config: dict, out_dir: Path) -> None:
    """Run full evaluation and generate HTML report."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    coefs_path = out_dir / "model_coefs.json"
    if not coefs_path.exists():
        logger.error("No model_coefs.json found. Run training first.")
        sys.exit(1)

    with open(coefs_path) as f:
        coefs = json.load(f)

    training = config.get("training", {})
    test_split = training.get("test_split", 0.15)
    years = training.get("years", 4)

    end_dt = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=365 * years)

    logger.info("Loading saved data artifacts...")
    prices = pd.read_parquet(out_dir / "fi_prices.parquet")["price_eur_mwh"]
    weather = pd.read_parquet(out_dir / "fi_weather.parquet")

    # Load cached tier 2/3 data if available
    neighbor_prices = None
    grid_data = None

    neighbor_cache = out_dir / "fi_neighbor_prices.parquet"
    grid_cache = out_dir / "fi_grid_data.parquet"

    if neighbor_cache.exists():
        logger.info("Loading cached neighbor prices from %s", neighbor_cache)
        ndf = pd.read_parquet(neighbor_cache)
        neighbor_prices = {col: ndf[col].dropna() for col in ndf.columns}
    else:
        logger.info("Fetching neighbor prices (no cache found)...")
        neighbor_prices = fetch_neighbor_prices(config, start_dt, end_dt)
        if neighbor_prices:
            ndf = pd.DataFrame(neighbor_prices)
            ndf.to_parquet(neighbor_cache)

    if grid_cache.exists():
        logger.info("Loading cached grid data from %s", grid_cache)
        gdf = pd.read_parquet(grid_cache)
        grid_data = {col: gdf[col].dropna() for col in gdf.columns}
    else:
        logger.info("Fetching grid data (no cache found)...")
        grid_data = fetch_grid_data(config, start_dt, end_dt)
        if grid_data:
            gdf = pd.DataFrame(grid_data)
            gdf.to_parquet(grid_cache)

    # Build features
    df, feature_cols = build_features(
        prices, weather, config,
        neighbor_prices=neighbor_prices if neighbor_prices else None,
        grid_data=grid_data if grid_data else None,
    )

    # Get base feature names from model
    all_names = coefs["feature_names"]
    base_features = [n for n in all_names if not n.startswith(("stage1_pred", "pw_relu_"))]

    # Verify features match -- fill missing with zeros
    missing = set(base_features) - set(df.columns)
    if missing:
        logger.warning("Missing features in data: %s -- filling with zeros", missing)
        for feat in missing:
            df[feat] = 0.0

    X = df[base_features].values.astype(np.float64)
    y_actual = df["price_eur_mwh"].values.astype(np.float64)
    y_clipped = df["price_clipped"].values.astype(np.float64)
    timestamps = df.index

    split = int(len(X) * (1.0 - test_split))

    # Full model predictions
    X_te = X[split:]
    y_te = y_actual[split:]
    y_te_clip = y_clipped[split:]
    ts_te = timestamps[split:]
    preds = _predict_model(X_te, coefs)

    # Also predict full dataset
    preds_all = _predict_model(X, coefs)

    # Overall test metrics
    mae = mean_absolute_error(y_te_clip, preds)
    rmse = np.sqrt(mean_squared_error(y_te_clip, preds))
    r2 = r2_score(y_te_clip, preds)
    mape_mask = np.abs(y_te_clip) > 1.0
    mape = float(np.mean(np.abs((y_te_clip[mape_mask] - preds[mape_mask]) / y_te_clip[mape_mask])) * 100)
    logger.info("Overall: MAE=%.2f, RMSE=%.2f, R2=%.4f, MAPE=%.1f%%", mae, rmse, r2, mape)

    # ── Tier contribution analysis ────────────────────────────────────
    logger.info("Computing tier contributions...")

    tier1_cols = [f for f in TIER1_FEATURES if f in df.columns]
    tier2_cols = [f for f in base_features if f.startswith(("import_potential_", "export_potential_"))]
    tier3_cols = [f for f in base_features if f.startswith(("nuclear_", "flow_fi_"))]

    tier_results = {}
    tier_results["T1 only"] = _train_tier_variant(df, tier1_cols, config)
    if tier2_cols:
        tier_results["T1+2"] = _train_tier_variant(df, tier1_cols + tier2_cols, config)
    if tier3_cols:
        tier_results["T1+2+3"] = _train_tier_variant(df, base_features, config)

    # ── Per-hour / per-month analysis (test set) ──────────────────────
    tz_offset = 2
    local_hours_te = (ts_te + pd.Timedelta(hours=tz_offset)).hour.to_numpy()
    local_months_te = (ts_te + pd.Timedelta(hours=tz_offset)).month.to_numpy()

    # Full dataset hours/months for full MAE
    local_hours_all = (timestamps + pd.Timedelta(hours=tz_offset)).hour.to_numpy()
    local_months_all = (timestamps + pd.Timedelta(hours=tz_offset)).month.to_numpy()

    hourly_mae_test = []
    hourly_mae_full = []
    for h in range(24):
        mask_te = local_hours_te == h
        mask_all = local_hours_all == h
        hourly_mae_test.append(
            float(mean_absolute_error(y_te_clip[mask_te], preds[mask_te]))
            if mask_te.sum() > 0 else 0.0
        )
        hourly_mae_full.append(
            float(mean_absolute_error(y_clipped[mask_all], preds_all[mask_all]))
            if mask_all.sum() > 0 else 0.0
        )

    monthly_mae_test = []
    monthly_mae_full = []
    for m in range(1, 13):
        mask_te = local_months_te == m
        mask_all = local_months_all == m
        monthly_mae_test.append(
            float(mean_absolute_error(y_te_clip[mask_te], preds[mask_te]))
            if mask_te.sum() > 10 else 0.0
        )
        monthly_mae_full.append(
            float(mean_absolute_error(y_clipped[mask_all], preds_all[mask_all]))
            if mask_all.sum() > 10 else 0.0
        )

    # ── Feature importance (normalized by feature std) ────────────────
    # Raw |coefficient| is misleading because features have different scales.
    # Normalized importance = |coefficient| × std(feature) shows actual
    # contribution to prediction variance in EUR/MWh.
    feature_importance = []
    for feat in coefs["features"]:
        name = feat["name"]
        c = feat["coef"]
        if name.startswith(("stage1_pred", "pw_relu_")):
            tier = "Stage2"
        elif name.startswith(("import_potential_", "export_potential_")):
            tier = "T2"
        elif name.startswith(("nuclear_", "flow_fi_")):
            tier = "T3"
        else:
            tier = "T1"

        # Compute feature std for normalization
        if name in df.columns:
            feat_std = float(df[name].std())
        elif name == "stage1_pred":
            # Approximate std of stage1 predictions
            s1_coefs_arr = np.array([f["coef"] for f in coefs["stage1"]["features"]])
            s1_preds = X @ s1_coefs_arr + coefs["stage1"]["intercept"]
            feat_std = float(np.std(s1_preds))
        elif name.startswith("pw_relu_"):
            # Piecewise ReLU std depends on stage1 predictions
            try:
                bp = float(name.split("_")[-1])
                s1_coefs_arr = np.array([f["coef"] for f in coefs["stage1"]["features"]])
                s1_preds = X @ s1_coefs_arr + coefs["stage1"]["intercept"]
                feat_std = float(np.std(np.maximum(0.0, s1_preds - bp)))
            except (ValueError, IndexError):
                feat_std = 1.0
        else:
            feat_std = 1.0

        impact = abs(c) * feat_std
        feature_importance.append({
            "name": name, "coef": c, "abs_coef": abs(c),
            "std": round(feat_std, 4), "impact": round(impact, 4), "tier": tier,
        })
    feature_importance.sort(key=lambda x: x["impact"], reverse=True)

    # ── Residual analysis ───────────────────────────────────────────────
    residuals = y_te_clip - preds

    # Histogram bins
    hist_bins = np.linspace(float(np.percentile(residuals, 1)),
                            float(np.percentile(residuals, 99)), 40)
    hist_counts, hist_edges = np.histogram(residuals, bins=hist_bins)
    hist_centers = [(hist_edges[i] + hist_edges[i + 1]) / 2 for i in range(len(hist_counts))]

    # ── Prepare DATA arrays for JS ─────────────────────────────────────
    # Test set arrays (hourly)
    test_timestamps = [d.strftime("%Y-%m-%dT%H:%M") for d in ts_te]
    test_actual = [round(float(v), 2) for v in y_te]
    test_predicted = [round(float(v), 2) for v in preds]
    test_local_hours = [int(h) for h in local_hours_te]
    test_months = [int(m) for m in local_months_te]

    # Full dataset arrays (hourly)
    full_timestamps = [d.strftime("%Y-%m-%dT%H:%M") for d in timestamps]
    full_actual = [round(float(v), 2) for v in y_actual]
    full_predicted = [round(float(v), 2) for v in preds_all]
    full_local_hours = [int(h) for h in local_hours_all]
    full_months = [int(m) for m in local_months_all]

    # Top 20 features for importance chart
    top_features = feature_importance[:20]

    # ── Generate HTML ──────────────────────────────────────────────────
    logger.info("Generating HTML report...")

    data_json = json.dumps({
        "test": {
            "timestamps": test_timestamps,
            "actual": test_actual,
            "predicted": test_predicted,
            "local_hours": test_local_hours,
            "months": test_months,
        },
        "full": {
            "timestamps": full_timestamps,
            "actual": full_actual,
            "predicted": full_predicted,
            "local_hours": full_local_hours,
            "months": full_months,
        },
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 1),
            "n_test": len(y_te),
            "n_train": split,
        },
        "hourly_mae_test": [round(v, 2) for v in hourly_mae_test],
        "hourly_mae_full": [round(v, 2) for v in hourly_mae_full],
        "monthly_mae_test": [round(v, 2) for v in monthly_mae_test],
        "monthly_mae_full": [round(v, 2) for v in monthly_mae_full],
        "histogram": {
            "centers": [round(c, 2) for c in hist_centers],
            "counts": [int(c) for c in hist_counts],
        },
        "tier_results": {k: {"mae": round(v["mae"], 2), "r2": round(v["r2"], 4),
                              "n_features": v["n_features"]}
                         for k, v in tier_results.items()},
        "feature_importance": [
            {"name": f["name"], "coef": round(f["coef"], 4),
             "abs_coef": round(f["abs_coef"], 4),
             "std": f.get("std", 1.0), "impact": f.get("impact", 0.0),
             "tier": f["tier"]}
            for f in top_features
        ],
        "split_index": split,
        "train_split_date": str(timestamps[split].date()) if split < len(timestamps) else "",
        "ymax_price": round(float(np.percentile(
            np.concatenate([np.array(full_actual), np.array(full_predicted)]), 99.5
        ) * 1.2), 1),
        "ymax_residual": round(float(np.percentile(
            np.abs(np.array(full_actual) - np.array(full_predicted)), 99
        ) * 1.5), 1),
    })

    html = _build_html(data_json)
    out_path = out_dir / "evaluation_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report saved: %s", out_path)


def _build_html(data_json: str) -> str:
    """Build the complete self-contained HTML dashboard."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spot Price Model Evaluation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0f1117; color: #e2e8f0; line-height: 1.5;
}}
.container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.6em; margin-bottom: 4px; color: #f1f5f9; }}
h2 {{
  font-size: 1.15em; color: #94a3b8; margin: 32px 0 12px;
  text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
}}
.subtitle {{ color: #64748b; font-size: 0.9em; margin-bottom: 20px; }}

/* Metric cards */
.metrics-row {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 16px 0 24px;
}}
.metric-card {{
  background: #1e2433; border: 1px solid #2d3748; border-radius: 10px;
  padding: 18px 14px; text-align: center;
}}
.metric-value {{ font-size: 2em; font-weight: 700; }}
.metric-label {{ font-size: 0.78em; color: #64748b; margin-top: 2px; }}
.v-green {{ color: #34d399; }}
.v-yellow {{ color: #fbbf24; }}
.v-blue {{ color: #60a5fa; }}
.v-pink {{ color: #f472b6; }}

/* Panels & charts */
.panel {{
  background: #1e2433; border: 1px solid #2d3748; border-radius: 10px;
  padding: 20px; margin: 16px 0;
}}
.chart-box {{
  background: #1e2433; border: 1px solid #2d3748; border-radius: 10px;
  padding: 16px; margin: 16px 0; position: relative;
}}
.chart-box canvas {{ width: 100% !important; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}

/* Slider controls */
.slider-controls {{
  display: flex; align-items: center; gap: 10px; margin: 10px 0 6px;
  flex-wrap: wrap;
}}
.slider-controls button {{
  background: #2d3748; color: #e2e8f0; border: 1px solid #4a5568;
  border-radius: 6px; padding: 4px 14px; cursor: pointer; font-size: 0.82em;
}}
.slider-controls button:hover {{ background: #4a5568; }}
.slider-controls button.active {{
  background: #60a5fa; color: #0f1117; border-color: #60a5fa;
}}
.slider-controls input[type=range] {{
  flex: 1; min-width: 120px; accent-color: #60a5fa;
}}
.slider-label {{ font-size: 0.8em; color: #64748b; min-width: 90px; text-align: right; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
th {{
  text-align: left; padding: 10px 12px; background: #2d3748;
  color: #94a3b8; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px;
}}
td {{ padding: 9px 12px; border-bottom: 1px solid #2d3748; }}
tr:hover td {{ background: rgba(96,165,250,0.06); }}
.improve {{ color: #34d399; }}
footer {{
  text-align: center; color: #475569; font-size: 0.75em;
  margin-top: 40px; padding: 16px;
}}
</style>
</head>
<body>
<div class="container">

<h1>Spot Price Prediction -- Evaluation Dashboard</h1>
<p class="subtitle" id="subtitle"></p>

<!-- Metric cards -->
<div class="metrics-row" id="metric-cards"></div>

<!-- Time series with slider -->
<h2>Time Series: Actual vs Predicted</h2>
<div class="chart-box">
  <div class="slider-controls">
    <button onclick="setWindow(7)" id="btn7">7d</button>
    <button onclick="setWindow(30)" id="btn30">30d</button>
    <button onclick="setWindow(90)" id="btn90">90d</button>
    <button onclick="setWindow(0)" id="btnAll">All</button>
    <input type="range" id="timeSlider" min="0" max="100" value="100">
    <span class="slider-label" id="sliderLabel"></span>
  </div>
  <canvas id="tsChart" height="110"></canvas>
</div>

<!-- Scatter plot -->
<h2>Scatter: Actual vs Predicted</h2>
<div class="two-col">
  <div class="chart-box"><canvas id="scatterChart" height="140"></canvas></div>
  <div class="chart-box"><canvas id="histChart" height="140"></canvas></div>
</div>

<!-- Tier Contribution -->
<h2>Tier Contribution</h2>
<div class="panel">
  <table id="tierTable">
    <thead><tr><th>Configuration</th><th>Features</th><th>MAE (EUR/MWh)</th><th>R&sup2;</th></tr></thead>
    <tbody></tbody>
  </table>
</div>

<!-- Feature Importance -->
<h2>Feature Importance (Top 20 by |Coefficient|)</h2>
<div class="chart-box" style="height:500px"><canvas id="fiChart"></canvas></div>

<!-- Hourly + Monthly MAE -->
<h2>Performance Breakdown</h2>
<div class="two-col">
  <div class="chart-box"><canvas id="hourlyChart" height="130"></canvas></div>
  <div class="chart-box"><canvas id="monthlyChart" height="130"></canvas></div>
</div>

<footer>
  Generated by HA-spot-price-predictor | Model v4.0 |
  <a href="https://github.com/watti-matti/HA-spot-price-predictor" style="color:#60a5fa">GitHub</a>
</footer>

</div>

<script>
// ── Inline data ──────────────────────────────────────────────────────
const DATA = {data_json};

// ── Chart.js defaults (dark theme) ───────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#2d3748';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

function initChart(ctx, cfg) {{
  cfg.options = cfg.options || {{}};
  cfg.options.responsive = true;
  cfg.options.maintainAspectRatio = false;
  cfg.options.plugins = cfg.options.plugins || {{}};
  cfg.options.plugins.legend = cfg.options.plugins.legend || {{ labels: {{ color: '#94a3b8', boxWidth: 12, padding: 10 }} }};
  if (cfg.options.scales) {{
    for (const axis of Object.values(cfg.options.scales)) {{
      axis.grid = axis.grid || {{}};
      axis.grid.color = axis.grid.color || '#2d3748';
      axis.ticks = axis.ticks || {{}};
      axis.ticks.color = axis.ticks.color || '#64748b';
    }}
  }}
  return new Chart(ctx, cfg);
}}

// ── Metric cards ─────────────────────────────────────────────────────
const m = DATA.metrics;
document.getElementById('subtitle').textContent =
  m.n_train.toLocaleString() + ' train / ' + m.n_test.toLocaleString() + ' test samples  |  Split: ' + DATA.train_split_date;

function renderMetrics(mae, rmse, r2, mape) {{
  document.getElementById('metric-cards').innerHTML =
    `<div class="metric-card"><div class="metric-value v-green">${{mae.toFixed(2)}}</div><div class="metric-label">MAE (EUR/MWh)</div></div>` +
    `<div class="metric-card"><div class="metric-value v-yellow">${{rmse.toFixed(2)}}</div><div class="metric-label">RMSE (EUR/MWh)</div></div>` +
    `<div class="metric-card"><div class="metric-value v-blue">${{r2.toFixed(4)}}</div><div class="metric-label">R-squared</div></div>` +
    `<div class="metric-card"><div class="metric-value v-pink">${{mape.toFixed(1)}}%</div><div class="metric-label">MAPE</div></div>`;
}}
renderMetrics(m.mae, m.rmse, m.r2, m.mape);

// ── Time series chart with slider ────────────────────────────────────
const tsCtx = document.getElementById('tsChart').getContext('2d');
let windowDays = 30;
let sliderPos = 100;

const tsData = DATA.test;

function getWindowSlice() {{
  const n = tsData.timestamps.length;
  if (windowDays === 0) return {{ start: 0, end: n }};
  const winSize = Math.min(windowDays * 24, n);
  const maxStart = n - winSize;
  const start = Math.round(maxStart * sliderPos / 100);
  return {{ start, end: start + winSize }};
}}

let tsChart = null;

function buildTsChart() {{
  const sl = getWindowSlice();
  const ts = tsData.timestamps.slice(sl.start, sl.end);
  const actual = tsData.actual.slice(sl.start, sl.end);
  const predicted = tsData.predicted.slice(sl.start, sl.end);
  const isAll = windowDays === 0;
  const step = isAll ? Math.max(1, Math.floor(actual.length / 600)) : 1;
  const sAct = [], sPred = [], sBand = [], sLab = [];
  for (let i = 0; i < actual.length; i += step) {{
    sAct.push(actual[i]);
    sPred.push(predicted[i]);
    sBand.push(Math.abs(predicted[i] - actual[i]));
    sLab.push(ts[i]);
  }}

  // Recompute metrics for visible window
  let sumAE = 0, sumSE = 0, sumAPE = 0, cntAPE = 0;
  for (let i = 0; i < actual.length; i++) {{
    const e = actual[i] - predicted[i];
    sumAE += Math.abs(e);
    sumSE += e * e;
    if (Math.abs(actual[i]) > 1) {{ sumAPE += Math.abs(e / actual[i]); cntAPE++; }}
  }}
  const n = actual.length;
  const wMAE = sumAE / n;
  const wRMSE = Math.sqrt(sumSE / n);
  const yMean = actual.reduce((a, b) => a + b, 0) / n;
  const ssTot = actual.reduce((s, v) => s + (v - yMean) ** 2, 0);
  const wR2 = 1 - sumSE / (ssTot || 1);
  const wMAPE = cntAPE > 0 ? sumAPE / cntAPE * 100 : 0;
  renderMetrics(wMAE, wRMSE, wR2, wMAPE);

  // Update slider label
  const label = ts.length > 0 ? ts[0].slice(0, 10) + ' .. ' + ts[ts.length - 1].slice(0, 10) : '';
  document.getElementById('sliderLabel').textContent = label;

  const data = {{
    labels: sLab,
    datasets: [
      {{ label: 'Actual', data: sAct, borderColor: '#f472b6',
         borderWidth: isAll ? 1 : 1.5, pointRadius: 0, fill: false, tension: 0.2, order: 1 }},
      {{ label: 'Predicted', data: sPred, borderColor: '#60a5fa',
         borderWidth: isAll ? 1 : 1.5, pointRadius: 0, fill: false, tension: 0.2,
         borderDash: isAll ? [] : [4, 2], order: 2 }},
      {{ label: '|Residual|', data: sBand, borderColor: 'transparent',
         backgroundColor: 'rgba(251,191,36,0.12)', fill: true, pointRadius: 0, tension: 0.2, order: 3 }},
    ]
  }};

  if (tsChart) {{
    tsChart.data = data;
    tsChart.update();
  }} else {{
    tsChart = new Chart(document.getElementById('tsChart').getContext('2d'), {{
      type: 'line', data,
      options: {{
        responsive: true,
        animation: {{ duration: 300 }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ mode: 'index', intersect: false,
            backgroundColor: '#1e2433', borderColor: '#374151', borderWidth: 1,
            titleColor: '#94a3b8', bodyColor: '#e2e8f0' }},
        }},
        scales: {{
          x: {{ grid: {{ color: '#1a2030' }}, ticks: {{ color: '#64748b', maxTicksLimit: 12, font: {{ size: 10 }} }} }},
          y: {{ grid: {{ color: '#1a2030' }}, ticks: {{ color: '#64748b', font: {{ size: 10 }} }} }},
        }},
      }}
    }});
  }}
}}

function setWindow(d) {{
  windowDays = d;
  document.querySelectorAll('.slider-controls button').forEach(b => b.classList.remove('active'));
  const id = d === 0 ? 'btnAll' : 'btn' + d;
  document.getElementById(id).classList.add('active');
  buildTsChart();
}}

document.getElementById('timeSlider').addEventListener('input', function() {{
  sliderPos = parseInt(this.value);
  buildTsChart();
}});

setWindow(30);

// ── Scatter plot ─────────────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById('scatterChart').getContext('2d');
  // Downsample for scatter
  const step = Math.max(1, Math.floor(tsData.actual.length / 2000));
  const pts = [];
  for (let i = 0; i < tsData.actual.length; i += step)
    pts.push({{ x: tsData.actual[i], y: tsData.predicted[i] }});
  const allVals = tsData.actual.concat(tsData.predicted);
  const lo = Math.min(...allVals.filter(v => v > -200));
  const hi = Math.max(...allVals.filter(v => v < 500));
  initChart(ctx, {{
    type: 'scatter',
    data: {{
      datasets: [
        {{ label: 'Predictions', data: pts,
           backgroundColor: 'rgba(96,165,250,0.3)', borderColor: 'rgba(96,165,250,0.6)',
           pointRadius: 2 }},
        {{ label: 'Perfect fit', data: [{{ x: lo, y: lo }}, {{ x: hi, y: hi }}],
           type: 'line', borderColor: '#4a5568', borderDash: [6, 4],
           borderWidth: 1.5, pointRadius: 0, fill: false }}
      ]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Actual (EUR/MWh)', color: '#64748b' }},
              min: lo, max: hi }},
        y: {{ title: {{ display: true, text: 'Predicted (EUR/MWh)', color: '#64748b' }},
              min: lo, max: hi }}
      }},
      plugins: {{ legend: {{ display: false }} }}
    }}
  }});
}})();

// ── Residual histogram ───────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById('histChart').getContext('2d');
  const centers = DATA.histogram.centers;
  const counts = DATA.histogram.counts;
  const maxAbs = Math.max(...centers.map(Math.abs));
  const colors = centers.map(c => {{
    const t = Math.abs(c) / (maxAbs || 1);
    const r = Math.round(52 + t * 192);
    const g = Math.round(211 - t * 140);
    const b = Math.round(153 - t * 50);
    return `rgba(${{r}},${{g}},${{b}},0.75)`;
  }});
  initChart(ctx, {{
    type: 'bar',
    data: {{
      labels: centers.map(c => c.toFixed(1)),
      datasets: [{{ label: 'Residual count', data: counts, backgroundColor: colors, borderWidth: 0 }}]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Residual (EUR/MWh)', color: '#64748b' }},
              ticks: {{ maxTicksLimit: 10 }} }},
        y: {{ title: {{ display: true, text: 'Count', color: '#64748b' }} }}
      }},
      plugins: {{ legend: {{ display: false }} }}
    }}
  }});
}})();

// ── Tier contribution table ──────────────────────────────────────────
(function() {{
  const tbody = document.querySelector('#tierTable tbody');
  const keys = Object.keys(DATA.tier_results);
  for (let i = 0; i < keys.length; i++) {{
    const k = keys[i];
    const v = DATA.tier_results[k];
    const improve = i > 0 ?
      ' <span class="improve">(' + (DATA.tier_results[keys[0]].mae - v.mae > 0 ? '-' : '+') +
        Math.abs(DATA.tier_results[keys[0]].mae - v.mae).toFixed(2) + ')</span>' : '';
    tbody.innerHTML += `<tr><td>${{k}}</td><td>${{v.n_features}}</td><td>${{v.mae.toFixed(2)}}${{improve}}</td><td>${{v.r2.toFixed(4)}}</td></tr>`;
  }}
}})();

// ── Feature importance horizontal bar chart ──────────────────────────
(function() {{
  const ctx = document.getElementById('fiChart').getContext('2d');
  const fi = DATA.feature_importance;
  const tierColors = {{ T1: '#34d399', T2: '#60a5fa', T3: '#fb923c', Stage2: '#c084fc' }};
  const labels = fi.map(f => f.name);
  const values = fi.map(f => f.impact || 0);
  const bgColors = fi.map(f => tierColors[f.tier] || '#94a3b8');
  initChart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ label: 'Normalized impact', data: values, backgroundColor: bgColors, borderWidth: 0 }}]
    }},
    options: {{
      indexAxis: 'y',
      scales: {{
        x: {{ title: {{ display: true, text: 'Impact: |coefficient| × std(feature)  [EUR/MWh]', color: '#64748b' }} }},
        y: {{ ticks: {{ color: '#e2e8f0', font: {{ size: 11, family: 'monospace' }} }} }}
      }},
      plugins: {{
        legend: {{ display: true,
          labels: {{
            color: '#e2e8f0',
            generateLabels: function() {{
              return [
                {{ text: 'T1: Base', fillStyle: '#34d399', strokeStyle: '#34d399', fontColor: '#e2e8f0' }},
                {{ text: 'T2: Cross-border', fillStyle: '#60a5fa', strokeStyle: '#60a5fa', fontColor: '#e2e8f0' }},
                {{ text: 'T3: Grid', fillStyle: '#fb923c', strokeStyle: '#fb923c', fontColor: '#e2e8f0' }},
                {{ text: 'Stage 2', fillStyle: '#c084fc', strokeStyle: '#c084fc', fontColor: '#e2e8f0' }},
              ];
            }}
          }}
        }},
        tooltip: {{
          callbacks: {{
            afterLabel: function(ctx) {{
              const f = fi[ctx.dataIndex];
              return 'Coef: ' + (f.coef > 0 ? '+' : '') + f.coef.toFixed(4) +
                     '  Std: ' + (f.std || 0).toFixed(3);
            }}
          }}
        }}
      }}
    }}
  }});
}})();

// ── Hourly MAE bar chart ─────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById('hourlyChart').getContext('2d');
  const labels = Array.from({{length: 24}}, (_, i) => String(i).padStart(2, '0') + ':00');
  const values = DATA.hourly_mae_full;
  const maxV = Math.max(...values);
  const colors = values.map(v => {{
    const t = v / (maxV || 1);
    return `rgba(${{Math.round(96 - t * 40)}},${{Math.round(165 - t * 60)}},${{Math.round(250 - t * 30)}},0.8)`;
  }});
  initChart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ label: 'MAE (full dataset)', data: values, backgroundColor: colors, borderWidth: 0 }}]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Hour (Finnish local)', color: '#64748b' }} }},
        y: {{ title: {{ display: true, text: 'MAE (EUR/MWh)', color: '#64748b' }} }}
      }}
    }}
  }});
}})();

// ── Monthly MAE bar chart ────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById('monthlyChart').getContext('2d');
  const labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const values = DATA.monthly_mae_full;
  const maxV = Math.max(...values);
  const colors = values.map(v => {{
    const t = v / (maxV || 1);
    return `rgba(${{Math.round(244 - t * 40)}},${{Math.round(114 - t * 40)}},${{Math.round(182 - t * 30)}},0.8)`;
  }});
  initChart(ctx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [{{ label: 'MAE (full dataset)', data: values, backgroundColor: colors, borderWidth: 0 }}]
    }},
    options: {{
      scales: {{
        x: {{ title: {{ display: true, text: 'Month', color: '#64748b' }} }},
        y: {{ title: {{ display: true, text: 'MAE (EUR/MWh)', color: '#64748b' }} }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>'''


# ── CLI entry point ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--region", default="finland")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    config_path = Path("config/regions") / f"{args.region}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_evaluation(config, Path(args.out_dir))


if __name__ == "__main__":
    main()
