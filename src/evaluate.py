"""
Model evaluation: metrics computation and HTML visualization.

Generates an interactive accuracy report comparing predictions vs actual
prices, with hourly, monthly, and segment-level breakdowns.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    coefs_path: str | Path,
    config: dict[str, Any],
) -> dict:
    """Evaluate trained model on the dataset and produce metrics.

    Args:
        df: DataFrame with features and price_clipped column.
        feature_cols: List of feature column names (Tier 1 base features only).
        coefs_path: Path to model_coefs.json.
        config: Region config dict.

    Returns:
        Dict of evaluation metrics.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    with open(coefs_path) as f:
        coefs = json.load(f)

    training = config.get("training", {})
    test_split = training.get("test_split", 0.15)
    pw_breaks = coefs.get("piecewise_breakpoints", [20, 40, 120])

    # Get all feature names from the model
    all_feature_names = coefs.get("feature_names", [])
    # Base features (before stage1_pred and pw_relu)
    base_features = [n for n in all_feature_names if not n.startswith(("stage1_pred", "pw_relu_"))]

    X = df[base_features].values.astype(np.float64)
    y = df["price_clipped"].values.astype(np.float64)

    # Time-ordered split
    split = int(len(X) * (1.0 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Stage 1 coefficients
    s1 = coefs["stage1"]
    s1_coefs = np.array([f["coef"] for f in s1["features"]], dtype=np.float64)
    s1_intercept = s1["intercept"]

    # Stage 1 predictions
    s1_te = X_te @ s1_coefs + s1_intercept

    # Stage 2: augment with piecewise features
    pw_te = np.column_stack(
        [s1_te] + [np.maximum(0.0, s1_te - bp) for bp in pw_breaks]
    )
    X_te_aug = np.hstack([X_te, pw_te])

    # Stage 2 coefficients
    s2_coefs = np.array([f["coef"] for f in coefs["features"]], dtype=np.float64)
    s2_intercept = coefs["intercept"]

    preds = X_te_aug @ s2_coefs + s2_intercept

    # Overall metrics
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)

    # Hourly breakdown
    tz_offset = 2
    test_idx = df.index[split:]
    local_hours = (test_idx + pd.Timedelta(hours=tz_offset)).hour

    hourly_mae = {}
    for h in range(24):
        mask = local_hours == h
        if mask.sum() > 0:
            hourly_mae[h] = float(mean_absolute_error(y_te[mask], preds[mask]))

    # Monthly breakdown
    test_months = (test_idx + pd.Timedelta(hours=tz_offset)).month
    monthly_mae = {}
    for m in range(1, 13):
        mask = test_months == m
        if mask.sum() > 0:
            monthly_mae[m] = float(mean_absolute_error(y_te[mask], preds[mask]))

    # Workday vs weekend
    test_dow = (test_idx + pd.Timedelta(hours=tz_offset)).dayofweek
    workday_mask = test_dow < 5
    weekend_mask = ~workday_mask

    metrics = {
        "overall": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_test": int(len(y_te)),
        },
        "hourly_mae": hourly_mae,
        "monthly_mae": monthly_mae,
        "segments": {
            "workday_mae": float(mean_absolute_error(y_te[workday_mask], preds[workday_mask]))
            if workday_mask.sum() > 0 else None,
            "weekend_mae": float(mean_absolute_error(y_te[weekend_mask], preds[weekend_mask]))
            if weekend_mask.sum() > 0 else None,
        },
        "tier_info": coefs.get("tier_info", {}),
        "feature_count": len(all_feature_names),
    }

    logger.info("Evaluation: MAE=%.2f, RMSE=%.2f, R2=%.4f (n=%d)",
                mae, rmse, r2, len(y_te))
    return metrics


def generate_html_report(
    metrics: dict,
    output_path: str | Path,
) -> None:
    """Generate an interactive HTML accuracy report.

    Args:
        metrics: Dict from evaluate_model().
        output_path: Path to write the HTML file.
    """
    overall = metrics["overall"]
    hourly = metrics.get("hourly_mae", {})
    monthly = metrics.get("monthly_mae", {})
    segments = metrics.get("segments", {})
    tier_info = metrics.get("tier_info", {})

    # Build hour labels and values
    hour_labels = [f"{h:02d}" for h in range(24)]
    hour_values = [hourly.get(h, 0) for h in range(24)]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_values = [monthly.get(m + 1, 0) for m in range(12)]

    tiers_active = []
    if tier_info.get("tier1"):
        tiers_active.append("Tier 1 (weather+demand)")
    if tier_info.get("tier2"):
        tiers_active.append("Tier 2 (cross-border)")
    if tier_info.get("tier3"):
        tiers_active.append("Tier 3 (grid)")
    tiers_str = " + ".join(tiers_active) if tiers_active else "Unknown"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Spot Price Prediction Accuracy Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ color: #1a1a2e; }}
  .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;
              margin: 20px 0; }}
  .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px;
                  text-align: center; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #e94560; }}
  .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
  .chart {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 20px 0; }}
  .bar-chart {{ display: flex; align-items: flex-end; gap: 4px; height: 200px; }}
  .bar {{ background: #e94560; border-radius: 3px 3px 0 0; min-width: 20px;
          position: relative; }}
  .bar-label {{ position: absolute; bottom: -20px; font-size: 10px;
                transform: rotate(-45deg); white-space: nowrap; }}
  .bar-value {{ position: absolute; top: -18px; font-size: 10px; width: 100%;
                text-align: center; }}
  .tiers {{ background: #e8f4e8; border-radius: 8px; padding: 15px; margin: 20px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Spot Price Prediction Accuracy</h1>
<div class="tiers">
  <strong>Active tiers:</strong> {tiers_str} |
  <strong>Features:</strong> {metrics.get('feature_count', '?')} |
  <strong>Test samples:</strong> {overall['n_test']:,}
</div>

<div class="metrics">
  <div class="metric-card">
    <div class="metric-value">{overall['mae']:.1f}</div>
    <div class="metric-label">MAE (EUR/MWh)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{overall['rmse']:.1f}</div>
    <div class="metric-label">RMSE (EUR/MWh)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{overall['r2']:.3f}</div>
    <div class="metric-label">R-squared</div>
  </div>
</div>

<div class="chart">
  <h3>MAE by Hour (Finnish local time)</h3>
  <div class="bar-chart">
    {''.join(f'<div class="bar" style="height:{max(2, v/max(max(hour_values),1)*180)}px;flex:1"><div class="bar-value">{v:.0f}</div><div class="bar-label">{hour_labels[i]}</div></div>' for i, v in enumerate(hour_values))}
  </div>
</div>

<div class="chart">
  <h3>MAE by Month</h3>
  <div class="bar-chart">
    {''.join(f'<div class="bar" style="height:{max(2, v/max(max(month_values),1)*180)}px;flex:1"><div class="bar-value">{v:.0f}</div><div class="bar-label">{month_names[i]}</div></div>' for i, v in enumerate(month_values))}
  </div>
</div>

<div class="chart">
  <h3>Segment Analysis</h3>
  <table>
    <tr><th>Segment</th><th>MAE (EUR/MWh)</th></tr>
    <tr><td>Workdays</td><td>{segments.get('workday_mae', 'N/A'):.1f if segments.get('workday_mae') else 'N/A'}</td></tr>
    <tr><td>Weekends</td><td>{segments.get('weekend_mae', 'N/A'):.1f if segments.get('weekend_mae') else 'N/A'}</td></tr>
  </table>
</div>

<p style="color:#999;font-size:0.8em;margin-top:40px;">
  Generated by HA-spot-price-predictor | Model v3.1
</p>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("HTML report saved: %s", output_path)
