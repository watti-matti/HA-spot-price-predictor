"""Study: does smoothing input data before training improve daily energy cost estimation?

Hypothesis: hourly price noise (spikes, scheduling artifacts) hurts the
regression. Smoothing both FI target and neighbor prices with trapezoidal
or other kernels before fitting may improve the underlying cost envelope
estimation, which is what matters for cheapest-day/block selection.

Test: smooth inputs with different kernels, train, evaluate on:
1. RAW hourly prices (how well does smoothed model predict actual prices?)
2. Smoothed hourly prices (internal consistency)
3. Daily mean price (the actual commercial objective)
4. Cheapest 4h block selection accuracy (the real use case)
"""
import pandas as pd
import numpy as np
import yaml
import sys
import math
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from src.features import build_features

prices = pd.read_parquet("output/fi_prices.parquet")["price_eur_mwh"]
neighbor = pd.read_parquet("output/fi_neighbor_prices.parquet")
weather = pd.read_parquet("output/fi_weather.parquet")
grid_df = pd.read_parquet("output/fi_grid_data.parquet")
with open("config/regions/finland.yaml") as f:
    config = yaml.safe_load(f)

df, feature_cols, _ = build_features(
    prices, weather, config,
    neighbor_prices={col: neighbor[col] for col in neighbor.columns},
    grid_data={col: grid_df[col] for col in grid_df.columns},
)

fi = df["price_eur_mwh"].values
se3 = pd.Series(neighbor["se3"].reindex(df.index).ffill().values, index=df.index)
se1 = pd.Series(neighbor["se1"].reindex(df.index).ffill().values, index=df.index)
ee = pd.Series(neighbor["ee"].reindex(df.index).ffill().values, index=df.index)

non_ar = [n for n in feature_cols if not n.startswith("ar_")]
X_base = df[non_ar].values.astype(np.float64)

split = int(len(fi) * 0.85)
LOG_OFFSET = 55
half_life = 120
alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)
age = np.arange(split - 1, -1, -1, dtype=np.float64)
weights = np.exp(-decay * age)
weights *= split / weights.sum()

local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
dates_te = local[split:].date


def train_ridge(Xtr, ytr, w):
    ws = w.sum()
    fm = (w[:, None] * Xtr).sum(0) / ws
    fv = (w[:, None] * (Xtr ** 2)).sum(0) / ws - fm ** 2
    fs = np.sqrt(np.maximum(0, fv))
    fs[fs < 1e-10] = 1.0
    ym = (w * ytr).sum() / ws
    Xs = (Xtr - fm) / fs
    ys = ytr - ym
    Xw = Xs * w[:, None]
    cs = np.linalg.solve(Xw.T @ Xs + alpha * np.eye(Xtr.shape[1]), Xw.T @ ys)
    coefs = cs / fs
    ic = ym - (fm / fs) @ cs
    return coefs, ic


def trapezoidal_kernel(n):
    """Trapezoidal smoothing kernel of width n."""
    if n <= 1:
        return np.array([1.0])
    if n == 2:
        return np.array([0.5, 0.5])
    # Trapezoidal: ramp up, flat top, ramp down
    ramp = max(1, n // 4)
    flat = n - 2 * ramp
    k = np.concatenate([
        np.linspace(0.5, 1.0, ramp),
        np.ones(flat),
        np.linspace(1.0, 0.5, ramp),
    ])
    return k / k.sum()


def smooth_series(vals, kernel):
    """Apply convolution smoothing with given kernel."""
    return np.convolve(vals, kernel, mode="same")


def cheapest_block_accuracy(preds, actuals, dates, N=4):
    """What fraction of days does the model correctly identify the cheapest N-hour block?"""
    correct = 0
    top2 = 0
    total = 0
    regrets = []
    for d in sorted(set(dates)):
        mask = np.array([dd == d for dd in dates])
        if mask.sum() < N + 1:
            continue
        p_d = preds[mask]
        y_d = actuals[mask]
        n = len(p_d)
        if n < N:
            continue
        p_blocks = [p_d[i:i+N].mean() for i in range(n - N + 1)]
        y_blocks = [y_d[i:i+N].mean() for i in range(n - N + 1)]

        pred_best = np.argmin(p_blocks)
        actual_rank = np.argsort(y_blocks)

        cost_pred = y_blocks[pred_best]
        cost_opt = min(y_blocks)
        cost_worst = max(y_blocks)

        if pred_best == actual_rank[0]:
            correct += 1
            top2 += 1
        elif len(actual_rank) > 1 and pred_best in actual_rank[:2]:
            top2 += 1

        if cost_worst > cost_opt + 0.01:
            regrets.append((cost_pred - cost_opt) / (cost_worst - cost_opt))
        total += 1

    return {
        "exact": correct / total if total > 0 else 0,
        "top2": top2 / total if total > 0 else 0,
        "avg_regret_pct": np.mean(regrets) * 100 if regrets else 0,
        "n_days": total,
    }


# ================================================================
print("=" * 80)
print("SMOOTHING KERNELS")
print("=" * 80)
print()

kernels = {
    "none": np.array([1.0]),
    "box_3h": np.ones(3) / 3,
    "box_5h": np.ones(5) / 5,
    "trap_3h": trapezoidal_kernel(3),
    "trap_5h": trapezoidal_kernel(5),
    "trap_7h": trapezoidal_kernel(7),
    "gauss_3h": np.exp(-0.5 * np.linspace(-1.5, 1.5, 3)**2),
    "gauss_5h": np.exp(-0.5 * np.linspace(-1.5, 1.5, 5)**2),
}
# Normalize gaussian kernels
for k in kernels:
    kernels[k] = kernels[k] / kernels[k].sum()

for name, k in kernels.items():
    print("  %-10s: %s" % (name, np.round(k, 3)))

# ================================================================
print()
print("=" * 80)
print("EXPERIMENT: SMOOTH TARGET AND/OR NEIGHBOR PRICES BEFORE TRAINING")
print("=" * 80)
print()

# For each smoothing config:
# 1. Smooth FI target prices (training only)
# 2. Smooth neighbor envelope
# 3. Train log-linear Ridge
# 4. Predict RAW hourly prices (NOT smoothed - this is the deployment case)
# 5. Evaluate: MAE on raw, daily mean accuracy, cheapest block accuracy

print("%-35s %6s %6s %7s %6s %6s %6s" %
      ("Config", "MAE", "R2", "Max", "Block", "Top2", "Regret"))
print("-" * 80)

for smooth_label, target_kernel_name, nb_kernel_name in [
    ("No smoothing (baseline)", "none", "none"),
    ("Smooth target 3h box", "box_3h", "none"),
    ("Smooth target 5h box", "box_5h", "none"),
    ("Smooth target 3h trap", "trap_3h", "none"),
    ("Smooth target 5h trap", "trap_5h", "none"),
    ("Smooth target 7h trap", "trap_7h", "none"),
    ("Smooth neighbor 3h box", "none", "box_3h"),
    ("Smooth neighbor 3h trap", "none", "trap_3h"),
    ("Smooth both 3h trap", "trap_3h", "trap_3h"),
    ("Smooth both 5h trap", "trap_5h", "trap_5h"),
    ("Smooth target 3h gauss", "gauss_3h", "none"),
    ("Smooth target 5h gauss", "gauss_5h", "none"),
    ("Smooth both 3h gauss", "gauss_3h", "gauss_3h"),
]:
    t_kernel = kernels[target_kernel_name]
    nb_kernel = kernels[nb_kernel_name]

    # Smooth FI target
    if len(t_kernel) > 1:
        fi_smooth = smooth_series(fi, t_kernel)
    else:
        fi_smooth = fi.copy()

    # Smooth neighbor envelopes
    if len(nb_kernel) > 1:
        se3_s = smooth_series(se3.values, nb_kernel) / 100
        se1_s = smooth_series(se1.values, nb_kernel) / 100
        ee_s = smooth_series(ee.values, nb_kernel) / 100
    else:
        se3_s = se3.values / 100
        se1_s = se1.values / 100
        ee_s = ee.values / 100

    # Build features
    extra = np.column_stack([se3_s, se1_s, ee_s])
    Xa = np.hstack([X_base, extra])

    # Train on smoothed target
    y_log_smooth = np.log(fi_smooth[:split] + LOG_OFFSET)
    c, ic = train_ridge(Xa[:split], y_log_smooth, weights)

    # Predict RAW hourly prices (deployment case)
    preds = np.maximum(0, np.exp(Xa[split:] @ c + ic) - LOG_OFFSET)

    mae = mean_absolute_error(fi[split:], preds)
    r2 = r2_score(fi[split:], preds)

    # Cheapest 4h block accuracy
    block = cheapest_block_accuracy(preds, fi[split:], dates_te, N=4)

    print("%-35s %6.3f %6.4f %7.1f %5.0f%% %5.0f%% %5.1f%%" %
          (smooth_label, mae, r2, preds.max(), block["exact"] * 100,
           block["top2"] * 100, block["avg_regret_pct"]))

# ================================================================
print()
print("=" * 80)
print("DAILY MEAN PRICE PREDICTION ACCURACY")
print("=" * 80)
print()

# For cheapest-DAY selection, evaluate how well models predict daily mean
print("%-35s %6s %6s %6s" % ("Config", "Daily MAE", "Daily R2", "DayRank"))
print("-" * 60)

for smooth_label, target_kernel_name, nb_kernel_name in [
    ("No smoothing", "none", "none"),
    ("Smooth target 3h trap", "trap_3h", "none"),
    ("Smooth both 3h trap", "trap_3h", "trap_3h"),
    ("Smooth target 5h trap", "trap_5h", "none"),
    ("Smooth both 5h trap", "trap_5h", "trap_5h"),
]:
    t_kernel = kernels[target_kernel_name]
    nb_kernel = kernels[nb_kernel_name]

    fi_smooth = smooth_series(fi, t_kernel) if len(t_kernel) > 1 else fi.copy()

    if len(nb_kernel) > 1:
        se3_s = smooth_series(se3.values, nb_kernel) / 100
        se1_s = smooth_series(se1.values, nb_kernel) / 100
        ee_s = smooth_series(ee.values, nb_kernel) / 100
    else:
        se3_s = se3.values / 100
        se1_s = se1.values / 100
        ee_s = ee.values / 100

    extra = np.column_stack([se3_s, se1_s, ee_s])
    Xa = np.hstack([X_base, extra])
    y_log_smooth = np.log(fi_smooth[:split] + LOG_OFFSET)
    c, ic = train_ridge(Xa[:split], y_log_smooth, weights)
    preds = np.maximum(0, np.exp(Xa[split:] @ c + ic) - LOG_OFFSET)

    # Aggregate to daily
    pred_daily = []
    actual_daily = []
    for d in sorted(set(dates_te)):
        mask = np.array([dd == d for dd in dates_te])
        if mask.sum() < 20:
            continue
        pred_daily.append(preds[mask].mean())
        actual_daily.append(fi[split:][mask].mean())

    pred_daily = np.array(pred_daily)
    actual_daily = np.array(actual_daily)
    daily_mae = np.abs(actual_daily - pred_daily).mean()
    daily_r2 = 1 - np.sum((actual_daily - pred_daily)**2) / np.sum((actual_daily - actual_daily.mean())**2)
    daily_rho, _ = spearmanr(pred_daily, actual_daily)

    print("%-35s %6.3f %6.4f %6.4f" %
          (smooth_label, daily_mae, daily_r2, daily_rho))
