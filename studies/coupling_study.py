"""Study cross-border price coupling architectures outside log-space."""
import pandas as pd
import numpy as np
import yaml
import sys
import math
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.features import build_features
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize as sp_min

prices = pd.read_parquet("output/fi_prices.parquet")["price_eur_mwh"]
weather = pd.read_parquet("output/fi_weather.parquet")
grid_df = pd.read_parquet("output/fi_grid_data.parquet")
neighbor_df = pd.read_parquet("output/fi_neighbor_prices.parquet")
with open("config/regions/finland.yaml") as f:
    config = yaml.safe_load(f)

df, feature_cols, _ = build_features(
    prices, weather, config,
    neighbor_prices={col: neighbor_df[col] for col in neighbor_df.columns},
    grid_data={col: grid_df[col] for col in grid_df.columns},
)

y_raw = df["price_eur_mwh"].values
split = int(len(y_raw) * 0.85)
y_tr, y_te = y_raw[:split], y_raw[split:]
X = df[feature_cols].values.astype(np.float64)
Xtr, Xte = X[:split], X[split:]

half_life = 120
alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)
age = np.arange(split - 1, -1, -1, dtype=np.float64)
weights = np.exp(-decay * age)
weights *= split / weights.sum()
LOG_OFFSET = 55

se3 = neighbor_df["se3"].reindex(df.index).ffill().values
se1 = neighbor_df["se1"].reindex(df.index).ffill().values
ee = neighbor_df["ee"].reindex(df.index).ffill().values


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


# Step 1: Actual FI vs neighbor price relationship
print("=" * 70)
print("ACTUAL FI vs NEIGHBOR PRICE RELATIONSHIP")
print("=" * 70)
print()
print("SE3 bin     FI avg   FI/SE3   Count")
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120),
               (120, 160), (160, 200), (200, 400)]:
    mask = (se3 >= lo) & (se3 < hi)
    if mask.sum() < 10:
        continue
    fi_avg = y_raw[mask].mean()
    mid = (lo + hi) / 2
    print("  %3d-%3d   %6.1f   %.3f    %5d" % (lo, hi, fi_avg, fi_avg / mid, mask.sum()))

# Step 2: Weather-only model
print()
print("=" * 70)
print("WEATHER-ONLY MODEL (no neighbor prices)")
print("=" * 70)

non_ar = [i for i, n in enumerate(feature_cols) if not n.startswith("ar_")]
non_ar_names = [feature_cols[i] for i in non_ar]
Xw_tr = Xtr[:, non_ar]
Xw_te = Xte[:, non_ar]
cw, icw = train_ridge(Xw_tr, np.log(y_tr + LOG_OFFSET), weights)
wp_tr = np.maximum(0, np.exp(Xw_tr @ cw + icw) - LOG_OFFSET)
wp_te = np.maximum(0, np.exp(Xw_te @ cw + icw) - LOG_OFFSET)
print("Weather-only: MAE=%.3f max=%.1f" % (mean_absolute_error(y_te, wp_te), wp_te.max()))

# Step 3: Full model baseline
coefs_full, ic_full = train_ridge(Xtr, np.log(y_tr + LOG_OFFSET), weights)
fp_te = np.maximum(0, np.exp(Xte @ coefs_full + ic_full) - LOG_OFFSET)
print("Full log-linear: MAE=%.3f max=%.1f" % (mean_absolute_error(y_te, fp_te), fp_te.max()))

# Step 4: Test hybrid architectures
print()
print("=" * 70)
print("HYBRID ARCHITECTURES: weather(log) + neighbor(linear)")
print("=" * 70)
print()

se3_tr, se3_te = se3[:split], se3[split:]
se1_tr, se1_te = se1[:split], se1[split:]

# A: Import price floor: FI = max(weather_pred, coupling * weighted_neighbor)
def obj_floor(params):
    c1, c2 = params
    if c1 < 0 or c2 < 0:
        return 1e6
    nb = c1 * se3_tr + c2 * se1_tr
    pred = np.maximum(wp_tr, nb)
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_a = sp_min(obj_floor, [0.1, 0.05], method="Nelder-Mead", options={"maxiter": 1000})
c_se3, c_se1 = opt_a.x
nb_te = c_se3 * se3_te + c_se1 * se1_te
pred_a = np.maximum(wp_te, nb_te)
mae_a = mean_absolute_error(y_te, pred_a)
print("A. Import floor: max(weather, %.3f*SE3 + %.3f*SE1)" % (c_se3, c_se1))
print("   MAE=%.3f max=%.1f" % (mae_a, pred_a.max()))

# B: Additive excess: FI = weather + alpha * max(0, beta * neighbor - weather)
def obj_excess(params):
    a, b = params
    if a < 0 or a > 1 or b < 0:
        return 1e6
    nb = b * (0.5 * se3_tr + 0.3 * se1_tr + 0.2 * (ee[:split] if len(ee) >= split else se3_tr))
    excess = np.maximum(0, nb - wp_tr)
    pred = wp_tr + a * excess
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_b = sp_min(obj_excess, [0.3, 0.15], method="Nelder-Mead", options={"maxiter": 1000})
a_b, b_b = opt_b.x
ee_te = ee[split:] if len(ee) > split else se3_te
nb_b = b_b * (0.5 * se3_te + 0.3 * se1_te + 0.2 * ee_te)
excess_b = np.maximum(0, nb_b - wp_te)
pred_b = wp_te + a_b * excess_b
mae_b = mean_absolute_error(y_te, pred_b)
print("B. Additive excess: weather + %.3f * max(0, %.3f*nb - weather)" % (a_b, b_b))
print("   MAE=%.3f max=%.1f" % (mae_b, pred_b.max()))

# C: Direct linear coupling with threshold
# FI = weather + c * max(0, SE3 - threshold)
def obj_threshold(params):
    c, t, c1 = params
    if c < 0 or t < 0 or c1 < 0:
        return 1e6
    contrib = c * np.maximum(0, se3_tr - t) + c1 * np.maximum(0, se1_tr - t)
    pred = wp_tr + contrib
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_c = sp_min(obj_threshold, [0.1, 30, 0.05], method="Nelder-Mead", options={"maxiter": 1000})
c_c, t_c, c1_c = opt_c.x
contrib_te = c_c * np.maximum(0, se3_te - t_c) + c1_c * np.maximum(0, se1_te - t_c)
pred_c = np.maximum(0, wp_te + contrib_te)
mae_c = mean_absolute_error(y_te, pred_c)
print("C. Threshold coupling: weather + %.3f*max(0,SE3-%.0f) + %.3f*max(0,SE1-%.0f)" %
      (c_c, t_c, c1_c, t_c))
print("   MAE=%.3f max=%.1f" % (mae_c, pred_c.max()))

# D: Power coupling: FI = weather + c * (SE3/100)^p
def obj_power_couple(params):
    c, p = params
    if c < 0 or p < 0.5 or p > 3:
        return 1e6
    contrib = c * np.power(se3_tr / 100 + 0.01, p)
    pred = np.maximum(0, wp_tr + contrib)
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_d = sp_min(obj_power_couple, [5, 1.5], method="Nelder-Mead", options={"maxiter": 1000})
c_d, p_d = opt_d.x
contrib_d = c_d * np.power(se3_te / 100 + 0.01, p_d)
pred_d = np.maximum(0, wp_te + contrib_d)
mae_d = mean_absolute_error(y_te, pred_d)
print("D. Power coupling: weather + %.2f * (SE3/100)^%.2f" % (c_d, p_d))
print("   MAE=%.3f max=%.1f" % (mae_d, pred_d.max()))

# Summary and simulations
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
models = [
    ("Weather only", wp_te),
    ("Full log-linear (current)", fp_te),
    ("A. Import floor", pred_a),
    ("B. Additive excess", pred_b),
    ("C. Threshold coupling", pred_c),
    ("D. Power coupling", pred_d),
]

from zoneinfo import ZoneInfo
dates_te = df.index[split:].tz_convert(ZoneInfo("Europe/Helsinki")).date

print("%-30s %6s %7s %7s %7s %7s" % ("Model", "MAE", "R2", "Max", "Bias20", "Rank4h"))
print("-" * 75)
for label, preds in models:
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    mask20 = (y_te >= 20) & (y_te < 50)
    b20 = (y_te[mask20] - preds[mask20]).mean() if mask20.sum() > 0 else 0
    # Quick rank
    conc = disc = 0
    for d in sorted(set(dates_te)):
        m = np.array([dd == d for dd in dates_te])
        if m.sum() < 5:
            continue
        p_d, y_d = preds[m], y_te[m]
        n = len(p_d)
        pb = np.array([p_d[i:i+4].mean() for i in range(n-3)])
        yb = np.array([y_d[i:i+4].mean() for i in range(n-3)])
        for i in range(len(pb)):
            for j in range(i+1, len(pb)):
                yd = yb[i] - yb[j]
                if abs(yd) < 5:
                    continue
                if (pb[i]-pb[j])*yd > 0:
                    conc += 1
                else:
                    disc += 1
    rank = 100*conc/(conc+disc) if (conc+disc) > 0 else 0
    print("%-30s %6.3f %7.4f %7.1f %+7.1f %6.1f%%" %
          (label, mae, r2, preds.max(), b20, rank))

# Simulate current conditions
print()
print("=" * 70)
print("SIMULATION: Spring, moderate wind, varying neighbor prices")
print("=" * 70)
print()
# Use avg spring weather prediction
wp_avg = 5.0  # typical spring weather prediction

print("%-10s  %-30s  %-30s  %-30s" %
      ("SE3", "C.Threshold", "D.Power", "A.Floor"))
for se3_val in [20, 40, 60, 80, 100, 120, 150, 200]:
    se1_val = se3_val * 0.5
    p_c = max(0, wp_avg + c_c * max(0, se3_val - t_c) + c1_c * max(0, se1_val - t_c))
    p_d = max(0, wp_avg + c_d * (se3_val / 100 + 0.01) ** p_d)
    p_a = max(wp_avg, c_se3 * se3_val + c_se1 * se1_val)
    cons_c = (p_c / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    cons_d = (p_d / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    cons_a = (p_a / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    print("SE3=%3d    %5.1f EUR = %4.1f c/kWh    %5.1f EUR = %4.1f c/kWh    %5.1f EUR = %4.1f c/kWh" %
          (se3_val, p_c, cons_c, p_d, cons_d, p_a, cons_a))
