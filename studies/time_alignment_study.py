"""Study time-elastic coupling methods for cross-border price alignment.

Problem: Hourly LS regression suppresses neighbor price coupling when
there's temporal misalignment between SE3/SE1 and FI price peaks.
The solver penalizes any phase offset, reducing coupling coefficients.

For cheapest-day/block selection, we need to know WHICH DAYS have
expensive energy, not the exact hour. This calls for multi-resolution
or time-elastic representations.

Theoretical approaches tested:
1. Multi-resolution features (daily/6h blocks instead of hourly)
2. Multi-lag features (include price at -6,-3,0,+3,+6 hour offsets)
3. Sliding window statistics (min/max/mean/p75 of last 24h)
4. Rank-based features (daily rank of neighbor price)
5. Frequency-domain coupling (low-pass filtered neighbor prices)
6. Dynamic envelope (daily min/max band of neighbor prices)
"""
import pandas as pd
import numpy as np
import yaml
import sys
import math
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.features import build_features
from src.holidays import build_holiday_set
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from zoneinfo import ZoneInfo

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

half_life = 120; alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)
age = np.arange(split-1,-1,-1,dtype=np.float64)
weights = np.exp(-decay*age); weights *= split/weights.sum()
LOG_OFFSET = 55

local_idx = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
hour_arr = local_idx.hour.values
dates_arr = local_idx.date


def train_ridge(Xtr, ytr, w):
    ws = w.sum()
    fm = (w[:,None]*Xtr).sum(0)/ws
    fv = (w[:,None]*(Xtr**2)).sum(0)/ws - fm**2
    fs = np.sqrt(np.maximum(0,fv)); fs[fs<1e-10]=1.0
    ym = (w*ytr).sum()/ws
    Xs = (Xtr-fm)/fs; ys = ytr-ym
    Xw = Xs*w[:,None]
    cs = np.linalg.solve(Xw.T@Xs + alpha*np.eye(Xtr.shape[1]), Xw.T@ys)
    coefs = cs/fs; ic = ym-(fm/fs)@cs
    return coefs, ic


# Weather-only features (no neighbor prices)
non_ar = [i for i, n in enumerate(feature_cols) if not n.startswith("ar_")]
non_ar_names = [feature_cols[i] for i in non_ar]
Xw_tr = Xtr[:, non_ar]
Xw_te = Xte[:, non_ar]


# ================================================================
print("=" * 80)
print("PART 1: TEMPORAL MISALIGNMENT ANALYSIS")
print("=" * 80)
print()

se3 = neighbor_df["se3"].reindex(df.index).ffill().values
se1 = neighbor_df["se1"].reindex(df.index).ffill().values
ee = neighbor_df["ee"].reindex(df.index).ffill().values

# Cross-correlation at different lags
print("FI-SE3 cross-correlation by lag:")
for lag in [-12, -6, -3, -1, 0, 1, 3, 6, 12, 24]:
    if lag >= 0:
        r = np.corrcoef(y_raw[lag:], se3[:len(y_raw)-lag])[0,1]
    else:
        r = np.corrcoef(y_raw[:len(y_raw)+lag], se3[-lag:])[0,1]
    print(f"  lag={lag:+3d}h: r={r:.4f}")

print()

# ================================================================
print("=" * 80)
print("PART 2: MULTI-RESOLUTION NEIGHBOR FEATURES")
print("=" * 80)
print()

# Build various time-resolution features from neighbor prices
se3_series = pd.Series(se3, index=df.index)
se1_series = pd.Series(se1, index=df.index)
ee_series = pd.Series(ee, index=df.index)

feature_sets = {}

# 2A: Current AR features (hourly, 24h exp-weighted)
# Already in feature_cols as ar_se1, ar_se3, ar_ee

# 2B: 6-hour block mean
for col, series in [("se3", se3_series), ("se1", se1_series), ("ee", ee_series)]:
    df[f"nb_{col}_6h_mean"] = series.rolling(6, min_periods=1).mean() / 100
    df[f"nb_{col}_6h_max"] = series.rolling(6, min_periods=1).max() / 100

# 2C: 24-hour daily statistics
for col, series in [("se3", se3_series), ("se1", se1_series), ("ee", ee_series)]:
    df[f"nb_{col}_24h_mean"] = series.rolling(24, min_periods=6).mean() / 100
    df[f"nb_{col}_24h_max"] = series.rolling(24, min_periods=6).max() / 100
    df[f"nb_{col}_24h_min"] = series.rolling(24, min_periods=6).min() / 100
    df[f"nb_{col}_24h_range"] = (df[f"nb_{col}_24h_max"] - df[f"nb_{col}_24h_min"])

# 2D: Multi-lag (include neighbor price at -6, -3, 0, +3 lags)
for col, series in [("se3", se3_series)]:
    for lag in [-6, -3, 0, 3, 6]:
        if lag >= 0:
            df[f"nb_{col}_lag{lag:+d}"] = series.shift(lag).ffill() / 100
        else:
            df[f"nb_{col}_lag{lag:+d}"] = series.shift(lag).bfill() / 100

# 2E: Low-pass filtered (48h exponential moving average)
for col, series in [("se3", se3_series), ("se1", se1_series)]:
    df[f"nb_{col}_48h_ema"] = series.ewm(span=48).mean() / 100

# 2F: Daily price level (same value for all hours in a day)
for col, series in [("se3", se3_series), ("se1", se1_series), ("ee", ee_series)]:
    daily_mean = series.resample("D").mean()
    df[f"nb_{col}_daily"] = daily_mean.reindex(df.index, method="ffill") / 100

# Fill NaN
for c in df.columns:
    if c.startswith("nb_"):
        df[c] = df[c].ffill().bfill().fillna(0)


# Define test configurations
configs = {
    "Baseline (weather only)": non_ar_names,
    "Current AR (hourly)": feature_cols,
    "6h block mean": non_ar_names + [f"nb_{c}_6h_mean" for c in ["se3","se1","ee"]],
    "6h block max": non_ar_names + [f"nb_{c}_6h_max" for c in ["se3","se1","ee"]],
    "24h daily mean": non_ar_names + [f"nb_{c}_24h_mean" for c in ["se3","se1","ee"]],
    "24h daily max": non_ar_names + [f"nb_{c}_24h_max" for c in ["se3","se1","ee"]],
    "24h range+mean": non_ar_names + [f"nb_{c}_24h_mean" for c in ["se3","se1","ee"]] + [f"nb_{c}_24h_range" for c in ["se3","se1","ee"]],
    "24h min+max": non_ar_names + [f"nb_{c}_24h_min" for c in ["se3","se1","ee"]] + [f"nb_{c}_24h_max" for c in ["se3","se1","ee"]],
    "Multi-lag SE3 (-6..+6)": non_ar_names + [f"nb_se3_lag{l:+d}" for l in [-6,-3,0,3,6]],
    "48h EMA": non_ar_names + [f"nb_{c}_48h_ema" for c in ["se3","se1"]],
    "Daily level": non_ar_names + [f"nb_{c}_daily" for c in ["se3","se1","ee"]],
    "Daily + 24h max": non_ar_names + [f"nb_{c}_daily" for c in ["se3","se1","ee"]] + [f"nb_{c}_24h_max" for c in ["se3","se1","ee"]],
}

dates_te = dates_arr[split:]

print("%-30s %6s %7s %7s %7s %7s" %
      ("Config", "MAE", "R2", "Max", "Bias20", "Rank4h"))
print("-" * 80)

for label, feat_names in configs.items():
    Xa = df[feat_names].values.astype(np.float64)
    Xa_tr, Xa_te = Xa[:split], Xa[split:]
    ca, ica = train_ridge(Xa_tr, np.log(y_tr + LOG_OFFSET), weights)
    pa = np.maximum(0, np.exp(Xa_te @ ca + ica) - LOG_OFFSET)
    mae = mean_absolute_error(y_te, pa)
    r2 = r2_score(y_te, pa)
    mask20 = (y_te >= 20) & (y_te < 50)
    b20 = (y_te[mask20] - pa[mask20]).mean() if mask20.sum() > 0 else 0

    # Quick rank
    conc = disc = 0
    for d in sorted(set(dates_te)):
        m = np.array([dd == d for dd in dates_te])
        if m.sum() < 5: continue
        p_d, y_d = pa[m], y_te[m]
        n = len(p_d)
        pb = np.array([p_d[i:i+4].mean() for i in range(n-3)])
        yb = np.array([y_d[i:i+4].mean() for i in range(n-3)])
        for i in range(len(pb)):
            for j in range(i+1, len(pb)):
                yd = yb[i]-yb[j]
                if abs(yd) < 5: continue
                if (pb[i]-pb[j])*yd > 0: conc += 1
                else: disc += 1
    rank = 100*conc/(conc+disc) if (conc+disc) > 0 else 0

    # Show neighbor coefs
    nb_coefs = ""
    for k in range(len(non_ar_names), len(feat_names)):
        nb_coefs += f" {feat_names[k].replace('nb_','')[:12]}={ca[k]:+.4f}"

    print("%-30s %6.3f %7.4f %7.1f %+7.1f %6.1f%%%s" %
          (label, mae, r2, pa.max(), b20, rank, nb_coefs[:60]))

# ================================================================
print()
print("=" * 80)
print("PART 3: ADDITIVE DAILY-LEVEL COUPLING (outside log-space)")
print("=" * 80)
print()

# Best approach from Part 2 + additive linear coupling
# Train weather model in log-space, then add daily neighbor price correction in EUR/MWh

cw, icw = train_ridge(Xw_tr, np.log(y_tr + LOG_OFFSET), weights)
wp_tr = np.maximum(0, np.exp(Xw_tr @ cw + icw) - LOG_OFFSET)
wp_te = np.maximum(0, np.exp(Xw_te @ cw + icw) - LOG_OFFSET)

# Daily-level neighbor prices
se3_daily = se3_series.resample("D").mean().reindex(df.index, method="ffill").values
se1_daily = se1_series.resample("D").mean().reindex(df.index, method="ffill").values
ee_daily = ee_series.resample("D").mean().reindex(df.index, method="ffill").values

# 24h max
se3_24max = se3_series.rolling(24, min_periods=6).max().values
se1_24max = se1_series.rolling(24, min_periods=6).max().values

from scipy.optimize import minimize as sp_min

# Approach A: weather + c * daily_SE3_mean
def obj_daily(params):
    c1, c2, c3, offset = params
    nb = c1 * se3_daily[:split] + c2 * se1_daily[:split] + c3 * ee_daily[:split] + offset
    pred = np.maximum(0, wp_tr + nb)
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_a = sp_min(obj_daily, [0.05, 0.03, 0.02, -3], method="Nelder-Mead",
               options={"maxiter": 2000})
c1, c2, c3, offset = opt_a.x
nb_te = c1 * se3_daily[split:] + c2 * se1_daily[split:] + c3 * ee_daily[split:] + offset
pred_a = np.maximum(0, wp_te + nb_te)
mae_a = mean_absolute_error(y_te, pred_a)
r2_a = r2_score(y_te, pred_a)
mask20 = (y_te >= 20) & (y_te < 50)
b20_a = (y_te[mask20] - pred_a[mask20]).mean()
print(f"A. Daily mean: weather + {c1:.4f}*SE3 + {c2:.4f}*SE1 + {c3:.4f}*EE + ({offset:.1f})")
print(f"   MAE={mae_a:.3f} R2={r2_a:.4f} max={pred_a.max():.1f} bias20={b20_a:+.1f}")
print(f"   Impact at SE3=100: +{c1*100:.1f} EUR/MWh")
print(f"   Impact at SE3=200: +{c1*200:.1f} EUR/MWh")

# Approach B: weather + c * 24h_max (captures peak price signal)
def obj_24max(params):
    c1, c2, offset = params
    nb = c1 * se3_24max[:split] + c2 * se1_24max[:split] + offset
    pred = np.maximum(0, wp_tr + nb)
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_b = sp_min(obj_24max, [0.05, 0.03, -5], method="Nelder-Mead",
               options={"maxiter": 2000})
c1b, c2b, offb = opt_b.x
nb_b = c1b * se3_24max[split:] + c2b * se1_24max[split:] + offb
pred_b = np.maximum(0, wp_te + nb_b)
mae_b = mean_absolute_error(y_te, pred_b)
r2_b = r2_score(y_te, pred_b)
b20_b = (y_te[mask20] - pred_b[mask20]).mean()
print(f"\nB. 24h max: weather + {c1b:.4f}*SE3_max + {c2b:.4f}*SE1_max + ({offb:.1f})")
print(f"   MAE={mae_b:.3f} R2={r2_b:.4f} max={pred_b.max():.1f} bias20={b20_b:+.1f}")

# Approach C: Combine daily + 24h max
def obj_combo(params):
    cd, cm, c1d, c1m, off = params
    nb = (cd * se3_daily[:split] + cm * se3_24max[:split] +
          c1d * se1_daily[:split] + c1m * se1_24max[:split] + off)
    pred = np.maximum(0, wp_tr + nb)
    return np.average(np.abs(y_tr - pred), weights=weights)

opt_c = sp_min(obj_combo, [0.03, 0.02, 0.02, 0.01, -3], method="Nelder-Mead",
               options={"maxiter": 2000})
cd, cm, c1d, c1m, off_c = opt_c.x
nb_c = (cd * se3_daily[split:] + cm * se3_24max[split:] +
        c1d * se1_daily[split:] + c1m * se1_24max[split:] + off_c)
pred_c = np.maximum(0, wp_te + nb_c)
mae_c = mean_absolute_error(y_te, pred_c)
r2_c = r2_score(y_te, pred_c)
b20_c = (y_te[mask20] - pred_c[mask20]).mean()
print(f"\nC. Daily+Max combo:")
print(f"   weather + {cd:.4f}*SE3_daily + {cm:.4f}*SE3_24max + {c1d:.4f}*SE1_daily + {c1m:.4f}*SE1_24max + ({off_c:.1f})")
print(f"   MAE={mae_c:.3f} R2={r2_c:.4f} max={pred_c.max():.1f} bias20={b20_c:+.1f}")

# Simulation
print()
print("Simulation (spring, moderate wind):")
wp_base = 5.0  # typical spring weather prediction
for se3_val, se1_val, label in [(30, 20, "Normal"), (80, 50, "Elevated"),
                                  (120, 70, "High"), (180, 100, "Spike")]:
    nb_daily = c1 * se3_val + c2 * se1_val + c3 * se3_val * 1.5 + offset  # EE ~ 1.5*SE3
    total = max(0, wp_base + nb_daily)
    consumer = (total/1000 + 0.0361 + 0.02325) * 1.255 * 100
    print(f"  {label:>8} SE3={se3_val}: nb_add={nb_daily:+.1f} -> FI={total:.1f} EUR/MWh = {consumer:.1f} c/kWh")

# ================================================================
print()
print("=" * 80)
print("PART 4: COMPARISON SUMMARY")
print("=" * 80)
print()

# Compare all approaches
all_models = [
    ("Weather only (log-linear)", wp_te),
    ("Current: AR in log-space", np.maximum(0, np.exp(Xte @ train_ridge(Xtr, np.log(y_tr+LOG_OFFSET), weights)[0] + train_ridge(Xtr, np.log(y_tr+LOG_OFFSET), weights)[1]) - LOG_OFFSET)),
    ("A: weather + daily linear", pred_a),
    ("B: weather + 24h max linear", pred_b),
    ("C: weather + daily+max combo", pred_c),
]

print("%-35s %6s %7s %7s %7s %7s" %
      ("Model", "MAE", "R2", "Max", "Bias20", "Rank4h"))
print("-" * 80)
for label, preds in all_models:
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    b20 = (y_te[mask20] - preds[mask20]).mean()
    conc = disc = 0
    for d in sorted(set(dates_te)):
        m = np.array([dd == d for dd in dates_te])
        if m.sum() < 5: continue
        p_d, y_d = preds[m], y_te[m]
        n = len(p_d)
        pb = np.array([p_d[i:i+4].mean() for i in range(n-3)])
        yb = np.array([y_d[i:i+4].mean() for i in range(n-3)])
        for i in range(len(pb)):
            for j in range(i+1, len(pb)):
                yd = yb[i]-yb[j]
                if abs(yd) < 5: continue
                if (pb[i]-pb[j])*yd > 0: conc += 1
                else: disc += 1
    rank = 100*conc/(conc+disc) if (conc+disc) > 0 else 0
    print("%-35s %6.3f %7.4f %7.1f %+7.1f %6.1f%%" %
          (label, mae, r2, preds.max(), b20, rank))
