"""Histogram-based price prediction: predict the price DISTRIBUTION
instead of the time-aligned price curve.

For load scheduling, the key information is:
- How many cheap hours are available in the next 24/48h?
- What price level can I expect for the cheapest 4h block?
- Is tomorrow going to be an expensive or cheap day?

All of these are properties of the price HISTOGRAM, not the time series.
By predicting histogram features, we avoid:
- Time alignment issues (histogram is permutation-invariant)
- Log compression (predict bin counts, not prices)
- Hourly noise (histogram smooths naturally)
"""
import pandas as pd, numpy as np, yaml, sys, math
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
se3 = neighbor["se3"].reindex(df.index).ffill().values
se1 = neighbor["se1"].reindex(df.index).ffill().values
ee = neighbor["ee"].reindex(df.index).ffill().values
non_ar = [n for n in feature_cols if not n.startswith("ar_")]
X_base = df[non_ar].values.astype(np.float64)

local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
dates = local.date
hours = local.hour.values

half_life = 120; alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)

def train_ridge(Xtr, ytr, w):
    ws = w.sum()
    fm = (w[:, None] * Xtr).sum(0) / ws
    fv = (w[:, None] * (Xtr ** 2)).sum(0) / ws - fm ** 2
    fs = np.sqrt(np.maximum(0, fv)); fs[fs < 1e-10] = 1.0
    ym = (w * ytr).sum() / ws
    Xs = (Xtr - fm) / fs; ys = ytr - ym
    Xw = Xs * w[:, None]
    cs = np.linalg.solve(Xw.T @ Xs + alpha * np.eye(Xtr.shape[1]), Xw.T @ ys)
    coefs = cs / fs; ic = ym - (fm / fs) @ cs
    return coefs, ic

# ================================================================
print("=" * 80)
print("PART 1: DAILY HISTOGRAM STRUCTURE")
print("=" * 80)
print()

# Build daily histograms: for each day, compute the price distribution
unique_dates = sorted(set(dates))
price_bins = [-10, 0, 2, 5, 8, 12, 18, 25, 40, 60, 100, 500]
bin_labels = ["%d-%d" % (price_bins[i], price_bins[i+1]) for i in range(len(price_bins)-1)]

daily_histograms = []
daily_features = []
daily_targets = []

for d in unique_dates:
    mask = np.array([dd == d for dd in dates])
    if mask.sum() < 20:
        continue
    fi_day = fi[mask]
    se3_day = se3[mask]
    se1_day = se1[mask]
    ee_day = ee[mask]
    x_day = X_base[mask]

    # Histogram targets
    hist, _ = np.histogram(fi_day, bins=price_bins)
    hist_frac = hist / hist.sum()

    # Summary statistics targets
    targets = {
        "mean": fi_day.mean(),
        "median": np.median(fi_day),
        "p25": np.percentile(fi_day, 25),
        "p75": np.percentile(fi_day, 75),
        "p90": np.percentile(fi_day, 90),
        "max": fi_day.max(),
        "min": fi_day.min(),
        "range": fi_day.max() - fi_day.min(),
        "cheap_hours": (fi_day < 5).sum(),     # hours below 5 EUR/MWh
        "expensive_hours": (fi_day > 15).sum(),  # hours above 15
        "hist": hist_frac,
    }

    # Daily features: aggregate weather and neighbor data for the day
    feats = {
        "x_mean": x_day.mean(axis=0),
        "se3_mean": se3_day.mean(),
        "se3_max": se3_day.max(),
        "se1_mean": se1_day.mean(),
        "ee_mean": ee_day.mean(),
    }

    daily_histograms.append(hist_frac)
    daily_features.append(feats)
    daily_targets.append(targets)

print("Daily data: %d days" % len(daily_targets))
print("Price bins: %s" % bin_labels)
print()

# Show average histogram
avg_hist = np.mean(daily_histograms, axis=0)
print("Average daily histogram (fraction of hours in each bin):")
for i, label in enumerate(bin_labels):
    bar = "#" * int(avg_hist[i] * 100)
    print("  %8s: %.3f (%d%%)  %s" % (label, avg_hist[i], avg_hist[i]*100, bar))

# ================================================================
print()
print("=" * 80)
print("PART 2: PREDICTING HISTOGRAM STATISTICS")
print("=" * 80)
print()

# Build feature matrix from daily aggregates
X_daily = np.column_stack([
    np.array([f["x_mean"] for f in daily_features]),
    np.array([[f["se3_mean"]/100, f["se3_max"]/100,
               f["se1_mean"]/100, f["ee_mean"]/100]
              for f in daily_features]),
])
feat_names_daily = non_ar + ["se3_mean", "se3_max", "se1_mean", "ee_mean"]

split_d = int(len(X_daily) * 0.85)
age_d = np.arange(split_d - 1, -1, -1, dtype=np.float64)
w_d = np.exp(-decay * age_d * 24)
w_d *= split_d / w_d.sum()

# Predict each histogram statistic
print("%-15s %8s %8s %8s %8s" % ("Target", "MAE", "R2", "Spear", "SE3_coef"))
print("-" * 55)

for target_name in ["mean", "median", "p25", "p75", "p90", "max",
                      "cheap_hours", "expensive_hours", "range"]:
    y = np.array([t[target_name] for t in daily_targets])
    cd, icd = train_ridge(X_daily[:split_d], y[:split_d], w_d)
    pred = X_daily[split_d:] @ cd + icd
    mae = mean_absolute_error(y[split_d:], pred)
    r2 = r2_score(y[split_d:], pred)
    rho, _ = spearmanr(pred, y[split_d:])
    se3_c = cd[len(non_ar)]

    print("%-15s %8.2f %8.4f %8.4f %+8.2f" %
          (target_name, mae, r2, rho, se3_c))

# ================================================================
print()
print("=" * 80)
print("PART 3: CAN WE PREDICT BIN COUNTS?")
print("=" * 80)
print()

# For each bin, predict: how many hours will be in this bin?
Y_hist = np.array(daily_histograms)

print("%-10s %8s %8s %8s %8s" % ("Bin", "Avg frac", "MAE", "R2", "SE3_coef"))
print("-" * 50)

for i, label in enumerate(bin_labels):
    y = Y_hist[:, i]
    cd, icd = train_ridge(X_daily[:split_d], y[:split_d], w_d)
    pred = np.clip(X_daily[split_d:] @ cd + icd, 0, 1)
    mae = mean_absolute_error(y[split_d:], pred)
    r2 = r2_score(y[split_d:], pred) if y[split_d:].std() > 0.01 else 0
    se3_c = cd[len(non_ar)]
    print("%-10s %8.3f %8.3f %8.4f %+8.4f" %
          (label, y.mean(), mae, r2, se3_c))

# ================================================================
print()
print("=" * 80)
print("PART 4: PRACTICAL VALUE - CHEAPEST HOURS PREDICTION")
print("=" * 80)
print()

# Key question: can histogram statistics predict how many
# cheap hours are available and at what price level?

# Predict cheapest 4h average price for each day
cheapest_4h = []
for d in unique_dates:
    mask = np.array([dd == d for dd in dates])
    if mask.sum() < 20:
        continue
    fi_day = fi[mask]
    n = len(fi_day)
    blocks = [fi_day[i:i+4].mean() for i in range(n - 3)]
    cheapest_4h.append(min(blocks))

cheapest_4h = np.array(cheapest_4h)

# Predict cheapest 4h cost
cd_c4, icd_c4 = train_ridge(X_daily[:split_d], cheapest_4h[:split_d], w_d)
pred_c4 = X_daily[split_d:] @ cd_c4 + icd_c4
mae_c4 = mean_absolute_error(cheapest_4h[split_d:], pred_c4)
r2_c4 = r2_score(cheapest_4h[split_d:], pred_c4)
rho_c4, _ = spearmanr(pred_c4, cheapest_4h[split_d:])

print("Cheapest 4h block cost prediction:")
print("  MAE=%.2f EUR/MWh, R2=%.4f, Spearman=%.4f" % (mae_c4, r2_c4, rho_c4))
print("  SE3 coef: %+.2f (per SE3/100)" % cd_c4[len(non_ar)])
print()

# Most expensive 4h
most_exp_4h = []
for d in unique_dates:
    mask = np.array([dd == d for dd in dates])
    if mask.sum() < 20:
        continue
    fi_day = fi[mask]
    n = len(fi_day)
    blocks = [fi_day[i:i+4].mean() for i in range(n - 3)]
    most_exp_4h.append(max(blocks))

most_exp_4h = np.array(most_exp_4h)

cd_e4, icd_e4 = train_ridge(X_daily[:split_d], most_exp_4h[:split_d], w_d)
pred_e4 = X_daily[split_d:] @ cd_e4 + icd_e4
mae_e4 = mean_absolute_error(most_exp_4h[split_d:], pred_e4)
r2_e4 = r2_score(most_exp_4h[split_d:], pred_e4)
rho_e4, _ = spearmanr(pred_e4, most_exp_4h[split_d:])

print("Most expensive 4h block cost prediction:")
print("  MAE=%.2f EUR/MWh, R2=%.4f, Spearman=%.4f" % (mae_e4, r2_e4, rho_e4))
print("  SE3 coef: %+.2f (per SE3/100)" % cd_e4[len(non_ar)])
print()

# Price spread: max 4h - min 4h (daily price spread)
spread_4h = most_exp_4h - cheapest_4h
cd_sp, icd_sp = train_ridge(X_daily[:split_d], spread_4h[:split_d], w_d)
pred_sp = X_daily[split_d:] @ cd_sp + icd_sp
mae_sp = mean_absolute_error(spread_4h[split_d:], pred_sp)
r2_sp = r2_score(spread_4h[split_d:], pred_sp)

print("Daily price spread (max4h - min4h):")
print("  MAE=%.2f EUR/MWh, R2=%.4f" % (mae_sp, r2_sp))
print("  Avg actual: %.1f EUR/MWh, Avg predicted: %.1f" %
      (spread_4h[split_d:].mean(), pred_sp.mean()))
print()

# Simulation: what does the histogram model predict?
print("Simulation - daily histogram prediction at different SE3 levels:")
x_avg = X_daily[:split_d].mean(axis=0)
for se3_val in [30, 80, 120, 200, 400]:
    x = x_avg.copy()
    x[len(non_ar)] = se3_val / 100
    x[len(non_ar) + 1] = se3_val * 1.5 / 100  # se3_max ~ 1.5x mean
    x[len(non_ar) + 2] = se3_val * 0.5 / 100
    x[len(non_ar) + 3] = se3_val * 1.5 / 100

    mean_pred = float(x @ train_ridge(X_daily[:split_d],
                      np.array([t["mean"] for t in daily_targets[:split_d]]), w_d)[0] +
                      train_ridge(X_daily[:split_d],
                      np.array([t["mean"] for t in daily_targets[:split_d]]), w_d)[1])
    cheap_pred = float(x @ cd_c4 + icd_c4)
    exp_pred = float(x @ cd_e4 + icd_e4)
    cons_mean = (mean_pred / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    cons_cheap = (max(0, cheap_pred) / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    cons_exp = (max(0, exp_pred) / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    print("  SE3=%3d: mean=%.1f (%.1fc) cheap4h=%.1f (%.1fc) exp4h=%.1f (%.1fc)" %
          (se3_val, mean_pred, cons_mean, cheap_pred, cons_cheap, exp_pred, cons_exp))
