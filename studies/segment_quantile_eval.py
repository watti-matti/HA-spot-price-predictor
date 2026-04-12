"""Segment-hierarchical distributional forecasting: implementation + evaluation.

Compares 3 methods for predicting FI price distributions per day-segment:
1. Quantile GBM (HistGradientBoostingRegressor with quantile loss)
2. NGBoost (LogNormal distributional regression)
3. Quantile Ridge (linear quantile regression)

Architecture:
- 4 segments: night(22-06), morning(06-10), midday(10-16), evening(16-22)
- Each segment uses its own SE3/SE1 price + weather features
- Segment histograms combined into daily histogram
"""
import pandas as pd, numpy as np, yaml, sys, json, math, time, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
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
local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
hours = local.hour.values
dates = local.date
dow = local.dayofweek.values
months = local.month.values

# Raw neighbor prices
se3 = neighbor["se3"].reindex(df.index).ffill().values
se1 = neighbor["se1"].reindex(df.index).ffill().values
ee = neighbor["ee"].reindex(df.index).ffill().values

# Nuclear deficit
nuc = grid_df.get("nuclear_mw")
nuc_aligned = nuc.reindex(df.index).ffill().bfill().fillna(0).values if nuc is not None else np.zeros(len(fi))
nuclear_deficit = np.maximum(0, 1.0 - nuc_aligned)

# Weather
wind = df["wind_speed_weighted"].values
solar = df["solar_irradiance_weighted"].values
temp = weather["temperature_weighted"].reindex(df.index).ffill().values
hdd = np.maximum(0, 17 - temp)

# Segment definitions
SEGMENTS = {
    "night":   list(range(22, 24)) + list(range(0, 6)),  # 8h
    "morning": list(range(6, 10)),                         # 4h
    "midday":  list(range(10, 16)),                        # 6h
    "evening": list(range(16, 22)),                        # 6h
}

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# ================================================================
# BUILD SEGMENT-LEVEL DAILY DATASET
# ================================================================
print("Building segment-level daily dataset...")

unique_dates = sorted(set(dates))
segment_data = {seg: [] for seg in SEGMENTS}

for d in unique_dates:
    d_mask = np.array([dd == d for dd in dates])
    if d_mask.sum() < 20:
        continue

    is_workday = 1.0 if dow[d_mask][0] < 5 else 0.0
    month_val = months[d_mask][0]

    for seg_name, seg_hours in SEGMENTS.items():
        seg_mask = d_mask & np.isin(hours, seg_hours)
        if seg_mask.sum() < len(seg_hours) // 2:
            continue

        # Features for this segment
        features = {
            "wind_mean": float(wind[seg_mask].mean()),
            "solar_mean": float(solar[seg_mask].mean()),
            "hdd_mean": float(hdd[seg_mask].mean()),
            "se3_mean": float(se3[seg_mask].mean()),
            "se1_mean": float(se1[seg_mask].mean()),
            "nuclear_deficit": float(nuclear_deficit[seg_mask].mean()),
            "is_workday": is_workday,
            "month_sin": math.sin(2 * math.pi * month_val / 12),
            "month_cos": math.cos(2 * math.pi * month_val / 12),
            "wind_log_scarcity": float(np.log1p(np.maximum(0, 8 - wind[seg_mask])).mean()),
        }

        # Targets: individual hourly prices in this segment
        fi_seg = fi[seg_mask]

        # Summary targets
        targets = {
            "prices": fi_seg.tolist(),
            "mean": float(fi_seg.mean()),
            "median": float(np.median(fi_seg)),
            "p10": float(np.percentile(fi_seg, 10)),
            "p25": float(np.percentile(fi_seg, 25)),
            "p75": float(np.percentile(fi_seg, 75)),
            "p90": float(np.percentile(fi_seg, 90)),
            "p95": float(np.percentile(fi_seg, 95)),
            "max": float(fi_seg.max()),
            "min": float(fi_seg.min()),
            "date": str(d),
        }

        segment_data[seg_name].append({"features": features, "targets": targets})

for seg in SEGMENTS:
    print("  %s: %d days" % (seg, len(segment_data[seg])))

# ================================================================
# PREPARE TRAINING DATA
# ================================================================
feature_names = ["wind_mean", "solar_mean", "hdd_mean", "se3_mean", "se1_mean",
                  "nuclear_deficit", "is_workday", "month_sin", "month_cos",
                  "wind_log_scarcity"]

def build_Xy(seg_data, target_name):
    X = np.array([[d["features"][f] for f in feature_names] for d in seg_data])
    y = np.array([d["targets"][target_name] for d in seg_data])
    return X, y

split_frac = 0.85

# Time-decay weights
half_life = 120
decay = np.log(2.0) / (half_life * 24.0)

# ================================================================
# METHOD 1: Quantile GBM
# ================================================================
print()
print("=" * 80)
print("METHOD 1: QUANTILE GBM (HistGradientBoostingRegressor)")
print("=" * 80)

t_start = time.time()
gbm_models = {}
gbm_results = {}

for seg_name in SEGMENTS:
    data = segment_data[seg_name]
    split = int(len(data) * split_frac)

    gbm_models[seg_name] = {}
    gbm_results[seg_name] = {}

    for q in QUANTILES:
        target_key = "p%d" % int(q * 100) if q in [0.10, 0.25, 0.75, 0.90, 0.95] else "median"
        if target_key not in data[0]["targets"]:
            target_key = "mean"

        # For quantile GBM, train on individual hourly prices (expanded)
        X_list, y_list = [], []
        for d in data[:split]:
            x = [d["features"][f] for f in feature_names]
            for price in d["targets"]["prices"]:
                X_list.append(x)
                y_list.append(price)

        X_tr = np.array(X_list)
        y_tr = np.array(y_list)

        # Weights (time-decay per day, broadcast to hours)
        w_days = np.exp(-decay * np.arange(split - 1, -1, -1, dtype=np.float64) * 24)
        w_hours = []
        for i, d in enumerate(data[:split]):
            w_hours.extend([w_days[i]] * len(d["targets"]["prices"]))
        w_hours = np.array(w_hours)
        w_hours *= len(w_hours) / w_hours.sum()

        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=10, random_state=42,
        )
        model.fit(X_tr, y_tr, sample_weight=w_hours)
        gbm_models[seg_name][q] = model

t_gbm = time.time() - t_start
print("Training time: %.1f seconds" % t_gbm)

# ================================================================
# METHOD 2: NGBoost
# ================================================================
print()
print("=" * 80)
print("METHOD 2: NGBoost (LogNormal distributional regression)")
print("=" * 80)

try:
    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal
    has_ngboost = True
except ImportError:
    print("NGBoost not installed. Skipping. Install with: pip install ngboost")
    has_ngboost = False

ngb_models = {}
t_start = time.time()

if has_ngboost:
    for seg_name in SEGMENTS:
        data = segment_data[seg_name]
        split = int(len(data) * split_frac)

        X_list, y_list = [], []
        for d in data[:split]:
            x = [d["features"][f] for f in feature_names]
            for price in d["targets"]["prices"]:
                X_list.append(x)
                y_list.append(max(price + 1, 0.1))  # LogNormal needs positive

        X_tr = np.array(X_list)
        y_tr = np.array(y_list)

        model = NGBRegressor(
            Dist=LogNormal, n_estimators=200,
            learning_rate=0.05, minibatch_frac=0.8,
            verbose=False, random_state=42,
        )
        model.fit(X_tr, y_tr)
        ngb_models[seg_name] = model

    t_ngb = time.time() - t_start
    print("Training time: %.1f seconds" % t_ngb)
else:
    t_ngb = 0

# ================================================================
# METHOD 3: Quantile Ridge
# ================================================================
print()
print("=" * 80)
print("METHOD 3: QUANTILE RIDGE (linear quantile regression)")
print("=" * 80)

t_start = time.time()
qr_models = {}

for seg_name in SEGMENTS:
    data = segment_data[seg_name]
    split = int(len(data) * split_frac)

    qr_models[seg_name] = {}

    X_list, y_list = [], []
    for d in data[:split]:
        x = [d["features"][f] for f in feature_names]
        for price in d["targets"]["prices"]:
            X_list.append(x)
            y_list.append(price)

    X_tr = np.array(X_list)
    y_tr = np.array(y_list)

    for q in QUANTILES:
        model = QuantileRegressor(
            quantile=q, alpha=1.0, solver="highs",
        )
        model.fit(X_tr, y_tr)
        qr_models[seg_name][q] = model

t_qr = time.time() - t_start
print("Training time: %.1f seconds" % t_qr)

# ================================================================
# EVALUATION
# ================================================================
print()
print("=" * 80)
print("COMPREHENSIVE EVALUATION")
print("=" * 80)
print()

def predict_quantiles_gbm(seg_name, features):
    x = np.array([[features[f] for f in feature_names]])
    return {q: float(gbm_models[seg_name][q].predict(x)[0]) for q in QUANTILES}

def predict_quantiles_ngb(seg_name, features):
    if not has_ngboost:
        return {}
    x = np.array([[features[f] for f in feature_names]])
    dist = ngb_models[seg_name].pred_dist(x)
    return {q: float(dist.ppf(q)[0]) - 1 for q in QUANTILES}  # undo +1 shift

def predict_quantiles_qr(seg_name, features):
    x = np.array([[features[f] for f in feature_names]])
    return {q: float(qr_models[seg_name][q].predict(x)[0]) for q in QUANTILES}

# Evaluate on test set
methods = [("GBM", predict_quantiles_gbm)]
if has_ngboost:
    methods.append(("NGBoost", predict_quantiles_ngb))
methods.append(("QRidge", predict_quantiles_qr))

# Per-segment evaluation
for method_name, predict_fn in methods:
    print("--- %s ---" % method_name)

    all_actual = []
    all_predicted_median = []
    all_pinball = {q: [] for q in QUANTILES}
    coverage = {q: 0 for q in QUANTILES}
    total_count = 0

    for seg_name in SEGMENTS:
        data = segment_data[seg_name]
        split = int(len(data) * split_frac)
        test_data = data[split:]

        seg_actual = []
        seg_pred_med = []

        for d in test_data:
            preds = predict_fn(seg_name, d["features"])
            if not preds:
                continue

            actual_prices = d["targets"]["prices"]
            pred_median = preds.get(0.50, 0)

            for price in actual_prices:
                seg_actual.append(price)
                seg_pred_med.append(pred_median)
                total_count += 1

                for q in QUANTILES:
                    loss = mean_pinball_loss([price], [preds[q]], alpha=q)
                    all_pinball[q].append(loss)
                    if price <= preds[q]:
                        coverage[q] += 1

        all_actual.extend(seg_actual)
        all_predicted_median.extend(seg_pred_med)

        # Segment-level metrics
        if seg_actual:
            mae = mean_absolute_error(seg_actual, seg_pred_med)
            rho, _ = spearmanr(seg_pred_med, seg_actual) if len(seg_actual) > 10 else (0, 0)
            print("  %-10s: MAE=%.2f Spearman=%.4f n=%d" %
                  (seg_name, mae, rho, len(seg_actual)))

    # Overall metrics
    mae_overall = mean_absolute_error(all_actual, all_predicted_median)
    rho_overall, _ = spearmanr(all_predicted_median, all_actual)

    print("  OVERALL:    MAE=%.2f Spearman=%.4f" % (mae_overall, rho_overall))

    # Pinball loss per quantile
    print("  Pinball loss: ", end="")
    for q in QUANTILES:
        avg_pb = np.mean(all_pinball[q])
        print("p%d=%.2f " % (int(q*100), avg_pb), end="")
    print()

    # Coverage calibration
    print("  Coverage:     ", end="")
    for q in QUANTILES:
        cov = coverage[q] / total_count if total_count > 0 else 0
        ideal = q
        print("p%d=%.0f%%(%.0f%%) " % (int(q*100), cov*100, ideal*100), end="")
    print()
    print()

# ================================================================
# DAILY HISTOGRAM RECONSTRUCTION
# ================================================================
print("=" * 80)
print("DAILY HISTOGRAM: COMBINE SEGMENT PREDICTIONS")
print("=" * 80)
print()

seg_hours = {"night": 8, "morning": 4, "midday": 6, "evening": 6}

# For each test day, reconstruct daily histogram from segment quantiles
daily_results = {m[0]: [] for m in methods}

for method_name, predict_fn in methods:
    # Group test data by date
    test_dates = {}
    for seg_name in SEGMENTS:
        data = segment_data[seg_name]
        split = int(len(data) * split_frac)
        for d in data[split:]:
            date = d["targets"]["date"]
            if date not in test_dates:
                test_dates[date] = {}
            preds = predict_fn(seg_name, d["features"])
            if preds:
                test_dates[date][seg_name] = {
                    "predicted": preds,
                    "actual_mean": d["targets"]["mean"],
                    "actual_prices": d["targets"]["prices"],
                }

    for date, segs in test_dates.items():
        if len(segs) < 4:
            continue

        # Weighted combination of segment quantiles
        total_hours = sum(seg_hours[s] for s in segs)
        daily_quantiles = {}
        for q in QUANTILES:
            daily_quantiles[q] = sum(
                segs[s]["predicted"].get(q, 0) * seg_hours[s]
                for s in segs
            ) / total_hours

        # Actual daily statistics
        all_prices = []
        for s in segs:
            all_prices.extend(segs[s]["actual_prices"])

        actual_mean = np.mean(all_prices)
        actual_cheapest_4h = min(
            np.mean(sorted(all_prices)[i:i+4])
            for i in range(len(all_prices) - 3)
        ) if len(all_prices) >= 4 else actual_mean

        daily_results[method_name].append({
            "date": date,
            "pred_median": daily_quantiles.get(0.50, 0),
            "pred_p10": daily_quantiles.get(0.10, 0),
            "pred_p90": daily_quantiles.get(0.90, 0),
            "pred_p95": daily_quantiles.get(0.95, 0),
            "actual_mean": actual_mean,
            "actual_cheapest_4h": actual_cheapest_4h,
        })

# Daily-level evaluation
print("%-8s %8s %8s %8s %8s %8s" %
      ("Method", "MAE_med", "R2_med", "Spear", "p90_cov", "Max_p95"))
print("-" * 55)

for method_name in [m[0] for m in methods]:
    results = daily_results[method_name]
    if not results:
        continue

    actual = np.array([r["actual_mean"] for r in results])
    pred_med = np.array([r["pred_median"] for r in results])
    pred_p90 = np.array([r["pred_p90"] for r in results])
    pred_p95 = np.array([r["pred_p95"] for r in results])

    mae = mean_absolute_error(actual, pred_med)
    r2 = 1 - np.sum((actual - pred_med)**2) / np.sum((actual - actual.mean())**2)
    rho, _ = spearmanr(pred_med, actual)
    p90_cov = np.mean(actual <= pred_p90)
    max_p95 = pred_p95.max()

    print("%-8s %8.2f %8.4f %8.4f %7.0f%% %8.1f" %
          (method_name, mae, r2, rho, p90_cov * 100, max_p95))

# ================================================================
# SE3 COUPLING ANALYSIS
# ================================================================
print()
print("=" * 80)
print("SE3 COUPLING: HOW STRONGLY DOES EACH METHOD RESPOND?")
print("=" * 80)
print()

# Simulate: hold everything constant, vary SE3
base_features = {
    "wind_mean": 5.0, "solar_mean": 100.0, "hdd_mean": 10.0,
    "se3_mean": 0.0,  # will vary
    "se1_mean": 0.0,
    "nuclear_deficit": 0.0,
    "is_workday": 1.0,
    "month_sin": math.sin(2 * math.pi * 4 / 12),
    "month_cos": math.cos(2 * math.pi * 4 / 12),
    "wind_log_scarcity": math.log1p(3),
}

print("%-8s" % "SE3", end="")
for method_name, _ in methods:
    print(" %8s_p50 %8s_p90" % (method_name, method_name), end="")
print()
print("-" * 70)

for se3_val in [20, 40, 60, 80, 100, 150, 200, 300, 400, 600]:
    feats = dict(base_features)
    feats["se3_mean"] = se3_val
    feats["se1_mean"] = se3_val * 0.5

    print("%-8d" % se3_val, end="")
    for method_name, predict_fn in methods:
        # Average across segments (weighted)
        p50_total = 0
        p90_total = 0
        total_h = 0
        for seg_name, n_hours in seg_hours.items():
            preds = predict_fn(seg_name, feats)
            if preds:
                p50_total += preds.get(0.50, 0) * n_hours
                p90_total += preds.get(0.90, 0) * n_hours
                total_h += n_hours
        if total_h > 0:
            p50 = p50_total / total_h
            p90 = p90_total / total_h
            cons_p50 = (max(0, p50) / 1000 + 0.0361 + 0.02325) * 1.255 * 100
            cons_p90 = (max(0, p90) / 1000 + 0.0361 + 0.02325) * 1.255 * 100
            print(" %5.1f/%4.1fc %5.1f/%4.1fc" % (p50, cons_p50, p90, cons_p90), end="")
    print()

# Timing summary
print()
print("=" * 80)
print("TIMING SUMMARY")
print("=" * 80)
print("  GBM:     %.1f seconds" % t_gbm)
if has_ngboost:
    print("  NGBoost: %.1f seconds" % t_ngb)
print("  QRidge:  %.1f seconds" % t_qr)

# QRidge coefficient analysis
print()
print("=" * 80)
print("QUANTILE RIDGE: SE3 COUPLING COEFFICIENTS PER SEGMENT × QUANTILE")
print("=" * 80)
print()
print("%-10s" % "Segment", end="")
for q in QUANTILES:
    print(" p%d" % int(q*100), end="     ")
print()
print("-" * 65)
for seg_name in SEGMENTS:
    print("%-10s" % seg_name, end="")
    for q in QUANTILES:
        se3_idx = feature_names.index("se3_mean")
        coef = qr_models[seg_name][q].coef_[se3_idx]
        print(" %+.4f" % coef, end=" ")
    print()
