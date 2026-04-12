"""Model Dashboard — Comprehensive model monitoring and evaluation.

Generates a single HTML dashboard combining:
  1. Duration model: D(k) curves, forgetting factor sweep, rolling Spearman
  2. Hourly model: feature importance, hourly/monthly MAE breakdown
  3. Scatter plots and contour visualizations

Usage:
    python model_dashboard.py [--region finland]
    Output: output/model_dashboard.html
"""
import pandas as pd, numpy as np, yaml, sys, json, math, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
from src.features import build_features

# ================================================================
# DATA LOADING
# ================================================================
print("Loading data...")
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

se3 = neighbor["se3"].reindex(df.index).ffill().values
se1 = neighbor["se1"].reindex(df.index).ffill().values
wind = df["wind_speed_weighted"].values
solar = df["solar_irradiance_weighted"].values
temp = weather["temperature_weighted"].reindex(df.index).ffill().values
hdd_threshold = config.get("demand", {}).get("hdd_threshold", 17.0)
hdd = np.maximum(0, hdd_threshold - temp)

nuc = grid_df.get("nuclear_mw")
nuc_vals = nuc.reindex(df.index).ffill().bfill().fillna(0).values if nuc is not None else np.zeros(len(fi))
nuclear_deficit = np.maximum(0, 1.0 - nuc_vals)

# Duration model config from YAML
dur_cfg = config.get("duration_model", {})
SEGMENTS = dur_cfg.get("segments", {
    "night":   [22, 23, 0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8, 9],
    "midday":  [10, 11, 12, 13, 14, 15],
    "evening": [16, 17, 18, 19, 20, 21],
})
SEG_HOURS = {seg: len(hrs) for seg, hrs in SEGMENTS.items()}
LOG_OFFSET = dur_cfg.get("log_offset", 55)
DURATION_LAMBDA_DEFAULT = dur_cfg.get("lambda", 0.990)
DURATION_RIDGE_ALPHA = dur_cfg.get("ridge_alpha", 1.0)
DURATION_EXP_CAP = dur_cfg.get("exp_cap", 20.0)

feature_names = dur_cfg.get("features", [
    "wind_mean", "solar_mean", "hdd_mean", "se3_mean", "se1_mean",
    "nuclear_deficit", "is_workday", "month_sin", "month_cos",
    "wind_log_scarcity"])
n_features = len(feature_names)

# Consumer price conversion from YAML
cons_cfg = config.get("consumer_pricing", {})
_vat = cons_cfg.get("vat_multiplier", 1.255)
_tax = cons_cfg.get("energy_tax_eur_kwh", 0.02325)
_default_op = cons_cfg.get("default_operator", "Elenia")
_operators = {op["name"]: op for op in cons_cfg.get("operators", [])}
_transfer = _operators.get(_default_op, {}).get("day_rate_eur_kwh", 0.0361)

def to_cons(spot):
    return (max(0.0, spot) / 1000 + _transfer + _tax) * _vat * 100

# ================================================================
# BUILD SEGMENT DATA WITH DURATION TARGETS
# ================================================================
_wind_scarcity_base = config.get("features", {}).get("wind_log_scarcity_base", 8.0)
print("Building segment data with duration targets...")
unique_dates = sorted(set(dates))
segment_data = {seg: [] for seg in SEGMENTS}
fullday_data = []

for d in unique_dates:
    d_mask = np.array([dd == d for dd in dates])
    if d_mask.sum() < 20:
        continue
    is_wd = 1.0 if dow[d_mask][0] < 5 else 0.0
    mo = months[d_mask][0]

    # Full-day duration curve (ground truth)
    day_prices = fi[d_mask]
    if len(day_prices) >= 23:
        sorted_prices = np.sort(day_prices)
        duration_curve = np.cumsum(sorted_prices) / np.arange(1, len(sorted_prices) + 1)
        fullday_data.append({
            "date": str(d),
            "duration_curve": duration_curve.tolist(),
            "sorted_prices": sorted_prices.tolist(),
        })

    # Per-segment duration curves
    for seg_name, seg_hours_list in SEGMENTS.items():
        seg_mask = d_mask & np.isin(hours, seg_hours_list)
        n_h = seg_mask.sum()
        if n_h != SEG_HOURS[seg_name]:
            continue
        seg_prices = fi[seg_mask]
        sorted_seg = np.sort(seg_prices)
        dur_curve = np.cumsum(sorted_seg) / np.arange(1, len(sorted_seg) + 1)
        segment_data[seg_name].append({
            "features": {
                "wind_mean": float(wind[seg_mask].mean()),
                "solar_mean": float(solar[seg_mask].mean()),
                "hdd_mean": float(hdd[seg_mask].mean()),
                "se3_mean": float(se3[seg_mask].mean()),
                "se1_mean": float(se1[seg_mask].mean()),
                "nuclear_deficit": float(nuclear_deficit[seg_mask].mean()),
                "is_workday": is_wd,
                "month_sin": math.sin(2 * math.pi * mo / 12),
                "month_cos": math.cos(2 * math.pi * mo / 12),
                "wind_log_scarcity": float(np.log1p(np.maximum(0, _wind_scarcity_base - wind[seg_mask])).mean()),
            },
            "date": str(d),
            "duration_curve": dur_curve.tolist(),
            "sorted_prices": sorted_seg.tolist(),
        })

print("  Full-day records: %d" % len(fullday_data))
for seg in SEGMENTS:
    print("  %s: %d records, %d hours/segment" % (seg, len(segment_data[seg]), SEG_HOURS[seg]))

# ================================================================
# PRE-BUILD MATRICES PER SEGMENT
# ================================================================
seg_matrices = {}
for seg_name in SEGMENTS:
    data = segment_data[seg_name]
    n = len(data)
    n_dur = SEG_HOURS[seg_name]
    X = np.array([[d["features"][f] for f in feature_names] for d in data])
    Y = np.zeros((n_dur, n))
    for k in range(n_dur):
        raw_vals = np.array([d["duration_curve"][k] + LOG_OFFSET for d in data])
        Y[k] = np.log(np.maximum(raw_vals, 1.0))
    seg_matrices[seg_name] = {"X": X, "Y": Y, "n": n, "n_dur": n_dur,
                               "dates": [d["date"] for d in data],
                               "curves": [d["duration_curve"] for d in data]}

# ================================================================
# TRAIN WITH lambda=0.990 (EXPANDING WINDOW)
# ================================================================
LAMBDA = DURATION_LAMBDA_DEFAULT
MIN_TRAIN = dur_cfg.get("min_train_days", 180)
ridge_alpha = DURATION_RIDGE_ALPHA
fd_lookup = {d["date"]: d for d in fullday_data}
best_lam = LAMBDA
best_hl = -math.log(2) / math.log(LAMBDA)
best_hl_str = "%.0f" % best_hl

print("\n" + "=" * 60)
print("TRAINING  lambda=%.3f  half-life=%s days" % (LAMBDA, best_hl_str))
print("=" * 60)

# Per-segment expanding window
seg_preds = {seg: {} for seg in SEGMENTS}
for seg_name in SEGMENTS:
    m = seg_matrices[seg_name]
    X, Y, n, n_dur = m["X"], m["Y"], m["n"], m["n_dur"]
    dates_s = m["dates"]
    iso = IsotonicRegression(increasing=True)
    print("  %s (%d days, %d levels)..." % (seg_name, n, n_dur), end=" ", flush=True)
    for t in range(MIN_TRAIN, n):
        w = LAMBDA ** np.arange(t - 1, -1, -1, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        # Augmented matrix [X|1] with no penalty on intercept
        X_aug = np.column_stack([X[:t], np.ones(t)])
        Xw_aug = X_aug * sqrt_w[:, None]
        A = Xw_aug.T @ Xw_aug + ridge_alpha * np.eye(n_features + 1)
        A[n_features, n_features] -= ridge_alpha  # don't penalise intercept
        x_test_aug = np.append(X[t], 1.0)
        raw = []
        for k in range(n_dur):
            yw = Y[k, :t] * sqrt_w
            beta = np.linalg.solve(A, Xw_aug.T @ yw)
            lp = float(beta @ x_test_aug)
            raw.append(max(0.0, math.exp(min(lp, DURATION_EXP_CAP)) - LOG_OFFSET))
        pava = iso.fit_transform(np.arange(n_dur), raw).tolist()
        seg_preds[seg_name][dates_s[t]] = pava
    print("done (%d predictions)" % len(seg_preds[seg_name]))

# Combine segments -> full-day
eval_dates = sorted(set.intersection(*[set(seg_preds[s].keys()) for s in SEGMENTS]))
best_eval = []
for date in eval_dates:
    fd = fd_lookup.get(date)
    if fd is None:
        continue
    pred_prices = []
    for seg_name in SEGMENTS:
        pc = seg_preds[seg_name][date]
        for k in range(len(pc)):
            if k == 0:
                pred_prices.append(pc[0])
            else:
                pred_prices.append(max(0.0, (k + 1) * pc[k] - k * pc[k - 1]))
    pred_prices.sort()
    nh = len(pred_prices)
    pred_dur = (np.cumsum(pred_prices) / np.arange(1, nh + 1)).tolist()
    actual_dur = fd["duration_curve"][:nh]
    best_eval.append({
        "date": date,
        "pred": pred_dur,
        "actual": actual_dur,
        "pred_sorted": pred_prices,
        "actual_sorted": fd["sorted_prices"][:nh],
    })

print("\nFull-day evaluation: %d days" % len(best_eval))

# Metrics
print("\nFull-day metrics (consumer c/kWh):")
for k in [1, 4, 6, 8, 12, 24]:
    if k > len(best_eval[0]["pred"]):
        continue
    pv = [r["pred"][k - 1] for r in best_eval]
    av = [r["actual"][k - 1] for r in best_eval]
    rho = spearmanr(pv, av).statistic
    mae_c = np.mean([abs(to_cons(p) - to_cons(a)) for p, a in zip(pv, av)])
    bias_c = np.mean([to_cons(p) - to_cons(a) for p, a in zip(pv, av)])
    use = {1: "cheapest 1h", 4: "cheapest 4h", 6: "cheapest 6h", 8: "cheapest 8h",
           12: "cheapest 12h", 24: "daily avg"}
    print("  D(%2d) %-14s: rho=%.4f  MAE=%.3f c/kWh  Bias=%+.3f" %
          (k, use.get(k, ""), rho, mae_c, bias_c))

# Last-year metrics
last_365 = best_eval[-365:]
print("\nLast-year metrics:")
for k in [1, 4, 8, 24]:
    pv = [r["pred"][k - 1] for r in last_365]
    av = [r["actual"][k - 1] for r in last_365]
    rho = spearmanr(pv, av).statistic
    mae_c = np.mean([abs(to_cons(p) - to_cons(a)) for p, a in zip(pv, av)])
    print("  D(%2d): rho=%.4f  MAE=%.3f c/kWh  (n=%d)" % (k, rho, mae_c, len(last_365)))

# Rolling Spearman (90-day window)
ROLL_W = 90
rolling_rho = []
for i in range(ROLL_W, len(best_eval)):
    window = best_eval[i - ROLL_W:i]
    date = window[-1]["date"]
    rv = {}
    for k in [1, 4, 8, 24]:
        pv = [r["pred"][k - 1] for r in window]
        av = [r["actual"][k - 1] for r in window]
        rv[k] = round(float(spearmanr(pv, av).statistic), 3)
    rolling_rho.append({"date": date, "d1": rv[1], "d4": rv[4], "d8": rv[8], "d24": rv[24]})

print("\nRolling Spearman (90-day window):")
print("  Latest:  D(4)=%.3f  D(8)=%.3f  D(24)=%.3f" %
      (rolling_rho[-1]["d4"], rolling_rho[-1]["d8"], rolling_rho[-1]["d24"]))
print("  Best:    D(4)=%.3f" % max(r["d4"] for r in rolling_rho))
print("  Worst:   D(4)=%.3f" % min(r["d4"] for r in rolling_rho))

# ================================================================
# LAMBDA SWEEP (live computation with augmented matrix)
# ================================================================
print("\nRunning lambda sweep...")
sweep_lambdas = [0.950, 0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000]
sweep_table = []
prod_table = []

for sweep_lam in sweep_lambdas:
    hl = -math.log(2) / math.log(sweep_lam) if sweep_lam < 1.0 else float("inf")
    hl_str = "%.0fd" % hl if hl < 1000 else "none"

    # Run expanding window for this lambda
    sweep_seg_preds = {seg: {} for seg in SEGMENTS}
    for seg_name in SEGMENTS:
        m = seg_matrices[seg_name]
        Xs, Ys, ns, nds = m["X"], m["Y"], m["n"], m["n_dur"]
        dates_s = m["dates"]
        iso_sw = IsotonicRegression(increasing=True)
        for t in range(MIN_TRAIN, ns):
            w = sweep_lam ** np.arange(t - 1, -1, -1, dtype=np.float64)
            sqrt_w = np.sqrt(w)
            X_aug = np.column_stack([Xs[:t], np.ones(t)])
            Xw_aug = X_aug * sqrt_w[:, None]
            A = Xw_aug.T @ Xw_aug + ridge_alpha * np.eye(n_features + 1)
            A[n_features, n_features] -= ridge_alpha
            x_test_aug = np.append(Xs[t], 1.0)
            raw = []
            for k in range(nds):
                yw = Ys[k, :t] * sqrt_w
                beta = np.linalg.solve(A, Xw_aug.T @ yw)
                lp = float(beta @ x_test_aug)
                raw.append(max(0.0, math.exp(min(lp, DURATION_EXP_CAP)) - LOG_OFFSET))
            pava = iso_sw.fit_transform(np.arange(nds), raw).tolist()
            sweep_seg_preds[seg_name][dates_s[t]] = pava

    # Combine segments -> full-day and evaluate
    sw_dates = sorted(set.intersection(*[set(sweep_seg_preds[s].keys()) for s in SEGMENTS]))
    sw_eval = []
    for date in sw_dates:
        fd = fd_lookup.get(date)
        if fd is None:
            continue
        pred_prices = []
        for seg_name in SEGMENTS:
            pc = sweep_seg_preds[seg_name][date]
            for k in range(len(pc)):
                if k == 0:
                    pred_prices.append(pc[0])
                else:
                    pred_prices.append(max(0.0, (k + 1) * pc[k] - k * pc[k - 1]))
        pred_prices.sort()
        nh = len(pred_prices)
        pred_dur = (np.cumsum(pred_prices) / np.arange(1, nh + 1)).tolist()
        actual_dur = fd["duration_curve"][:nh]
        sw_eval.append({"pred": pred_dur, "actual": actual_dur})

    # Compute rho at key levels
    def _cons(s):
        return (max(0.0, s) / 1000 + 0.0361 + 0.02325) * 1.255 * 100

    def _rho_at(records, k_idx):
        pv = [_cons(r["pred"][k_idx]) for r in records]
        av = [_cons(r["actual"][k_idx]) for r in records]
        return spearmanr(pv, av).statistic

    def _mae_at(records, k_idx):
        return float(np.mean([abs(_cons(r["pred"][k_idx]) - _cons(r["actual"][k_idx])) for r in records]))

    d1_rho = _rho_at(sw_eval, 0)
    d4_rho = _rho_at(sw_eval, 3)
    d8_rho = _rho_at(sw_eval, 7)
    d24_rho = _rho_at(sw_eval, 23)
    d4_mae = _mae_at(sw_eval, 3)
    composite = (d1_rho + d4_rho + d8_rho + d24_rho) / 4

    sweep_table.append({
        "lam": sweep_lam, "halflife": hl_str, "composite": round(composite, 4),
        "d1_rho": round(d1_rho, 4), "d4_rho": round(d4_rho, 4),
        "d8_rho": round(d8_rho, 4), "d24_rho": round(d24_rho, 4),
        "d4_mae": round(d4_mae, 3),
    })

    # Production sweep (last 365 days only)
    sw_eval_365 = sw_eval[-365:] if len(sw_eval) > 365 else sw_eval
    d1_r365 = _rho_at(sw_eval_365, 0)
    d4_r365 = _rho_at(sw_eval_365, 3)
    d8_r365 = _rho_at(sw_eval_365, 7)
    d24_r365 = _rho_at(sw_eval_365, 23)
    d4_m365 = _mae_at(sw_eval_365, 3)
    comp365 = (d1_r365 + d4_r365 + d8_r365 + d24_r365) / 4

    prod_table.append({
        "lam": sweep_lam, "hl": hl_str, "comp": round(comp365, 4),
        "m": {
            1: {"rho": round(d1_r365, 3), "mae": round(d4_m365, 3)},
            4: {"rho": round(d4_r365, 3), "mae": round(d4_m365, 3)},
            8: {"rho": round(d8_r365, 3), "mae": 0.0},
            24: {"rho": round(d24_r365, 3), "mae": 0.0},
        },
    })

    print("  lambda=%.3f hl=%-5s  rho: D1=%.3f D4=%.3f D8=%.3f D24=%.3f  (365d: D4=%.3f)" % (
        sweep_lam, hl_str, d1_rho, d4_rho, d8_rho, d24_rho, d4_r365))

# ================================================================
# HOURLY MODEL: FEATURE IMPORTANCE + HOURLY/MONTHLY MAE
# ================================================================
print("\nComputing hourly model metrics...")
from src.train_model import _make_time_weights, _batched_stats, _solve_normal_eq, _predict
from sklearn.metrics import mean_absolute_error as sk_mae, r2_score

coefs_path = "output/model_coefs.json"
with open(coefs_path) as _f:
    model_coefs = json.load(_f)

base_features = model_coefs["feature_names"]
# Ensure all features present
for feat_name in base_features:
    if feat_name not in df.columns:
        df[feat_name] = 0.0

X_hourly = df[base_features].values.astype(np.float64)
y_hourly = df["price_eur_mwh"].values.astype(np.float64)
y_clipped = np.minimum(y_hourly, 500.0)

training_cfg = config.get("training", {})
test_split = training_cfg.get("test_split", 0.15)
split_idx = int(len(X_hourly) * (1.0 - test_split))

# Predict using saved model
feature_coefs_arr = np.array([f["coef"] for f in model_coefs["features"]])
intercept = model_coefs["intercept"]
log_offset = model_coefs.get("log_offset", 55)
power_scale = model_coefs.get("power_scale", 1.0)
power_exp = model_coefs.get("power_exp", 1.0)
log_pred = X_hourly @ feature_coefs_arr + intercept
raw_pred = np.maximum(0, np.exp(np.minimum(log_pred, 20.0)) - log_offset)
preds_all = power_scale * np.power(raw_pred + 1e-10, power_exp)

X_te = X_hourly[split_idx:]
y_te = y_clipped[split_idx:]
preds_te = preds_all[split_idx:]
ts_all = df.index

# Overall metrics
hourly_mae = float(sk_mae(y_te, preds_te))
hourly_rmse = float(np.sqrt(np.mean((y_te - preds_te) ** 2)))
hourly_r2 = float(r2_score(y_te, preds_te))
print("  Hourly model: MAE=%.2f, RMSE=%.2f, R2=%.4f" % (hourly_mae, hourly_rmse, hourly_r2))

# Hourly and monthly MAE
local_hours_all = (ts_all + pd.Timedelta(hours=3)).hour.to_numpy()
local_months_all = (ts_all + pd.Timedelta(hours=3)).month.to_numpy()

hourly_mae_vals = []
for h in range(24):
    mask = local_hours_all == h
    hourly_mae_vals.append(float(sk_mae(y_clipped[mask], preds_all[mask])) if mask.sum() > 0 else 0.0)

monthly_mae_vals = []
for m in range(1, 13):
    mask = local_months_all == m
    monthly_mae_vals.append(float(sk_mae(y_clipped[mask], preds_all[mask])) if mask.sum() > 10 else 0.0)

# Feature importance (|coef| * std)
from src.features import BASE_FEATURES
feature_importance = []
for feat in model_coefs["features"]:
    name = feat["name"]
    c = feat["coef"]
    if name.startswith(("import_potential_", "export_potential_", "ar_")):
        group = "cross-border"
    elif name.startswith(("nuclear_", "flow_fi_")):
        group = "nuclear"
    else:
        group = "base"
    feat_std = float(df[name].std()) if name in df.columns else 1.0
    impact = abs(c) * feat_std
    feature_importance.append({
        "name": name, "coef": round(c, 4), "abs_coef": round(abs(c), 4),
        "std": round(feat_std, 4), "impact": round(impact, 4), "group": group,
    })
feature_importance.sort(key=lambda x: x["impact"], reverse=True)
top_features = feature_importance[:20]

print("  Top 5 features by impact:")
for fi in top_features[:5]:
    print("    %-25s impact=%.4f  coef=%+.4f  group=%s" % (fi["name"], fi["impact"], fi["coef"], fi["group"]))

# ================================================================
# BUILD HTML DASHBOARD
# ================================================================
print("\nBuilding dashboard...")

chart_daily = []
for r in best_eval:
    chart_daily.append({
        "date": r["date"],
        "pred": [round(v, 2) for v in r["pred"]],
        "actual": [round(v, 2) for v in r["actual"]],
        "pred_sorted": [round(v, 2) for v in r["pred_sorted"]],
        "actual_sorted": [round(v, 2) for v in r["actual_sorted"]],
    })

sweep_chart = []
for row in sweep_table:
    sweep_chart.append({
        "lambda": row["lam"],
        "halflife": row["halflife"],
        "composite": row["composite"],
        "d1_rho": row.get("d1_rho", 0),
        "d4_rho": row.get("d4_rho", 0),
        "d8_rho": row.get("d8_rho", 0),
        "d24_rho": row.get("d24_rho", 0),
        "d4_mae": row.get("d4_mae", 0),
    })

prod_sweep_chart = []
for row in prod_table:
    prod_sweep_chart.append({
        "lambda": row["lam"], "halflife": row["hl"], "composite": row["comp"],
        "d1_rho": row["m"][1]["rho"], "d4_rho": row["m"][4]["rho"],
        "d8_rho": row["m"][8]["rho"], "d24_rho": row["m"][24]["rho"],
        "d4_mae": row["m"][4]["mae"],
    })

chart_json = json.dumps({
    "daily": chart_daily,
    "sweep": sweep_chart,
    "prod_sweep": prod_sweep_chart,
    "rolling": rolling_rho,
    "best_lambda": best_lam,
    "best_halflife": best_hl_str,
    "hourly_model": {
        "mae": round(hourly_mae, 2),
        "rmse": round(hourly_rmse, 2),
        "r2": round(hourly_r2, 4),
        "hourly_mae": [round(v, 2) for v in hourly_mae_vals],
        "monthly_mae": [round(v, 2) for v in monthly_mae_vals],
        "feature_importance": [
            {"name": f["name"], "coef": f["coef"], "impact": f["impact"],
             "std": f["std"], "group": f["group"]}
            for f in top_features
        ],
        "model_version": model_coefs.get("model_version", "?"),
        "n_features": model_coefs.get("feature_count", len(base_features)),
    },
})

html = '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Dashboard — Spot Price Predictor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:20px; }
h1 { color:#f8fafc; margin-bottom:5px; font-size:22px; }
h2 { color:#94a3b8; margin:25px 0 10px; font-size:16px; border-bottom:1px solid #334155; padding-bottom:6px; }
.sub { color:#64748b; font-size:12px; margin-bottom:12px; }
.box { background:#1e293b; border-radius:8px; padding:15px; margin:12px 0; }
canvas { width:100%!important; }
.grid-4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:15px; }
.mrow { display:flex; gap:10px; margin:10px 0; flex-wrap:wrap; }
.met { background:#1e293b; border-radius:8px; padding:8px 12px; text-align:center; min-width:100px; flex:1; }
.mv { font-size:17px; font-weight:700; }
.ml { font-size:10px; color:#94a3b8; margin-top:2px; }
.vc { color:#22d3ee; } .vy { color:#facc15; } .vo { color:#f97316; }
.vr { color:#ef4444; } .vg { color:#4ade80; } .vp { color:#a78bfa; }
.sc { display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
.sc button { background:#334155; color:#94a3b8; border:none; border-radius:4px;
  padding:4px 12px; font-size:12px; cursor:pointer; transition:all 0.15s; }
.sc button:hover { background:#475569; color:#e2e8f0; }
.sc button.active { background:#60a5fa; color:#0f172a; font-weight:600; }
.sc input[type="range"] { flex:1; min-width:120px; accent-color:#60a5fa; }
.slbl { color:#64748b; font-size:11px; min-width:140px; text-align:right; }
footer { margin-top:30px; padding:12px 0; text-align:center; color:#475569; font-size:11px;
  border-top:1px solid #1e293b; }
footer a { color:#60a5fa; text-decoration:none; }
.refs { margin-top:30px; }
.refs h2 { color:#94a3b8; }
.refs h3 { color:#e2e8f0; font-size:13px; margin:16px 0 6px; }
.refs ol { padding-left:22px; margin:0 0 8px; }
.refs li { color:#cbd5e1; font-size:11.5px; line-height:1.55; margin-bottom:4px; }
.refs li em { color:#94a3b8; font-style:italic; }
.refs .lim { background:#1e293b; border-radius:6px; padding:10px 14px; margin:8px 0 16px; }
.refs .lim ul { padding-left:18px; margin:0; }
.refs .lim li { color:#94a3b8; font-size:11px; line-height:1.5; }
.refs .lim li strong { color:#e2e8f0; font-weight:600; }
.refs table { width:100%; border-collapse:collapse; margin:10px 0; font-size:11px; }
.refs th { text-align:left; padding:6px 8px; background:#334155; color:#e2e8f0; font-weight:600; border:1px solid #475569; }
.refs td { padding:5px 8px; border:1px solid #334155; color:#cbd5e1; }
.refs td:first-child { color:#e2e8f0; font-weight:500; }
</style></head>
<body>
<h1>Model Dashboard — Spot Price Predictor</h1>
<p class="sub">Duration model: D(k) Ridge + PAVA | Hourly model: log-linear Ridge |
All parameters from <code>config/regions/finland.yaml</code></p>

<div class="mrow" id="metrics"></div>

<h2>Duration Curve: Actual vs Predicted (Consumer c/kWh)</h2>
<p class="sub">X = cheapest hours used, Y = average price.
Solid = actual, dashed = predicted.  Slider syncs all views.</p>
<div class="box">
  <div class="sc">
    <button onclick="setWin(7)" id="b7">7d</button>
    <button onclick="setWin(30)" id="b30">30d</button>
    <button onclick="setWin(90)" id="b90">90d</button>
    <button onclick="setWin(365)" id="b365">1y</button>
    <button onclick="setWin(0)" id="bAll">All</button>
    <input type="range" id="sl" min="0" max="100" value="100">
    <span class="slbl" id="slbl"></span>
  </div>
  <canvas id="durCurve" height="310"></canvas>
</div>

<h2>Forgetting Factor Sweep: Spearman &rho; vs &lambda;</h2>
<p class="sub">Left: full-history evaluation.
Right: <strong>production sweep</strong> — only last 365 days evaluated (answers: what &lambda; is best for a model retrained now?).</p>
<div class="grid-2">
  <div class="box"><canvas id="sweepChart" height="220"></canvas></div>
  <div class="box"><canvas id="prodSweep" height="220"></canvas></div>
</div>

<h2>Rolling Spearman &rho; (90-day window)</h2>
<p class="sub">Shows how rank prediction accuracy evolves over time.
Early period (limited training data + 2022 crisis) degrades overall average.</p>
<div class="box"><canvas id="rollChart" height="230"></canvas></div>

<h2>Marginal Cost Waterfall (Consumer c/kWh)</h2>
<p class="sub">Sorted hour price — cost of adding each additional cheapest hour.
Reveals price jump structure.</p>
<div class="box"><canvas id="marginal" height="240"></canvas></div>

<h2>Duration Contour: Predicted Price by Duration Level</h2>
<p class="sub">Contour lines at integer c/kWh. Bottom = cheapest 1h, top = 24h avg.
Scrolls with slider.</p>
<div class="box" style="overflow-x:auto"><canvas id="contour" height="380"></canvas></div>

<h2>Scatter: Predicted vs Actual per Duration Level</h2>
<div class="grid-4">
  <div class="box"><div style="font-size:12px;color:#e2e8f0;font-weight:600;margin-bottom:4px">D(1) Cheapest 1h</div><canvas id="sc1" height="200"></canvas></div>
  <div class="box"><div style="font-size:12px;color:#e2e8f0;font-weight:600;margin-bottom:4px">D(4) Cheapest 4h</div><canvas id="sc4" height="200"></canvas></div>
  <div class="box"><div style="font-size:12px;color:#e2e8f0;font-weight:600;margin-bottom:4px">D(8) Cheapest 8h</div><canvas id="sc8" height="200"></canvas></div>
  <div class="box"><div style="font-size:12px;color:#e2e8f0;font-weight:600;margin-bottom:4px">D(24) Daily Avg</div><canvas id="sc24" height="200"></canvas></div>
</div>

<h2 style="margin-top:40px;border-top:2px solid #60a5fa;padding-top:20px">
  Hourly Model — Feature Importance &amp; Error Analysis</h2>
<p class="sub">Log-linear Ridge regression. Impact = |coefficient| &times; std(feature) [EUR/MWh].
Color by source: <span style="color:#34d399">Base (weather)</span> |
<span style="color:#60a5fa">Cross-border</span> |
<span style="color:#fb923c">Nuclear</span></p>
<div class="mrow" id="hourly-metrics"></div>
<div class="box" style="height:500px"><canvas id="fiChart"></canvas></div>

<h2>Hourly &amp; Monthly MAE Breakdown</h2>
<p class="sub">Mean Absolute Error by hour-of-day and month across the full dataset.</p>
<div class="grid-2">
  <div class="box"><canvas id="hourlyChart" height="200"></canvas></div>
  <div class="box"><canvas id="monthlyChart" height="200"></canvas></div>
</div>

<div class="refs">
<h2>Scientific References &amp; Methodology</h2>
<p class="sub">This model combines six established methods.
Each component has a solid theoretical foundation; the novelty lies in their composition
for hourly electricity price rank prediction.</p>

<h3>1. Price Duration Curves</h3>
<p class="sub">D(k) = (1/k) &Sigma;<sub>i=1</sub><sup>k</sup> p<sub>(i)</sub> &mdash;
average consumer price for the k cheapest hours, adapted from Load Duration Curves (LDC).</p>
<ol>
<li>Stoft, S. (2002). <em>Power System Economics: Designing Markets for Electricity.</em>
IEEE/Wiley. &mdash; Foundational definition of load duration curves and their role in electricity market analysis.</li>
<li>Weron, R. (2014). &ldquo;Electricity price forecasting: A review of the state-of-the-art
with a look into the future.&rdquo; <em>International Journal of Forecasting, 30</em>(4), 1030&ndash;1044.
&mdash; Comprehensive review; &sect;3&ndash;4 cover distributional approaches to price modeling.</li>
<li>Joskow, P.L. (2007). &ldquo;Competitive Electricity Markets and Investment in New Generating
Capacity.&rdquo; In <em>The New Energy Paradigm</em>, Oxford University Press.
&mdash; Uses price duration curves to analyze generation investment signals.</li>
</ol>
<div class="lim"><ul>
<li><strong>Temporal ordering discarded</strong> &mdash; D(k) tells you the cheapest k hours' average
but not <em>when</em> those hours occur.</li>
<li><strong>Segment&rarr;full-day reconstruction not bijective</strong> &mdash; merging 4 segments'
sorted prices and re-sorting introduces compounding bias.</li>
<li><strong>Intra-segment stationarity assumed</strong> &mdash; all hours within a segment are treated
as exchangeable.</li>
</ul></div>

<h3>2. Exponentially Weighted Ridge Regression (&lambda; forgetting factor)</h3>
<p class="sub">Sample i weighted by &lambda;<sup>days_ago</sup>, implemented as
W<sup>&frac12;</sup>X &rarr; standard Ridge. Equivalent to Recursive Least Squares (RLS) with forgetting.</p>
<ol start="4">
<li>Hoerl, A.E. &amp; Kennard, R.W. (1970). &ldquo;Ridge Regression: Biased Estimation for
Nonorthogonal Problems.&rdquo; <em>Technometrics, 12</em>(1), 55&ndash;67.
&mdash; Original Ridge regression paper.</li>
<li>Ljung, L. (1999). <em>System Identification: Theory for the User</em> (2nd ed.). Prentice Hall.
&mdash; &sect;11.3: recursive least squares with exponential forgetting. Canonical reference for the &lambda; mechanism.</li>
<li>Haykin, S. (2002). <em>Adaptive Filter Theory</em> (4th ed.). Prentice Hall.
&mdash; Ch. 13: RLS with forgetting factor, convergence analysis.</li>
<li>Gaillard, P., Goude, Y. &amp; Nedellec, R. (2016). &ldquo;Additive models and robust aggregation
for GEFCom2014 probabilistic electric load and electricity price forecasting.&rdquo;
<em>Int. J. Forecasting, 32</em>(3), 1038&ndash;1050.
&mdash; Applies exponential forgetting to electricity price forecasting specifically.</li>
</ol>
<div class="lim"><ul>
<li><strong>Single &lambda; for all features</strong> &mdash; optimal decay rate may differ for structural
(seasonality) vs. volatile (wind) features.</li>
<li><strong>Half-life = 17 days</strong> &mdash; the model structurally cannot learn seasonal patterns
with period &gt; ~2 months; annual winter&ndash;summer seasonality is progressively forgotten (mitigated by month_sin/cos features).</li>
<li><strong>No regime detection</strong> &mdash; &lambda; decays smoothly regardless of market regime shifts.
A sudden shock takes ~17 days to fully adapt.</li>
<li><strong>Equispaced assumption</strong> &mdash; missing dates (DST, data gaps) violate
the &lambda;<sup>n</sup> weighting premise.</li>
</ul></div>

<h3>3. Log-Linear Target Transformation</h3>
<p class="sub">y = log(D(k) + 55). Shifted logarithm to stabilize variance and keep
the argument positive for negative spot prices.</p>
<ol start="8">
<li>Box, G.E.P. &amp; Cox, D.R. (1964). &ldquo;An Analysis of Transformations.&rdquo;
<em>J. Royal Statistical Society, Series B, 26</em>(2), 211&ndash;252.
&mdash; Formal framework for power/log transformations; log(y + c) is shifted Box-Cox with &lambda;=0.</li>
<li>Weron, R. (2006). <em>Modeling and Forecasting Electricity Loads and Prices:
A Statistical Approach.</em> Wiley. &mdash; &sect;6: log and shifted-log transforms for electricity prices.</li>
<li>Duan, N. (1983). &ldquo;Smearing Estimate: A Nonparametric Retransformation Method.&rdquo;
<em>JASA, 78</em>(383), 605&ndash;610.
&mdash; Correction for back-transformation bias E[log(Y)] &ne; log(E[Y]). <strong>Not applied in this model.</strong></li>
</ol>
<div class="lim"><ul>
<li><strong>Offset 55 is ad hoc</strong> &mdash; chosen empirically to keep log argument positive for Finnish
prices (~&minus;50 &euro;/MWh floor). The optimal offset is data-dependent and non-stationary.</li>
<li><strong>Compresses peak prices</strong> &mdash; the model sees less gradient signal for extreme prices;
errors at the expensive end are under-penalized.</li>
<li><strong>Back-transformation bias</strong> &mdash; exp(E[log Y]) &lt; E[Y] systematically.
Duan&rsquo;s smearing correction [10] is not applied, leading to slight underestimation of means.</li>
</ul></div>

<h3>4. Isotonic Regression (PAVA) Post-Processing</h3>
<p class="sub">Pool Adjacent Violators Algorithm enforces D(1) &le; D(2) &le; &hellip; &le; D(N),
projecting raw predictions onto the monotone cone.</p>
<ol start="11">
<li>Barlow, R.E., Bartholomew, D.J., Bremner, J.M. &amp; Brunk, H.D. (1972).
<em>Statistical Inference Under Order Restrictions.</em> Wiley.
&mdash; Original PAVA monograph.</li>
<li>de Leeuw, J., Hornik, K. &amp; Mair, P. (2009). &ldquo;Isotone Optimization in R:
Pool-Adjacent-Violators Algorithm (PAVA) and Active Set Methods.&rdquo;
<em>J. Statistical Software, 32</em>(5), 1&ndash;24.</li>
<li>Zadrozny, B. &amp; Elkan, C. (2002). &ldquo;Transforming Classifier Scores into Accurate
Multiclass Probability Estimates.&rdquo; <em>KDD &rsquo;02.</em>
&mdash; Isotonic regression for post-hoc calibration (analogous to monotonicity correction).</li>
</ol>
<div class="lim"><ul>
<li><strong>Per-segment only</strong> &mdash; monotonicity is enforced within each segment independently;
after merging, global ordering can be violated.</li>
<li><strong>Scale mismatch</strong> &mdash; PAVA projects on linear scale after back-transforming from log;
it does not minimize log-scale error.</li>
<li><strong>No smoothness constraint</strong> &mdash; PAVA produces piecewise-constant corrections with
possible flat plateaus where adjacent levels are pooled.</li>
</ul></div>

<h3>5. Segment-to-Full-Day Curve Reconstruction</h3>
<p class="sub">4 segment predictions &rarr; extract sorted prices via
p<sub>(k)</sub> = (k+1)&middot;D(k) &minus; k&middot;D(k&minus;1) &rarr; merge &rarr; re-sort &rarr; full-day D(k).</p>
<ol start="14">
<li>Hong, T., Pinson, P. &amp; Fan, S. (2014). &ldquo;Global Energy Forecasting Competition
2012.&rdquo; <em>Int. J. Forecasting, 30</em>(2), 357&ndash;363.
&mdash; Hierarchical forecasting with reconciliation; segment&rarr;day assembly is bottom-up hierarchical.</li>
<li>Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G. &amp; Shang, H.L. (2011).
&ldquo;Optimal Combination Forecasts for Hierarchical Time Series.&rdquo;
<em>Computational Statistics &amp; Data Analysis, 55</em>(9), 2579&ndash;2589.
&mdash; Formal framework for coherent hierarchical forecasts.</li>
</ol>
<div class="lim"><ul>
<li><strong>Error amplification</strong> &mdash; extracting p<sub>(k)</sub> from predicted D(k) amplifies
small duration-curve errors into large marginal-price errors at the extremes.</li>
<li><strong>Cross-segment correlation ignored</strong> &mdash; night and midday prices share common
fundamentals but are predicted independently; no covariance structure modeled.</li>
<li><strong>Non-coherent merge</strong> &mdash; a predicted &ldquo;cheapest night hour&rdquo; may exceed a predicted
&ldquo;most expensive midday hour&rdquo;; re-sorting hides this inconsistency.</li>
<li><strong>Unequal segment sizes</strong> (8+4+6+6) &mdash; morning (4h) has less statistical power
but contributes equally to reconstruction noise.</li>
</ul></div>

<h3>6. Spearman Rank Correlation as Evaluation Metric</h3>
<p class="sub">Nonparametric rank correlation with averaged ranks for ties,
computed as Pearson correlation of rank vectors.</p>
<ol start="16">
<li>Spearman, C. (1904). &ldquo;The proof and measurement of association between
two things.&rdquo; <em>American J. Psychology, 15</em>(1), 72&ndash;101. &mdash; Original paper.</li>
<li>Conover, W.J. (1999). <em>Practical Nonparametric Statistics</em> (3rd ed.). Wiley.
&mdash; &sect;5.4: rank correlation with ties (averaged-ranks implementation).</li>
</ol>
<div class="lim"><ul>
<li><strong>Ordinal agreement only</strong> &mdash; perfect ranks with wrong magnitudes scores &rho;=1.0.
Sufficient for load-shifting optimization but not for budgeting or hedging.</li>
<li><strong>Insensitive to calibration</strong> &mdash; two models with identical Spearman but different
MAE are indistinguishable by this metric alone.</li>
<li><strong>Not directly optimizable</strong> &mdash; the model minimizes MSE (Ridge) and evaluates Spearman
post-hoc; no guarantee MSE minimization maximizes rank accuracy.</li>
</ul></div>

<h3>Structural Limitation Summary</h3>
<table>
<tr><th>Limitation</th><th>Impact</th><th>Possible mitigation</th></tr>
<tr><td>Segment reconstruction amplifies bias</td>
    <td>D(1)&gt;D(24) Spearman artifact in full history</td>
    <td>Direct full-day D(k) model (skip segments)</td></tr>
<tr><td>Single &lambda; forgets seasonality</td>
    <td>Poor winter prediction after long summer</td>
    <td>Explicit seasonal features (month_sin/cos &mdash; partially done)</td></tr>
<tr><td>No regime detection</td>
    <td>~17 days to adapt after market shocks</td>
    <td>Regime-switching model or adaptive &lambda;</td></tr>
<tr><td>Log back-transform bias</td>
    <td>Systematic underestimation of means</td>
    <td>Duan&rsquo;s smearing estimator [10]</td></tr>
<tr><td>Cross-segment independence</td>
    <td>Ignores night&harr;midday price correlation</td>
    <td>Joint multivariate model or copula</td></tr>
<tr><td>Spearman&ndash;MSE mismatch</td>
    <td>Optimizing MSE &ne; maximizing rank accuracy</td>
    <td>Learning-to-rank methods (e.g. LambdaRank)</td></tr>
</table>
</div>

<footer>Model Dashboard — Spot Price Predictor | Duration: Ridge + PAVA | Hourly: log-linear Ridge |
  <a href="https://github.com/watti-matti/HA-spot-price-predictor">GitHub</a></footer>

<script>
const D = ''' + chart_json + ''';
const A = D.daily, N = A.length;
function toC(s){return (Math.max(0,s)/1000+0.0361+0.02325)*1.255*100;}

// ── Proper Spearman with averaged ranks (matches scipy) ──
function avgRanks(arr){
  const n=arr.length;
  const s=arr.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v);
  const r=new Array(n);
  let i=0;
  while(i<n){let j=i;
    while(j<n-1&&s[j+1].v===s[j].v)j++;
    const avg=(i+j)/2+1;
    for(let k=i;k<=j;k++)r[s[k].i]=avg;
    i=j+1;}
  return r;
}
function spRho(x,y){
  const n=x.length;if(n<3)return 0;
  const rx=avgRanks(x),ry=avgRanks(y);
  let mx=0,my=0;for(let i=0;i<n;i++){mx+=rx[i];my+=ry[i];}
  mx/=n;my/=n;
  let num=0,dx2=0,dy2=0;
  for(let i=0;i<n;i++){const a=rx[i]-mx,b=ry[i]-my;num+=a*b;dx2+=a*a;dy2+=b*b;}
  return num/Math.sqrt(dx2*dy2+1e-15);
}

// ── slider state ──
let winD=30, slP=100;
function getS(){
  if(!winD) return {s:0,e:N};
  const w=Math.min(winD,N), mx=N-w, s=Math.round(mx*slP/100);
  return {s,e:s+w};
}

// ── metrics ──
function updMet(sl){
  const n=sl.length; let h='';
  [{k:1,l:'D(1) 1h',c:'vc'},{k:4,l:'D(4) 4h',c:'vy'},
   {k:8,l:'D(8) 8h',c:'vo'},{k:24,l:'D(24)',c:'vr'}].forEach(lv=>{
    if(lv.k>(sl[0]?.pred?.length||0)) return;
    const pv=sl.map(d=>d.pred[lv.k-1]), av=sl.map(d=>d.actual[lv.k-1]);
    const rho = spRho(pv, av);
    // MAE
    let sAE=0; sl.forEach(d=>{sAE+=Math.abs(toC(d.pred[lv.k-1])-toC(d.actual[lv.k-1]));});
    const mae=(sAE/n).toFixed(3);
    h+='<div class="met"><div class="mv '+lv.c+'">'+rho.toFixed(3)+
        '</div><div class="ml">&rho; '+lv.l+'</div></div>';
    h+='<div class="met"><div class="mv '+lv.c+'">'+mae+
        '</div><div class="ml">MAE '+lv.l+' c/kWh</div></div>';
  });
  h+='<div class="met"><div class="mv vg">'+n+'</div><div class="ml">Days</div></div>';
  document.getElementById('metrics').innerHTML=h;
}

// ── Duration Curve Chart ──
let durChart=null;
function buildDur(){
  const {s,e}=getS(), sl=A.slice(s,e);
  if(!sl.length) return;
  document.getElementById('slbl').textContent=sl[0].date+' .. '+sl[sl.length-1].date;
  updMet(sl);
  const nH=sl[0].pred.length;
  const labels=Array.from({length:nH},(_,i)=>(i+1)+'h');
  const avgA=new Array(nH).fill(0), avgP=new Array(nH).fill(0);
  sl.forEach(d=>{for(let k=0;k<nH;k++){avgA[k]+=toC(d.actual[k]);avgP[k]+=toC(d.pred[k]);}});
  for(let k=0;k<nH;k++){avgA[k]/=sl.length;avgP[k]/=sl.length;}
  const ds=[
    {label:'Avg Actual',data:avgA,borderColor:'#60a5fa',borderWidth:2.5,pointRadius:3,
     pointBackgroundColor:'#60a5fa',fill:false,tension:0.2},
    {label:'Avg Predicted',data:avgP,borderColor:'#facc15',borderWidth:2.5,pointRadius:3,
     pointBackgroundColor:'#facc15',fill:false,tension:0.2,borderDash:[6,3]},
  ];
  if(sl.length<=14) sl.forEach((d,i)=>{
    const a=Math.max(0.15,0.6-i*0.04);
    ds.push({label:'_a'+i,data:d.actual.map(v=>toC(v)),borderColor:'rgba(96,165,250,'+a+')',borderWidth:0.8,pointRadius:0,fill:false,tension:0.3});
    ds.push({label:'_p'+i,data:d.pred.map(v=>toC(v)),borderColor:'rgba(250,204,21,'+a+')',borderWidth:0.8,pointRadius:0,fill:false,tension:0.3,borderDash:[4,2]});
  });
  const data={labels,datasets:ds};
  if(durChart){durChart.data=data;durChart.update();}
  else durChart=new Chart(document.getElementById('durCurve').getContext('2d'),{type:'line',data,
    options:{responsive:true,animation:{duration:200},interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:true,labels:{color:'#e2e8f0',font:{size:10},
        filter:i=>!i.text.startsWith('_')}},
        tooltip:{mode:'index',intersect:false,backgroundColor:'#1e2433',borderColor:'#374151',
          borderWidth:1,titleColor:'#e2e8f0',bodyColor:'#e2e8f0',
          filter:i=>!i.dataset.label.startsWith('_'),
          callbacks:{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(2)+' c/kWh'}}},
      scales:{x:{title:{display:true,text:'Cheapest hours used',color:'#e2e8f0'},
        grid:{color:'#1e293b'},ticks:{color:'#e2e8f0',font:{size:10}}},
        y:{title:{display:true,text:'Avg consumer price (c/kWh)',color:'#e2e8f0'},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});
  buildMarg(sl);
  buildContour(sl);
}

// ── Marginal Cost ──
let margC=null;
function buildMarg(sl){
  const nH=sl[0]?.pred?.length||24;
  const labels=Array.from({length:nH},(_,i)=>(i+1)+'h');
  const aM=new Array(nH).fill(0), pM=new Array(nH).fill(0);
  sl.forEach(d=>{for(let k=0;k<nH;k++){aM[k]+=toC(d.actual_sorted[k]);pM[k]+=toC(d.pred_sorted[k]);}});
  for(let k=0;k<nH;k++){aM[k]/=sl.length;pM[k]/=sl.length;}
  const data={labels,datasets:[
    {label:'Actual (sorted price)',data:aM,backgroundColor:'#60a5fa66',borderColor:'#60a5fa',borderWidth:1},
    {label:'Predicted (sorted price)',data:pM,backgroundColor:'#facc1566',borderColor:'#facc15',borderWidth:1}]};
  if(margC){margC.data=data;margC.update();}
  else margC=new Chart(document.getElementById('marginal').getContext('2d'),{type:'bar',data,
    options:{responsive:true,animation:{duration:200},
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}},
        tooltip:{callbacks:{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(2)+' c/kWh'}}},
      scales:{x:{title:{display:true,text:'Hour rank (cheapest to most expensive)',color:'#e2e8f0'},
        grid:{color:'#1e293b'},ticks:{color:'#e2e8f0',font:{size:9}}},
        y:{title:{display:true,text:'Sorted hour price (c/kWh)',color:'#e2e8f0'},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}}}});
}

// ── Marching Squares for contour lines ──
function marchSq(grid,nR,nC,level){
  const segs=[];
  const lerp=(v1,v2,r1,c1,r2,c2)=>{
    const t=Math.abs(v2-v1)<1e-10?0.5:(level-v1)/(v2-v1);
    return [r1+t*(r2-r1),c1+t*(c2-c1)];};
  for(let r=0;r<nR-1;r++) for(let c=0;c<nC-1;c++){
    const tl=grid[r][c],tr=grid[r][c+1],bl=grid[r+1][c],br=grid[r+1][c+1];
    let idx=0;
    if(tl>=level)idx|=8; if(tr>=level)idx|=4; if(br>=level)idx|=2; if(bl>=level)idx|=1;
    if(idx===0||idx===15) continue;
    const T=lerp(tl,tr,r,c,r,c+1), R=lerp(tr,br,r,c+1,r+1,c+1);
    const B=lerp(bl,br,r+1,c,r+1,c+1), L=lerp(tl,bl,r,c,r+1,c);
    const cs={1:[[L,B]],2:[[B,R]],3:[[L,R]],4:[[T,R]],5:[[L,T],[B,R]],
      6:[[T,B]],7:[[L,T]],8:[[T,L]],9:[[T,B]],10:[[T,R],[L,B]],
      11:[[T,R]],12:[[L,R]],13:[[B,R]],14:[[L,B]]};
    (cs[idx]||[]).forEach(s=>segs.push(s));
  }
  return segs;
}

// ── Contour Plot ──
function buildContour(slice){
  const cv=document.getElementById('contour');
  const ctx=cv.getContext('2d');
  const nDays=slice.length, nH=slice[0]?.pred?.length||24;
  const plotX=50, plotY=15;
  const maxCvW=1100;
  cv.width=Math.min(Math.max(nDays*15+100,820),maxCvW);
  cv.height=380;
  const plotW=cv.width-100, plotH=cv.height-45;
  const cellW=plotW/nDays, cellH=plotH/nH;

  // Build grid: grid[row][col], row 0=top(24h), row nH-1=bottom(1h)
  const grid=[];
  let minP=Infinity,maxP=-Infinity;
  for(let row=0;row<nH;row++){grid[row]=[];
    for(let d=0;d<nDays;d++){
      const v=toC(slice[d].pred[nH-1-row]);
      grid[row][d]=v;
      if(v<minP)minP=v; if(v>maxP)maxP=v;
    }}

  // Color function
  function pCol(c){
    const t=Math.min(1,Math.max(0,(c-minP)/(maxP-minP+0.01)));
    if(t<0.25){const s=t/0.25;
      return 'rgb('+Math.round(15+s*5)+','+Math.round(50+s*120)+','+Math.round(90+s*80)+')';}
    else if(t<0.5){const s=(t-0.25)/0.25;
      return 'rgb('+Math.round(20+s*50)+','+Math.round(170+s*40)+','+Math.round(170-s*40)+')';}
    else if(t<0.75){const s=(t-0.5)/0.25;
      return 'rgb('+Math.round(70+s*180)+','+Math.round(210-s*20)+','+Math.round(130-s*100)+')';}
    else{const s=(t-0.75)/0.25;
      return 'rgb('+Math.round(250)+','+Math.round(190-s*150)+','+Math.round(30-s*20)+')';}
  }

  // Clear
  ctx.fillStyle='#0f172a'; ctx.fillRect(0,0,cv.width,cv.height);

  // Draw filled cells
  const cw=Math.max(cellW,1.2);
  for(let row=0;row<nH;row++) for(let d=0;d<nDays;d++){
    ctx.fillStyle=pCol(grid[row][d]);
    ctx.fillRect(plotX+d*cellW,plotY+row*cellH,cw,cellH+0.5);
  }

  // Draw contour lines at integer levels (only for windows <= 90 days)
  const minLvl=Math.ceil(minP), maxLvl=Math.floor(maxP);
  const placed=[];
  const drawLines = nDays <= 90;
  for(let level=minLvl;level<=maxLvl;level++){
    if(!drawLines) continue;
    const segs=marchSq(grid,nH,nDays,level);
    if(!segs.length) continue;
    ctx.strokeStyle='rgba(255,255,255,0.55)';
    ctx.lineWidth=1.2;
    ctx.beginPath();
    segs.forEach(([p1,p2])=>{
      ctx.moveTo(plotX+p1[1]*cellW,plotY+p1[0]*cellH);
      ctx.lineTo(plotX+p2[1]*cellW,plotY+p2[0]*cellH);});
    ctx.stroke();

    // Place label — find segment near horizontal center, avoiding overlap
    const ctrC=nDays*0.45;
    let best=null,bestDist=Infinity;
    segs.forEach(s=>{
      const mc=(s[0][1]+s[1][1])/2, d=Math.abs(mc-ctrC);
      if(d<bestDist){bestDist=d;best=s;}});
    if(best){
      const lr=(best[0][0]+best[1][0])/2, lc=(best[0][1]+best[1][1])/2;
      const lx=plotX+lc*cellW, ly=plotY+lr*cellH;
      const tooClose=placed.some(([px,py])=>Math.abs(px-lx)<35&&Math.abs(py-ly)<16);
      if(!tooClose){
        ctx.font='bold 10px sans-serif';
        const txt=level+'';
        const tw=ctx.measureText(txt).width+6;
        ctx.fillStyle='rgba(15,23,42,0.8)';
        ctx.fillRect(lx-tw/2,ly-8,tw,16);
        ctx.fillStyle='#fff';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(txt,lx,ly);
        placed.push([lx,ly]);
      }
    }
  }

  // Y-axis
  ctx.fillStyle='#e2e8f0'; ctx.font='10px sans-serif';
  ctx.textAlign='right'; ctx.textBaseline='middle';
  for(let k=0;k<nH;k+=Math.max(1,Math.floor(nH/8))){
    const row=nH-1-k;
    ctx.fillText((k+1)+'h',plotX-5,plotY+row*cellH+cellH/2);}

  // X-axis — adaptive label spacing
  ctx.textAlign='center'; ctx.textBaseline='top';
  const minLblPx=55;
  const lblStep=Math.max(1,Math.ceil(minLblPx/cellW));
  for(let d=0;d<nDays;d+=lblStep){
    const dt=slice[d].date;
    const lbl=nDays<=14?dt.substring(5):nDays<=400?dt.substring(5,10):dt.substring(0,7);
    ctx.fillText(lbl,plotX+d*cellW+cellW/2,cv.height-15);}

  // Y-axis title
  ctx.save(); ctx.translate(12,plotH/2+plotY); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('Duration (cheapest hours)',0,0); ctx.restore();

  // Color legend bar on right edge
  const legX=cv.width-40, legY=plotY, legH=plotH;
  for(let i=0;i<legH;i++){
    const t=1-i/legH;
    const c=minP+t*(maxP-minP);
    ctx.fillStyle=pCol(c);
    ctx.fillRect(legX,legY+i,15,1.5);}
  ctx.fillStyle='#e2e8f0'; ctx.font='9px sans-serif';
  ctx.textAlign='left'; ctx.textBaseline='top';
  ctx.fillText(maxP.toFixed(1),legX,legY-12);
  ctx.textBaseline='bottom';
  ctx.fillText(minP.toFixed(1)+' c/kWh',legX-5,legY+legH+14);
}

// ── Scatter plots ──
function scPlot(id,kI,color){
  const pts=A.map(d=>({x:toC(d.pred[kI]),y:toC(d.actual[kI])}));
  const vs=pts.flatMap(p=>[p.x,p.y]);
  const mn=Math.min(...vs),mx=Math.max(...vs);
  const n=pts.length;
  let sAE=0,sE2=0,yM=0,sB=0;
  pts.forEach(p=>{sAE+=Math.abs(p.x-p.y);yM+=p.y;sB+=(p.x-p.y);});
  yM/=n; let sT=0;
  pts.forEach(p=>{sE2+=(p.x-p.y)**2;sT+=(p.y-yM)**2;});
  const r2=1-sE2/(sT||1), mae=sAE/n, bias=sB/n;
  const rho=spRho(pts.map(p=>p.x),pts.map(p=>p.y));
  new Chart(document.getElementById(id).getContext('2d'),{type:'scatter',
    data:{datasets:[
      {data:pts,backgroundColor:color+'66',borderColor:color,pointRadius:2.5,borderWidth:0.5},
      {data:[{x:mn,y:mn},{x:mx,y:mx}],borderColor:'#94a3b8',borderDash:[4,4],pointRadius:0,showLine:true,fill:false}]},
    options:{responsive:true,animation:false,aspectRatio:1,
      plugins:{legend:{display:false},
        title:{display:true,
          text:'\\u03C1='+rho.toFixed(3)+'  MAE='+mae.toFixed(2)+'  Bias='+(bias>=0?'+':'')+bias.toFixed(2),
          color:'#e2e8f0',font:{size:10,weight:'normal'}}},
      scales:{x:{title:{display:true,text:'Predicted (c/kWh)',color:'#e2e8f0',font:{size:9}},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0',font:{size:9}}},
        y:{title:{display:true,text:'Actual (c/kWh)',color:'#e2e8f0',font:{size:9}},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0',font:{size:9}}}}}});
}

// ── Lambda sweep charts ──
(function(){
  const sw=D.sweep;
  const labels=sw.map(s=>{
    if(s.lambda>=1) return '\\u03BB=1.000';
    return '\\u03BB='+s.lambda.toFixed(3)+'\\n('+s.halflife+')';});

  // Spearman chart
  new Chart(document.getElementById('sweepChart').getContext('2d'),{type:'line',
    data:{labels,datasets:[
      {label:'D(1) Cheapest',data:sw.map(s=>s.d1_rho),borderColor:'#22d3ee',borderWidth:2,pointRadius:4,fill:false},
      {label:'D(4) 4h',data:sw.map(s=>s.d4_rho),borderColor:'#facc15',borderWidth:2,pointRadius:4,fill:false},
      {label:'D(8) 8h',data:sw.map(s=>s.d8_rho),borderColor:'#f97316',borderWidth:2,pointRadius:4,fill:false},
      {label:'D(24) Daily',data:sw.map(s=>s.d24_rho),borderColor:'#ef4444',borderWidth:2,pointRadius:4,fill:false},
      {label:'Composite',data:sw.map(s=>s.composite),borderColor:'#a78bfa',borderWidth:3,pointRadius:5,
       fill:false,borderDash:[6,3]}]},
    options:{responsive:true,animation:false,
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}},
        title:{display:true,color:'#4ade80',font:{size:12},
          text:'Spearman \\u03C1 — best \\u03BB='+D.best_lambda.toFixed(3)+' (half-life '+D.best_halflife+'d)'}},
      scales:{x:{ticks:{color:'#e2e8f0',font:{size:9},maxRotation:0},grid:{color:'#1e293b'}},
        y:{title:{display:true,text:'Spearman \\u03C1',color:'#e2e8f0'},
          ticks:{color:'#e2e8f0'},grid:{color:'#334155'}}}}});

  // Production sweep chart (last 365 days)
  const ps=D.prod_sweep;
  const plabels=ps.map(s=>{
    if(s.lambda>=1) return '\\u03BB=1.000';
    return '\\u03BB='+s.lambda.toFixed(3)+'\\n('+s.halflife+')';});
  new Chart(document.getElementById('prodSweep').getContext('2d'),{type:'line',
    data:{labels:plabels,datasets:[
      {label:'D(4) 4h',data:ps.map(s=>s.d4_rho),borderColor:'#facc15',borderWidth:2,pointRadius:4,fill:false},
      {label:'D(8) 8h',data:ps.map(s=>s.d8_rho),borderColor:'#f97316',borderWidth:2,pointRadius:4,fill:false},
      {label:'D(24) Daily',data:ps.map(s=>s.d24_rho),borderColor:'#ef4444',borderWidth:2,pointRadius:4,fill:false},
      {label:'Composite',data:ps.map(s=>s.composite),borderColor:'#a78bfa',borderWidth:3,pointRadius:5,
       fill:false,borderDash:[6,3]}]},
    options:{responsive:true,animation:false,
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10}}},
        title:{display:true,color:'#4ade80',font:{size:11},
          text:'Production (last 365d) — best \\u03BB='+D.best_lambda.toFixed(3)}},
      scales:{x:{ticks:{color:'#e2e8f0',font:{size:9},maxRotation:0},grid:{color:'#1e293b'}},
        y:{title:{display:true,text:'Spearman \\u03C1',color:'#e2e8f0'},
          ticks:{color:'#e2e8f0'},grid:{color:'#334155'}}}}});
})();

// ── Rolling Spearman chart (with date x-axis) ──
(function(){
  const rl=D.rolling;
  if(!rl||!rl.length) return;
  // Full date labels — let Chart.js autoSkip handle spacing
  const labels=rl.map(r=>r.date);
  new Chart(document.getElementById('rollChart').getContext('2d'),{type:'line',
    data:{labels,datasets:[
      {label:'D(1) Cheapest',data:rl.map(r=>r.d1),borderColor:'#22d3ee',borderWidth:1.5,pointRadius:0,fill:false,tension:0.3},
      {label:'D(4) 4h',data:rl.map(r=>r.d4),borderColor:'#facc15',borderWidth:2,pointRadius:0,fill:false,tension:0.3},
      {label:'D(8) 8h',data:rl.map(r=>r.d8),borderColor:'#f97316',borderWidth:1.5,pointRadius:0,fill:false,tension:0.3},
      {label:'D(24) Daily Avg',data:rl.map(r=>r.d24),borderColor:'#ef4444',borderWidth:1.5,pointRadius:0,fill:false,tension:0.3},
      // zero line
      {label:'_zero',data:rl.map(()=>0),borderColor:'#475569',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false},
    ]},
    options:{responsive:true,animation:false,
      plugins:{legend:{labels:{color:'#e2e8f0',font:{size:10},filter:i=>!i.text.startsWith('_')}}},
      scales:{x:{ticks:{color:'#e2e8f0',font:{size:9},maxRotation:45,autoSkip:true,maxTicksLimit:18,
        callback:function(val,idx){const d=labels[idx];if(!d)return'';
          // Show YYYY-MM for first point of each month
          const parts=d.split('-');return parts[0]+'-'+parts[1];}},
        grid:{color:'#1e293b'}},
        y:{title:{display:true,text:'Spearman \\u03C1 (90-day window)',color:'#e2e8f0'},
          min:-0.5,max:1.0,ticks:{color:'#e2e8f0'},grid:{color:'#334155'}}}}});
})();

// ── Controls ──
function setWin(d){
  winD=d;
  document.querySelectorAll('.sc button').forEach(b=>b.classList.remove('active'));
  const id=d===0?'bAll':d===365?'b365':'b'+d;
  document.getElementById(id).classList.add('active');
  buildDur();
}
document.getElementById('sl').addEventListener('input',function(){slP=parseInt(this.value);buildDur();});

// ── Hourly model metrics cards ──
(function(){
  const hm=D.hourly_model;
  if(!hm) return;
  let h='<div class="met"><div class="mv vg">'+hm.model_version+'</div><div class="ml">Model version</div></div>';
  h+='<div class="met"><div class="mv vc">'+hm.n_features+'</div><div class="ml">Features</div></div>';
  h+='<div class="met"><div class="mv vy">'+hm.mae.toFixed(2)+'</div><div class="ml">MAE EUR/MWh</div></div>';
  h+='<div class="met"><div class="mv vo">'+hm.rmse.toFixed(2)+'</div><div class="ml">RMSE EUR/MWh</div></div>';
  h+='<div class="met"><div class="mv vp">'+hm.r2.toFixed(4)+'</div><div class="ml">R-squared</div></div>';
  document.getElementById('hourly-metrics').innerHTML=h;
})();

// ── Feature importance horizontal bar ──
(function(){
  const hm=D.hourly_model;
  if(!hm||!hm.feature_importance) return;
  const fi=hm.feature_importance;
  const groupColors={base:'#34d399','cross-border':'#60a5fa',nuclear:'#fb923c'};
  new Chart(document.getElementById('fiChart').getContext('2d'),{
    type:'bar',
    data:{labels:fi.map(f=>f.name),
      datasets:[{label:'Impact',data:fi.map(f=>f.impact),
        backgroundColor:fi.map(f=>groupColors[f.group]||'#94a3b8'),borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',animation:false,
      scales:{
        x:{title:{display:true,text:'Impact: |coefficient| \\u00D7 std(feature) [EUR/MWh]',color:'#e2e8f0'},
          grid:{color:'#334155'},ticks:{color:'#e2e8f0'}},
        y:{ticks:{color:'#e2e8f0',font:{size:11,family:'monospace'}},grid:{color:'#1e293b'}}},
      plugins:{
        legend:{display:true,labels:{color:'#e2e8f0',
          generateLabels:function(){return[
            {text:'Base (weather)',fillStyle:'#34d399',strokeStyle:'#34d399',fontColor:'#e2e8f0'},
            {text:'Cross-border',fillStyle:'#60a5fa',strokeStyle:'#60a5fa',fontColor:'#e2e8f0'},
            {text:'Nuclear',fillStyle:'#fb923c',strokeStyle:'#fb923c',fontColor:'#e2e8f0'}];}}},
        tooltip:{callbacks:{afterLabel:function(ctx){
          const f=fi[ctx.dataIndex];
          return 'Coef: '+(f.coef>0?'+':'')+f.coef.toFixed(4)+'  Std: '+f.std.toFixed(3);}}}}}});
})();

// ── Hourly MAE bar chart ──
(function(){
  const hm=D.hourly_model;
  if(!hm||!hm.hourly_mae) return;
  const labels=Array.from({length:24},(_,i)=>String(i).padStart(2,'0')+':00');
  const vals=hm.hourly_mae;
  const mx=Math.max(...vals);
  const cols=vals.map(v=>{const t=v/(mx||1);
    return 'rgba('+Math.round(96-t*40)+','+Math.round(165-t*60)+','+Math.round(250-t*30)+',0.8)';});
  new Chart(document.getElementById('hourlyChart').getContext('2d'),{type:'bar',
    data:{labels,datasets:[{label:'MAE (EUR/MWh)',data:vals,backgroundColor:cols,borderWidth:0}]},
    options:{responsive:true,animation:false,
      scales:{x:{title:{display:true,text:'Hour (Finnish local)',color:'#e2e8f0'},
        grid:{color:'#1e293b'},ticks:{color:'#e2e8f0',font:{size:9}}},
        y:{title:{display:true,text:'MAE (EUR/MWh)',color:'#e2e8f0'},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}},
      plugins:{legend:{display:false}}}});
})();

// ── Monthly MAE bar chart ──
(function(){
  const hm=D.hourly_model;
  if(!hm||!hm.monthly_mae) return;
  const labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const vals=hm.monthly_mae;
  const mx=Math.max(...vals);
  const cols=vals.map(v=>{const t=v/(mx||1);
    return 'rgba('+Math.round(244-t*40)+','+Math.round(114-t*40)+','+Math.round(182-t*30)+',0.8)';});
  new Chart(document.getElementById('monthlyChart').getContext('2d'),{type:'bar',
    data:{labels,datasets:[{label:'MAE (EUR/MWh)',data:vals,backgroundColor:cols,borderWidth:0}]},
    options:{responsive:true,animation:false,
      scales:{x:{title:{display:true,text:'Month',color:'#e2e8f0'},
        grid:{color:'#1e293b'},ticks:{color:'#e2e8f0',font:{size:9}}},
        y:{title:{display:true,text:'MAE (EUR/MWh)',color:'#e2e8f0'},
        grid:{color:'#334155'},ticks:{color:'#e2e8f0'}}},
      plugins:{legend:{display:false}}}});
})();

// Initialize
setWin(30);
scPlot('sc1',0,'#22d3ee');
scPlot('sc4',3,'#facc15');
scPlot('sc8',7,'#f97316');
scPlot('sc24',23,'#ef4444');
</script></body></html>'''

with open("output/model_dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print("\nDashboard saved to: output/model_dashboard.html")
