"""Multi-horizon architecture: lower resolution for longer forecasts."""
import pandas as pd, numpy as np, yaml, sys, math
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize as sp_min
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

LOG_OFFSET = 55
half_life = 120
alpha = 1.0
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
print("MODELS AT DIFFERENT TIME RESOLUTIONS")
print("=" * 80)
print()
print("%-6s %8s %8s %6s %6s %6s %8s %8s %8s" %
      ("Res", "N_train", "N_test", "MAE", "R2", "Max", "SE3_c", "SE1_c", "EE_c"))
print("-" * 80)

for res_h in [1, 3, 6, 12, 24]:
    fi_r = pd.Series(fi, index=df.index).resample(str(res_h) + "h").mean()
    se3_r = pd.Series(se3, index=df.index).resample(str(res_h) + "h").mean() / 100
    se1_r = pd.Series(se1, index=df.index).resample(str(res_h) + "h").mean() / 100
    ee_r = pd.Series(ee, index=df.index).resample(str(res_h) + "h").mean() / 100

    Xb_r = pd.DataFrame(X_base, index=df.index, columns=non_ar).resample(
        str(res_h) + "h").mean()

    combined = pd.concat([Xb_r, se3_r.rename("se3"), se1_r.rename("se1"),
                           ee_r.rename("ee"), fi_r.rename("fi")], axis=1).dropna()

    Xa_r = combined[non_ar + ["se3", "se1", "ee"]].values
    fi_rv = combined["fi"].values
    split_r = int(len(fi_rv) * 0.85)

    age_r = np.arange(split_r - 1, -1, -1, dtype=np.float64)
    w_r = np.exp(-decay * age_r * res_h)
    w_r *= split_r / w_r.sum()

    y_log_r = np.log(fi_rv[:split_r] + LOG_OFFSET)
    cr, icr = train_ridge(Xa_r[:split_r], y_log_r, w_r)
    preds_r = np.maximum(0, np.exp(Xa_r[split_r:] @ cr + icr) - LOG_OFFSET)
    mae_r = mean_absolute_error(fi_rv[split_r:], preds_r)
    r2_r = r2_score(fi_rv[split_r:], preds_r)

    se3_c = cr[len(non_ar)]
    se1_c = cr[len(non_ar) + 1]
    ee_c = cr[len(non_ar) + 2]

    print("%-6s %8d %8d %6.3f %6.4f %6.1f %+8.4f %+8.4f %+8.4f" %
          (str(res_h) + "h", split_r, len(fi_rv) - split_r,
           mae_r, r2_r, preds_r.max(), se3_c, se1_c, ee_c))

# ================================================================
print()
print("=" * 80)
print("DAILY MODEL + POWER STRETCH: MAX PRICE RANGE")
print("=" * 80)
print()

fi_d = pd.Series(fi, index=df.index).resample("D").mean()
se3_d = pd.Series(se3, index=df.index).resample("D").mean() / 100
se1_d = pd.Series(se1, index=df.index).resample("D").mean() / 100
ee_d = pd.Series(ee, index=df.index).resample("D").mean() / 100
Xb_d = pd.DataFrame(X_base, index=df.index, columns=non_ar).resample("D").mean()

combined_d = pd.concat([Xb_d, se3_d.rename("se3"), se1_d.rename("se1"),
                         ee_d.rename("ee"), fi_d.rename("fi")], axis=1).dropna()
Xa_d = combined_d[non_ar + ["se3", "se1", "ee"]].values
fi_dv = combined_d["fi"].values
split_d = int(len(fi_dv) * 0.85)

age_d = np.arange(split_d - 1, -1, -1, dtype=np.float64)
w_d = np.exp(-decay * age_d * 24)
w_d *= split_d / w_d.sum()

y_log_d = np.log(fi_dv[:split_d] + LOG_OFFSET)
cd, icd = train_ridge(Xa_d[:split_d], y_log_d, w_d)

raw_d_tr = np.maximum(0, np.exp(Xa_d[:split_d] @ cd + icd) - LOG_OFFSET)
raw_d_te = np.maximum(0, np.exp(Xa_d[split_d:] @ cd + icd) - LOG_OFFSET)

# Power stretch
def obj_pow(params):
    s, p = params
    if s <= 0 or p < 0.5 or p > 3:
        return 1e6
    pred = s * np.power(raw_d_tr + 1e-10, p)
    return np.average(np.abs(np.maximum(fi_dv[:split_d], 0) - pred), weights=w_d)

opt = sp_min(obj_pow, [1.0, 1.0], method="Nelder-Mead", options={"maxiter": 1000})
ps, pp = opt.x
stretched_d = ps * np.power(raw_d_te + 1e-10, pp)

mae_raw = mean_absolute_error(fi_dv[split_d:], raw_d_te)
mae_str = mean_absolute_error(fi_dv[split_d:], stretched_d)
r2_str = r2_score(fi_dv[split_d:], stretched_d)

print("Daily model: stretch scale=%.4f power=%.4f" % (ps, pp))
print("MAE raw=%.3f MAE stretched=%.3f R2=%.4f" % (mae_raw, mae_str, r2_str))
print("Max raw=%.1f Max stretched=%.1f EUR/MWh" % (raw_d_te.max(), stretched_d.max()))
print("Max consumer=%.1f c/kWh" % ((stretched_d.max() / 1000 + 0.0361 + 0.02325) * 1.255 * 100))
print()

print("Neighbor coefficients (daily resolution, log-space):")
print("  SE3: %+.4f" % cd[len(non_ar)])
print("  SE1: %+.4f" % cd[len(non_ar) + 1])
print("  EE:  %+.4f" % cd[len(non_ar) + 2])
print()

# Simulation
print("Daily predictions at different neighbor price levels:")
x_avg = Xa_d[:split_d].mean(axis=0)
for se3_val in [20, 40, 60, 80, 100, 150, 200, 300, 400, 600]:
    x = x_avg.copy()
    x[len(non_ar)] = se3_val / 100
    x[len(non_ar) + 1] = se3_val * 0.5 / 100
    x[len(non_ar) + 2] = se3_val * 1.5 / 100
    raw = max(0, math.exp(min(float(x @ cd + icd), 20)) - LOG_OFFSET)
    st = ps * (raw + 1e-10) ** pp
    cons = (st / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    ratio = st / se3_val if se3_val > 0 else 0
    print("  SE3=%3d: raw=%5.1f stretched=%6.1f EUR/MWh = %5.1f c/kWh  (FI/SE3=%.3f)" %
          (se3_val, raw, st, cons, ratio))

# ================================================================
print()
print("=" * 80)
print("MULTI-HORIZON ARCHITECTURE CONCEPT")
print("=" * 80)
print()
print("Horizon 0-24h:  1h resolution, weather model + hourly neighbor data")
print("Horizon 24-72h: 6h resolution, weather forecast + 6h envelope")
print("Horizon 72-170h: daily resolution, weather trend + daily envelope")
print()
print("Each horizon uses the appropriate time resolution.")
print("Longer horizons have LARGER neighbor coupling because")
print("daily-averaged data removes phase misalignment artifacts.")
print()
print("Resolution   SE3 coupling coefficient (log-space)")
c_by_res = {}
for res_h in [1, 3, 6, 12, 24]:
    fi_r = pd.Series(fi, index=df.index).resample(str(res_h) + "h").mean()
    se3_r = pd.Series(se3, index=df.index).resample(str(res_h) + "h").mean() / 100
    se1_r = pd.Series(se1, index=df.index).resample(str(res_h) + "h").mean() / 100
    ee_r = pd.Series(ee, index=df.index).resample(str(res_h) + "h").mean() / 100
    Xb_r = pd.DataFrame(X_base, index=df.index, columns=non_ar).resample(
        str(res_h) + "h").mean()
    combined = pd.concat([Xb_r, se3_r.rename("se3"), se1_r.rename("se1"),
                           ee_r.rename("ee"), fi_r.rename("fi")], axis=1).dropna()
    Xa_r = combined[non_ar + ["se3", "se1", "ee"]].values
    fi_rv = combined["fi"].values
    split_r = int(len(fi_rv) * 0.85)
    age_r = np.arange(split_r - 1, -1, -1, dtype=np.float64)
    w_r = np.exp(-decay * age_r * res_h)
    w_r *= split_r / w_r.sum()
    cr, _ = train_ridge(Xa_r[:split_r], np.log(fi_rv[:split_r] + LOG_OFFSET), w_r)
    c_by_res[res_h] = cr[len(non_ar)]
    total_nb = cr[len(non_ar)] + cr[len(non_ar)+1] + cr[len(non_ar)+2]
    print("  %3dh:  SE3=%+.4f  SE1=%+.4f  EE=%+.4f  total=%+.4f" %
          (res_h, cr[len(non_ar)], cr[len(non_ar)+1], cr[len(non_ar)+2], total_nb))
