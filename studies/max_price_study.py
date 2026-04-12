"""Study max achievable price vs envelope resolution."""
import pandas as pd, numpy as np, yaml, sys, math
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize as sp_min
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
age = np.arange(split-1, -1, -1, dtype=np.float64)
weights = np.exp(-decay * age)
weights *= split / weights.sum()
y_log_tr = np.log(fi[:split] + LOG_OFFSET)

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

print("=" * 80)
print("MAX ACHIEVABLE PRICE vs ENVELOPE RESOLUTION")
print("=" * 80)
print()

print("%-6s %8s %8s %8s %8s %8s %8s" %
      ("Block", "MAE_raw", "Max_raw", "MAE_str", "Max_str", "c/kWh", "Bias20"))
print("-" * 70)

for block_h in [1, 2, 3, 4, 6, 8, 12, 24]:
    if block_h == 1:
        se3_e = se3.values / 100
        se1_e = se1.values / 100
        ee_e = ee.values / 100
    else:
        se3_e = se3.rolling(block_h, center=True, min_periods=max(1, block_h//2)).mean().ffill().bfill().values / 100
        se1_e = se1.rolling(block_h, center=True, min_periods=max(1, block_h//2)).mean().ffill().bfill().values / 100
        ee_e = ee.rolling(block_h, center=True, min_periods=max(1, block_h//2)).mean().ffill().bfill().values / 100

    extra = np.column_stack([se3_e, se1_e, ee_e])
    Xa = np.hstack([X_base, extra])
    c, ic = train_ridge(Xa[:split], y_log_tr, weights)
    raw = np.maximum(0, np.exp(Xa[split:] @ c + ic) - LOG_OFFSET)
    mae_r = mean_absolute_error(fi[split:], raw)

    # Power stretch
    raw_tr = np.maximum(0, np.exp(Xa[:split] @ c + ic) - LOG_OFFSET)
    def obj(params):
        s, p = params
        if s <= 0 or p < 0.5 or p > 3: return 1e6
        return np.average(np.abs(np.maximum(fi[:split], 0) - s * np.power(raw_tr + 1e-10, p)), weights=weights)
    opt = sp_min(obj, [1.0, 1.0], method="Nelder-Mead", options={"maxiter": 500})
    ps, pp = opt.x
    stretched = ps * np.power(raw + 1e-10, pp)
    mae_s = mean_absolute_error(fi[split:], stretched)
    mask20 = (fi[split:] >= 20) & (fi[split:] < 50)
    b20 = (fi[split:][mask20] - stretched[mask20]).mean() if mask20.sum() > 0 else 0
    cmax = (stretched.max() / 1000 + 0.0361 + 0.02325) * 1.255 * 100

    print("%-6s %8.3f %8.1f %8.3f %8.1f %7.1f %+7.1f  s=%.3f p=%.3f" %
          ("%dh" % block_h, mae_r, raw.max(), mae_s, stretched.max(), cmax, b20, ps, pp))

# Detailed analysis for best configs
print()
print("=" * 80)
print("DETAILED SIMULATION: 3h ENVELOPE + POWER STRETCH")
print("=" * 80)
print()

se3_3h = se3.rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
se1_3h = se1.rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
ee_3h = ee.rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
extra = np.column_stack([se3_3h, se1_3h, ee_3h])
Xa = np.hstack([X_base, extra])
c_best, ic_best = train_ridge(Xa[:split], y_log_tr, weights)
raw_tr = np.maximum(0, np.exp(Xa[:split] @ c_best + ic_best) - LOG_OFFSET)
raw_te = np.maximum(0, np.exp(Xa[split:] @ c_best + ic_best) - LOG_OFFSET)

def obj(params):
    s, p = params
    if s <= 0 or p < 0.5 or p > 3: return 1e6
    return np.average(np.abs(np.maximum(fi[:split], 0) - s * np.power(raw_tr + 1e-10, p)), weights=weights)
opt = sp_min(obj, [1.0, 1.0], method="Nelder-Mead", options={"maxiter": 1000})
ps, pp = opt.x
print("Power stretch: scale=%.4f power=%.4f" % (ps, pp))

stretched_te = ps * np.power(raw_te + 1e-10, pp)
mae = mean_absolute_error(fi[split:], stretched_te)
r2 = r2_score(fi[split:], stretched_te)
print("MAE=%.3f R2=%.4f" % (mae, r2))
print("Max: %.1f EUR/MWh = %.1f c/kWh consumer" %
      (stretched_te.max(), (stretched_te.max()/1000 + 0.0361 + 0.02325) * 1.255 * 100))
print()

# Price range
print("Price range:")
for tag, lo, hi in [("<5", -999, 5), ("5-10", 5, 10), ("10-20", 10, 20),
                     ("20-30", 20, 30), ("30-50", 30, 50), (">50", 50, 999)]:
    mask = (fi[split:] >= lo) & (fi[split:] < hi)
    if mask.sum() == 0: continue
    m = np.abs(fi[split:][mask] - stretched_te[mask]).mean()
    b = (fi[split:][mask] - stretched_te[mask]).mean()
    print("  %-6s n=%4d MAE=%5.1f Bias=%+5.1f" % (tag, mask.sum(), m, b))

# Simulation
print()
print("Prediction at various conditions:")
for label, wind, temp, s3, s1, ees, nuc_def in [
    ("Normal spring", 6, 8, 30, 20, 80, 0.0),
    ("Low wind spring", 2, 5, 70, 40, 130, 0.0),
    ("Low wind, SE3 high", 2, 5, 120, 70, 180, 0.21),
    ("Crisis: SE3=200", 1, -5, 200, 110, 300, 0.21),
    ("Extreme: SE3=400", 0.5, -15, 400, 200, 600, 0.40),
    ("Max: SE3=600 winter", 0.5, -20, 600, 300, 800, 0.50),
]:
    h = max(0, 17 - temp)
    feats = {n: 0 for n in non_ar}
    feats.update({
        "wind_speed_weighted": wind,
        "solar_irradiance_weighted": max(0, 200 - h * 10),
        "hour_sin": math.sin(2*math.pi*9/24),
        "hour_cos": math.cos(2*math.pi*9/24),
        "month_sin": math.sin(2*math.pi*(1 if temp < 0 else 4)/12),
        "month_cos": math.cos(2*math.pi*(1 if temp < 0 else 4)/12),
        "is_holiday": 0,
        "hdd_sq": h**2,
        "wind_log_scarcity": math.log1p(max(0, 8-wind)),
        "wind_calm_x_peak_am": max(0, 6-wind) * 0.9,
        "wind_calm_x_peak_pm": 0,
        "export_potential_se3": 0.01,
        "nuclear_x_scarcity": nuc_def * max(0, 5-wind) * h * 0.9,
        "nuclear_deficit": nuc_def,
    })
    x_base = np.array([feats.get(n, 0) for n in non_ar])
    x_env = np.array([s3/100, s1/100, ees/100])
    x = np.concatenate([x_base, x_env])
    raw = max(0, math.exp(min(float(x @ c_best + ic_best), 20)) - LOG_OFFSET)
    st = ps * (raw + 1e-10) ** pp
    cons = (st / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    print("  %-25s raw=%5.1f stretched=%6.1f EUR/MWh = %5.1f c/kWh" %
          (label, raw, st, cons))
