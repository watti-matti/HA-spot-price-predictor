"""Study optimal resolution for envelope coupling."""
import pandas as pd
import numpy as np
import yaml
import sys
from zoneinfo import ZoneInfo
from sklearn.metrics import mean_absolute_error, r2_score

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

local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
hours = local.hour.values

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
y_log_tr = np.log(fi[:split] + LOG_OFFSET)


def train_eval(Xtr, Xte, label):
    ws = weights.sum()
    fm = (weights[:, None] * Xtr).sum(0) / ws
    fv = (weights[:, None] * (Xtr ** 2)).sum(0) / ws - fm ** 2
    fs = np.sqrt(np.maximum(0, fv))
    fs[fs < 1e-10] = 1.0
    ym = (weights * y_log_tr).sum() / ws
    Xs = (Xtr - fm) / fs
    ys = y_log_tr - ym
    Xw = Xs * weights[:, None]
    cs = np.linalg.solve(Xw.T @ Xs + alpha * np.eye(Xtr.shape[1]), Xw.T @ ys)
    coefs = cs / fs
    ic = ym - (fm / fs) @ cs
    preds = np.maximum(0, np.exp(Xte @ coefs + ic) - LOG_OFFSET)
    mae = mean_absolute_error(fi[split:], preds)
    r2 = r2_score(fi[split:], preds)
    mask20 = (fi[split:] >= 20) & (fi[split:] < 50)
    b20 = (fi[split:][mask20] - preds[mask20]).mean() if mask20.sum() > 0 else 0

    new_coefs = ""
    for k in range(len(non_ar), len(coefs)):
        new_coefs += " %+.4f" % coefs[k]

    print("  %-45s MAE=%.3f R2=%.4f max=%5.1f b20=%+.1f%s" %
          (label, mae, r2, preds.max(), b20, new_coefs[:50]))
    return preds


# ================================================================
print("=" * 80)
print("PART 1: COUPLING COEFFICIENT vs ENVELOPE RESOLUTION")
print("=" * 80)
print()
print("%-20s %8s %8s %8s %8s" % ("Resolution", "Corr FI", "Coef", "R2", "% of 0.10"))
print("-" * 55)

for label, block_h in [("1h (raw)", 1), ("2h", 2), ("3h", 3), ("4h", 4),
                         ("6h", 6), ("8h", 8), ("12h", 12), ("24h", 24), ("48h", 48)]:
    if block_h == 1:
        env = se3.values
    else:
        env = se3.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values

    valid = np.isfinite(env) & np.isfinite(fi)
    corr = np.corrcoef(env[valid], fi[valid])[0, 1]
    A = np.column_stack([np.ones(valid.sum()), env[valid]])
    coefs = np.linalg.lstsq(A, fi[valid], rcond=None)[0]
    pred = A @ coefs
    r2 = 1 - np.sum((fi[valid] - pred) ** 2) / np.sum((fi[valid] - fi[valid].mean()) ** 2)
    pct = coefs[1] / 0.10 * 100
    print("%-20s %8.4f %8.4f %8.4f %7.0f%%" % (label, corr, coefs[1], r2, pct))

# ================================================================
print()
print("=" * 80)
print("PART 2: REGIME-BASED BLOCKS (natural price regimes)")
print("=" * 80)
print()

# Test blocks aligned to market regimes
regime_defs = {
    "Night 22-06": list(range(22, 24)) + list(range(0, 6)),
    "Morning 06-10": list(range(6, 10)),
    "Midday 10-16": list(range(10, 16)),
    "Evening 16-22": list(range(16, 22)),
}

print("%-15s %6s %6s %6s %8s %8s" %
      ("Regime", "Hours", "FI avg", "SE3 avg", "Corr", "Coef"))
print("-" * 55)
for name, h_list in regime_defs.items():
    mask = np.isin(hours, h_list)
    fi_r = fi[mask]
    se3_r = se3.values[mask]
    valid = np.isfinite(se3_r) & np.isfinite(fi_r)
    corr = np.corrcoef(se3_r[valid], fi_r[valid])[0, 1]
    A = np.column_stack([np.ones(valid.sum()), se3_r[valid]])
    c = np.linalg.lstsq(A, fi_r[valid], rcond=None)[0]
    print("%-15s %6d %6.1f %7.1f %8.4f %8.4f" %
          (name, len(h_list), fi_r.mean(), se3_r.mean(), corr, c[1]))

# ================================================================
print()
print("=" * 80)
print("PART 3: LOG-LINEAR MODEL WITH DIFFERENT ENVELOPE FEATURES")
print("=" * 80)
print()

# Build envelope features at different resolutions
train_eval(X_base[:split], X_base[split:], "Weather only (baseline)")

# Test each resolution
for block_h in [3, 4, 6, 8, 12, 24]:
    # SE3 and SE1 envelopes
    se3_env = se3.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values / 100
    se1_env = se1.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values / 100
    ee_env = ee.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values / 100
    extra = np.column_stack([se3_env, se1_env, ee_env])
    Xa = np.hstack([X_base, extra])
    train_eval(Xa[:split], Xa[split:], "+ SE3+SE1+EE %dh envelope" % block_h)

# Multi-resolution: combine short + long
for short, long in [(3, 24), (4, 24), (6, 24), (6, 12)]:
    se3_s = se3.rolling(short, center=True, min_periods=1).mean().ffill().bfill().values / 100
    se3_l = se3.rolling(long, center=True, min_periods=1).mean().ffill().bfill().values / 100
    se1_s = se1.rolling(short, center=True, min_periods=1).mean().ffill().bfill().values / 100
    se1_l = se1.rolling(long, center=True, min_periods=1).mean().ffill().bfill().values / 100
    extra = np.column_stack([se3_s, se3_l, se1_s, se1_l])
    Xa = np.hstack([X_base, extra])
    train_eval(Xa[:split], Xa[split:], "+ SE3+SE1 %dh + %dh" % (short, long))

# ================================================================
print()
print("=" * 80)
print("PART 4: HYBRID - LOG-SPACE WEATHER + LINEAR-SPACE ENVELOPE")
print("=" * 80)
print()
print("Architecture: FI = max(0, exp(weather_log) - 55 + coupling * envelope)")
print("Coupling optimized via grid search")
print()

# Weather-only prediction
cw, icw = np.linalg.lstsq(
    np.column_stack([np.ones(split), X_base[:split]]),
    y_log_tr, rcond=None)[0]
# Simpler: use train_eval weights
ws = weights.sum()
fm = (weights[:, None] * X_base[:split]).sum(0) / ws
fv = (weights[:, None] * (X_base[:split] ** 2)).sum(0) / ws - fm ** 2
fs = np.sqrt(np.maximum(0, fv))
fs[fs < 1e-10] = 1.0
ym = (weights * y_log_tr).sum() / ws
Xs = (X_base[:split] - fm) / fs
ys = y_log_tr - ym
Xw = Xs * weights[:, None]
cs = np.linalg.solve(Xw.T @ Xs + alpha * np.eye(X_base.shape[1]), Xw.T @ ys)
coefs_w = cs / fs
ic_w = ym - (fm / fs) @ cs

wp_tr = np.maximum(0, np.exp(X_base[:split] @ coefs_w + ic_w) - LOG_OFFSET)
wp_te = np.maximum(0, np.exp(X_base[split:] @ coefs_w + ic_w) - LOG_OFFSET)

# Test hybrid with different envelope resolutions and coupling strengths
from scipy.optimize import minimize as sp_min

for block_h in [6, 12, 24]:
    se3_env = se3.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values
    se1_env = se1.rolling(block_h, center=True, min_periods=max(1, block_h // 2)).mean().ffill().bfill().values

    def obj(params):
        c3, c1, off = params
        nb = c3 * se3_env[:split] + c1 * se1_env[:split] + off
        pred = np.maximum(0, wp_tr + nb)
        return np.average(np.abs(fi[:split] - pred), weights=weights)

    opt = sp_min(obj, [0.05, 0.03, -5], method="Nelder-Mead", options={"maxiter": 2000})
    c3, c1, off = opt.x
    nb_te = c3 * se3_env[split:] + c1 * se1_env[split:] + off
    pred = np.maximum(0, wp_te + nb_te)
    mae = mean_absolute_error(fi[split:], pred)
    r2 = r2_score(fi[split:], pred)
    mask20 = (fi[split:] >= 20) & (fi[split:] < 50)
    b20 = (fi[split:][mask20] - pred[mask20]).mean() if mask20.sum() > 0 else 0

    print("  %dh envelope: c_SE3=%.4f c_SE1=%.4f off=%.1f -> MAE=%.3f R2=%.4f max=%.1f b20=%+.1f" %
          (block_h, c3, c1, off, mae, r2, pred.max(), b20))

    # Simulate
    if block_h == 6:
        print("    Simulation (weather=5):")
        for s3, s1, lab in [(30, 20, "Normal"), (80, 50, "Elev"),
                             (120, 70, "High"), (200, 110, "Spike")]:
            nb = c3 * s3 + c1 * s1 + off
            total = max(0, 5 + nb)
            cons = (total / 1000 + 0.0361 + 0.02325) * 1.255 * 100
            print("      %5s SE3=%3d: nb=%+.1f -> FI=%.1f = %.1f c/kWh" %
                  (lab, s3, nb, total, cons))
