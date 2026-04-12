"""Optimize nonlinearity with smoothed/resampled input data.

Key insight: train at lower temporal resolution where the price
distribution is less skewed, then optimize the output nonlinearity
to map back to the actual price range.

Architecture:
1. Resample all data to N-hour blocks (3h, 6h, 12h, 24h)
2. Train LINEAR Ridge at that resolution (direct EUR coupling, no log)
3. Optimize output nonlinearity via Nelder-Mead to map linear
   predictions to actual prices at the ORIGINAL hourly resolution

This separates the coupling estimation (at low resolution where
it's not suppressed) from the output shaping (where nonlinearity
handles the skewed distribution).
"""
import pandas as pd, numpy as np, yaml, sys, math
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize as sp_min
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

half_life = 120; alpha = 1.0
decay = np.log(2.0) / (half_life * 24.0)
split_h = int(len(fi) * 0.85)  # hourly split

local = df.index.tz_convert(ZoneInfo("Europe/Helsinki"))
dates_te = local[split_h:].date

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

def block_rank_4h(preds, actuals, dates):
    conc = disc = 0
    for d in sorted(set(dates)):
        m = np.array([dd == d for dd in dates])
        if m.sum() < 5: continue
        p_d, y_d = preds[m], actuals[m]
        n = len(p_d)
        pb = np.array([p_d[i:i+4].mean() for i in range(n-3)])
        yb = np.array([y_d[i:i+4].mean() for i in range(n-3)])
        for i in range(len(pb)):
            for j in range(i+1, len(pb)):
                yd = yb[i]-yb[j]
                if abs(yd) < 5: continue
                if (pb[i]-pb[j])*yd > 0: conc += 1
                else: disc += 1
    return 100*conc/(conc+disc) if (conc+disc) > 0 else 0


# ================================================================
print("=" * 80)
print("ARCHITECTURE: LINEAR Ridge at lower resolution + optimized nonlinearity")
print("=" * 80)
print()

for train_res in [3, 6, 12, 24]:
    print("-" * 70)
    print("TRAINING RESOLUTION: %dh" % train_res)
    print("-" * 70)

    # Resample to training resolution
    fi_r = pd.Series(fi, index=df.index).resample(str(train_res) + "h").mean()
    se3_r = pd.Series(se3, index=df.index).resample(str(train_res) + "h").mean() / 100
    se1_r = pd.Series(se1, index=df.index).resample(str(train_res) + "h").mean() / 100
    ee_r = pd.Series(ee, index=df.index).resample(str(train_res) + "h").mean() / 100
    Xb_r = pd.DataFrame(X_base, index=df.index, columns=non_ar).resample(
        str(train_res) + "h").mean()

    combined = pd.concat([Xb_r, se3_r.rename("se3"), se1_r.rename("se1"),
                           ee_r.rename("ee"), fi_r.rename("fi")], axis=1).dropna()
    Xa_r = combined[non_ar + ["se3", "se1", "ee"]].values
    y_r = combined["fi"].values
    split_r = int(len(y_r) * 0.85)

    age_r = np.arange(split_r - 1, -1, -1, dtype=np.float64)
    w_r = np.exp(-decay * age_r * train_res)
    w_r *= split_r / w_r.sum()

    # Train LINEAR Ridge at this resolution
    cr, icr = train_ridge(Xa_r[:split_r], np.maximum(y_r[:split_r], 0), w_r)

    se3_c = cr[len(non_ar)]
    se1_c = cr[len(non_ar) + 1]
    ee_c = cr[len(non_ar) + 2]
    print("  Linear coefficients: SE3=%+.2f SE1=%+.2f EE=%+.2f" % (se3_c, se1_c, ee_c))
    print("  SE3+100 -> FI %+.1f EUR/MWh" % (se3_c * 1.0))

    # Now apply this model to HOURLY data
    # Build hourly features with envelope at training resolution
    se3_env = pd.Series(se3, index=df.index).rolling(
        train_res, center=True, min_periods=max(1, train_res // 2)).mean().ffill().bfill().values / 100
    se1_env = pd.Series(se1, index=df.index).rolling(
        train_res, center=True, min_periods=max(1, train_res // 2)).mean().ffill().bfill().values / 100
    ee_env = pd.Series(ee, index=df.index).rolling(
        train_res, center=True, min_periods=max(1, train_res // 2)).mean().ffill().bfill().values / 100

    extra_h = np.column_stack([se3_env, se1_env, ee_env])
    Xa_h = np.hstack([X_base, extra_h])

    # Linear prediction at hourly resolution using low-res coefficients
    linear_h = Xa_h @ cr + icr
    linear_h_tr = linear_h[:split_h]
    linear_h_te = linear_h[split_h:]

    # Baseline: just max(0, linear)
    preds_raw = np.maximum(0, linear_h_te)
    mae_raw = mean_absolute_error(fi[split_h:], preds_raw)
    r2_raw = r2_score(fi[split_h:], preds_raw)
    rank_raw = block_rank_4h(preds_raw, fi[split_h:], dates_te)
    print("  Raw linear (max 0): MAE=%.3f R2=%.4f max=%.1f rank=%.1f%%" %
          (mae_raw, r2_raw, preds_raw.max(), rank_raw))

    # Optimize nonlinearity: f(x) = a + b*ln(c*x + d)
    actual_tr = fi[:split_h]
    w_h = np.exp(-decay * np.arange(split_h - 1, -1, -1, dtype=np.float64))
    w_h *= split_h / w_h.sum()

    # A: Power: f(x) = scale * max(0, x)^power
    def obj_power(params):
        s, p = params
        if s <= 0 or p < 0.5 or p > 3: return 1e6
        pred = s * np.power(np.maximum(linear_h_tr, 0) + 1e-10, p)
        return np.average(np.abs(actual_tr - pred), weights=w_h)

    opt_p = sp_min(obj_power, [1.0, 1.0], method="Nelder-Mead", options={"maxiter": 1000})
    ps, pp = opt_p.x
    preds_power = ps * np.power(np.maximum(linear_h_te, 0) + 1e-10, pp)
    mae_p = mean_absolute_error(fi[split_h:], preds_power)
    r2_p = r2_score(fi[split_h:], preds_power)
    rank_p = block_rank_4h(preds_power, fi[split_h:], dates_te)
    print("  Power (s=%.3f p=%.3f): MAE=%.3f R2=%.4f max=%.1f rank=%.1f%%" %
          (ps, pp, mae_p, r2_p, preds_power.max(), rank_p))

    # B: Generalized log: f(x) = a + b * ln(c*x + d)
    def obj_genlog(params):
        a, b, c, d = params
        if b <= 0 or c <= 0 or d <= 0: return 1e6
        inner = c * np.maximum(linear_h_tr, 0) + d
        pred = a + b * np.log(inner)
        return np.average(np.abs(actual_tr - pred), weights=w_h)

    best_gl = None
    for init in [(-10, 5, 0.1, 1), (0, 3, 0.5, 5), (-20, 8, 0.05, 2), (0, 10, 0.02, 0.5)]:
        try:
            opt = sp_min(obj_genlog, init, method="Nelder-Mead", options={"maxiter": 1000})
            a, b, c, d = opt.x
            if b > 0 and c > 0 and d > 0:
                inner = c * np.maximum(linear_h_te, 0) + d
                pred = a + b * np.log(inner)
                mae = mean_absolute_error(fi[split_h:], pred)
                if best_gl is None or mae < best_gl[0]:
                    best_gl = (mae, a, b, c, d, pred)
        except Exception:
            pass

    if best_gl:
        mae_gl, a, b, c, d, preds_gl = best_gl
        r2_gl = r2_score(fi[split_h:], preds_gl)
        rank_gl = block_rank_4h(preds_gl, fi[split_h:], dates_te)
        print("  GenLog (a=%.1f b=%.1f c=%.3f d=%.1f): MAE=%.3f R2=%.4f max=%.1f rank=%.1f%%" %
              (a, b, c, d, mae_gl, r2_gl, preds_gl.max(), rank_gl))

    # C: Power log: f(x) = a * (ln(b*x + 1))^p
    def obj_powerlog(params):
        a, b, p = params
        if a <= 0 or b <= 0 or p <= 0.3: return 1e6
        inner = b * np.maximum(linear_h_tr, 0) + 1
        pred = a * np.power(np.log(inner), p)
        if not np.all(np.isfinite(pred)): return 1e6
        return np.average(np.abs(actual_tr - pred), weights=w_h)

    best_pl = None
    for init in [(5, 0.1, 1.5), (10, 0.05, 2), (3, 0.5, 1), (8, 0.02, 1.2)]:
        try:
            opt = sp_min(obj_powerlog, init, method="Nelder-Mead", options={"maxiter": 1000})
            a, b, p = opt.x
            if a > 0 and b > 0 and p > 0.3:
                inner = b * np.maximum(linear_h_te, 0) + 1
                pred = a * np.power(np.log(inner), p)
                mae = mean_absolute_error(fi[split_h:], pred)
                if best_pl is None or mae < best_pl[0]:
                    best_pl = (mae, a, b, p, pred)
        except Exception:
            pass

    if best_pl:
        mae_pl, a, b, p, preds_pl = best_pl
        r2_pl = r2_score(fi[split_h:], preds_pl)
        rank_pl = block_rank_4h(preds_pl, fi[split_h:], dates_te)
        print("  PowerLog (a=%.1f b=%.3f p=%.2f): MAE=%.3f R2=%.4f max=%.1f rank=%.1f%%" %
              (a, b, p, mae_pl, r2_pl, preds_pl.max(), rank_pl))

    # Simulation with best nonlinearity
    best_preds = preds_power  # use power as default
    if best_gl and best_gl[0] < mae_p:
        best_preds = best_gl[5]
    if best_pl and best_pl[0] < mae_p:
        best_preds = best_pl[4]

    print()
    print("  Simulation (avg spring weather, workday AM peak):")
    x_avg = Xa_r[:split_r].mean(axis=0)
    for se3_val in [30, 80, 120, 200, 400, 600]:
        x = x_avg.copy()
        x[len(non_ar)] = se3_val / 100
        x[len(non_ar) + 1] = se3_val * 0.5 / 100
        x[len(non_ar) + 2] = se3_val * 1.5 / 100
        lin = max(0, float(x @ cr + icr))
        pw = ps * (lin + 1e-10) ** pp
        cons = (pw / 1000 + 0.0361 + 0.02325) * 1.255 * 100
        print("    SE3=%3d: linear=%5.1f power=%6.1f EUR/MWh = %5.1f c/kWh" %
              (se3_val, lin, pw, cons))
    print()

# Summary comparison
print("=" * 80)
print("SUMMARY: BEST MODEL AT EACH TRAINING RESOLUTION")
print("=" * 80)
print()

# Also include the previous log-linear model for comparison
LOG_OFFSET = 55
se3_3h = pd.Series(se3, index=df.index).rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
se1_3h = pd.Series(se1, index=df.index).rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
ee_3h = pd.Series(ee, index=df.index).rolling(3, center=True, min_periods=1).mean().ffill().bfill().values / 100
extra_log = np.column_stack([se3_3h, se1_3h, ee_3h])
Xa_log = np.hstack([X_base, extra_log])
age_log = np.arange(split_h - 1, -1, -1, dtype=np.float64)
w_log = np.exp(-decay * age_log)
w_log *= split_h / w_log.sum()
cl, icl = train_ridge(Xa_log[:split_h], np.log(fi[:split_h] + LOG_OFFSET), w_log)
preds_log = np.maximum(0, np.exp(Xa_log[split_h:] @ cl + icl) - LOG_OFFSET)
mae_log = mean_absolute_error(fi[split_h:], preds_log)
r2_log = r2_score(fi[split_h:], preds_log)
rank_log = block_rank_4h(preds_log, fi[split_h:], dates_te)
print("Current v1.6 log-linear 3h: MAE=%.3f R2=%.4f max=%.1f rank=%.1f%%" %
      (mae_log, r2_log, preds_log.max(), rank_log))
