"""Do we need log transform at lower time resolutions?"""
import pandas as pd, numpy as np, math, sys, yaml
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import skew, kurtosis
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

# Price distribution at different resolutions
print("=" * 80)
print("PRICE DISTRIBUTION BY RESOLUTION")
print("=" * 80)
print()
print("%-6s %7s %7s %7s %7s %7s %7s" %
      ("Res", "Mean", "Std", "Min", "Max", "Skew", "Neg%"))
print("-" * 52)
for res_h in [1, 3, 6, 12, 24]:
    fi_r = pd.Series(fi, index=df.index).resample(str(res_h) + "h").mean().dropna().values
    print("%-6s %7.1f %7.1f %7.1f %7.1f %7.2f %6.1f%%" %
          (str(res_h) + "h", fi_r.mean(), fi_r.std(), fi_r.min(), fi_r.max(),
           skew(fi_r), 100 * (fi_r < 0).mean()))

# LINEAR vs LOG at each resolution
print()
print("=" * 80)
print("LINEAR vs LOG-LINEAR: WHICH IS BETTER AT EACH RESOLUTION?")
print("=" * 80)
print()
print("%-5s | %-30s | %-30s" % ("", "LINEAR (raw target)", "LOG-LINEAR (log target)"))
print("%-5s | %6s %6s %6s %7s | %6s %6s %6s %7s" %
      ("Res", "MAE", "R2", "Max", "SE3/100", "MAE", "R2", "Max", "SE3/100"))
print("-" * 75)

for res_h in [1, 3, 6, 12, 24]:
    fi_r = pd.Series(fi, index=df.index).resample(str(res_h) + "h").mean()
    se3_r = pd.Series(se3, index=df.index).resample(str(res_h) + "h").mean() / 100
    se1_r = pd.Series(se1, index=df.index).resample(str(res_h) + "h").mean() / 100
    ee_r = pd.Series(ee, index=df.index).resample(str(res_h) + "h").mean() / 100
    Xb_r = pd.DataFrame(X_base, index=df.index, columns=non_ar).resample(
        str(res_h) + "h").mean()
    combined = pd.concat([Xb_r, se3_r.rename("se3"), se1_r.rename("se1"),
                           ee_r.rename("ee"), fi_r.rename("fi")], axis=1).dropna()
    Xa = combined[non_ar + ["se3", "se1", "ee"]].values
    y = combined["fi"].values
    split = int(len(y) * 0.85)
    age_r = np.arange(split - 1, -1, -1, dtype=np.float64)
    w = np.exp(-decay * age_r * res_h)
    w *= split / w.sum()

    # LINEAR
    cl, icl = train_ridge(Xa[:split], np.maximum(y[:split], 0), w)
    pl = np.maximum(0, Xa[split:] @ cl + icl)
    mae_l = mean_absolute_error(y[split:], pl)
    r2_l = r2_score(y[split:], pl)
    se3_l = cl[len(non_ar)]

    # LOG-LINEAR
    cg, icg = train_ridge(Xa[:split], np.log(y[:split] + LOG_OFFSET), w)
    pg = np.maximum(0, np.exp(Xa[split:] @ cg + icg) - LOG_OFFSET)
    mae_g = mean_absolute_error(y[split:], pg)
    r2_g = r2_score(y[split:], pg)
    se3_g = cg[len(non_ar)]

    print("%-5s | %6.3f %6.4f %6.1f %+7.2f | %6.3f %6.4f %6.1f %+7.4f" %
          (str(res_h) + "h", mae_l, r2_l, pl.max(), se3_l,
           mae_g, r2_g, pg.max(), se3_g))

# LINEAR daily model with direct coupling
print()
print("=" * 80)
print("DAILY LINEAR MODEL: DIRECT EUR/MWh COUPLING")
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
y_d = combined_d["fi"].values
split_d = int(len(y_d) * 0.85)
age_d = np.arange(split_d - 1, -1, -1, dtype=np.float64)
w_d = np.exp(-decay * age_d * 24)
w_d *= split_d / w_d.sum()

cd, icd = train_ridge(Xa_d[:split_d], np.maximum(y_d[:split_d], 0), w_d)
pd_te = np.maximum(0, Xa_d[split_d:] @ cd + icd)
mae_d = mean_absolute_error(y_d[split_d:], pd_te)
r2_d = r2_score(y_d[split_d:], pd_te)

se3_c = cd[len(non_ar)]
se1_c = cd[len(non_ar) + 1]
ee_c = cd[len(non_ar) + 2]

print("MAE=%.3f R2=%.4f max=%.1f" % (mae_d, r2_d, pd_te.max()))
print("SE3 coef: %+.2f EUR FI per (SE3/100)" % se3_c)
print("SE1 coef: %+.2f EUR FI per (SE1/100)" % se1_c)
print("EE coef:  %+.2f EUR FI per (EE/100)" % ee_c)
print()
print("When SE3 increases by 100 EUR/MWh: FI increases by %+.1f EUR/MWh" % (se3_c * 1.0))
print("When ALL neighbors +100: FI increases by %+.1f EUR/MWh" %
      ((se3_c + se1_c + ee_c) * 1.0))
print()

# Simulation
print("Daily LINEAR predictions:")
x_avg = Xa_d[:split_d].mean(axis=0)
for se3_val in [20, 40, 60, 80, 100, 150, 200, 300, 400, 600]:
    x = x_avg.copy()
    x[len(non_ar)] = se3_val / 100
    x[len(non_ar) + 1] = se3_val * 0.5 / 100
    x[len(non_ar) + 2] = se3_val * 1.5 / 100
    pred = max(0, float(x @ cd + icd))
    cons = (pred / 1000 + 0.0361 + 0.02325) * 1.255 * 100
    ratio = pred / se3_val if se3_val > 0 else 0
    print("  SE3=%3d: FI=%6.1f EUR/MWh = %5.1f c/kWh  FI/SE3=%.3f" %
          (se3_val, pred, cons, ratio))
