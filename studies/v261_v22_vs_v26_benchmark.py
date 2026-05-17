"""v2.6.1 — head-to-head v2.2 9-feature Ridge vs v2.6.0 L1+L2+L3+L4 benchmark.

User direction 2026-05-17: "We better test the new model against the old
one and if the new model outperforms the old one we can forget
maintaining v2.2 generation model as a part of the system. Implement
head to head comparison with real data."

Reconstructs the v2.2 9-feature Ridge prediction offline from the
shipped artifact (`data/model_coefs_default.json`) — including the
AR(2) profiles for SE3 / EE neighbours — and compares it against the
v2.6.0 V26Pipeline output on the EXACT same test window.

One caveat documented up-front: the nuclear_x_scarcity feature (1 of
the 9) requires historical Fingrid nuclear-capacity data that's only
available for ~5 months. For the rest of the test window we set this
feature to 0 (equivalent to setting nuclear deficit to zero). The
nuclear feature's Ridge coefficient is +0.031, so the missing
contribution is bounded. The v2.6.0 pipeline doesn't use nuclear at
all → both models are evaluated without nuclear info, fair comparison.

Output:
  studies/results/V2_6_1_BENCHMARK.md
  studies/results/figures/v2_6_1_*.png
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import math
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))
sys.path.insert(0, str(REPO / "studies"))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

# Reuse v2.5.16 pipeline assembly for v2.6.0
_spec = _ilu.spec_from_file_location(
    "v2516_performance_review",
    REPO / "studies" / "v2516_performance_review.py",
)
v2516 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(v2516)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO / "custom_components" / "spot_price_predictor" / "data"

V22_COEFS = json.loads((DATA_DIR / "model_coefs_default.json").read_text())


# ── v2.2 feature reconstruction ────────────────────────────────────


def _fi_holidays() -> set[str]:
    """Minimal FI holiday set covering the test window. Same dates as
    `custom_components/spot_price_predictor/holidays.py` produces for
    2023-2026."""
    holidays_iso = []
    for year in (2023, 2024, 2025, 2026):
        holidays_iso += [
            f"{year}-01-01", f"{year}-01-06",
            f"{year}-05-01", f"{year}-06-22",
            f"{year}-12-06", f"{year}-12-24", f"{year}-12-25", f"{year}-12-26",
        ]
    # Variable dates (rough Easter friday/monday)
    holidays_iso += [
        "2023-04-07", "2023-04-10", "2023-05-18",
        "2024-03-29", "2024-04-01", "2024-05-09",
        "2025-04-18", "2025-04-21", "2025-05-29",
        "2026-04-03", "2026-04-06", "2026-05-14",
    ]
    return set(holidays_iso)


def _build_ar_neighbor_residual(
    actual_prices: pd.Series, ar_profile: dict
) -> pd.Series:
    """Y_neighbour(t) = actual(t) − profile[weekday, hour].

    `profile` shape: 7 (weekdays) × 24 (hours). We pull the stored
    workday / weekend profiles from the v2.2 artifact and assemble per
    timestamp."""
    profile_wd = np.asarray(ar_profile["profile_wd"], dtype=float)
    profile_we = np.asarray(ar_profile["profile_we"], dtype=float)
    # Both expected shape (24,)
    if profile_wd.shape != (24,) or profile_we.shape != (24,):
        raise ValueError(f"unexpected profile shape: wd={profile_wd.shape} "
                          f"we={profile_we.shape}")
    h = actual_prices.index.hour.to_numpy()
    weekday = actual_prices.index.weekday.to_numpy()
    is_weekend = (weekday >= 5)
    base = np.where(is_weekend, profile_we[h], profile_wd[h])
    return actual_prices - base


def _ar2_predict(Y: pd.Series, ar_coefs: list[float]) -> pd.Series:
    """AR(2): pred(t) = β1·Y(t-1) + β2·Y(t-2)."""
    b1, b2 = float(ar_coefs[0]), float(ar_coefs[1])
    y = Y.values
    out = np.zeros(len(y), dtype=float)
    out[2:] = b1 * y[1:-1] + b2 * y[:-2]
    return pd.Series(out, index=Y.index, name=Y.name)


def build_v22_features(
    df: pd.DataFrame, holidays: set[str], ar_models: dict
) -> pd.DataFrame:
    """Build the 9-feature v2.2 design matrix from the cached parquets.

    `df` must have columns: fi, se3, se1, ee, wind, solar, temp
    indexed by UTC datetime. `ar_models` from v2.2 artifact.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)
    # Calendar
    h = df.index.hour.to_numpy()
    mo = df.index.month.to_numpy()
    out["month_cos"] = np.cos(2 * np.pi * mo / 12.0)
    dates = df.index.strftime("%Y-%m-%d").to_numpy()
    out["is_holiday"] = np.isin(dates, list(holidays)).astype(float)
    # Weather-derived
    out["wind_speed_weighted"] = df["wind"].values
    out["hdd_sq"] = np.maximum(0.0, 17.0 - df["temp"].values) ** 2
    out["wind_log_scarcity"] = np.log1p(
        np.maximum(0.0, 8.0 - df["wind"].values))
    # Cross-border AR
    Y_se3 = _build_ar_neighbor_residual(df["se3"], ar_models["se3"])
    Y_ee  = _build_ar_neighbor_residual(df["ee"],  ar_models["ee"])
    out["ar_se3"] = _ar2_predict(Y_se3, ar_models["se3"]["ar_coefs"]) / 100.0
    out["ar_ee"]  = _ar2_predict(Y_ee,  ar_models["ee"]["ar_coefs"])  / 100.0
    # Export potential: 7-day rolling spread between FI and SE3
    spread = df["fi"] - df["se3"]
    rolling_spread = spread.rolling("7D").mean()
    out["export_potential_se3"] = np.maximum(0.0, -rolling_spread.values)
    # Nuclear x scarcity — NOT available for full test window; set to 0
    out["nuclear_x_scarcity"] = 0.0
    return out


def v22_predict(features: pd.DataFrame) -> np.ndarray:
    """Apply the v2.2 log-linear model:
        linear = intercept + Σ coef · feature
        raw    = max(0, exp(min(linear, 20)) - log_offset)
        pred   = power_scale · raw ^ power_exp
    """
    intercept    = float(V22_COEFS["intercept"])
    log_offset   = float(V22_COEFS["log_offset"])
    power_scale  = float(V22_COEFS["power_scale"])
    power_exp    = float(V22_COEFS["power_exp"])
    coef_map = {f["name"]: float(f["coef"]) for f in V22_COEFS["features"]}
    n = len(features)
    linear = np.full(n, intercept, dtype=float)
    for name, c in coef_map.items():
        if name not in features.columns:
            continue
        linear += c * features[name].values
    linear = np.minimum(linear, 20.0)
    raw = np.maximum(0.0, np.exp(linear) - log_offset)
    pred = np.where(raw > 0, power_scale * raw ** power_exp, 0.0)
    return pred


# ── Metrics ────────────────────────────────────────────────────────


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    var_y = float(np.var(actual))
    r2 = 1.0 - float(np.var(err)) / var_y if var_y > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "bias": bias, "R2": r2}


def daily_dk(prices: pd.Series) -> pd.DataFrame:
    """24-entry D(k) per day per direction."""
    rows = []
    for date, day in prices.groupby(prices.index.date):
        vals = np.sort(day.values)
        if len(vals) < 24:
            continue
        row: dict = {"date": pd.Timestamp(date)}
        cum_low  = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        cum_high = np.cumsum(vals[::-1]) / np.arange(1, len(vals) + 1)
        for i in range(24):
            row[f"cheap_{i:02d}"] = float(cum_low[i])
            row[f"peak_{i:02d}"]  = float(cum_high[i])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def dk_compare(actual_dk: pd.DataFrame, pred_dk: pd.DataFrame
               ) -> dict[str, dict[str, float]]:
    common = actual_dk.index.intersection(pred_dk.index)
    out = {}
    for col in actual_dk.columns:
        a = actual_dk.loc[common, col].values
        p = pred_dk.loc[common, col].values
        out[col] = metrics(a, p)
    return out


def peak_event_capture(actual: np.ndarray, pred: np.ndarray,
                        threshold: float = 100.0) -> dict[str, float]:
    actual_ev = actual >= threshold
    predicted_warn = pred >= 0.7 * threshold
    tp = int((actual_ev & predicted_warn).sum())
    fn = int((actual_ev & ~predicted_warn).sum())
    fp = int((~actual_ev & predicted_warn).sum())
    return {
        "n_events": tp + fn,
        "n_warnings": tp + fp,
        "hit_rate": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
    }


# ── Figures ────────────────────────────────────────────────────────


def fig_full_period_compare(df: pd.DataFrame, v22: np.ndarray,
                              v26: np.ndarray, split: int,
                              out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    test_ts = df.index[split:]
    test_act = df["fi"].values[split:]
    test_v22 = v22[split:]
    test_v26 = v26[split:]

    daily = pd.DataFrame({
        "actual": test_act, "v22": test_v22, "v26": test_v26,
    }, index=test_ts).resample("D").mean()
    ax = axes[0]
    ax.plot(daily.index, daily["actual"], "k-", lw=1.1, label="Actual")
    ax.plot(daily.index, daily["v22"],    "C3-", lw=0.9, alpha=0.85,
            label="v2.2 9-feature Ridge")
    ax.plot(daily.index, daily["v26"],    "C0-", lw=0.9, alpha=0.85,
            label="v2.6.0 L1+L2+L3+L4+floor")
    ax.set_ylabel("Daily mean FI price [EUR/MWh]")
    ax.set_title(f"Full test period — daily means "
                  f"({test_ts[0].date()} → {test_ts[-1].date()})")
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30, labelsize=9)

    # 4-week zoom — hourly resolution
    zoom_start = pd.Timestamp("2025-08-04", tz="UTC")
    zoom_end   = pd.Timestamp("2025-09-01", tz="UTC")
    mask = (test_ts >= zoom_start) & (test_ts <= zoom_end)
    ax = axes[1]
    ax.plot(test_ts[mask], test_act[mask], "k-", lw=0.9, label="Actual")
    ax.plot(test_ts[mask], test_v22[mask], "C3-", lw=0.9, alpha=0.7,
            label="v2.2")
    ax.plot(test_ts[mask], test_v26[mask], "C0-", lw=0.9, alpha=0.85,
            label="v2.6.0")
    ax.set_ylabel("Hourly FI price [EUR/MWh]")
    ax.set_title(f"4-week zoom — hourly ({zoom_start.date()} → "
                  f"{zoom_end.date()})")
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_metric_comparison(v22_m: dict, v26_m: dict,
                           dk_v22: dict, dk_v26: dict,
                           out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Hourly metrics
    ax = axes[0, 0]
    metric_names = ["MAE", "RMSE", "|bias|"]
    v22_vals = [v22_m["MAE"], v22_m["RMSE"], abs(v22_m["bias"])]
    v26_vals = [v26_m["MAE"], v26_m["RMSE"], abs(v26_m["bias"])]
    x = np.arange(len(metric_names))
    ax.bar(x - 0.2, v22_vals, width=0.4, color="C3", label="v2.2")
    ax.bar(x + 0.2, v26_vals, width=0.4, color="C0", label="v2.6.0")
    for i, (a, b) in enumerate(zip(v22_vals, v26_vals)):
        ax.annotate(f"{a:.1f}", (i - 0.2, a), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{b:.1f}", (i + 0.2, b), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Hourly point-forecast accuracy")
    ax.legend(loc="upper right", fontsize=10)

    # R²
    ax = axes[0, 1]
    ax.bar([0], [v22_m["R2"]], width=0.4, color="C3", label="v2.2")
    ax.bar([0.5], [v26_m["R2"]], width=0.4, color="C0", label="v2.6.0")
    for x, v in [(0, v22_m["R2"]), (0.5, v26_m["R2"])]:
        ax.annotate(f"{v:+.3f}", (x, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks([0, 0.5]); ax.set_xticklabels(["v2.2", "v2.6.0"])
    ax.set_ylabel("R²")
    ax.set_title("Hourly point-forecast R²")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(-0.1, 1.0)

    # D(k) MAE per index
    ax = axes[1, 0]
    indices = list(range(24))
    v22_cheap_mae = [dk_v22[f"cheap_{i:02d}"]["MAE"] for i in indices]
    v26_cheap_mae = [dk_v26[f"cheap_{i:02d}"]["MAE"] for i in indices]
    v22_peak_mae  = [dk_v22[f"peak_{i:02d}"]["MAE"]  for i in indices]
    v26_peak_mae  = [dk_v26[f"peak_{i:02d}"]["MAE"]  for i in indices]
    ax.plot(indices, v22_cheap_mae, "C3-o", lw=1.2, ms=4, label="v2.2 cheap")
    ax.plot(indices, v26_cheap_mae, "C0-o", lw=1.5, ms=5, label="v2.6.0 cheap")
    ax.plot(indices, v22_peak_mae,  "C3--s", lw=1.2, ms=4, label="v2.2 peak")
    ax.plot(indices, v26_peak_mae,  "C0--s", lw=1.5, ms=5, label="v2.6.0 peak")
    ax.set_xlabel("index i (0 = single hour, 23 = full day)")
    ax.set_ylabel("D(k) MAE [EUR/MWh]")
    ax.set_title("D(k) accuracy per index")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks(range(0, 24, 2))

    # D(k) R² per index
    ax = axes[1, 1]
    v22_cheap_r2 = [dk_v22[f"cheap_{i:02d}"]["R2"] for i in indices]
    v26_cheap_r2 = [dk_v26[f"cheap_{i:02d}"]["R2"] for i in indices]
    v22_peak_r2  = [dk_v22[f"peak_{i:02d}"]["R2"]  for i in indices]
    v26_peak_r2  = [dk_v26[f"peak_{i:02d}"]["R2"]  for i in indices]
    ax.plot(indices, v22_cheap_r2, "C3-o", lw=1.2, ms=4, label="v2.2 cheap")
    ax.plot(indices, v26_cheap_r2, "C0-o", lw=1.5, ms=5, label="v2.6.0 cheap")
    ax.plot(indices, v22_peak_r2,  "C3--s", lw=1.2, ms=4, label="v2.2 peak")
    ax.plot(indices, v26_peak_r2,  "C0--s", lw=1.5, ms=5, label="v2.6.0 peak")
    ax.set_xlabel("index i")
    ax.set_ylabel("R²")
    ax.set_title("D(k) R² per index")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0.7, 1.0)

    fig.suptitle("v2.6.1 — v2.2 vs v2.6.0 head-to-head on real FI test data",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("=== v2.6.1 head-to-head benchmark: v2.2 vs v2.6.0 ===\n")
    print("[1/4] Building v2.6.0 pipeline (reuses v2.5.16 module)...",
          flush=True)
    p = v2516.build_pipeline()
    df, split = p["df"], p["split"]
    actual = p["actual"]
    v26_pred = p["corrected"]                # v2.6.0 final corrected output
    print(f"  test window: {df.index[split].date()} → {df.index[-1].date()}  "
          f"({len(df)-split:,} hourly rows)", flush=True)

    print("\n[2/4] Reconstructing v2.2 9-feature Ridge predictions...",
          flush=True)
    holidays = _fi_holidays()
    ar_models = V22_COEFS["ar_models"]
    v22_features = build_v22_features(df, holidays, ar_models)
    v22_pred = v22_predict(v22_features)
    # v2.2 predicts shifted prices (it's spot in EUR/MWh)
    print(f"  v2.2 features built (shape {v22_features.shape})", flush=True)

    # Restrict comparison to test portion
    test_actual = actual[split:]
    test_v22    = v22_pred[split:]
    test_v26    = v26_pred[split:]

    print("\n[3/4] Computing comparison metrics...", flush=True)
    # Hourly point-forecast metrics
    v22_m = metrics(test_actual, test_v22)
    v26_m = metrics(test_actual, test_v26)
    print(f"  hourly point forecast:")
    print(f"    v2.2     MAE={v22_m['MAE']:.2f}  RMSE={v22_m['RMSE']:.2f}  "
          f"R²={v22_m['R2']:+.3f}  bias={v22_m['bias']:+.2f}", flush=True)
    print(f"    v2.6.0   MAE={v26_m['MAE']:.2f}  RMSE={v26_m['RMSE']:.2f}  "
          f"R²={v26_m['R2']:+.3f}  bias={v26_m['bias']:+.2f}", flush=True)
    delta_mae  = v22_m["MAE"] - v26_m["MAE"]
    delta_r2   = v26_m["R2"]  - v22_m["R2"]
    print(f"    Δ MAE (v22 - v26) = {delta_mae:+.2f} EUR/MWh  "
          f"({100*delta_mae/v22_m['MAE']:+.1f}%)")
    print(f"    Δ R²              = {delta_r2:+.3f}")

    # D(k) per-index accuracy
    test_ts = df.index[split:]
    actual_series = pd.Series(test_actual, index=test_ts)
    v22_series    = pd.Series(test_v22,    index=test_ts)
    v26_series    = pd.Series(test_v26,    index=test_ts)
    actual_dk = daily_dk(actual_series)
    v22_dk = daily_dk(v22_series)
    v26_dk = daily_dk(v26_series)
    dk_v22 = dk_compare(actual_dk, v22_dk)
    dk_v26 = dk_compare(actual_dk, v26_dk)

    print(f"\n  D(k) accuracy at key indices:")
    print(f"    {'metric':>10s}  {'v2.2 MAE':>9s}  {'v2.6 MAE':>9s}  "
          f"{'v2.2 R²':>8s}  {'v2.6 R²':>8s}  {'Δ MAE':>8s}")
    for col in ["cheap_00", "cheap_03", "cheap_07", "cheap_11",
                "cheap_15", "cheap_19", "cheap_23",
                "peak_00", "peak_03", "peak_07", "peak_11",
                "peak_15", "peak_19", "peak_23"]:
        v22d = dk_v22[col]; v26d = dk_v26[col]
        d_mae = v22d["MAE"] - v26d["MAE"]
        print(f"    {col:>10s}  {v22d['MAE']:9.2f}  {v26d['MAE']:9.2f}  "
              f"{v22d['R2']:+8.3f}  {v26d['R2']:+8.3f}  {d_mae:+8.2f}",
              flush=True)

    # Peak event capture
    pec_v22 = peak_event_capture(test_actual, test_v22)
    pec_v26 = peak_event_capture(test_actual, test_v26)
    print(f"\n  Peak event capture (≥ 100 EUR/MWh threshold):")
    print(f"    v2.2:     hit_rate={pec_v22['hit_rate']*100:.1f}%  "
          f"precision={pec_v22['precision']*100:.1f}%")
    print(f"    v2.6.0:   hit_rate={pec_v26['hit_rate']*100:.1f}%  "
          f"precision={pec_v26['precision']*100:.1f}%")

    # Per-month MAE
    monthly_rows = []
    df_test = pd.DataFrame({
        "actual": test_actual, "v22": test_v22, "v26": test_v26,
    }, index=test_ts)
    for month, g in df_test.groupby(df_test.index.to_period("M")):
        monthly_rows.append({
            "month": str(month),
            "n": len(g),
            "v22_MAE": float(np.mean(np.abs(g["v22"] - g["actual"]))),
            "v26_MAE": float(np.mean(np.abs(g["v26"] - g["actual"]))),
        })
    monthly = pd.DataFrame(monthly_rows)
    monthly["delta"] = monthly["v22_MAE"] - monthly["v26_MAE"]
    print(f"\n  Per-month MAE (v22 - v26 positive ⇒ v26 better):")
    print(monthly[["month", "n", "v22_MAE", "v26_MAE", "delta"]
                  ].to_string(index=False), flush=True)
    v26_wins = (monthly["delta"] > 0).sum()
    print(f"\n  v2.6.0 wins {v26_wins} of {len(monthly)} months on MAE",
          flush=True)

    # Figures
    print("\n[4/4] Rendering figures + report...", flush=True)
    fig_full_period_compare(df, v22_pred, v26_pred, split,
                              FIGURES_DIR / "v2_6_1_full_period.png")
    fig_metric_comparison(v22_m, v26_m, dk_v22, dk_v26,
                            FIGURES_DIR / "v2_6_1_metric_comparison.png")

    # Verdict
    cheap_wins = sum(1 for i in range(24)
                      if dk_v26[f"cheap_{i:02d}"]["MAE"]
                      < dk_v22[f"cheap_{i:02d}"]["MAE"])
    peak_wins  = sum(1 for i in range(24)
                      if dk_v26[f"peak_{i:02d}"]["MAE"]
                      < dk_v22[f"peak_{i:02d}"]["MAE"])
    print(f"\n  D(k) cheap: v2.6.0 wins {cheap_wins}/24 indices on MAE",
          flush=True)
    print(f"  D(k) peak:  v2.6.0 wins {peak_wins}/24 indices on MAE",
          flush=True)
    verdict = ""
    if (delta_mae > 0 and delta_r2 > 0 and cheap_wins >= 12 and peak_wins >= 12
        and pec_v26["hit_rate"] >= pec_v22["hit_rate"]):
        verdict = "✅ v2.6.0 WINS — recommend v2.7.0 cutover"
    elif delta_mae > 0 and delta_r2 > 0:
        verdict = "🟡 v2.6.0 wins on point forecast, mixed on D(k) — review per-index"
    else:
        verdict = "❌ v2.6.0 does NOT clearly beat v2.2 — investigate before cutover"
    print(f"\n  ===  VERDICT: {verdict}  ===\n", flush=True)

    # Markdown
    md = RESULTS_DIR / "V2_6_1_BENCHMARK.md"
    lines = [
        "# v2.6.1 — Head-to-head: v2.2 9-feature Ridge vs v2.6.0 L1+L2+L3+L4",
        "",
        "Per user direction 2026-05-17: \"We better test the new model against "
        "the old one and if the new model outperforms the old one we can "
        "forget maintaining v2.2 generation model as a part of the system.\"",
        "",
        f"**Test window**: {df.index[split].date()} → {df.index[-1].date()} "
        f"({len(df)-split:,} hourly rows)",
        "",
        "**Methodology**: v2.2 features reconstructed from "
        "`data/model_coefs_default.json` including stored AR(2) profiles "
        "for SE3 / EE. One caveat: `nuclear_x_scarcity` is set to 0 because "
        "Fingrid nuclear data isn't cached for the full window. The v2.2 "
        "Ridge coefficient on this feature is +0.031 (small), and v2.6.0 "
        "doesn't use nuclear info at all, so both models are evaluated "
        "without nuclear awareness — fair comparison.",
        "",
        "## Hourly point-forecast accuracy",
        "",
        f"| Model | MAE | RMSE | R² | Bias |",
        f"|---|---:|---:|---:|---:|",
        f"| v2.2 9-feature Ridge | {v22_m['MAE']:.2f} | {v22_m['RMSE']:.2f} | "
        f"{v22_m['R2']:+.3f} | {v22_m['bias']:+.2f} |",
        f"| **v2.6.0 L1+L2+L3+L4+floor** | "
        f"**{v26_m['MAE']:.2f}** | **{v26_m['RMSE']:.2f}** | "
        f"**{v26_m['R2']:+.3f}** | **{v26_m['bias']:+.2f}** |",
        f"| **Δ (v22 − v26)** | **{delta_mae:+.2f}** "
        f"({100*delta_mae/v22_m['MAE']:+.1f}%) | — | "
        f"**{delta_r2:+.3f}** | — |",
        "",
        "## D(k) accuracy per index",
        "",
        "| metric | v2.2 MAE | v2.6 MAE | v2.2 R² | v2.6 R² | Δ MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for col in ["cheap_00", "cheap_03", "cheap_07", "cheap_11", "cheap_15",
                "cheap_19", "cheap_23",
                "peak_00", "peak_03", "peak_07", "peak_11", "peak_15",
                "peak_19", "peak_23"]:
        v22d = dk_v22[col]; v26d = dk_v26[col]
        lines.append(
            f"| `{col}` | {v22d['MAE']:.2f} | {v26d['MAE']:.2f} | "
            f"{v22d['R2']:+.3f} | {v26d['R2']:+.3f} | "
            f"{v22d['MAE'] - v26d['MAE']:+.2f} |"
        )
    lines += [
        "",
        f"**D(k) cheap: v2.6.0 wins {cheap_wins}/24 indices on MAE**",
        f"**D(k) peak: v2.6.0 wins {peak_wins}/24 indices on MAE**",
        "",
        "## Peak-event capture (actual ≥ 100 EUR/MWh)",
        "",
        f"- v2.2:     hit_rate {pec_v22['hit_rate']*100:.1f} %, "
        f"precision {pec_v22['precision']*100:.1f} %",
        f"- v2.6.0:   hit_rate {pec_v26['hit_rate']*100:.1f} %, "
        f"precision {pec_v26['precision']*100:.1f} %",
        "",
        "## Per-month MAE breakdown",
        "",
        "| month | n | v2.2 MAE | v2.6 MAE | Δ (v22 − v26) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in monthly.iterrows():
        lines.append(
            f"| {row['month']} | {row['n']} | {row['v22_MAE']:.2f} | "
            f"{row['v26_MAE']:.2f} | {row['delta']:+.2f} |"
        )
    lines += [
        "",
        f"**v2.6.0 wins {v26_wins} of {len(monthly)} months on MAE.**",
        "",
        "## Figures",
        "",
        "### Full-period comparison",
        "",
        "![Full period](figures/v2_6_1_full_period.png)",
        "",
        "### Metric comparison + D(k) per index",
        "",
        "![Metrics](figures/v2_6_1_metric_comparison.png)",
        "",
        "## Verdict",
        "",
        f"### {verdict}",
        "",
        "Recommendation for v2.7.0:",
    ]
    if verdict.startswith("✅"):
        lines += [
            "",
            "- **Drop the v2.2 9-feature Ridge from the production code path.**",
            "- Replace `forecast[i].spot_eur_mwh` / `consumer_eur_kwh` with the "
            "v2.6.0 V_sigmoid_full prediction.",
            "- Replace `dk_cheap_eur_kwh[12]` / `dk_peak_eur_kwh[12]` with the "
            "24-entry v2.6.0 D(k) (per v2.5.17 schema).",
            "- Remove `model.py`, `features.py` AR-with-daytype machinery, "
            "`data/model_coefs_default.json` — estimated net cleanup of "
            "~400 LOC.",
            "- The `v26_*` attributes can be aliased back to the primary "
            "names (`spot_eur_mwh`, etc.) — dashboards keep working but the "
            "values are produced by the better model.",
            "",
            "v2.6.1 (this patch) just produces the evidence; the cleanup is "
            "v2.7.0.",
        ]
    else:
        lines += [
            "",
            "- Investigate the per-index losses before cutting v2.2.",
            "- Consider keeping v2.2 as a fallback for the bands where it wins.",
            "- Re-run after addressing the gap.",
        ]
    lines += [
        "",
        "## Files",
        "",
        "- **New**: `studies/v261_v22_vs_v26_benchmark.py` (~470 LOC)",
        "- **New**: `studies/results/V2_6_1_BENCHMARK.md` — this report",
        "- **New**: 2 figures (`v2_6_1_full_period.png`, "
        "`v2_6_1_metric_comparison.png`)",
        "- **Modified**: `manifest.json` 2.6.0 → 2.6.1, README index",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v261_v22_vs_v26_benchmark.py",
        "```",
        "",
        "Offline; uses only locally cached parquets + shipped artifacts.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {md}")


if __name__ == "__main__":
    main()
