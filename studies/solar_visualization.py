"""Rich visual analysis of the v2.5.3 solar production sub-model.

Reads the cached Fingrid + Open-Meteo data left by
`solar_clear_sky_submodel.py` and the deployed artifact, then renders
three figures covering different facets of the forecast story:

  studies/results/figures/
    solar_tracking.png       — four seasonal two-week windows: actual,
                               deployed prediction, cloud cover overlay
    solar_accuracy.png       — scatter density, residual distribution
                               + Q-Q, error vs cloud cover, error vs
                               clear-sky GHI level
    solar_temporal.png       — monthly MAE / bias / capacity drift,
                               cumulative error envelope, deployed-vs-
                               training-fit comparison

No Fingrid API call is performed — the script is purely offline. The
caches under studies/.cache/ are populated by the upstream training
script.

Usage:
    python studies/solar_visualization.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import solar_clear_sky as scs  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE_DIR = REPO / "studies" / ".cache"
FIGURES_DIR = REPO / "studies" / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = (REPO / "custom_components" / "spot_price_predictor"
                 / "data" / "solar_submodel_default.json")


# ── Data loaders (reuse cached files; no network) ─────────────────


def _load_fingrid(ds_id: int) -> pd.Series:
    """Locate the cached Fingrid response for dataset `ds_id` and return
    the hourly-mean series (UTC index, MW values)."""
    matches = list(CACHE_DIR.glob(f"fingrid_ds{ds_id}_*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No cached Fingrid ds{ds_id} response found in {CACHE_DIR}. "
            f"Run studies/solar_clear_sky_submodel.py first."
        )
    rows = json.loads(matches[0].read_text())
    bucket: dict[datetime, list[float]] = defaultdict(list)
    for row in rows:
        ts_str = row.get("startTime")
        if not ts_str:
            continue
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        h = ts.replace(minute=0, second=0, microsecond=0)
        try:
            bucket[h].append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    idx = pd.DatetimeIndex(sorted(bucket.keys()), tz="UTC")
    vals = [sum(bucket[t]) / len(bucket[t]) for t in idx]
    return pd.Series(vals, index=idx, dtype=float)


def _load_cloud_cover_weighted(sites: list[dict]) -> pd.Series:
    by_loc: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for loc in sites:
        sw = float(loc.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        name = loc["name"]
        key = name.replace(" ", "_").replace("/", "_")
        matches = list(CACHE_DIR.glob(f"openmeteo_cloud_{key}_*.json"))
        if not matches:
            continue
        payload = json.loads(matches[0].read_text())
        h = payload.get("hourly") or {}
        idx = pd.to_datetime(h.get("time", []), utc=True)
        vals = np.array(h.get("cloud_cover", []), dtype=float)
        if len(idx) == 0:
            continue
        by_loc[name] = pd.Series(np.nan_to_num(vals, nan=50.0), index=idx)
        weights[name] = sw
    if not by_loc:
        raise FileNotFoundError(
            "No cached Open-Meteo cloud_cover responses found. "
            "Run studies/solar_clear_sky_submodel.py first."
        )
    common = None
    for s in by_loc.values():
        common = s.index if common is None else common.intersection(s.index)
    w_total = sum(weights.values())
    return sum(by_loc[n].reindex(common) * (weights[n] / w_total)
               for n in by_loc).rename("cloud_cover_w")


def _build_dataframe() -> tuple[pd.DataFrame, dict]:
    """Return aligned hourly df with actual, capacity, cloud, prediction
    columns + the loaded artifact dict."""
    artifact = json.loads(ARTIFACT_PATH.read_text())
    s_actual = _load_fingrid(248).rename("actual_mw")
    s_cap    = _load_fingrid(267).rename("capacity_mw")
    s_cap    = s_cap.reindex(s_actual.index).ffill().bfill()
    s_cloud  = _load_cloud_cover_weighted(artifact["sites"])
    df = pd.concat([s_actual, s_cap, s_cloud], axis=1).dropna()
    df["pred_mw"] = scs.predict_solar_mw(
        df.index.values, df["cloud_cover_w"].values, artifact)
    df["resid_mw"] = df["pred_mw"] - df["actual_mw"]
    df["ghi_w_m2"] = scs.predict_solar_mw(
        df.index.values, np.zeros(len(df)), artifact) - artifact["alpha"]
    df["ghi_w_m2"] /= max(artifact["K"], 1e-9)
    # Day / night flag for plotting filters
    df["is_daylight"] = df["ghi_w_m2"] > 5.0
    # Split mask matching the upstream 70/30 chronological split
    split_at = df.index[int(len(df) * 0.70)]
    df["split"] = np.where(df.index < split_at, "train", "test")
    return df, artifact


# ── Figure 1: seasonal tracking windows ────────────────────────────


SEASONAL_WINDOWS = [
    ("Winter (Jan)",  "2025-01-15", 14),
    ("Spring (Apr)",  "2025-04-10", 14),
    ("Summer (Jul)",  "2025-07-01", 14),
    ("Autumn (Oct)",  "2025-10-05", 14),
]


def fig_tracking(df: pd.DataFrame, artifact: dict, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=False)

    for ax, (label, start_str, days) in zip(axes.ravel(), SEASONAL_WINDOWS):
        start = pd.Timestamp(start_str, tz="UTC")
        end   = start + pd.Timedelta(days=days)
        win = df.loc[start:end]
        if win.empty:
            ax.set_title(f"{label} — no data in cache")
            continue

        ax.fill_between(win.index, 0, win["actual_mw"],
                        color="#cccccc", alpha=0.6, label="Actual")
        ax.plot(win.index, win["actual_mw"], color="#666666",
                lw=0.9, label="_nolegend_")
        ax.plot(win.index, win["pred_mw"], color="C0", lw=1.3,
                label="Deployed prediction")

        # Cloud cover overlay on a secondary axis
        ax2 = ax.twinx()
        ax2.plot(win.index, win["cloud_cover_w"], color="C3",
                 lw=0.6, alpha=0.5, label="Cloud cover (%)")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Cloud cover [%]", color="C3", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="C3", labelsize=8)
        ax2.grid(False)

        peak = float(win["actual_mw"].max())
        ax.set_ylim(0, max(50, peak * 1.1))
        ax.set_title(f"{label} 2025 — peak {peak:.0f} MW")
        ax.set_ylabel("PV production [MW]")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if ax is axes[0, 0]:
            ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Deployed solar sub-model — four seasonal two-week samples",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Figure 2: accuracy diagnostics ────────────────────────────────


def fig_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    test = df[df["split"] == "test"]
    test_day = test[test["is_daylight"]]

    # (a) Scatter density: predicted vs actual (test, daylight only)
    ax = axes[0, 0]
    a = test_day["actual_mw"].values
    p = test_day["pred_mw"].values
    hb = ax.hexbin(a, p, gridsize=55, mincnt=1, cmap="Blues", bins="log")
    lim = max(a.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=0.8, label="y = x (ideal)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Actual production [MW]")
    ax.set_ylabel("Deployed prediction [MW]")
    r = np.corrcoef(a, p)[0, 1] ** 2
    mae = float(np.mean(np.abs(p - a)))
    ax.set_title(f"Scatter density (test, daylight) — R²={r:.3f}, MAE={mae:.0f} MW")
    ax.legend(loc="upper left", fontsize=8)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log hours", fontsize=8); cb.ax.tick_params(labelsize=7)

    # (b) Residual histogram + KDE
    ax = axes[0, 1]
    resid = test_day["resid_mw"].values
    ax.hist(resid, bins=80, color="C0", alpha=0.6, edgecolor="white",
            density=True, label="Residual (pred − actual)")
    # Light overlay: gaussian with same mean/std
    mu, sd = resid.mean(), resid.std()
    xs = np.linspace(resid.min(), resid.max(), 200)
    gauss = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(xs, gauss, "k--", lw=1.0, label=f"N({mu:+.1f}, {sd:.1f}²)")
    ax.axvline(0, color="r", lw=0.8, alpha=0.7)
    ax.set_xlabel("Residual [MW]"); ax.set_ylabel("Density")
    ax.set_title(f"Residual distribution — mean {mu:+.1f}, σ {sd:.1f} MW")
    ax.legend(loc="upper right", fontsize=8)

    # (c) Mean error vs cloud cover (binned)
    ax = axes[1, 0]
    bins = np.arange(0, 101, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cc = test_day["cloud_cover_w"].values
    counts, _ = np.histogram(cc, bins=bins)
    mean_resid = np.array([
        resid[(cc >= bins[i]) & (cc < bins[i+1])].mean()
        if counts[i] > 0 else np.nan for i in range(len(bins) - 1)
    ])
    std_resid = np.array([
        resid[(cc >= bins[i]) & (cc < bins[i+1])].std()
        if counts[i] > 0 else np.nan for i in range(len(bins) - 1)
    ])
    ax.bar(bin_centers, mean_resid, width=8, yerr=std_resid,
           color="C2", alpha=0.7, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    for c, n in zip(bin_centers, counts):
        ax.annotate(f"{n}", (c, 0), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color="#444444")
    ax.set_xlabel("Cloud cover bin [%]")
    ax.set_ylabel("Mean residual [MW] (bars = ±1σ)")
    ax.set_title("Residual conditional on cloud cover")

    # (d) Mean error vs daylight GHI level (binned)
    ax = axes[1, 1]
    ghi = test_day["ghi_w_m2"].values
    g_bins = np.linspace(0, max(50, ghi.max()), 11)
    g_centers = (g_bins[:-1] + g_bins[1:]) / 2
    g_counts, _ = np.histogram(ghi, bins=g_bins)
    g_mean = np.array([
        resid[(ghi >= g_bins[i]) & (ghi < g_bins[i+1])].mean()
        if g_counts[i] > 0 else np.nan for i in range(len(g_bins) - 1)
    ])
    g_std = np.array([
        resid[(ghi >= g_bins[i]) & (ghi < g_bins[i+1])].std()
        if g_counts[i] > 0 else np.nan for i in range(len(g_bins) - 1)
    ])
    width = (g_bins[1] - g_bins[0]) * 0.85
    ax.bar(g_centers, g_mean, width=width, yerr=g_std, color="C1",
           alpha=0.7, capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Capacity-weighted clear-sky GHI [W/m²]")
    ax.set_ylabel("Mean residual [MW] (bars = ±1σ)")
    ax.set_title("Residual conditional on solar elevation (GHI)")

    fig.suptitle("Solar sub-model — accuracy diagnostics (test set, daylight only)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Figure 3: temporal accuracy + capacity drift ─────────────────


def fig_temporal(df: pd.DataFrame, artifact: dict, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    # Monthly aggregates
    monthly = df.groupby(pd.Grouper(freq="MS")).agg(
        actual_total=("actual_mw", "sum"),
        pred_total=("pred_mw", "sum"),
        mae=("resid_mw", lambda r: float(np.mean(np.abs(r)))),
        bias=("resid_mw", "mean"),
        capacity_eom=("capacity_mw", "last"),
    )
    train_end = df[df["split"] == "train"].index[-1]

    # (a) Monthly produced energy: actual vs predicted (GWh)
    ax = axes[0]
    actual_gwh = monthly["actual_total"] / 1000
    pred_gwh = monthly["pred_total"] / 1000
    x = monthly.index
    width = 18
    ax.bar(x - pd.Timedelta(days=width/2 - 1), actual_gwh, width=width,
           label="Actual (Fingrid 248)", color="#666666", alpha=0.85)
    ax.bar(x + pd.Timedelta(days=width/2 - 1), pred_gwh, width=width,
           label="Deployed prediction", color="C0", alpha=0.85)
    ax.axvline(train_end, color="r", lw=1.0, ls="--",
               label="Train / test split")
    ax.set_ylabel("Monthly produced energy [GWh]")
    ax.set_title("Monthly aggregate: actual vs deployed prediction")
    ax.legend(loc="upper left", fontsize=9)

    # (b) Monthly MAE + bias overlaid on a second axis
    ax = axes[1]
    ax.bar(monthly.index, monthly["mae"], width=18, color="C2",
           alpha=0.7, label="Monthly MAE")
    ax.set_ylabel("MAE [MW]", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, monthly["bias"], "C3-o", lw=1.0, ms=4,
             label="Monthly bias")
    ax2.axhline(0, color="C3", lw=0.5)
    ax2.set_ylabel("Bias [MW] (pred − actual)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)
    ax.axvline(train_end, color="r", lw=1.0, ls="--")
    ax.set_title("Monthly MAE and bias")

    # (c) End-of-month installed capacity + the K-baked reference
    ax = axes[2]
    ax.fill_between(monthly.index, 0, monthly["capacity_eom"],
                    color="C4", alpha=0.4)
    ax.plot(monthly.index, monthly["capacity_eom"], color="C4", lw=1.2,
            label="Fingrid 267 installed PV capacity")
    ax.axhline(artifact["capacity_ref_mw"], color="k", lw=1.0, ls="--",
               label=f"capacity_ref baked into artifact "
                     f"({artifact['capacity_ref_mw']:.0f} MW)")
    ax.axvline(train_end, color="r", lw=1.0, ls="--")
    ax.set_ylabel("Installed capacity [MW]")
    ax.set_xlabel("Date")
    ax.set_title("Installed FI PV capacity — drift since the artifact was fitted")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.suptitle("Solar sub-model — temporal accuracy and capacity drift",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    print("Loading cached data + artifact...", flush=True)
    df, artifact = _build_dataframe()
    print(f"  aligned: {len(df):,} hourly rows  "
          f"({df.index[0].date()} -> {df.index[-1].date()})", flush=True)

    print("[1/3] Rendering tracking figure...", flush=True)
    fig_tracking(df, artifact, FIGURES_DIR / "solar_tracking.png")

    print("[2/3] Rendering accuracy diagnostics figure...", flush=True)
    fig_accuracy(df, FIGURES_DIR / "solar_accuracy.png")

    print("[3/3] Rendering temporal-accuracy figure...", flush=True)
    fig_temporal(df, artifact, FIGURES_DIR / "solar_temporal.png")

    # Headline numbers for the console
    test = df[df["split"] == "test"]
    test_day = test[test["is_daylight"]]
    resid = test_day["resid_mw"].values
    print(f"\nTest (daylight) headline metrics under deployed model:")
    print(f"  rows: {len(test_day):,}  ({test_day.index[0].date()} -> "
          f"{test_day.index[-1].date()})")
    print(f"  R²:   {1 - np.var(resid) / np.var(test_day['actual_mw']):.3f}")
    print(f"  MAE:  {float(np.mean(np.abs(resid))):.2f} MW")
    print(f"  Bias: {float(np.mean(resid)):+.2f} MW")
    print(f"  σ:    {float(np.std(resid)):.2f} MW")

    print(f"\nFigures written to: {FIGURES_DIR}")
    for name in ("solar_tracking.png", "solar_accuracy.png",
                 "solar_temporal.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
