"""Sanity-check the input → seasonal → residual → prediction pipeline.

User asked 2026-05-17: *"Can you provide me normalized input and outputs
for visual inspection because I am not sure if all parameters are
properly mapped to the model?"*

For every candidate input to the FI Ridge, this script renders three
parallel z-score-normalized panels over a representative test window:

   raw X(t)
   seasonal reconstruction = Σ P_components(X)
   residual Y_X = X − seasonal

so the user can visually verify that:

1. Each input is read from the correct parquet / cache.
2. The artifact's per-input depth (P_hour / P_day / P_week) is applied
   correctly — seasonal panel should reproduce raw's main shape.
3. The residual is what's left after seasonal — zero-mean, smaller-σ.
4. The final FI prediction (seasonal + Ridge-on-residual) matches actual.

Output:
  studies/results/figures/input_output_mapping_zscore.png  (grid panel)
  studies/results/figures/input_output_mapping_native.png  (same plot in
        native units — sanity for absolute values)
  studies/results/figures/fi_prediction_decomposition.png  (final
        prediction split into seasonal vs residual contribution)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))
sys.path.insert(0, str(REPO / "studies"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402
from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE_DIR  = REPO / "studies" / ".cache"
RESULTS_DIR = REPO / "studies" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = (REPO / "custom_components" / "spot_price_predictor"
                 / "data" / "seasonal_components_default.json")
SOLAR_ART = (REPO / "custom_components" / "spot_price_predictor"
             / "data" / "solar_submodel_default.json")

UNITS = {
    "fi": "EUR/MWh", "se3": "EUR/MWh", "se1": "EUR/MWh", "ee": "EUR/MWh",
    "wind": "m/s", "solar": "W/m²", "temp": "°C", "cloud": "%",
    "ghi_cs": "W/m²",
}

# Two-week window for the time-series panels — late summer 2025 has a
# good mix of cloud-pass volatility, weekend transitions, and price
# spikes.
SAMPLE_START = pd.Timestamp("2025-08-04", tz="UTC")
SAMPLE_END   = pd.Timestamp("2025-08-18", tz="UTC")


def _build_aligned() -> tuple[pd.DataFrame, dict]:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    solar_art = json.loads(SOLAR_ART.read_text())

    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())

    import yaml
    region = yaml.safe_load((REPO / "custom_components"
        / "spot_price_predictor" / "data" / "finland.yaml").read_text())
    sites = region["weather_source"]["locations"]
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)

    # Clear-sky GHI on the weather grid
    if wea:
        ws_idx = None
        for s in wea.values():
            ws_idx = s.index if ws_idx is None else ws_idx.intersection(s.index)
        ts_np = ws_idx.values
        ghi = np.zeros(len(ws_idx), dtype=float)
        w_total = 0.0
        for site in solar_art["sites"]:
            sw = float(site.get("solar_weight", 0.0))
            if sw <= 0:
                continue
            ghi += sw * scs.clear_sky_series(
                ts_np, lat_deg=float(site["lat"]),
                lon_deg=float(site["lon"]),
                model=solar_art["clear_sky_model"])
            w_total += sw
        if w_total > 0:
            ghi /= w_total
        inputs["ghi_cs"] = pd.Series(ghi, index=ws_idx, name="ghi_cs")

    # Intersect on a common hourly grid
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common).dropna() for k, s in inputs.items()}
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common) for k, s in inputs.items()}

    df = pd.concat(inputs.values(), axis=1)
    df.columns = list(inputs.keys())

    # Apply v2.5.8 deployed seasonal decomposition to every input
    ts_np = pd.DatetimeIndex(common, tz="UTC").values
    for name in df.columns:
        if name not in artifact["components"]:
            continue
        components = artifact["components"][name]
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, components)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    return df, artifact


# ── Figure 1: per-input raw / seasonal / residual on z-score axes ──


def fig_zscore_grid(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    inputs_to_show = ["fi", "se3", "se1", "ee",
                      "wind", "solar", "temp", "cloud"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)

    sample = df.loc[SAMPLE_START:SAMPLE_END]
    for ax, name in zip(axes.ravel(), inputs_to_show):
        if name not in df.columns:
            ax.set_title(f"{name} — not in artifact")
            continue
        raw_full = df[name].dropna()
        # z-score on the full window so the figure scale is comparable
        mu, sd_ = float(raw_full.mean()), float(raw_full.std())
        if sd_ <= 0:
            ax.set_title(f"{name} — degenerate (σ=0)")
            continue
        z_raw  = (sample[name]                - mu) / sd_
        z_seas = (sample[f"seasonal_{name}"]  - mu) / sd_
        z_res  =  sample[f"Y_{name}"]               / sd_
        ax.plot(sample.index, z_raw,  "k-",  lw=0.9, alpha=0.85,
                label="raw X (z)")
        ax.plot(sample.index, z_seas, "C2-", lw=1.4,
                label="seasonal (z)")
        ax.plot(sample.index, z_res,  "C0-", lw=0.7, alpha=0.7,
                label="residual Y (z)")
        ax.axhline(0, color="grey", lw=0.4)
        ax.set_ylim(-4, 4)
        ax.set_title(f"{name} (μ={mu:.2f} {UNITS.get(name,'')}, "
                     f"σ={sd_:.2f})")
        ax.set_ylabel("z-score")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if ax is axes[0, 0]:
            ax.legend(loc="upper right", fontsize=8, ncol=3,
                      framealpha=0.95)

    fig.suptitle(
        f"Input → seasonal → residual mapping (z-score) — "
        f"sample {SAMPLE_START.date()} to {SAMPLE_END.date()}\n"
        f"raw (black), seasonal reconstruction (green), residual Y (blue)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Figure 2: native units version (same data, no z-score) ─────────


def fig_native_grid(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    inputs_to_show = ["fi", "se3", "se1", "ee",
                      "wind", "solar", "temp", "cloud"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), sharex=True)
    sample = df.loc[SAMPLE_START:SAMPLE_END]
    for ax, name in zip(axes.ravel(), inputs_to_show):
        if name not in df.columns:
            ax.set_title(f"{name} — not in artifact"); continue
        ax.plot(sample.index, sample[name], "k-", lw=0.9, alpha=0.85,
                label="raw X")
        ax.plot(sample.index, sample[f"seasonal_{name}"], "C2-",
                lw=1.4, label="seasonal")
        ax.plot(sample.index, sample[f"Y_{name}"], "C0-", lw=0.7,
                alpha=0.7, label="residual Y")
        ax.axhline(0, color="grey", lw=0.4)
        unit = UNITS.get(name, "")
        ax.set_title(f"{name} [{unit}]")
        ax.set_ylabel(unit)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if ax is axes[0, 0]:
            ax.legend(loc="upper right", fontsize=8, ncol=3,
                      framealpha=0.95)
    fig.suptitle(
        f"Input → seasonal → residual mapping (native units) — "
        f"sample {SAMPLE_START.date()} to {SAMPLE_END.date()}\n"
        f"raw (black), seasonal reconstruction (green), residual Y (blue)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Figure 3: FI prediction decomposition (seasonal + Ridge residual)──


def fig_fi_prediction_decomposition(df: pd.DataFrame, out_path: Path) -> None:
    """Fit the v2.5.6 winner (Y_fi_lag168 + is_workday) so the user can
    see exactly what the deployed pipeline would produce."""
    plt.style.use("seaborn-v0_8-whitegrid")
    df = df.copy()
    df["Y_fi_lag168"] = df["Y_fi"].shift(168)
    df["is_workday"] = (df.index.weekday < 5).astype(float)
    df = df.dropna(subset=["Y_fi", "Y_fi_lag168", "is_workday"])

    # Chronological 55/45 train/test (mirror v2.5.6)
    split = int(len(df) * 0.55)
    train = df.iloc[:split]
    X_train = np.column_stack([
        np.ones(len(train)), train["Y_fi_lag168"], train["is_workday"]])
    y_train = train["Y_fi"].values
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_all = np.column_stack([
        np.ones(len(df)), df["Y_fi_lag168"], df["is_workday"]])
    ridge_pred = X_all @ coef
    df["fi_pred"] = df["seasonal_fi"].values + ridge_pred
    df["ridge_contribution"] = ridge_pred

    sample = df.loc[SAMPLE_START:SAMPLE_END]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(sample.index, sample["fi"],          "k-",  lw=1.3,
            label="actual FI price")
    ax.plot(sample.index, sample["fi_pred"],     "C0-", lw=1.3,
            label="predicted FI price = seasonal + Ridge")
    ax.plot(sample.index, sample["seasonal_fi"], "C2-", lw=1.0, alpha=0.7,
            label="Layer 1: seasonal_fi only")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("FI price — actual vs predicted (v2.5.6 winner)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    ax = axes[1]
    ax.fill_between(sample.index, 0, sample["ridge_contribution"],
                     color="C0", alpha=0.4)
    ax.plot(sample.index, sample["ridge_contribution"], "C0-", lw=1.0,
            label="Layer 2 Ridge contribution (= β·Y_fi_lag168 + γ·is_workday)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("EUR/MWh added on top of seasonal")
    ax.set_title("Ridge residual contribution (what Layer 2 adds to Layer 1)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    test = df.iloc[split:]
    mae = float(np.mean(np.abs(test["fi_pred"] - test["fi"])))
    r2 = 1.0 - float(np.var(test["fi_pred"] - test["fi"])) / float(np.var(test["fi"]))
    fig.suptitle(
        f"FI prediction decomposition — Ridge coefficients: "
        f"intercept={coef[0]:+.2f}, β_lag168={coef[1]:+.3f}, "
        f"γ_workday={coef[2]:+.2f}  |  test MAE={mae:.2f} EUR/MWh, "
        f"R²={r2:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return mae, r2, coef.tolist()


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Building aligned input/output dataframe...", flush=True)
    df, artifact = _build_aligned()
    print(f"  {len(df):,} hourly rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})", flush=True)
    print(f"  columns: {list(df.columns)}", flush=True)

    print(f"\nRendering z-score mapping figure "
          f"({SAMPLE_START.date()} → {SAMPLE_END.date()})...", flush=True)
    fig_zscore_grid(df, FIGURES_DIR / "input_output_mapping_zscore.png")

    print("Rendering native-units mapping figure...", flush=True)
    fig_native_grid(df, FIGURES_DIR / "input_output_mapping_native.png")

    print("Rendering FI prediction decomposition figure...", flush=True)
    mae, r2, coef = fig_fi_prediction_decomposition(
        df, FIGURES_DIR / "fi_prediction_decomposition.png")
    print(f"  Ridge coefficients: intercept={coef[0]:+.3f}, "
          f"β_lag168={coef[1]:+.3f}, γ_workday={coef[2]:+.3f}",
          flush=True)
    print(f"  Test MAE = {mae:.2f} EUR/MWh, R² = {r2:.3f}", flush=True)

    # Sanity-check that each X reconstructs as seasonal + residual
    print("\nVerifying X == seasonal + residual (per input)...", flush=True)
    for name in ["fi", "se3", "se1", "ee", "wind", "solar",
                 "temp", "cloud", "ghi_cs"]:
        if name not in df.columns:
            continue
        recon = df[f"seasonal_{name}"] + df[f"Y_{name}"]
        max_err = float(np.max(np.abs(recon - df[name])))
        flag = "OK" if max_err < 1e-9 else f"MISMATCH max |Δ|={max_err:.3e}"
        print(f"  {name:8s}  {flag}", flush=True)

    # Final stats per input
    print("\nPer-input stats (full window):", flush=True)
    print(f"  {'name':8s}  {'mean':>10s}  {'σ raw':>8s}  "
          f"{'σ Y':>8s}  {'var_red':>8s}")
    for name in ["fi", "se3", "se1", "ee", "wind", "solar",
                 "temp", "cloud", "ghi_cs"]:
        if name not in df.columns:
            continue
        raw_std = float(df[name].std())
        res_std = float(df[f"Y_{name}"].std())
        vr = 1.0 - (res_std / raw_std) ** 2 if raw_std > 0 else 0.0
        print(f"  {name:8s}  {float(df[name].mean()):10.2f}  "
              f"{raw_std:8.2f}  {res_std:8.2f}  {100*vr:7.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
