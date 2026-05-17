"""Render per-input figures from the SHIPPED v2.5.8 artifact, so what
you see matches what's deployed.

The v2.5.4 `per_sensor_seasonality_audit.py` figures were built from a
self-contained `fit_seasonal_hdw` call with no smoothing and the recent
window — i.e. they show the v2.5.4 decomposition, not the v2.5.8 one
that the integration actually ships.

This script overwrites `studies/results/figures/per_sensor_components_*.png`
so every figure reflects the components currently sitting in
`data/seasonal_components_default.json`.

Output:
  studies/results/figures/per_sensor_components_<NAME>.png  (overwritten)
  studies/results/figures/seasonal_artifact_overview.png    (new — all
       inputs on one page)
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

import matplotlib.pyplot as plt  # noqa: E402

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402
from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    PRICE_WINDOW_START, WEATHER_WINDOW_START, WINDOW_END,
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

UNITS = {
    "fi": "EUR/MWh", "se3": "EUR/MWh", "se1": "EUR/MWh", "ee": "EUR/MWh",
    "wind": "m/s", "solar": "W/m²", "temp": "°C", "cloud": "%",
    "ghi_cs": "W/m²",
}

EVAL_START = pd.Timestamp("2023-01-01", tz="UTC")
EVAL_END   = pd.Timestamp("2026-04-28", tz="UTC")


# ── Load inputs aligned with shipped artifact's data range ─────────


def _load_all_inputs() -> dict[str, pd.Series]:
    """Build the same per-input series the artifact was fit on.

    Prices use the 2023+ parquets; weather uses the 8.3 y Open-Meteo
    cache. Each series is returned on its OWN time grid (don't intersect
    yet — the artifact uses per-input windows)."""
    import yaml
    region_yaml = (REPO / "custom_components" / "spot_price_predictor"
                   / "data" / "finland.yaml")
    region = yaml.safe_load(region_yaml.read_text())
    sites = region["weather_source"]["locations"]

    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)

    # Compute clear-sky GHI on the wind/solar/temp/cloud index
    if wea:
        ws_idx = None
        for s in wea.values():
            ws_idx = s.index if ws_idx is None else \
                ws_idx.intersection(s.index)
        artifact_solar = json.loads((REPO / "custom_components"
            / "spot_price_predictor" / "data"
            / "solar_submodel_default.json").read_text())
        ts_np = ws_idx.values
        ghi = np.zeros(len(ws_idx), dtype=float)
        w_total = 0.0
        for site in artifact_solar["sites"]:
            sw = float(site.get("solar_weight", 0.0))
            if sw <= 0:
                continue
            ghi += sw * scs.clear_sky_series(
                ts_np, lat_deg=float(site["lat"]),
                lon_deg=float(site["lon"]),
                model=artifact_solar["clear_sky_model"])
            w_total += sw
        if w_total > 0:
            ghi /= w_total
        inputs["ghi_cs"] = pd.Series(ghi, index=ws_idx, name="ghi_cs")

    return inputs


def _acf(y: np.ndarray, lags: int = 73) -> np.ndarray:
    y = y - y.mean()
    var = float(np.var(y))
    out = np.empty(lags, dtype=float)
    n = len(y)
    out[0] = 1.0
    for k in range(1, lags):
        if n - k <= 1 or var <= 0:
            out[k] = 0.0
        else:
            out[k] = float(np.dot(y[:-k], y[k:])) / ((n - k) * var)
    return out


def _ljung_box(y: np.ndarray, max_lag: int = 24) -> float:
    n = len(y)
    if n <= max_lag + 1:
        return 0.0
    y_c = y - y.mean()
    var = float(np.var(y_c))
    if var <= 0:
        return 0.0
    Q = 0.0
    for k in range(1, max_lag + 1):
        rho = float(np.dot(y_c[:-k], y_c[k:])) / ((n - k) * var)
        Q += rho ** 2 / (n - k)
    return n * (n + 2) * Q


# ── Per-input figure ────────────────────────────────────────────────


def render_input(name: str, series: pd.Series, artifact: dict,
                  out_path: Path) -> dict:
    """Render the 4-panel figure for one input using SHIPPED components.

    Returns diagnostic stats (residual std, Ljung-Box, etc.).
    """
    components = artifact["components"][name]
    depths = artifact["depths"][name]
    unit = UNITS.get(name, "")

    # Drop NaN rows so the variance / Ljung-Box are not contaminated by
    # missing data (e.g. EE has scattered NaN values in the parquet).
    series = series.dropna()
    mu = float(series.mean())
    raw_std = float(series.std())

    ts_np = pd.DatetimeIndex(series.index, tz="UTC").values
    Y = sd.compute_residual(series.values, ts_np, components)
    res_std = float(np.std(Y))
    var_red = 1.0 - (res_std ** 2 / raw_std ** 2) if raw_std > 0 else 0.0
    lb = _ljung_box(Y)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) P_hour profile
    ax = axes[0, 0]
    if "P_hour" in components:
        p = np.array(components["P_hour"])
        ax.bar(range(24), p - p.mean(), color="C0")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel(f"Deviation from mean [{unit}]")
        ax.set_title(f"P_hour (length {len(p)}, σ {p.std():.2f})")
    else:
        ax.text(0.5, 0.5, "P_hour DROPPED\n(per v2.5.4 audit)",
                ha="center", va="center", fontsize=12, color="grey")
        ax.set_title("P_hour")
        ax.set_xticks([]); ax.set_yticks([])

    # (b) P_day profile — always shown as deviation from mean
    ax = axes[0, 1]
    if "P_day" in components:
        p = np.array(components["P_day"])
        ax.bar(range(7), p - p.mean(), color="C1")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        ax.set_ylabel(f"Deviation from mean [{unit}]")
        ax.set_title(f"P_day (length {len(p)}, σ {p.std():.2f})")
    else:
        ax.text(0.5, 0.5, "P_day DROPPED\n(per v2.5.4 audit)",
                ha="center", va="center", fontsize=12, color="grey")
        ax.set_title("P_day")
        ax.set_xticks([]); ax.set_yticks([])

    # (c) P_week profile — always shown as deviation from mean so the
    # P_hour-then-week vs P_week-only depths look comparable on the page.
    ax = axes[1, 0]
    if "P_week" in components:
        p = np.array(components["P_week"])
        ax.bar(range(53), p - p.mean(), color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Week of year")
        ax.set_ylabel(f"Deviation from mean [{unit}]")
        sm_label = ""
        # Try to label the smoothing if known
        from build_seasonal_components import DEFAULT_SMOOTH
        smooth_meta = DEFAULT_SMOOTH.get(name, {}).get("P_week")
        if smooth_meta and smooth_meta > 1:
            sm_label = f", smoothed {smooth_meta}-bin"
        ax.set_title(f"P_week (length {len(p)}, σ {p.std():.2f}{sm_label})")
    else:
        ax.text(0.5, 0.5, "P_week DROPPED",
                ha="center", va="center", fontsize=12, color="grey")
        ax.set_title("P_week")

    # (d) Residual ACF
    ax = axes[1, 1]
    acf = _acf(Y, lags=73)
    lags = np.arange(0, 73)
    ax.vlines(lags, 0, acf, color="C3")
    ax.scatter(lags, acf, color="C3", s=10)
    ax.axhline(0, color="k", lw=0.5)
    n = len(Y)
    bound = 1.96 / np.sqrt(n)
    ax.axhline( bound, color="grey", ls="--", lw=0.7,
                label=f"±1.96/√n = ±{bound:.3f}")
    ax.axhline(-bound, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("Lag [hours]")
    ax.set_ylabel("Y_t ACF")
    ax.set_title(f"Residual ACF  (LB Q24={lb:.0f})")
    ax.legend(loc="upper right", fontsize=8)

    fit_window = artifact.get("per_input_fit_window", {}).get(name, ("?", "?"))
    depth_label = " + ".join(depths) if depths else "(none)"
    fig.suptitle(
        f"{name} (deployed v{artifact.get('version','?')}) — depth {depth_label}\n"
        f"window {fit_window[0][:10]}…{fit_window[1][:10]} | "
        f"mean {mu:.2f} {unit} | σ_raw {raw_std:.2f} → σ_Y {res_std:.2f} | "
        f"var_red {100*var_red:.1f}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return {
        "name": name, "var_red": var_red, "ljung_box": lb,
        "raw_std": raw_std, "res_std": res_std, "mean": mu,
    }


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    print(f"Artifact v{artifact.get('version','?')} loaded "
          f"({len(artifact['components'])} inputs)", flush=True)

    print("\nLoading raw input series...", flush=True)
    inputs = _load_all_inputs()
    print(f"  {list(inputs.keys())}", flush=True)

    print("\nRendering per-input figures from shipped components...",
          flush=True)
    stats: list[dict] = []
    for name in artifact["components"].keys():
        if name not in inputs:
            print(f"  {name:8s}: skipped (no data loaded)", flush=True)
            continue
        out = FIGURES_DIR / f"per_sensor_components_{name}.png"
        s = render_input(name, inputs[name], artifact, out)
        stats.append(s)
        print(f"  {name:8s}  var_red {100*s['var_red']:5.1f}%  "
              f"LB {s['ljung_box']:.0f}  → {out.name}", flush=True)

    # Overview figure: variance shares (computed from artifact stats)
    print("\nRendering overview figure...", flush=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    names = [s["name"] for s in stats]
    raws  = np.array([s["raw_std"] for s in stats])
    ress  = np.array([s["res_std"] for s in stats])
    x = np.arange(len(names))
    ax.bar(x - 0.18, raws, width=0.36, color="C7", alpha=0.7,
           label="Raw input σ")
    ax.bar(x + 0.18, ress, width=0.36, color="C0",
           label="Residual σ (after subtracting seasonal)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("σ in native units (LOG y-axis — mixed units)")
    ax.set_yscale("log")
    ax.set_title(f"Deployed v{artifact.get('version','?')} seasonal "
                 f"decomposition — raw vs residual σ per input")
    for i, s in enumerate(stats):
        ax.annotate(f"-{100 * (1 - s['res_std'] / s['raw_std']):.0f}%",
                    (i + 0.18, s["res_std"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonal_artifact_overview.png", dpi=110)
    plt.close(fig)
    print(f"  written: seasonal_artifact_overview.png", flush=True)


if __name__ == "__main__":
    main()
