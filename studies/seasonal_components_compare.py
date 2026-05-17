"""v2.5.7/v2.5.8 visualization — before / after seasonal-component comparison.

For every weather input (cloud, wind, solar, temperature, ghi_cs) we
render the v2.5.5 (3.3 y, no smoothing) P_week against the current
v2.5.8 (8.3 y + stronger circular smoothing) P_week. v2.5.8 also adds
a single grid figure with all five panels for at-a-glance comparison.

Output:
  studies/results/figures/seasonal_compare_<NAME>.png   (per-input panels)
  studies/results/figures/seasonal_compare_all.png      (5-panel grid)
  studies/results/seasonal_components_compare.md  (auto-generated)
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

# Recent (v2.5.5) window — for the BEFORE side
SHORT_START = pd.Timestamp("2023-01-01", tz="UTC")
SHORT_END   = pd.Timestamp("2026-04-28", tz="UTC")


def _load_recent_weather() -> dict[str, pd.Series]:
    """Re-build the v2.5.5-era SHORT-window weather series from the
    existing parquet (wind/solar/temp) and the v2.5.3 cloud cache."""
    import yaml
    region_yaml = (REPO / "custom_components" / "spot_price_predictor"
                   / "data" / "finland.yaml")
    region = yaml.safe_load(region_yaml.read_text())
    sites = region["weather_source"]["locations"]

    out: dict[str, pd.Series] = {}
    # Wind / solar / temp from the recent parquet (already loaded by the
    # production pipeline in 2022-04 → 2026-04 form).
    wea = pd.read_parquet(REPO / "output" / "fi_weather.parquet")
    wea = wea.loc[SHORT_START:SHORT_END]
    out["wind"]  = wea["wind_speed_weighted"].rename("wind")
    out["solar"] = wea["solar_irradiance_weighted"].rename("solar")
    out["temp"]  = wea["temperature_weighted"].rename("temp")

    # Cloud from the v2.5.3 cache (3.3 y window).
    by_loc, weights = {}, {}
    for loc in sites:
        sw = float(loc.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        key = loc["name"].replace(" ", "_").replace("/", "_")
        matches = [p for p in CACHE_DIR.glob(f"openmeteo_cloud_{key}_*.json")
                   if "2023" in p.name]
        if not matches:
            continue
        payload = json.loads(matches[0].read_text())
        h = payload.get("hourly") or {}
        idx = pd.to_datetime(h.get("time", []), utc=True)
        vals = np.array(h.get("cloud_cover", []), dtype=float)
        by_loc[loc["name"]] = pd.Series(
            np.nan_to_num(vals, nan=50.0), index=idx)
        weights[loc["name"]] = sw
    if by_loc:
        common = None
        for s in by_loc.values():
            common = s.index if common is None else common.intersection(s.index)
        w_total = sum(weights.values())
        out["cloud"] = sum(
            by_loc[n].reindex(common) * (weights[n] / w_total)
            for n in by_loc).loc[SHORT_START:SHORT_END]
    return out


def _fit_short_window(series: pd.Series, depth: tuple[str, ...]) -> dict:
    """Replicate the v2.5.5 behaviour: short window, no smoothing."""
    ts = pd.DatetimeIndex(series.index, tz="UTC").values
    return sd.fit_components(series.values, ts, depth=depth)


UNITS = {
    "wind":  "m/s",
    "solar": "W/m²",
    "temp":  "°C",
    "cloud": "%",
}


def main() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    components_now = artifact["components"]

    print("Loading v2.5.5-era short-window weather series...", flush=True)
    recent = _load_recent_weather()
    if not recent:
        print("  ERROR: no weather data available; aborting", flush=True)
        return

    inputs = ["cloud", "wind", "solar", "temp"]
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Per-input figures + summary table ──────────────────────────
    summary_rows: list[tuple[str, float, float, int]] = []
    for name in inputs:
        if name not in recent or name not in components_now:
            print(f"  {name}: skipping (missing data)", flush=True)
            continue
        unit = UNITS.get(name, "")
        # Match the v2.5.8 depth so the comparison is depth-consistent
        # (otherwise P_week of a hour-then-week fit lives in the
        # residual mean while a P_week-only fit lives in the raw mean).
        depth_now = tuple(artifact["depths"][name])
        comp_before = _fit_short_window(recent[name], depth=depth_now)
        comp_after  = components_now[name]
        p_before = np.array(comp_before["P_week"])
        p_after  = np.array(comp_after["P_week"])
        # Smoothing window used per DEFAULT_SMOOTH (in build_seasonal_components)
        smooth_meta = None
        for k, v in [(7, 'cloud'), (7, 'wind'), (7, 'solar'), (9, 'temp')]:
            if v == name:
                smooth_meta = k
                break
        smooth_meta = smooth_meta or 7

        fig, ax = plt.subplots(figsize=(13, 5.5))
        w = np.arange(53)
        ax.plot(w, p_before, "C3-o", lw=1.0, ms=4, alpha=0.7,
                label=f"v2.5.5 — 3.3 y window, no smoothing  "
                      f"(σ={p_before.std():.2f} {unit})")
        ax.plot(w, p_after, "C0-o", lw=1.8, ms=5,
                label=f"v2.5.8 — 8.3 y window + circular smooth "
                      f"{smooth_meta} weeks  (σ={p_after.std():.2f} {unit})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Week of year")
        ax.set_ylabel(f"P_week deviation from mean {name} [{unit}]")
        ax.set_title(f"{name.upper()} P_week — before / after extended "
                     f"window + smoothing")
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.set_xlim(-0.5, 52.5)
        fig.tight_layout()
        out_path = FIGURES_DIR / f"seasonal_compare_{name}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)

        reduction = 100.0 * (1.0 - p_after.std() / p_before.std()) \
            if p_before.std() > 0 else float("nan")
        summary_rows.append((name, p_before.std(), p_after.std(), reduction))
        print(f"  {name:6s}  σ {p_before.std():7.3f} → {p_after.std():7.3f}"
              f"  ({reduction:+5.1f}% noise change) → "
              f"{out_path.name}", flush=True)

    # ── Combined 4-panel grid figure ───────────────────────────────
    print("\nRendering combined 4-panel grid...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, name in zip(axes.ravel(), inputs):
        if name not in recent or name not in components_now:
            ax.set_title(f"{name} — no data")
            continue
        unit = UNITS.get(name, "")
        # Match the v2.5.8 depth so the comparison is depth-consistent
        # (otherwise P_week of a hour-then-week fit lives in the
        # residual mean while a P_week-only fit lives in the raw mean).
        depth_now = tuple(artifact["depths"][name])
        comp_before = _fit_short_window(recent[name], depth=depth_now)
        p_before = np.array(comp_before["P_week"])
        p_after  = np.array(components_now[name]["P_week"])
        w = np.arange(53)
        ax.plot(w, p_before, "C3-o", lw=1.0, ms=3, alpha=0.65,
                label=f"v2.5.5 raw  σ={p_before.std():.2f}")
        ax.plot(w, p_after, "C0-o", lw=1.8, ms=4,
                label=f"v2.5.8 smoothed  σ={p_after.std():.2f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Week of year")
        ax.set_ylabel(f"deviation [{unit}]")
        ax.set_title(f"{name.upper()}")
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
        ax.set_xlim(-0.5, 52.5)
    fig.suptitle("Weather P_week — v2.5.5 raw vs v2.5.8 (8.3 y + smoothing)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonal_compare_all.png", dpi=110)
    plt.close(fig)

    # ── Markdown summary ───────────────────────────────────────────
    md = RESULTS_DIR / "seasonal_components_compare.md"
    lines = [
        "# Seasonal components — v2.5.5 vs v2.5.8 comparison (all weather inputs)",
        "",
        "User observations 2026-05-17:",
        "",
        "1. *\"seasonal decomposition for Cloud seems to provide quite noisy*",
        "   *estimate for P_week ... if the averaging window would be longer*",
        "   *the model should be smoother. This model does not have regional*",
        "   *changes so if longer dataset is available I would prefer to*",
        "   *improve the model as this version will modulate residual noise.\"*",
        "2. *\"also other weather related seasonal estimates could benefit*",
        "   *from circular averaging as, unlike consumption that could*",
        "   *relate to holiday patterns, wind or solar is not expected to*",
        "   *contain this amount of seasonal noise.\"*",
        "",
        "v2.5.7 extended the fit window for weather inputs to 8.3 y and",
        "smoothed cloud aggressively (7-bin). v2.5.8 extends the same",
        "treatment to wind / solar / temp with stronger smoothing windows",
        "consistent with each input's physical smoothness.",
        "",
        "## DEFAULT_SMOOTH (v2.5.8)",
        "",
        "| Input | P_week smoothing | Rationale |",
        "|---|---:|---|",
        "| `wind` | 7 weeks | Annual circulation pattern is smooth |",
        "| `solar` | 7 weeks | Annual day-length cycle is smooth |",
        "| `temp` | 9 weeks | Annual temperature cycle is the smoothest input |",
        "| `cloud` | 7 weeks | Unchanged from v2.5.7 (already adequate) |",
        "| `ghi_cs` | (none) | Deterministic clear-sky has zero noise |",
        "",
        "## Per-input noise-reduction summary",
        "",
        "| Input | σ(P_week) v2.5.5 | σ(P_week) v2.5.8 | Noise reduction |",
        "|---|---:|---:|---:|",
    ]
    for name, sd_b, sd_a, red in summary_rows:
        lines.append(
            f"| `{name}` | {sd_b:.3f} {UNITS.get(name, '')} | "
            f"{sd_a:.3f} {UNITS.get(name, '')} | "
            f"**{red:+.1f} %** |"
        )

    lines += [
        "",
        "## Combined comparison figure",
        "",
        "![All weather inputs](figures/seasonal_compare_all.png)",
        "",
        "## Per-input panels",
        "",
    ]
    for name, *_ in summary_rows:
        lines.append(f"### {name.upper()}")
        lines.append("")
        lines.append(f"![{name} P_week comparison]"
                     f"(figures/seasonal_compare_{name}.png)")
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- The bin-to-bin oscillations in the v2.5.5 (red) curves are",
        "  sampling noise: a single year of bad weather in week 8 inflates",
        "  that bin's mean while week 9 might happen to be calmer.",
        "  Physically wind/solar/temp cannot differ meaningfully between",
        "  consecutive calendar weeks; the smoothed v2.5.8 (blue) curves",
        "  reflect the underlying climatology.",
        "- The variance reduction reported by the v2.5.4 audit was therefore",
        "  *partly* sampling noise being captured as seasonal signal. The",
        "  smoothed v2.5.8 components attribute that noise correctly to the",
        "  stochastic residual `Y_X`, which is what the v2.5.6 hedge-gated",
        "  sweep operates on.",
        "- Prices (FI / SE3 / SE1 / EE) are kept un-smoothed: their week-to-",
        "  week variation includes real signal from scheduled outages, hydro",
        "  releases, holiday demand, and the like. Smoothing those would",
        "  hide real economic structure.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/build_seasonal_components.py   # refit + ship artifact",
        "python studies/seasonal_components_compare.py # render comparison figures",
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")
    print(f"Grid figure: {FIGURES_DIR / 'seasonal_compare_all.png'}")


if __name__ == "__main__":
    main()
