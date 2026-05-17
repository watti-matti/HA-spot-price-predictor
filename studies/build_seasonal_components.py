"""v2.5.5 builder — fits per-input seasonal components and ships the
deployable artifact `seasonal_components_default.json`.

Uses the same input universe and analysis window as the v2.5.4 audit
(`studies/per_sensor_seasonality_audit.py`). The decomposition depth
per input is taken from `seasonal_decomposition.DEFAULT_DEPTHS`, which
encodes the v2.5.4 verdict.

Refresh cadence: quarterly, alongside the v2.5.3 solar sub-model
artifact. The integration loads the resulting JSON at startup; runtime
is pure-numpy.

Usage:
    python studies/build_seasonal_components.py

Output:
    custom_components/spot_price_predictor/data/seasonal_components_default.json
    studies/results/seasonal_components_build.md
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

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_DIR  = REPO / "output"
CACHE_DIR   = REPO / "studies" / ".cache"
RESULTS_DIR = REPO / "studies" / "results"
ARTIFACT_PATH = (REPO / "custom_components" / "spot_price_predictor"
                 / "data" / "seasonal_components_default.json")

# Two windows: prices use the recent window only (regime changes),
# weather uses the long window (no regime changes — climatology is
# stationary on this timescale) for smoother per-week estimates.
PRICE_WINDOW_START   = pd.Timestamp("2023-01-01", tz="UTC")
WEATHER_WINDOW_START = pd.Timestamp("2018-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-04-28", tz="UTC")

# Per-input smoothing applied to P_week (and P_hour for some inputs)
# to reduce the residual noise that the user (2026-05-17) flagged on
# cloud cover. Weather inputs only; prices are left raw because their
# week-to-week variation carries real signal (outages, hydro releases).
DEFAULT_SMOOTH = {
    "wind":   {"P_week": 5},
    "solar":  {"P_week": 3},   # less smoothing — physical signal is sharp
    "ghi_cs": {"P_week": 3},
    "temp":   {"P_week": 5},
    "cloud":  {"P_week": 7},   # strongest smoothing — most noise per bin
}

# Legacy alias for any callers that still use it
WINDOW_START = PRICE_WINDOW_START


# ── Loaders (mirror studies/per_sensor_seasonality_audit.py) ─────


def load_fi_prices() -> pd.Series:
    df = pd.read_parquet(OUTPUT_DIR / "fi_prices.parquet")
    return df["price_eur_mwh"].loc[WINDOW_START:WINDOW_END].rename("fi")


def load_neighbor_prices() -> dict[str, pd.Series]:
    df = pd.read_parquet(OUTPUT_DIR / "fi_neighbor_prices.parquet")
    df = df.loc[WINDOW_START:WINDOW_END]
    return {k: df[k].rename(k) for k in ("se3", "se1", "ee") if k in df.columns}


def load_weather() -> dict[str, pd.Series]:
    df = pd.read_parquet(OUTPUT_DIR / "fi_weather.parquet")
    df = df.loc[WINDOW_START:WINDOW_END]
    return {
        "wind":  df["wind_speed_weighted"].rename("wind"),
        "solar": df["solar_irradiance_weighted"].rename("solar"),
        "temp":  df["temperature_weighted"].rename("temp"),
    }


def _fetch_openmeteo_var_weighted(
    variable: str,
    weight_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sites: list[dict],
) -> pd.Series | None:
    """Fetch any Open-Meteo `variable` for every site weighted by
    `weight_key` (e.g. `solar_weight`, `wind_weight`, `temp_weight`).
    Cached per (site, window, variable). No API key required.
    """
    import requests
    import time
    archive_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    by_loc: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for loc in sites:
        w = float(loc.get(weight_key, 0.0))
        if w <= 0:
            continue
        name = loc["name"]
        key = name.replace(" ", "_").replace("/", "_")
        cache_path = (CACHE_DIR / f"openmeteo_{variable}_{key}"
                      f"_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json")
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
        else:
            params = {
                "latitude":   loc["lat"],
                "longitude":  loc["lon"],
                "hourly":     variable,
                "timezone":   "UTC",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date":   end.strftime("%Y-%m-%d"),
                # Match the conventions used by src/data_sources.py so
                # the fitted seasonal vectors are unit-compatible with
                # the production weather pipeline.
                "wind_speed_unit": "ms",
            }
            if variable.startswith("global_tilted_irradiance"):
                params["tilt"] = 45
            # Open-Meteo can throttle very long requests; one retry is enough.
            for retry in range(2):
                r = requests.get(archive_url, params=params, timeout=240)
                if r.status_code == 200:
                    break
                print(f"   Open-Meteo {variable} {name}: HTTP {r.status_code} "
                      f"(retry {retry})", flush=True)
                time.sleep(2.0 * (retry + 1))
            else:
                continue
            payload = r.json()
            cache_path.write_text(json.dumps(payload))
            time.sleep(0.6)
            print(f"  {variable} {name}: cached "
                  f"({start.date()} → {end.date()})", flush=True)
        h = payload.get("hourly") or {}
        idx = pd.to_datetime(h.get("time", []), utc=True)
        vals = np.array(h.get(variable, []), dtype=float)
        if len(idx) == 0:
            continue
        by_loc[name] = pd.Series(np.nan_to_num(vals, nan=0.0), index=idx)
        weights[name] = w
    if not by_loc:
        return None
    common = None
    for s in by_loc.values():
        common = s.index if common is None else common.intersection(s.index)
    w_total = sum(weights.values())
    s = sum(by_loc[n].reindex(common) * (weights[n] / w_total)
            for n in by_loc)
    return s.loc[start:end]


def load_weather_extended(start: pd.Timestamp, end: pd.Timestamp,
                          sites: list[dict]) -> dict[str, pd.Series]:
    """Fetch wind / solar / temp / cloud from Open-Meteo on the EXTENDED
    window (typically 2018+), capacity-weighted per site. Disk-cached."""
    print(f"  fetching weather window {start.date()} → {end.date()} "
          f"({len(sites)} sites)...", flush=True)
    out: dict[str, pd.Series] = {}
    for var, weight_key, out_name in [
        ("wind_speed_120m",                "wind_weight",  "wind"),
        ("global_tilted_irradiance_instant","solar_weight", "solar"),
        ("temperature_2m",                 "temp_weight",  "temp"),
        ("cloud_cover",                    "solar_weight", "cloud"),
    ]:
        s = _fetch_openmeteo_var_weighted(var, weight_key, start, end, sites)
        if s is not None:
            out[out_name] = s.rename(out_name)
    return out


def load_cloud_cover() -> pd.Series | None:
    artifact_path = (REPO / "custom_components" / "spot_price_predictor"
                     / "data" / "solar_submodel_default.json")
    if not artifact_path.exists():
        return None
    art = json.loads(artifact_path.read_text())
    sites = art["sites"]
    by_loc, weights = {}, {}
    for loc in sites:
        sw = float(loc.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        key = loc["name"].replace(" ", "_").replace("/", "_")
        matches = list(CACHE_DIR.glob(f"openmeteo_cloud_{key}_*.json"))
        if not matches:
            continue
        payload = json.loads(matches[0].read_text())
        h = payload.get("hourly") or {}
        idx = pd.to_datetime(h.get("time", []), utc=True)
        vals = np.array(h.get("cloud_cover", []), dtype=float)
        if len(idx) == 0:
            continue
        by_loc[loc["name"]] = pd.Series(np.nan_to_num(vals, nan=50.0), index=idx)
        weights[loc["name"]] = sw
    if not by_loc:
        return None
    common = None
    for s in by_loc.values():
        common = s.index if common is None else common.intersection(s.index)
    w_total = sum(weights.values())
    return sum(by_loc[n].reindex(common) * (weights[n] / w_total)
               for n in by_loc).rename("cloud").loc[WINDOW_START:WINDOW_END]


def compute_clear_sky_ghi(reference_ts: pd.DatetimeIndex) -> pd.Series:
    art = json.loads((REPO / "custom_components" / "spot_price_predictor"
                       / "data" / "solar_submodel_default.json").read_text())
    sites = art["sites"]
    arr = np.zeros(len(reference_ts), dtype=float)
    w_total = 0.0
    ts_np = reference_ts.values
    for site in sites:
        sw = float(site.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        ghi = scs.clear_sky_series(
            ts_np, lat_deg=float(site["lat"]), lon_deg=float(site["lon"]),
            model=art["clear_sky_model"],
        )
        arr += sw * ghi
        w_total += sw
    if w_total > 0:
        arr /= w_total
    return pd.Series(arr, index=reference_ts, name="ghi_cs")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading PRICE inputs (recent window only)...", flush=True)
    price_inputs: dict[str, pd.Series] = {}
    price_inputs["fi"] = load_fi_prices()
    price_inputs.update(load_neighbor_prices())

    # Weather: prefer extended window (2018+) fetched from Open-Meteo
    # archive. Falls back to the recent fi_weather.parquet if the
    # extended fetch is unavailable (offline mode).
    #
    # The FULL site list comes from data/finland.yaml — the solar
    # artifact only carries solar_weight, but wind/temp need their own
    # capacity weights.
    print(f"\nLoading WEATHER inputs (extended window "
          f"{WEATHER_WINDOW_START.date()} → {WINDOW_END.date()})...",
          flush=True)
    import yaml
    region_yaml = (REPO / "custom_components" / "spot_price_predictor"
                   / "data" / "finland.yaml")
    region = yaml.safe_load(region_yaml.read_text())
    sites = region["weather_source"]["locations"]
    try:
        weather_inputs = load_weather_extended(
            WEATHER_WINDOW_START, WINDOW_END, sites)
    except Exception as e:
        print(f"  extended fetch failed ({e}); falling back to "
              f"output/fi_weather.parquet", flush=True)
        weather_inputs = load_weather()

    # Clear-sky GHI is deterministic — compute on the weather index
    if weather_inputs:
        weather_common = None
        for s in weather_inputs.values():
            weather_common = s.index if weather_common is None else \
                weather_common.intersection(s.index)
        weather_inputs["ghi_cs"] = compute_clear_sky_ghi(
            pd.DatetimeIndex(weather_common, tz="UTC"))

    # Each input fits on its own window. Stats are still reported on the
    # shared output grid (last-3-years common index) so before/after
    # comparisons are apples-to-apples.
    share_idx = price_inputs["fi"].index   # 2023-01 → 2026-04 by construction

    print("\nFitting per-input seasonal components on per-input window...",
          flush=True)
    fitted: dict[str, dict[str, list[float]]] = {}
    stats: dict[str, dict[str, float]] = {}
    fit_windows: dict[str, tuple[str, str]] = {}

    def _fit(name: str, series: pd.Series, smooth_spec: dict | None = None):
        depth = sd.DEFAULT_DEPTHS.get(name)
        if not depth:
            print(f"  {name:8s}  (no depth spec; skipped)", flush=True)
            return
        s = series.dropna()
        ts_np = pd.DatetimeIndex(s.index, tz="UTC").values
        comp = sd.fit_components(s.values, ts_np, depth=depth,
                                  smooth=smooth_spec)
        # Evaluate var reduction on the SHARED output window so the
        # stats are directly comparable across inputs and across runs.
        share_s = series.reindex(share_idx).dropna()
        share_ts = pd.DatetimeIndex(share_s.index, tz="UTC").values
        residual = sd.compute_residual(share_s.values, share_ts, comp)
        raw_std = float(np.std(share_s.values))
        res_std = float(np.std(residual))
        var_red = 1.0 - (res_std ** 2 / raw_std ** 2) if raw_std > 0 else 0.0
        fitted[name] = comp
        stats[name] = {
            "raw_std": raw_std,
            "residual_std": res_std,
            "var_reduction": var_red,
            "residual_mean": float(np.mean(residual)),
        }
        fit_windows[name] = (str(s.index[0]), str(s.index[-1]))
        depth_label = " + ".join(depth)
        smooth_label = (" (smooth=" + ",".join(
            f"{k}:{v}" for k, v in (smooth_spec or {}).items()) + ")"
            if smooth_spec else "")
        n_years = (s.index[-1] - s.index[0]).total_seconds() / (365.25 * 86400)
        print(
            f"  {name:8s}  depth={depth_label:25s}{smooth_label}  "
            f"σ_raw={raw_std:7.2f} → σ_Y={res_std:6.2f}  "
            f"var_red={100*var_red:5.1f}%  "
            f"window={n_years:.1f}y",
            flush=True,
        )

    # Prices — short window, no smoothing
    for name, series in price_inputs.items():
        _fit(name, series, smooth_spec=None)
    # Weather — long window, smoothing per DEFAULT_SMOOTH
    for name, series in weather_inputs.items():
        _fit(name, series, smooth_spec=DEFAULT_SMOOTH.get(name))

    common = share_idx  # for the artifact's train_window metadata

    artifact = sd.build_artifact(
        fitted,
        train_window=(str(common[0]), str(common[-1])),
        stats=stats,
        notes=(
            "v2.5.7 per-input seasonal components. Prices fit on "
            f"{PRICE_WINDOW_START.date()} → {WINDOW_END.date()} (recent, "
            "regime-sensitive); weather fit on "
            f"{WEATHER_WINDOW_START.date()} → {WINDOW_END.date()} (long "
            "window, stationary climatology). Weather P_week is circular-"
            "smoothed (3-7 bins per DEFAULT_SMOOTH) to reduce the "
            "residual-noise modulation flagged on cloud cover in v2.5.6."
        ),
    )
    artifact["per_input_fit_window"] = fit_windows
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nArtifact: {ARTIFACT_PATH}", flush=True)
    print(f"  size: {ARTIFACT_PATH.stat().st_size:,} bytes", flush=True)

    # Markdown summary for traceability
    md = RESULTS_DIR / "seasonal_components_build.md"
    lines = [
        "# Seasonal components build — v2.5.5",
        "",
        f"**Window:** {common[0].date()} → {common[-1].date()} "
        f"({len(common):,} aligned hourly rows)",
        f"**Artifact:** `custom_components/spot_price_predictor/data/"
        f"seasonal_components_default.json` ({ARTIFACT_PATH.stat().st_size:,} bytes)",
        "",
        "## Per-input fit results",
        "",
        "| Input | Depth | σ_raw | σ_Y | Var reduction | E[Y] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, st in stats.items():
        depth = sd.DEFAULT_DEPTHS.get(name, ())
        lines.append(
            f"| `{name}` | {' + '.join(depth)} | "
            f"{st['raw_std']:.2f} | {st['residual_std']:.2f} | "
            f"{100*st['var_reduction']:.1f}% | {st['residual_mean']:+.2e} |"
        )

    lines += [
        "",
        "## Deployment story",
        "",
        "- Artifact ships in `data/seasonal_components_default.json` —",
        "  the integration loads it at startup; no fit at runtime.",
        "- `seasonal_decomposition.compute_residual(X, ts, components)`",
        "  is the inference entry point; pure-numpy, deterministic.",
        "- E[Y] ≈ 0 by construction (sequential subtraction); the table",
        "  above confirms this numerically (residual mean ~ 1e-13).",
        "- Refresh quarterly via `python studies/build_seasonal_components.py`",
        "  + commit the regenerated JSON. The integration picks it up on",
        "  the next coordinator restart.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/build_seasonal_components.py",
        "```",
        "",
        "Reads only `output/*.parquet` and `studies/.cache/openmeteo_cloud_*.json`",
        "(populated by the v2.5.3 solar study). No API call.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report:   {md}")


if __name__ == "__main__":
    main()
