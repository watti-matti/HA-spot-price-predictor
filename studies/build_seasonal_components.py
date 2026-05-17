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

WINDOW_START = pd.Timestamp("2023-01-01", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-04-28", tz="UTC")


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
    print("Loading input series...", flush=True)
    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    inputs.update(load_weather())
    cloud = load_cloud_cover()
    if cloud is not None:
        inputs["cloud"] = cloud
    inputs["ghi_cs"] = compute_clear_sky_ghi(
        pd.DatetimeIndex(inputs["fi"].index, tz="UTC"))

    # Align on a common hourly index
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common).dropna() for k, s in inputs.items()}
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common) for k, s in inputs.items()}
    print(f"  aligned: {len(common):,} hourly rows  "
          f"({common[0].date()} -> {common[-1].date()})", flush=True)

    # Fit + collect stats
    print("\nFitting per-input seasonal components...", flush=True)
    ts_np = common.values  # numpy datetime64
    fitted: dict[str, dict[str, list[float]]] = {}
    stats: dict[str, dict[str, float]] = {}
    for name, series in inputs.items():
        depth = sd.DEFAULT_DEPTHS.get(name)
        if not depth:
            print(f"  {name:8s}  (no depth spec; skipped)", flush=True)
            continue
        comp = sd.fit_components(series.values, ts_np, depth=depth)
        residual = sd.compute_residual(series.values, ts_np, comp)
        raw_std = float(np.std(series.values))
        res_std = float(np.std(residual))
        var_red = 1.0 - (res_std ** 2 / raw_std ** 2) if raw_std > 0 else 0.0
        fitted[name] = comp
        stats[name] = {
            "raw_std": raw_std,
            "residual_std": res_std,
            "var_reduction": var_red,
            "residual_mean": float(np.mean(residual)),
        }
        depth_label = " + ".join(depth)
        print(
            f"  {name:8s}  depth={depth_label:25s}  "
            f"σ_raw={raw_std:7.2f} → σ_Y={res_std:6.2f}  "
            f"var_red={100*var_red:5.1f}%  "
            f"E[Y]={stats[name]['residual_mean']:+.3e}",
            flush=True,
        )

    artifact = sd.build_artifact(
        fitted,
        train_window=(str(common[0]), str(common[-1])),
        stats=stats,
        notes=(
            "v2.5.5 per-input seasonal components. Fit window from the "
            "v2.5.4 audit. Runtime helper: "
            "seasonal_decomposition.compute_residual(X, ts, components). "
            "Refresh quarterly alongside the v2.5.3 solar artifact."
        ),
    )
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
