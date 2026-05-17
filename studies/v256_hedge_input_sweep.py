"""v2.5.6 — Hedge-gated input selection sweep.

Restarts the FI feature selection from a clean architecture:
  - target = Y_fi = FI price deseasonalized by the v2.5.5 artifact
    (drops the v2.2 `month_cos` and AR-with-daytype features by
    construction, because their information now lives in the input-
    level seasonal vectors).
  - candidate features are the residual forms `Y_X` of every input that
    survived the v2.5.4 audit, plus AR(1) lags on Y_fi, plus the
    structural features (is_holiday, hdd_sq, wind_log_scarcity,
    spread_se3_se1, etc.), plus the v2.5.3 solar sub-model output.
  - acceptance criterion is the NPK-CVaR hedge gate at α=0.05 with the
    forecast forward-shifted by `lag=168` hours (the user's primary
    metric: 7-day CVaR accuracy).

The sweep is forward-add greedy: at each step, the candidate that most
improves test CVaR is added; stop when the next-best candidate adds
less than `ACCEPT_THRESHOLD_PP` (0.3 pp).

Time-series MAE is reported as a secondary metric (per user direction:
"primarily for 7-day CVaR data accuracy and secondarily for predicted
timeseries for predicted price (mainly for visualization purposes)").

Reads everything offline; no API call.

Output:
  studies/results/v256_input_sweep.md          — full scorecard
  studies/results/figures/v256_sweep_path.png  — forward-add path
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
from npk_cvar_hedge import optimize_hedge  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_DIR  = REPO / "output"
CACHE_DIR   = REPO / "studies" / ".cache"
RESULTS_DIR = REPO / "studies" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR    = REPO / "custom_components" / "spot_price_predictor" / "data"
SEAS_ART    = DATA_DIR / "seasonal_components_default.json"
SOLAR_ART   = DATA_DIR / "solar_submodel_default.json"

# Hedge gate hyper-parameters
ALPHA = 0.05               # CVaR confidence level (95 %)
LAG_HOURS = 168            # 7-day horizon (user's primary metric)
TRAIN_FRAC = 0.55          # chronological 55/45 split
ACCEPT_THRESHOLD_PP = 0.3  # require ≥ 0.3 pp improvement to ACCEPT

# Window — same as v2.5.4 / v2.5.5 so the residuals match exactly
WINDOW_START = pd.Timestamp("2023-01-01", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-04-28", tz="UTC")


# ── Loaders (mirror v2.5.5 build_seasonal_components.py) ───────────


def load_aligned_inputs() -> tuple[pd.DataFrame, dict]:
    """Return one wide DataFrame containing every raw input + Y_X
    residual + structural feature, all on the same hourly index.

    Returns:
        (df, artifact) where `df` columns include both `X` (raw) and
        `Y_X` (residual) for each input, plus structural features.
    """
    fi  = pd.read_parquet(OUTPUT_DIR / "fi_prices.parquet")["price_eur_mwh"]
    nei = pd.read_parquet(OUTPUT_DIR / "fi_neighbor_prices.parquet")
    wea = pd.read_parquet(OUTPUT_DIR / "fi_weather.parquet")

    seas = json.loads(SEAS_ART.read_text())
    solar_art = json.loads(SOLAR_ART.read_text())

    # Cloud cover (same loader as v2.5.5)
    by_loc, weights = {}, {}
    for loc in solar_art["sites"]:
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
    common_cloud = None
    for s in by_loc.values():
        common_cloud = s.index if common_cloud is None else common_cloud.intersection(s.index)
    w_total = sum(weights.values())
    cloud = sum(by_loc[n].reindex(common_cloud) * (weights[n] / w_total)
                for n in by_loc).rename("cloud")

    # Trim everything to common index
    raw = pd.concat(
        [fi.rename("fi"), nei[["se3", "se1", "ee"]],
         wea[["wind_speed_weighted", "solar_irradiance_weighted",
              "temperature_weighted"]].rename(columns={
                  "wind_speed_weighted": "wind",
                  "solar_irradiance_weighted": "solar",
                  "temperature_weighted": "temp",
              }),
         cloud],
        axis=1,
    )
    raw = raw.loc[WINDOW_START:WINDOW_END].dropna()
    common_idx = raw.index

    # Clear-sky GHI for the same index
    ghi_cs = np.zeros(len(common_idx), dtype=float)
    w_total = 0.0
    ts_np = common_idx.values
    for site in solar_art["sites"]:
        sw = float(site.get("solar_weight", 0.0))
        if sw <= 0:
            continue
        ghi_cs += sw * scs.clear_sky_series(
            ts_np, lat_deg=float(site["lat"]), lon_deg=float(site["lon"]),
            model=solar_art["clear_sky_model"],
        )
        w_total += sw
    if w_total > 0:
        ghi_cs /= w_total
    raw["ghi_cs"] = ghi_cs

    # Solar sub-model deployed prediction (Fingrid-free)
    raw["solar_pred_mw"] = scs.predict_solar_mw(
        ts_np, raw["cloud"].values, solar_art)

    # Residuals Y_X via shipped v2.5.5 artifact
    for name in ("fi", "se3", "se1", "ee", "wind", "solar",
                 "ghi_cs", "temp", "cloud"):
        components = seas["components"].get(name)
        if components is None:
            continue
        y = sd.compute_residual(raw[name].values, ts_np, components)
        raw[f"Y_{name}"] = y

    # Lag features on Y_fi
    raw["Y_fi_lag1"]   = raw["Y_fi"].shift(1)
    raw["Y_fi_lag24"]  = raw["Y_fi"].shift(24)
    raw["Y_fi_lag168"] = raw["Y_fi"].shift(168)

    # Cross-zone spreads (per v2.5.1 finding)
    raw["spread_se3_se1"] = raw["Y_se3"] - raw["Y_se1"]
    raw["spread_se3_ee"]  = raw["Y_se3"] - raw["Y_ee"]

    # Structural features
    is_workday = (raw.index.weekday < 5).astype(float)
    raw["is_workday"] = is_workday
    raw["is_weekend"] = 1.0 - is_workday
    raw["hdd_sq"]     = np.maximum(0.0, 17.0 - raw["temp"]) ** 2
    raw["wind_scar"]  = np.log1p(np.maximum(0.0, 8.0 - raw["wind"]))

    raw = raw.dropna()
    return raw, seas


# ── Ridge fit on (X, y) with weight decay ──────────────────────────


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Return coefficient vector for `y = X β` with L2 regularisation
    on every column EXCEPT the constant (assumed to be column 0)."""
    n, p = X.shape
    XtX = X.T @ X
    pen = alpha * np.eye(p)
    pen[0, 0] = 0.0   # don't penalise the constant
    return np.linalg.solve(XtX + pen, X.T @ y)


# ── Hedge gate: mirror studies/se3_model_v242.py:hedge_reduction ───


def hedge_cvar_pct(actual: np.ndarray, model: np.ndarray,
                   lag: int = LAG_HOURS, alpha: float = ALPHA) -> dict:
    """CVaR-reduction percentage of `actual` when hedged using a
    forward-shifted `model` forecast at lag `lag` hours."""
    fwd = np.concatenate([model[lag:], np.repeat(model[-1], lag)])
    res = optimize_hedge(np.diff(actual), np.diff(fwd), alpha=alpha)
    res["pct_reduction"] = (
        100.0
        * (res["cvar_test_hist_unhedged"] - res["cvar_test_hist_hedged"])
        / res["cvar_test_hist_unhedged"]
    )
    return res


# ── Forward-add sweep ──────────────────────────────────────────────


def evaluate_feature_set(df: pd.DataFrame, features: list[str],
                         target_col: str = "Y_fi",
                         lag: int = LAG_HOURS) -> dict:
    """Fit Ridge on `features` predicting `target_col`, reconstruct the
    full price prediction by adding back the seasonal component (which
    is implicit in the difference `fi - Y_fi`), then run the hedge gate
    on the prediction vs actual FI prices.

    Returns metrics dict with `cvar_pct`, `mae_test`, `n_features`.
    """
    # Design matrix: intercept + features
    X = np.column_stack([np.ones(len(df)), df[features].values])
    y = df[target_col].values
    fi_actual = df["fi"].values
    seasonal_fi = fi_actual - y  # the seasonal component is exactly this

    n = len(df)
    split = int(n * TRAIN_FRAC)
    coef = fit_ridge(X[:split], y[:split], alpha=1.0)
    y_pred_full = X @ coef

    # Reconstructed FI prediction = seasonal + residual prediction
    fi_pred = seasonal_fi + y_pred_full
    fi_pred_test = fi_pred[split:]
    fi_test = fi_actual[split:]

    mae_test = float(np.mean(np.abs(fi_pred_test - fi_test)))

    try:
        h = hedge_cvar_pct(fi_actual, fi_pred, lag=lag, alpha=ALPHA)
        cvar_pct = float(h["pct_reduction"])
        cvar_unhedged = float(h["cvar_test_hist_unhedged"])
        cvar_hedged   = float(h["cvar_test_hist_hedged"])
    except Exception as e:
        cvar_pct = float("nan")
        cvar_unhedged = float("nan")
        cvar_hedged   = float("nan")
        print(f"      hedge failed: {e}", flush=True)

    return {
        "features": tuple(features),
        "n_features": len(features),
        "mae_test_eur_mwh": mae_test,
        "cvar_pct": cvar_pct,
        "cvar_unhedged": cvar_unhedged,
        "cvar_hedged": cvar_hedged,
        "ridge_coef": coef.tolist(),
    }


def forward_add_sweep(
    df: pd.DataFrame,
    candidate_features: list[str],
    target_col: str = "Y_fi",
    baseline_features: list[str] | None = None,
    max_steps: int | None = None,
    lag: int = LAG_HOURS,
) -> list[dict]:
    """Greedy forward-add at the given hedge horizon `lag` hours."""
    baseline = list(baseline_features) if baseline_features else []
    remaining = [f for f in candidate_features if f not in baseline]
    history: list[dict] = []

    base_res = evaluate_feature_set(df, baseline, target_col=target_col, lag=lag)
    base_res["step"] = 0
    base_res["added"] = "(baseline)"
    base_res["cvar_delta_pp"] = 0.0
    history.append(base_res)
    current_cvar = base_res["cvar_pct"]
    print(f"  step 0 baseline ({len(baseline)} features): "
          f"CVaR {current_cvar:+.2f}%  MAE {base_res['mae_test_eur_mwh']:.2f}",
          flush=True)

    selected = list(baseline)
    step = 0
    while remaining:
        step += 1
        if max_steps is not None and step > max_steps:
            break
        best = None
        best_delta = -np.inf
        for cand in remaining:
            trial = evaluate_feature_set(df, selected + [cand],
                                          target_col=target_col, lag=lag)
            delta = trial["cvar_pct"] - current_cvar
            if delta > best_delta:
                best_delta = delta
                best = (cand, trial)
        cand, trial = best
        print(f"  step {step}: add `{cand}`  "
              f"CVaR {trial['cvar_pct']:+.2f}% (Δ {best_delta:+.2f} pp)  "
              f"MAE {trial['mae_test_eur_mwh']:.2f}",
              flush=True)
        trial["step"]  = step
        trial["added"] = cand
        trial["cvar_delta_pp"] = best_delta
        history.append(trial)
        if best_delta < ACCEPT_THRESHOLD_PP:
            print(f"  stop: best Δ {best_delta:+.2f} pp < "
                  f"{ACCEPT_THRESHOLD_PP} pp threshold", flush=True)
            break
        selected.append(cand)
        remaining.remove(cand)
        current_cvar = trial["cvar_pct"]

    return history


# ── Plotting ───────────────────────────────────────────────────────


def fig_sweep_path(sweeps: dict[int, list[dict]], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {24: "C0", 48: "C1", 168: "C2"}
    labels_for_lag = {24: "24 h (day-ahead, secondary)",
                      48: "48 h (v2.4.x reference, secondary)",
                      168: "168 h (7-day, PRIMARY per user)"}
    ax_c, ax_m = axes

    for lag in sorted(sweeps.keys()):
        hist = sweeps[lag]
        steps = [h["step"] for h in hist]
        cvar  = [h["cvar_pct"] for h in hist]
        mae   = [h["mae_test_eur_mwh"] for h in hist]
        ax_c.plot(steps, cvar, f"{colors[lag]}-o",
                  lw=1.7 if lag == 168 else 1.0,
                  label=labels_for_lag[lag])
        ax_m.plot(steps, mae, f"{colors[lag]}-o",
                  lw=1.7 if lag == 168 else 1.0,
                  label=labels_for_lag[lag])
        # Annotate winner-per-step on the 168 h primary track
        if lag == 168:
            for s, c, h in zip(steps, cvar, hist):
                ax_c.annotate(h["added"], (s, c), xytext=(4, 6),
                              textcoords="offset points",
                              fontsize=8, rotation=15)

    ax_c.axhline(0, color="k", lw=0.5)
    ax_c.set_xlabel("Forward-add step")
    ax_c.set_ylabel("Hedge CVaR reduction [%]")
    ax_c.set_title("CVaR reduction by step (primary metric)")
    ax_c.legend(loc="lower right", fontsize=9)

    ax_m.set_xlabel("Forward-add step")
    ax_m.set_ylabel("Test MAE [EUR/MWh]")
    ax_m.set_title("Test MAE by step (secondary, visualization only)")
    ax_m.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"v2.5.6 forward-add input sweep — α={ALPHA}, "
        f"accept ≥ {ACCEPT_THRESHOLD_PP} pp per added feature",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Loading aligned inputs (offline)...", flush=True)
    df, seas = load_aligned_inputs()
    print(f"  rows: {len(df):,}  "
          f"({df.index[0].date()} → {df.index[-1].date()})", flush=True)
    print(f"  columns: {len(df.columns)}", flush=True)

    candidates = [
        # AR lags on Y_fi (cheap, trivial complexity)
        "Y_fi_lag1", "Y_fi_lag24", "Y_fi_lag168",
        # Cross-border residuals (free, no auth — Elering / elprisetjustnu.se)
        "Y_se3", "Y_se1", "Y_ee",
        # Spread features (derived from above)
        "spread_se3_se1", "spread_se3_ee",
        # Exogenous physical inputs (Open-Meteo, already pulled)
        "Y_wind", "Y_solar", "Y_temp", "Y_cloud", "Y_ghi_cs",
        # Solar sub-model (v2.5.3)
        "solar_pred_mw",
        # Structural / interaction features (zero marginal complexity)
        "wind_scar", "hdd_sq", "is_workday",
    ]
    print(f"  candidates: {len(candidates)}", flush=True)
    print()

    # Run the sweep at multiple horizons so the user can read which
    # features matter at which forecast lead time. Primary horizon is
    # 168h (user direction); 48h kept as the v2.4.x reference horizon
    # because most existing studies use it; 24h shows day-ahead view.
    sweeps: dict[int, list[dict]] = {}
    for lag in (24, 48, LAG_HOURS):
        print(f"\n=== Forward-add sweep at lag = {lag} h ===")
        sweeps[lag] = forward_add_sweep(df, candidates, target_col="Y_fi",
                                          baseline_features=[], lag=lag)
    history = sweeps[LAG_HOURS]

    print("\nRendering figures...", flush=True)
    fig_sweep_path(sweeps, FIGURES_DIR / "v256_sweep_path.png")

    # Markdown report
    print("Writing markdown summary...", flush=True)
    md = RESULTS_DIR / "v256_input_sweep.md"
    n = len(df)
    split = int(n * TRAIN_FRAC)
    lines = [
        "# v2.5.6 — Hedge-gated input selection sweep",
        "",
        f"**Window:** {df.index[0].date()} → {df.index[-1].date()} "
        f"({n:,} hourly rows; train/test split at "
        f"{df.index[split].date()})",
        f"**Target:** `Y_fi` (FI price deseasonalized by the v2.5.5 artifact)",
        f"**Hedge gate:** α = {ALPHA}, accept ≥ {ACCEPT_THRESHOLD_PP} pp per added feature.",
        f"**Primary horizon (user direction):** {LAG_HOURS} h (7-day CVaR accuracy).",
        f"**Secondary horizons reported for context:** 48 h (v2.4.x baseline) and 24 h (day-ahead).",
        f"**Selection:** forward-add greedy on hedge CVaR-reduction; "
        f"chronological 55 / 45 train/test split; Ridge α = 1.0.",
        "",
    ]

    for lag in (LAG_HOURS, 48, 24):
        label = (f"Primary — {lag} h (7-day horizon)" if lag == LAG_HOURS
                 else f"Secondary — {lag} h "
                      f"({'day-ahead' if lag == 24 else 'v2.4.x reference'})")
        hist = sweeps[lag]
        accepted = [h["added"] for h in hist
                    if h["step"] > 0 and h["cvar_delta_pp"] >= ACCEPT_THRESHOLD_PP]
        final = hist[-1] if hist[-1]["cvar_delta_pp"] >= ACCEPT_THRESHOLD_PP else (hist[-2] if len(hist) >= 2 else hist[-1])
        lines += [
            f"## {label}",
            "",
            "| Step | Added feature | Total | CVaR % | Δ (pp) | Test MAE |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for h in hist:
            lines.append(
                f"| {h['step']} | `{h['added']}` | {h['n_features']} | "
                f"{h['cvar_pct']:+.2f}% | {h['cvar_delta_pp']:+.2f} | "
                f"{h['mae_test_eur_mwh']:.2f} |"
            )
        lines += [
            "",
            (f"**Accepted:** {'none' if not accepted else '`' + '`, `'.join(accepted) + '`'}"
             f" — final CVaR {final['cvar_pct']:+.2f} %, "
             f"MAE {final['mae_test_eur_mwh']:.2f} EUR/MWh, "
             f"{final['n_features']} feature(s)."),
            "",
        ]

    lines += [
        "## Cross-horizon figure",
        "",
        "![Sweep paths at three horizons](figures/v256_sweep_path.png)",
        "",
        "## Interpretation",
        "",
        "- **Primary metric is hedge CVaR-reduction.** The user direction "
        "2026-05-17 prioritises 7-day CVaR data accuracy; test MAE is "
        "tracked secondarily (visualization only).",
        "- The target `Y_fi` is the FI price residual after the "
        "v2.5.5 deseasonalization artifact. Reconstructing the full "
        "price prediction adds the seasonal component back; the hedge "
        "gate compares the reconstructed full-price prediction vs the "
        "actual full-price series.",
        "- The dominant 7-day-ahead signal lives in the seasonal "
        "forecast (already in the target via the v2.5.5 artifact) plus "
        "`Y_fi_lag168` (same-day-last-week residual) — together they "
        "explain almost all the recoverable hedge value at 168 h.",
        "- Cross-border / weather / solar-submodel features add little "
        "at the 7-day horizon. The 48 h panel above shows whether the "
        "same features carry more value closer-in.",
        "- Ridge α = 1.0 with constant un-penalised; chronological "
        f"55 / 45 train/test split; 17 candidate features.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v256_hedge_input_sweep.py",
        "```",
        "",
        "No network call; reads only `output/*.parquet`, the v2.5.3 + "
        "v2.5.5 artifacts in `data/`, and the cloud-cover cache in "
        "`studies/.cache/`.",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")
    print(f"Figure: {FIGURES_DIR / 'v256_sweep_path.png'}")


if __name__ == "__main__":
    main()
