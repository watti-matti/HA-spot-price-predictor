"""Experiment — does adding nuclear-deficit and/or cross-border features to
the v2.8.1 L2 Ridge improve the spot forecast?

This is research code that runs OFF the production path:
  * does NOT modify any artifact under data/
  * does NOT touch RIDGE_FEATURES, the pipeline, or sensors
  * writes findings to studies/results/experiment_extra_l2_features.{md,json}

Variants
--------
  B0  baseline           (v2.8.1 six features — sanity replication)
  B1  + nuclear_deficit  (1 - normalised nuclear_mw, centred)
  B2  + cross-border     (deseasonalised SE3 / EE prices + export-potential SE3)
  B3  B1 + B2 combined

For each variant we:
  1. Fit a Ridge on the train split (matches v2513 TRAIN_FRAC).
  2. Fit AR(1) on the Ridge residual.
  3. Compose L1 + L2 + L3 prediction on the test split.
  4. Report MAE, R², and the same metrics restricted to the extreme-
     price subset (|spot| > 100 EUR/MWh) where the v2.5.13 work showed
     the model is weakest.

The harness reuses the same loaders, L1 components, and physics helpers
as `studies/v2513_layer4_spike_model.py` so the B0 variant should
exactly reproduce the production model's training-time MAE on the held-
out window.

Prerequisites: cached parquets under output/ (fi_prices, fi_weather,
fi_neighbor_prices, fi_grid_data).
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

from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)
from v2510_layer3_ar_wind import fit_ridge, fit_ar1, TRAIN_FRAC  # noqa: E402
from v2512_sigmoid_turbine_curve import sigmoid_turbine_rho  # noqa: E402
from v2511_physics_features import solar_effective  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


REPO_DATA = REPO / "custom_components" / "spot_price_predictor" / "data"
RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = REPO / "output"

# Shipped seasonal artifact — same L1 components the production Pipeline uses.
SEASONAL_ARTIFACT = json.loads(
    (REPO_DATA / "seasonal_components_default.json").read_text()
)
SOLAR_ARTIFACT = json.loads(
    (REPO_DATA / "solar_submodel_default.json").read_text()
)


# ── Data assembly ────────────────────────────────────────────────────


def load_grid() -> dict[str, pd.Series]:
    """Fingrid streams (nuclear_mw is already normalised to [0, 1])."""
    df = pd.read_parquet(OUTPUT_DIR / "fi_grid_data.parquet")
    return {col: df[col] for col in df.columns}


def build_dataframe() -> pd.DataFrame:
    """Build the same dataframe `v2513_layer4_spike_model._build_dataframe`
    produces, plus the candidate extra features (`nuclear_deficit`,
    `ar_se3`, `ar_ee`, `export_potential_se3`)."""
    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    inputs.update(load_grid())

    import yaml
    region = yaml.safe_load((REPO_DATA / "finland.yaml").read_text())
    sites = region["weather_source"]["locations"]
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)

    # GHI clear-sky proxy (matches v2513) — used for solar_effective.
    if wea:
        ws_idx = None
        for s in wea.values():
            ws_idx = s.index if ws_idx is None else ws_idx.intersection(s.index)
        ts_np = ws_idx.values
        ghi = np.zeros(len(ws_idx), dtype=float)
        w_total = 0.0
        for site in SOLAR_ARTIFACT["sites"]:
            sw = float(site.get("solar_weight", 0.0))
            if sw <= 0:
                continue
            ghi += sw * scs.clear_sky_series(
                ts_np, lat_deg=float(site["lat"]),
                lon_deg=float(site["lon"]),
                model=SOLAR_ARTIFACT["clear_sky_model"])
            w_total += sw
        if w_total > 0:
            ghi /= w_total
        inputs["ghi_cs"] = pd.Series(ghi, index=ws_idx, name="ghi_cs")

    # Inner join on shared timestamps.
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
    ts_np = pd.DatetimeIndex(common, tz="UTC").values

    # L1 deseasonalisation using the shipped components.
    for name in df.columns:
        if name not in SEASONAL_ARTIFACT["components"]:
            continue
        components = SEASONAL_ARTIFACT["components"][name]
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, components)
        df[f"Y_{name}"] = df[name].values - df[f"seasonal_{name}"].values

    # L2 physics features.
    df["Y_fi_lag168"] = df["Y_fi"].shift(168)
    df["is_workday"] = (df.index.weekday < 5).astype(float)
    df["sigmoid_wind_rho"] = sigmoid_turbine_rho(
        df["wind"].values, df["temp"].values)
    df["solar_effective"] = solar_effective(
        df["solar"].values, df["temp"].values)
    for name in ("sigmoid_wind_rho", "solar_effective"):
        comp = sd.fit_components(df[name].values, ts_np,
                                  depth=("P_hour", "P_week"),
                                  smooth={"P_week": 7})
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, comp)
        df[f"Y_{name}"] = df[name].values - df[f"seasonal_{name}"].values

    # Candidate extra features.
    # nuclear_deficit ∈ [0, 1]: bigger ⇒ more outage. Mean-centre so the
    # Ridge intercept stays meaningful when added/removed across variants.
    nuc_def = np.clip(1.0 - df["nuclear_mw"].values, 0.0, 1.0)
    df["nuclear_deficit"] = nuc_def - nuc_def[~np.isnan(nuc_def)].mean()

    # Cross-border: deseasonalised neighbour prices.
    # legacy v2.2 used a proper AR(2) day-type model; here we reuse the
    # L1 components already shipped for se1/se3/ee, which is the same
    # deseasonalising step the FI side uses. Result: Y_se3 ≈ "AR(2) -ish"
    # deviation of SE3 from its hour-of-week climatology.
    for nb in ("se3", "ee", "se1"):
        col = f"Y_{nb}"
        if col not in df.columns:
            df[col] = 0.0

    # export_potential_se3 ≈ max(0, -(fi - se3)) on raw prices (legacy
    # definition). Mean-centre across the full dataset for stability.
    ep_raw = np.maximum(0.0, -(df["fi"].values - df["se3"].values))
    df["export_potential_se3"] = ep_raw - np.nanmean(ep_raw)

    return df.dropna()


# ── Variant fitting and evaluation ───────────────────────────────────


VARIANTS: dict[str, list[str]] = {
    "B0_baseline": [
        "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
    ],
    "B1_nuclear": [
        "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
        "nuclear_deficit",
    ],
    "B2_cross_border": [
        "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
        "Y_se3", "Y_ee", "export_potential_se3",
    ],
    "B3_combined": [
        "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
        "nuclear_deficit",
        "Y_se3", "Y_ee", "export_potential_se3",
    ],
}


def fit_and_evaluate(df: pd.DataFrame, features: list[str],
                     alpha: float = 1.0) -> dict:
    """Fit Ridge + AR(1) on the train split, evaluate on the test split.

    The forecast under test mirrors `Pipeline.compute_forecast` for a
    cold-start single-step horizon: ŷ(t) = L1_seasonal(t) + L2_ridge(t)
    + φ·η(t−1). Bias EMA / fan chart / softplus floor are intentionally
    not applied here — we want to isolate the impact of the L2 feature
    set on the point forecast.
    """
    n = len(df)
    split = int(n * TRAIN_FRAC)

    y = df["Y_fi"].values
    X = np.column_stack(
        [np.ones(n)] + [df[f].values for f in features]
    )

    coef = fit_ridge(X[:split], y[:split], alpha=alpha)
    ridge_pred = X @ coef
    eps = y - ridge_pred
    phi, _ = fit_ar1(eps[:split])

    # AR(1) one-step-ahead on the test split.
    ar_contribution = np.zeros(n, dtype=float)
    ar_contribution[1:] = phi * eps[:-1]

    pred_residual = ridge_pred + ar_contribution
    spot_actual = df["fi"].values
    spot_pred = df["seasonal_fi"].values + pred_residual

    err = spot_actual - spot_pred

    def _metrics(mask: np.ndarray) -> dict:
        e = err[mask]
        y_ = spot_actual[mask]
        if e.size == 0:
            return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                    "r2": float("nan")}
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
        return {
            "n": int(e.size),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    test_mask = np.zeros(n, dtype=bool)
    test_mask[split:] = True
    extreme_mask = test_mask & (np.abs(spot_actual) > 100.0)

    return {
        "features": features,
        "n_features_with_intercept": len(features) + 1,
        "ridge_coef": coef.tolist(),
        "phi": float(phi),
        "test_overall": _metrics(test_mask),
        "test_extreme_gt100": _metrics(extreme_mask),
        "train_size": split,
        "test_size": n - split,
    }


# ── Reporting ────────────────────────────────────────────────────────


def write_findings_md(results: dict, out_path: Path) -> None:
    rows = []
    for name, res in results.items():
        ov = res["test_overall"]
        ex = res["test_extreme_gt100"]
        rows.append(
            f"| {name} | {res['n_features_with_intercept']} | "
            f"{ov['mae']:.2f} | {ov['rmse']:.2f} | {ov['r2']:+.3f} | "
            f"{ex['mae']:.2f} | {ex['r2']:+.3f} | {res['phi']:+.3f} |"
        )
    table = "\n".join(rows)

    baseline = results["B0_baseline"]
    deltas = []
    for name, res in results.items():
        if name == "B0_baseline":
            continue
        d_mae = res["test_overall"]["mae"] - baseline["test_overall"]["mae"]
        d_r2 = res["test_overall"]["r2"] - baseline["test_overall"]["r2"]
        d_mae_ext = (res["test_extreme_gt100"]["mae"]
                     - baseline["test_extreme_gt100"]["mae"])
        # Verdict logic. Two independent wins matter:
        #   * overall MAE: small but broad accuracy gain
        #   * extreme-price MAE: cuts the worst error mode
        # Either alone, if material, is enough to recommend a follow-up
        # refit; a degradation in either reverses the recommendation.
        overall_win = d_mae <= -0.05
        extreme_win = d_mae_ext <= -1.0
        overall_loss = d_mae > +0.05
        extreme_loss = d_mae_ext > +1.0
        if overall_loss or extreme_loss:
            verdict = "**reject**"
        elif overall_win and extreme_win:
            verdict = "**accept (both metrics)**"
        elif extreme_win:
            verdict = "**accept (extreme-tail)**"
        elif overall_win:
            verdict = "**accept (overall)**"
        else:
            verdict = "neutral"
        deltas.append(
            f"| {name} | {d_mae:+.2f} | {d_mae_ext:+.2f} | "
            f"{d_r2:+.4f} | {verdict} |"
        )
    deltas_table = "\n".join(deltas)

    md = f"""# Experiment — extra L2 features (nuclear deficit, cross-border)

Branch: `experiment/extra-l2-features`. Off-tree research only — no
production artefact change.

## Variants

| Variant | n_feat | MAE | RMSE | R² | MAE (|spot|>100) | R² (|spot|>100) | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table}

## Delta vs B0 baseline (test split)

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ R² | Verdict |
|---|:---:|:---:|:---:|:---:|
{deltas_table}

**Decision rule.** Two independent wins matter:

- *overall MAE* — broad accuracy across all test hours.
- *extreme-price MAE* — accuracy on the rare but expensive spike hours
  (|spot| > 100 EUR/MWh), where the v2.5.13 work showed the model is
  weakest.

Accept if either improves materially (overall Δ ≤ −0.05 EUR/MWh, or
extreme-tail Δ ≤ −1.0 EUR/MWh) without the other regressing materially
(overall Δ > +0.05, or extreme-tail Δ > +1.0). Reject if either
regresses materially. Otherwise: neutral — keep the production model.

## Method

- Data: cached parquets under `output/` (≈ 4 years, hourly).
- Train/test: time-ordered, `TRAIN_FRAC = {TRAIN_FRAC}`
  (train = first {results['B0_baseline']['train_size']} hours,
   test  = last  {results['B0_baseline']['test_size']} hours).
- Ridge α = 1.0, intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then applied
  one-step-ahead on the test split.
- Forecast under test: L1 seasonal + L2 ridge + φ·ε(t−1).
  Hourly-bias EMA, softplus floor, and L4 GPD POT bands are NOT applied
  here — the goal is to isolate the impact of the L2 feature set on
  the point forecast.
- Extreme-price bucket: test hours with |spot| > 100 EUR/MWh, where the
  v2.5.13 work showed the model is weakest.

## Candidate features (legacy v2.2 lineage)

- `nuclear_deficit ∈ [0, 1]` — `max(0, 1 − nuclear_mw)` where
  `nuclear_mw` is Fingrid #188 normalised by max-fleet 4 372 MW.
  Mean-centred for stable Ridge weighting.
- `Y_se3`, `Y_ee` — neighbour spot prices deseasonalised against the
  shipped per-zone hourly+weekly L1 components. The legacy v2.2
  `ar_se3` / `ar_ee` used a proper AR(2) daytype-deviation; this is a
  simpler analogue.
- `export_potential_se3` — `max(0, −(fi − se3))`, mean-centred. When
  FI is cheaper than SE3 (negative spread), export pressure pulls the
  FI price up; when FI is more expensive there is no export pressure
  (clipped to zero).

## Open follow-ups (deferred to separate experiments)

- **SE3 / SE1 / EE transit-capacity saturation.** A continuous
  saturation indicator (e.g. `min(|spread|, cap) / cap` calibrated on
  historical transit-capacity-out data) was *not* tested here. The
  existing `Y_se3` / `Y_ee` already proxy coupling; a saturation
  indicator only adds information when transit *de*couples (large
  spread regime). Belongs in a follow-up if B2 / B3 are accepted.
- **`nuclear_x_scarcity` interaction.** Legacy v2.2 multiplied
  `nuclear_deficit × wind_log_scarcity` to amplify outage impact under
  cold-and-windless conditions. Not tested here; can be a follow-up if
  B1 is accepted.
"""
    out_path.write_text(md, encoding="utf-8")


def main() -> None:
    print("Building dataframe…", flush=True)
    df = build_dataframe()
    print(f"  rows = {len(df):,}, span = "
          f"{df.index[0].date()} → {df.index[-1].date()}", flush=True)

    results = {}
    for name, features in VARIANTS.items():
        print(f"Fitting {name} ({len(features) + 1} feats incl. intercept)…",
              flush=True)
        results[name] = fit_and_evaluate(df, features)
        ov = results[name]["test_overall"]
        ex = results[name]["test_extreme_gt100"]
        print(f"  test MAE = {ov['mae']:.2f}  R² = {ov['r2']:+.3f}  "
              f"|spot|>100 MAE = {ex['mae']:.2f}", flush=True)

    md_path = RESULTS_DIR / "experiment_extra_l2_features.md"
    json_path = RESULTS_DIR / "experiment_extra_l2_features.json"
    write_findings_md(results, md_path)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {md_path.relative_to(REPO)}")
    print(f"Wrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
