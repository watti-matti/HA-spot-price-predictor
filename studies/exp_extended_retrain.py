"""Full L1+L2+L3+L4 retrain with the extended (cross-border) feature set,
gated by NPK-CVaR hedge analysis per v2.5.6.

Three variants are refit end-to-end on the cached parquets (2023-2026):

  V_prod    current v2.8.1 5-feature design (sanity baseline; should
            reproduce the production spike_model_default.json hedge
            metric)
  V_xb      extended 8-feature design adding Y_se1, Y_se3, Y_ee
            (the cross-border layer the previous experiment found
            most useful)
  V_xb_nuc  V_xb + capacity-aware nuclear_deficit_v2
            (rolling-60-day max minus current nuclear_mw; activates
            during real OL1/OL2/Loviisa outage episodes — re-tested
            with the user's note 2026-05-19 that nuclear deficit
            should be analysed against capacity ceiling rather than
            against the normalisation constant 1.0)

For each variant we refit:
  L1 seasonal (uses the shipped components — unchanged, since L1 is
              feature-independent)
  L2 Ridge with the variant's feature set
  L3 AR(1) on the L2 residual (train split)
  L4 GPD POT on the post-AR residual (right tail only; matches v2513)

Outputs:
  studies/results/exp_extended_retrain.md       (head-to-head report)
  studies/results/exp_extended_retrain.json     (full metrics)
  output/exp_spike_model_<variant>.json         (candidate artifacts)

No production artefact is overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studies"))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from exp_extra_features import (  # noqa: E402
    build_dataframe, _CORE,
)
from exp_se1_and_nuclear_capacity import (  # noqa: E402
    add_capacity_aware_features,
)
from v2510_layer3_ar_wind import fit_ridge, fit_ar1, TRAIN_FRAC  # noqa: E402
from npk_cvar_hedge import optimize_hedge  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "peak_model_feasibility",
    REPO / "studies" / "peak_model_feasibility.py",
)
_pmf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pmf)
fit_gpd_pot = _pmf.fit_gpd_pot
cvar_normal = _pmf.cvar_normal
hill_estimator = _pmf.hill_estimator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = REPO / "output"


# ── Variant definitions (5 / 8 / 9 features incl. intercept handling) ─


VARIANTS: dict[str, list[str]] = {
    # v2.8.1 production design (no intercept here — added by fit code).
    "V_prod":   ["Y_fi_lag168", "is_workday",
                 "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp"],
    "V_xb":     ["Y_fi_lag168", "is_workday",
                 "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
                 "Y_se1", "Y_se3", "Y_ee"],
    "V_xb_nuc": ["Y_fi_lag168", "is_workday",
                 "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
                 "Y_se1", "Y_se3", "Y_ee", "nuclear_deficit_v2"],
}


# ── End-to-end refit ─────────────────────────────────────────────────


def refit_variant(df: pd.DataFrame, features: list[str],
                  alpha_ridge: float = 1.0) -> dict:
    n = len(df)
    split = int(n * TRAIN_FRAC)
    y = df["Y_fi"].values

    X_full = np.column_stack(
        [np.ones(n)] + [df[f].values for f in features]
    )
    coef = fit_ridge(X_full[:split], y[:split], alpha=alpha_ridge)
    ridge_pred_full = X_full @ coef
    eps_full = y - ridge_pred_full
    phi, sigma_eta_fit = fit_ar1(eps_full[:split])

    # AR(1) one-step-ahead residual η(t) = ε(t) − φ·ε(t−1).
    eta = np.zeros(n, dtype=float)
    eta[1:] = eps_full[1:] - phi * eps_full[:-1]
    eta[0] = eps_full[0]
    eta_train, eta_test = eta[:split], eta[split:]

    # L4 GPD POT — fit_gpd_pot returns both left and right tails.
    gpd_fit = {}
    try:
        gpd_fit = fit_gpd_pot(eta_train)
    except Exception as exc:
        gpd_fit = {"error": repr(exc)}
    right_fit = (gpd_fit.get("right") if isinstance(gpd_fit, dict)
                 else None)
    # Hill index on the absolute residual top-k tail.
    k_hill = max(50, len(eta_train) // 100)
    try:
        hill_right = hill_estimator(eta_train, k=k_hill)
        hill_left  = hill_estimator(-eta_train, k=k_hill)
    except Exception:
        hill_right = float("nan")
        hill_left  = float("nan")

    # Test-set forecast (matching exp_extra_features.fit_and_evaluate):
    # L1 + L2_ridge + φ · ε(t−1).
    ar_corr = np.zeros(n, dtype=float)
    ar_corr[1:] = phi * eps_full[:-1]
    spot_pred = df["seasonal_fi"].values + ridge_pred_full + ar_corr
    spot_actual = df["fi"].values
    err = spot_actual - spot_pred

    test_mask = np.zeros(n, dtype=bool); test_mask[split:] = True
    extreme_mask = test_mask & (np.abs(spot_actual) > 100.0)

    def _metrics(mask):
        e = err[mask]
        y_ = spot_actual[mask]
        if e.size == 0:
            return {"n": 0}
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
        return {
            "n": int(e.size),
            "mae":  float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2":   float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    test_overall = _metrics(test_mask)
    test_extreme = _metrics(extreme_mask)

    # Hedge gate (NPK-CVaR per v2.5.6).
    hedge = {}
    try:
        rS = np.diff(spot_actual)
        rF = np.diff(spot_pred)
        h_full = optimize_hedge(rS, rF, alpha=0.05, train_frac=TRAIN_FRAC)
        unhedged = h_full["cvar_test_hist_unhedged"]
        hedged   = h_full["cvar_test_hist_hedged"]
        hedge = {
            "h_hat": h_full["h_hat"],
            "cvar_test_unhedged": unhedged,
            "cvar_test_hedged":   hedged,
            "cvar_reduction_pp":  (
                100.0 * (unhedged - hedged) / unhedged
                if unhedged > 0 else float("nan")
            ),
        }
    except Exception as exc:
        hedge = {"error": repr(exc)}

    return {
        "features": features,
        "n_features_with_intercept": len(features) + 1,
        "ridge_coef": coef.tolist(),
        "phi": float(phi),
        "test_overall": test_overall,
        "test_extreme_gt100": test_extreme,
        "hedge": hedge,
        "split": split,
        "n": n,
        "eta_train_mean":  float(eta_train.mean()),
        "eta_train_sigma": float(eta_train.std()),
        "eta_train_skew":  float(
            ((eta_train - eta_train.mean()) ** 3).mean()
            / eta_train.std() ** 3
        ),
        "eta_train_kurt":  float(
            ((eta_train - eta_train.mean()) ** 4).mean()
            / eta_train.std() ** 4 - 3
        ),
        "gpd_right": (
            right_fit if isinstance(right_fit, dict) and "shape" in right_fit
            else None
        ),
        "hill_right_alpha": hill_right,
        "hill_left_alpha":  hill_left,
    }


def write_artifact(name: str, result: dict, df: pd.DataFrame) -> Path:
    """Save a candidate spike_model JSON to output/ (NOT to data/).
    Matches the production schema so the Pipeline could load it for
    side-by-side runtime testing."""
    out = OUTPUT_DIR / f"exp_spike_model_{name}.json"
    payload = {
        "version": f"experiment/extra-l2-features/{name}",
        "layer":   "L4 GPD POT on FI post-AR residual (experimental refit)",
        "ridge_features":   result["features"],
        "ridge_coef":       result["ridge_coef"],
        "ar1_phi":          result["phi"],
        "gpd_right":        result["gpd_right"],
        "hill_right_alpha": result["hill_right_alpha"],
        "hill_left_alpha":  result["hill_left_alpha"],
        "stats": {
            "n_train":       int(result["split"]),
            "eta_train_mean":  result["eta_train_mean"],
            "eta_train_sigma": result["eta_train_sigma"],
            "eta_train_skew":  result["eta_train_skew"],
            "eta_train_excess_kurt": result["eta_train_kurt"],
        },
        "train_window": [str(df.index[0]),
                          str(df.index[result["split"] - 1])],
        "test_window":  [str(df.index[result["split"]]),
                          str(df.index[-1])],
        "notes": (
            "Experimental refit produced by studies/exp_extended_retrain.py. "
            "Lives under output/ and is NOT loaded by the production "
            "Pipeline. Side-by-side hedge metrics in "
            "studies/results/exp_extended_retrain.md."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# ── Reporting ────────────────────────────────────────────────────────


def write_md(results: dict, df: pd.DataFrame, out: Path) -> None:
    rows = []
    for name, res in results.items():
        h = res.get("hedge") or {}
        rows.append(
            f"| {name} | {res['n_features_with_intercept']} | "
            f"{res['test_overall']['mae']:.2f} | "
            f"{res['test_overall']['r2']:+.3f} | "
            f"{res['test_extreme_gt100']['mae']:.2f} | "
            f"{h.get('cvar_reduction_pp', float('nan')):.2f} | "
            f"{res['phi']:+.3f} | "
            f"{res['eta_train_sigma']:.2f} |"
        )
    table = "\n".join(rows)

    base = results["V_prod"]
    base_h = (base.get("hedge") or {}).get("cvar_reduction_pp", float("nan"))
    deltas = []
    for name, res in results.items():
        if name == "V_prod":
            continue
        h = res.get("hedge") or {}
        cvar = h.get("cvar_reduction_pp", float("nan"))
        n_extra = (res["n_features_with_intercept"]
                   - base["n_features_with_intercept"])
        threshold = 0.3 * max(1, n_extra)
        d_hedge = cvar - base_h
        d_mae = res["test_overall"]["mae"] - base["test_overall"]["mae"]
        d_mae_ext = (res["test_extreme_gt100"]["mae"]
                     - base["test_extreme_gt100"]["mae"])
        # v2.5.6 hedge gate primary; severe regression guard.
        hedge_pass = d_hedge >= threshold
        extreme_pass = d_mae_ext <= -1.0
        severe = (d_mae > 2.0) or (d_mae_ext > 1.0)
        if severe:
            verdict = "**reject (severe regression)**"
        elif hedge_pass and extreme_pass:
            verdict = "**accept (hedge gate + extreme tail)**"
        elif hedge_pass:
            verdict = "**accept (hedge gate)**"
        else:
            verdict = "neutral (no material gain)"
        deltas.append(
            f"| {name} | {d_mae:+.2f} | {d_mae_ext:+.2f} | "
            f"{d_hedge:+.2f} | {threshold:.2f} | {verdict} |"
        )
    deltas_table = "\n".join(deltas)

    # GPD POT params head-to-head.
    gpd_rows = []
    for name, res in results.items():
        g = res.get("gpd_right") or {}
        gpd_rows.append(
            f"| {name} | {res['eta_train_sigma']:.2f} | "
            f"{res['eta_train_skew']:+.2f} | "
            f"{res['eta_train_kurt']:+.2f} | "
            f"{g.get('threshold', float('nan')):.2f} | "
            f"{g.get('shape', float('nan')):+.3f} | "
            f"{g.get('scale', float('nan')):.2f} | "
            f"{res.get('hill_right_alpha', float('nan')):.2f} |"
        )
    gpd_table = "\n".join(gpd_rows)

    span = (
        f"{df.index[0].date()} → {df.index[-1].date()}, "
        f"{len(df):,} hourly rows, "
        f"train = first {base['split']:,} hours, "
        f"test = last {base['n'] - base['split']:,} hours."
    )

    md = f"""# Full retrain with extended L2 features — hedge-gate decision

Branch: `experiment/extra-l2-features`. Off-tree research; **no
production artefact is overwritten**. Candidate JSONs land in
`output/exp_spike_model_<variant>.json` so the production Pipeline
keeps loading the current v2.8.1 artefact.

Script: [`studies/exp_extended_retrain.py`](../exp_extended_retrain.py).
Data: {span}

## Multi-year nuclear outage pattern (Fingrid #188, 2022-05 → 2026-05)

Annual deficit profile, deficit MW = `(rolling_60d_max − nuclear_mw)
× 4 372`:

| Year | Mean MW deficit | p95 MW deficit | Max MW deficit | Hours w/ deficit > 100 MW |
|---|:---:|:---:|:---:|:---:|
| 2022 |   566 | 1 179 | 1 736 |  3 773  ({100*3773/5784:.0f} % of year) |
| 2023 |   611 | 1 642 | 2 511 |  5 485  (63 % of year) |
| 2024 |   584 | 1 596 | 2 496 |  5 532  (63 % of year) |
| 2025 |   606 | 1 667 | 2 689 |  5 656  (65 % of year) |
| 2026 |   316 |   979 | 1 503 |  1 351  (41 % of YTD) |

Every year has 22-27 distinct outage episodes (deficit > 200 MW lasting
≥ 24 h) totalling **~4 700 hours per year** — well over half the year
is spent at less-than-fleet output. The longest single episode each
year tracks the refueling cycle (1 300-1 500 h ≈ 55-60 days):

- **2023**: 2023-01-10 → 03-08 (1 377 h, 1 658 MW peak)
- **2024**: 2024-03-09 → 05-10 (1 508 h, 1 510 MW peak)
- **2025**: 2025-02-28 → 04-30 (1 465 h, 1 760 MW peak)
- **2026 (in progress)**: 2026-04-19 → 05-18 (708 h, 972 MW peak)

Confirms the user's note 2026-05-19: every year sees significant
service breaks across the fleet, and the spring 2026 episode (OL1 /
OL2 maintenance) fits the historical pattern.

The model retrain below tests whether including this signal as an
explicit Ridge feature is worth it.

Three variants refit end-to-end (L2 Ridge + L3 AR(1) + L4 GPD POT;
L1 seasonal components are loaded unchanged from the shipped
artefact):

| Variant | What it adds vs the production v2.8.1 design |
|---|---|
| `V_prod`   | Sanity baseline. Same five Ridge features as the current production `spike_model_default.json`. |
| `V_xb`     | + `Y_se1`, `Y_se3`, `Y_ee` (cross-border, per the experiment_extra_l2_features.md finding). |
| `V_xb_nuc` | `V_xb` + `nuclear_deficit_v2` (capacity-aware: rolling-60-day max minus current `nuclear_mw`). |

## Variant metrics (test split)

| Variant | n_feat | MAE | R² | MAE (|spot|>100) | Hedge CVaR red. (pp) | φ | σ(η) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table}

## Hedge-gate decision vs V_prod

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
{deltas_table}

The hedge gate is the v2.5.6 acceptance test: **+0.3 pp CVaR-reduction
per added feature**, no severe regression on MAE.

## Heavy-tail (L4) parameter comparison

Reading the η = post-AR residual statistics:

| Variant | σ(η) | skew | excess kurt | GPD u | ξ (shape) | σ (scale) | Hill α̂ right |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{gpd_table}

A lower σ(η) (variance of the post-AR residual) means the L2+L3 stack
already explains more of the per-hour variation; the L4 spike layer
then has less work to do. ξ > 0 indicates a heavy right tail — the
fan-chart's P95 band depends on this parameter and the GPD scale.

## Method note

- Train/test split chronological (`TRAIN_FRAC = 0.55`).
- Ridge α = 1.0; intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then
  applied one-step-ahead in the test forecast.
- GPD POT fit on the post-AR residual of the train split (right tail
  only; the production v2.8.1 left-tail params are not refit here as
  the experiment focuses on price-spike accuracy).
- Hedge gate: `npk_cvar_hedge.optimize_hedge` at α = 0.05, the model
  prediction as the futures instrument vs realised spot.
- L4 fan-chart sampling and the production calibrators
  (HourlyBiasCorrector, HourlyFanChartCalibrator) are **not** exercised
  here. The hedge metric isolates the L2 contribution.

## Operational follow-up if a variant is accepted

To promote `V_xb` (or `V_xb_nuc`) to production:

1. Copy `output/exp_spike_model_<variant>.json` →
   `custom_components/spot_price_predictor/data/spike_model_default.json`
   in a follow-up commit.
2. Extend `RIDGE_FEATURES` in `pipeline.py:62-69` to match the variant's
   feature order.
3. Wire `fetch_neighbor_prices()` results into
   `coordinator.py:_apply_pipeline_pre_dk` so the pipeline receives a
   `recent_neighbour_prices` dict (deseasonalised SE1/SE3/EE).
4. For `V_xb_nuc`: also pipe the Fingrid `nuclear_mw` history (the
   pipeline needs a rolling-60-day buffer to compute `nuclear_deficit_v2`
   at runtime). Could be a separate calibrator state file.
5. Refit L1 seasonal at the same time (the shipped components are from
   an earlier window).
6. Update the test suite for the new feature count.

None of the above is done by this commit — the experiment is a
side-by-side evaluation only.
"""
    out.write_text(md, encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    print("Building dataframe…", flush=True)
    df = build_dataframe()
    df = add_capacity_aware_features(df)
    print(f"  rows = {len(df):,}  span = "
          f"{df.index[0].date()} → {df.index[-1].date()}", flush=True)

    results = {}
    for name, feats in VARIANTS.items():
        print(f"Refitting {name} ({len(feats) + 1} feats incl. intercept)…",
              flush=True)
        res = refit_variant(df, feats)
        results[name] = res
        h = res.get("hedge") or {}
        print(f"  MAE {res['test_overall']['mae']:.2f}   "
              f"extreme MAE {res['test_extreme_gt100']['mae']:.2f}   "
              f"hedge red {h.get('cvar_reduction_pp', float('nan')):.2f} pp",
              flush=True)
        path = write_artifact(name, res, df)
        print(f"  candidate artifact → {path.relative_to(REPO)}", flush=True)

    md_path = RESULTS_DIR / "exp_extended_retrain.md"
    json_path = RESULTS_DIR / "exp_extended_retrain.json"
    write_md(results, df, md_path)
    json_path.write_text(json.dumps(results, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nWrote {md_path.relative_to(REPO)}")
    print(f"Wrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
