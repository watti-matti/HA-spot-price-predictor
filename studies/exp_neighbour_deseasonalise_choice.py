"""Is the deseasonalisation of Y_se1 / Y_se3 / Y_ee empirically justified?

User question (2026-05-19): SE hydro reservoirs and SE3 prices have
clear seasonality (winter peak / spring melt crash / summer trough /
autumn rise). FennoSkan-1/2 transmission CAPACITY between FI and SE3
is fixed, so the cross-border *influence* on FI is determined by:
  (a) the SE-side price level (seasonal), and
  (b) whether the cable is binding (event-driven, not seasonal).

The current production v2.9.0 design uses deseasonalised neighbour
prices (Y_se* = se* − seasonal_se*) on the assumption that FI's own
L1 seasonal_fi already absorbs the SE-correlated seasonal component.
That assumption was never tested. This script tests four variants
under the same v2.5.6 hedge gate used in exp_extended_retrain.py.

Variants (each adds the SE/EE block to the 5-feature core):

  V_xb         deseasonalised: Y_se1, Y_se3, Y_ee
               (current production v2.9.0)
  V_xb_raw     raw mean-centred: se1, se3, ee
               (lets Ridge weigh the seasonal SE level directly; if
               FI's seasonal_fi does NOT absorb SE seasonality this
               should win)
  V_xb_hybrid  both: Y_se1, Y_se3, Y_ee  +  seas_se1, seas_se3, seas_ee
               (decoupled — the climatology and the deviation get
               independent weights; a forensic test for whether the
               seasonal part carries explanatory power that the
               deseasonalised form throws away)
  V_xb_spread  inter-zone spreads + raw_se3:
               raw_se3_mc, spread_se1_se3_v2, spread_se3_ee_v2
               (SE3 carries the FI-cable level; spreads carry
               transit-saturation. Tests whether the operative
               signal is "absolute SE3 price + transit imbalance"
               rather than three deseasonalised neighbour series.)

Acceptance: must beat V_xb hedge CVaR reduction by +0.3 pp per ADDED
feature relative to V_xb (the v2.5.6 threshold). Hybrid adds 3, raw
adds 0 (same count, different content), spread adds 0.
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

from exp_extra_features import build_dataframe, _CORE  # noqa: E402
from v2510_layer3_ar_wind import fit_ridge, fit_ar1, TRAIN_FRAC  # noqa: E402
from npk_cvar_hedge import optimize_hedge  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Alternative neighbour feature forms ─────────────────────────────


def add_alternative_neighbour_forms(df: pd.DataFrame) -> pd.DataFrame:
    """Adds raw / seasonal / spread variants alongside Y_se* already
    present in `df` (which build_dataframe() produced)."""
    for nb in ("se1", "se3", "ee"):
        raw = df[nb].values.astype(float)
        # NaN-safe mean-centre (no leakage; centring is symmetric).
        mu = float(np.nanmean(raw))
        df[f"raw_{nb}_mc"] = np.where(np.isnan(raw), 0.0, raw - mu)
        seas = df[f"seasonal_{nb}"].values.astype(float)
        mu_s = float(np.nanmean(seas))
        df[f"seas_{nb}_mc"] = np.where(np.isnan(seas), 0.0, seas - mu_s)

    # Inter-zone *raw* spreads — capture transit-saturation regardless
    # of whether the level is seasonal. Mean-centred for Ridge stability.
    for left, right in (("se1", "se3"), ("se3", "ee")):
        sp = df[left].values - df[right].values
        df[f"spread_{left}_{right}_v2"] = sp - float(np.nanmean(sp))
    return df


# ── Variant feature sets ─────────────────────────────────────────────

VARIANTS: dict[str, list[str]] = {
    "V_xb":        _CORE + ["Y_se1", "Y_se3", "Y_ee"],
    "V_xb_raw":    _CORE + ["raw_se1_mc", "raw_se3_mc", "raw_ee_mc"],
    "V_xb_hybrid": _CORE + ["Y_se1", "Y_se3", "Y_ee",
                             "seas_se1_mc", "seas_se3_mc", "seas_ee_mc"],
    "V_xb_spread": _CORE + ["raw_se3_mc",
                             "spread_se1_se3_v2", "spread_se3_ee_v2"],
}


# ── Fit + evaluate (mirrors exp_extra_features.fit_and_evaluate) ────


def fit_and_evaluate(df: pd.DataFrame, features: list[str],
                     alpha: float = 1.0) -> dict:
    n = len(df)
    split = int(n * TRAIN_FRAC)
    y = df["Y_fi"].values
    X = np.column_stack([np.ones(n)] + [df[f].values for f in features])
    coef = fit_ridge(X[:split], y[:split], alpha=alpha)
    ridge_pred = X @ coef
    eps = y - ridge_pred
    phi, _ = fit_ar1(eps[:split])
    ar_contribution = np.zeros(n, dtype=float)
    ar_contribution[1:] = phi * eps[:-1]
    spot_pred = df["seasonal_fi"].values + ridge_pred + ar_contribution
    spot_actual = df["fi"].values
    err = spot_actual - spot_pred

    def _m(mask: np.ndarray) -> dict:
        e = err[mask]; y_ = spot_actual[mask]
        if e.size == 0:
            return {"n": 0}
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
        return {
            "n": int(e.size),
            "mae":  float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2":   float(1.0 - ss_res / ss_tot) if ss_tot > 0
                    else float("nan"),
        }

    test_mask = np.zeros(n, dtype=bool); test_mask[split:] = True
    extreme_mask = test_mask & (np.abs(spot_actual) > 100.0)

    hedge: dict = {}
    try:
        rS = np.diff(spot_actual)
        rF = np.diff(spot_pred)
        h = optimize_hedge(rS, rF, alpha=0.05, train_frac=TRAIN_FRAC)
        unhedged = h["cvar_test_hist_unhedged"]
        hedged   = h["cvar_test_hist_hedged"]
        hedge = {
            "h_hat": h["h_hat"],
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
        "test_overall": _m(test_mask),
        "test_extreme_gt100": _m(extreme_mask),
        "hedge": hedge,
    }


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
            f"{res['phi']:+.3f} |"
        )
    table = "\n".join(rows)

    # Coefficient lines for the SE/EE block.
    coef_lines = []
    for name, res in results.items():
        feats = ["intercept"] + res["features"]
        coefs = res["ridge_coef"]
        pairs = []
        for f, c in zip(feats, coefs):
            if any(tok in f for tok in
                   ("se1", "se3", "ee", "spread", "seas_", "raw_")):
                pairs.append(f"{f}={c:+.3f}")
        coef_lines.append(f"- **{name}**: " + ", ".join(pairs))
    coef_block = "\n".join(coef_lines)

    base = results["V_xb"]
    base_hedge = (base.get("hedge") or {}).get(
        "cvar_reduction_pp", float("nan"))
    base_n = base["n_features_with_intercept"]

    delta_rows = []
    for name, res in results.items():
        if name == "V_xb":
            continue
        h = res.get("hedge") or {}
        cvar = h.get("cvar_reduction_pp", float("nan"))
        n_extra = res["n_features_with_intercept"] - base_n
        threshold = 0.3 * max(1, abs(n_extra)) if n_extra > 0 else 0.3
        d_hedge = cvar - base_hedge
        d_mae = (res["test_overall"]["mae"]
                 - base["test_overall"]["mae"])
        d_mae_ext = (res["test_extreme_gt100"]["mae"]
                     - base["test_extreme_gt100"]["mae"])
        verdict = ("✅ beats V_xb" if (n_extra > 0 and d_hedge >= threshold)
                   or (n_extra <= 0 and d_hedge >= 0.3)
                   else "❌ does not justify replacing V_xb")
        delta_rows.append(
            f"| {name} | {n_extra:+d} | {d_hedge:+.2f} | "
            f"{threshold:.2f} | {d_mae:+.2f} | {d_mae_ext:+.2f} | "
            f"{verdict} |"
        )
    delta_block = "\n".join(delta_rows)

    md = f"""# Should Y_se1 / Y_se3 / Y_ee be deseasonalised?

Branch: `experiment/extra-l2-features`. Off-tree report. Script:
[`studies/exp_neighbour_deseasonalise_choice.py`](../exp_neighbour_deseasonalise_choice.py).

Tests whether the v2.9.0 production choice to deseasonalise the
cross-border neighbour prices (Y_se* = se* − seasonal_se*) is the
right one, or whether raw / hybrid / spread forms encode the
SE-FI coupling better.

## Conceptual framing

- SE1 (Luleå area, hydro-dominated) and SE3 (central, includes
  Stockholm) couple to FI through FennoSkan-1/2 cables (~1100 MW) at
  the **SE3** node. There is no direct FI↔SE1 cable.
- SE3 price has hydro-driven seasonality: winter peak (heating + low
  inflow), spring crash (snowmelt fills reservoirs), summer trough,
  autumn climb.
- Transit capacity is constant, but whether the cable BINDS is
  event-driven: it binds when the SE↔FI spread is large enough
  relative to capacity, which happens disproportionately during
  spike hours (low SE3 hydro + high FI demand or low FI nuclear).
- The deseasonalisation question: does FI's own `seasonal_fi`
  artefact already absorb the SE-correlated seasonal signal (so
  feeding the FI Ridge the *residual* Y_se* is sufficient), or does
  the seasonal SE3 level carry independent information about FI
  cable flow that gets discarded by L1 subtraction?

## Setup

- Data window: same as `exp_extended_retrain.md` (cached parquets,
  2023-01-08 → 2026-04-26).
- 55 / 45 chronological train / test.
- L1 (seasonal_fi) unchanged; L2 Ridge alpha=1.0; L3 AR(1) refit
  per variant. L4 not refit — the comparison is on point error and
  hedge CVaR.

## Headline metrics

| Variant | k (incl. intercept) | MAE | R² | MAE \\|spot\\|>100 | Hedge ΔCVaR % | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
{table}

## SE/EE block coefficients

{coef_block}

## Δ vs V_xb (the production design) — v2.5.6 gate

Threshold: +0.3 pp hedge CVaR reduction per ADDED feature
(or just +0.3 pp if feature count is unchanged but the form
differs).

| Variant | Δ features | Δhedge pp | Threshold pp | ΔMAE | ΔMAE>100 | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|---|
{delta_block}

## Interpretation

The verdict above answers the empirical question directly. The
*conceptual* read-out independent of the numbers:

- **If V_xb_raw wins**: FI's seasonal_fi does NOT fully absorb
  SE-driven seasonality. The seasonal SE3 LEVEL carries information
  the deseasonalised residual loses. Production should switch to
  raw mean-centred neighbour prices.
- **If V_xb_hybrid wins by ≥0.9 pp** (3 extra features × 0.3 pp):
  the seasonal and residual SE components carry independent
  signal. Production should expand to both, or at minimum revisit
  the L1 components for SE3 (the current SE3 climatology may be
  miscalibrated).
- **If V_xb_spread wins**: the operative cross-border signal is
  raw SE3 level + transit-saturation spreads, not three
  deseasonalised neighbour series. This would be the cleanest
  win — fewer features, more physical, no leakage on FI itself.
- **If V_xb is best**: the deseasonalisation choice is justified.
  Seasonal_fi + Y_se* is sufficient because FI seasonality already
  encodes the heating-demand component that SE3 hydro
  seasonality correlates with.

Whichever variant wins, the underlying mechanism documented in
`studies/results/exp_extended_retrain.md` (SE3 carries FI cable
state, SE1 carries upstream hydro inflow, EE carries the
Baltic-side coupling) is unchanged — only the encoding moves.
"""
    out.write_text(md, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading dataframe…")
    df = build_dataframe()
    df = add_alternative_neighbour_forms(df)
    print(f"  rows: {len(df):,}  span: {df.index[0]} → {df.index[-1]}")

    results: dict[str, dict] = {}
    for name, feats in VARIANTS.items():
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"  [{name}] SKIP — missing columns: {missing}")
            continue
        print(f"  fitting {name} ({len(feats)+1} features incl. intercept)…")
        results[name] = fit_and_evaluate(df, feats)

    out_md   = RESULTS_DIR / "exp_neighbour_deseasonalise_choice.md"
    out_json = RESULTS_DIR / "exp_neighbour_deseasonalise_choice.json"
    write_md(results, df, out_md)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
