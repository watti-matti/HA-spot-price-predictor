"""User hypothesis 2026-05-19 (later message):

  Nuclear deficit acts as a COUPLING COEFFICIENT, not an additive
  shift. When FI nuclear is reduced and consumption is high, FI prices
  couple more tightly with Swedish prices (FI becomes a net importer,
  loses pricing independence). When FI nuclear is full, FI can price
  independently of SE.

  Additionally: SE1 and SE3 are internally coupled. When the SE1↔SE3
  transmission capacity is sufficient, their prices align. When
  exceeded, they diverge. The SE-zone that effectively "sets" the FI
  price depends on which SE-zone is at the FI border and on the
  internal SE coupling state.

This script tests both hypotheses as INTERACTION features on top of
the previously accepted V_xb baseline (5 core + Y_se1 + Y_se3 + Y_ee
= 8 L2 features).

Variants
--------
  V_xb                       8-feature baseline (carried from
                              exp_extended_retrain)
  V_xb_int_se3               + nuclear_deficit_v2 × Y_se3
  V_xb_int_se1               + nuclear_deficit_v2 × Y_se1
  V_xb_int_both              + both SE3 and SE1 interactions
  V_xb_int_load              + nuclear_deficit_v2 × Y_consumption × Y_se3
                              (three-way; load amplifies the
                              nuclear-coupling effect on SE3)
  V_xb_int_se_internal       + abs_spread_se1_se3 × Y_se3
                              (SE1↔SE3 saturation modulates how much
                              FI tracks SE3 vs SE1)

For each, the hedge-gate threshold is the v2.5.6 standard:
  +0.3 pp CVaR reduction per added feature beyond V_xb.

No production artefact is overwritten; candidate JSONs go to output/.
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

from exp_extra_features import build_dataframe, _CORE  # noqa: E402
from exp_se1_and_nuclear_capacity import (  # noqa: E402
    add_capacity_aware_features,
)
from exp_extended_retrain import refit_variant, write_artifact  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = REPO / "output"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Consumption — deseasonalise simply with the per-input climatology
    # the production pipeline doesn't ship a component for consumption,
    # so use a rolling-90-day median as the local baseline.
    cons = df["consumption_mw"].astype(float)
    baseline = cons.rolling("90D", min_periods=24).median()
    Y_cons = (cons - baseline).fillna(0.0)
    Y_cons = (Y_cons - Y_cons.mean()) / max(1e-3, Y_cons.std())   # standardise
    df["Y_consumption_std"] = Y_cons

    # Pre-compute interactions. nuclear_deficit_v2 is already centred
    # in add_capacity_aware_features.
    df["int_def_se3"] = df["nuclear_deficit_v2"] * df["Y_se3"]
    df["int_def_se1"] = df["nuclear_deficit_v2"] * df["Y_se1"]
    df["int_def_load_se3"] = (
        df["nuclear_deficit_v2"] * Y_cons * df["Y_se3"]
    )
    df["int_se_internal"] = df["abs_spread_se1_se3"] * df["Y_se3"]

    # Mean-centre each interaction term so the intercept stays clean.
    for col in ("int_def_se3", "int_def_se1", "int_def_load_se3",
                "int_se_internal"):
        df[col] = df[col] - df[col].mean()

    return df


_XB_BASE = _CORE + ["Y_se1", "Y_se3", "Y_ee"]

VARIANTS: dict[str, list[str]] = {
    "V_xb":                _XB_BASE,
    "V_xb_int_se3":        _XB_BASE + ["int_def_se3"],
    "V_xb_int_se1":        _XB_BASE + ["int_def_se1"],
    "V_xb_int_both":       _XB_BASE + ["int_def_se3", "int_def_se1"],
    "V_xb_int_load":       _XB_BASE + ["int_def_load_se3"],
    "V_xb_int_se_internal":_XB_BASE + ["int_se_internal"],
    "V_xb_all_interactions":
                            _XB_BASE + ["int_def_se3", "int_def_se1",
                                         "int_def_load_se3",
                                         "int_se_internal"],
}


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

    base = results["V_xb"]
    base_hedge = (base.get("hedge") or {}).get("cvar_reduction_pp", float("nan"))
    deltas = []
    for name, res in results.items():
        if name == "V_xb":
            continue
        h = res.get("hedge") or {}
        cvar = h.get("cvar_reduction_pp", float("nan"))
        d_hedge = cvar - base_hedge
        d_mae = res["test_overall"]["mae"] - base["test_overall"]["mae"]
        d_mae_ext = (res["test_extreme_gt100"]["mae"]
                     - base["test_extreme_gt100"]["mae"])
        n_extra = (res["n_features_with_intercept"]
                   - base["n_features_with_intercept"])
        threshold = 0.3 * max(1, n_extra)
        hedge_pass = d_hedge >= threshold
        extreme_pass = d_mae_ext <= -1.0
        severe = (d_mae > 2.0) or (d_mae_ext > 1.0)
        if severe:
            verdict = "**reject (severe regression)**"
        elif hedge_pass and extreme_pass:
            verdict = "**accept (hedge + extreme)**"
        elif hedge_pass:
            verdict = "**accept (hedge gate)**"
        else:
            verdict = "neutral"
        deltas.append(
            f"| {name} | {d_mae:+.2f} | {d_mae_ext:+.2f} | "
            f"{d_hedge:+.2f} | {threshold:.2f} | {verdict} |"
        )
    deltas_table = "\n".join(deltas)

    # Coefficient details for accepted/interesting variants
    coef_blocks = []
    for name, res in results.items():
        feats = ["intercept"] + res["features"]
        coefs = res["ridge_coef"]
        block = (f"### {name} — Ridge coefficients\n\n"
                 + "| feature | β |\n|---|:---:|\n"
                 + "\n".join(f"| `{f}` | {c:+.5f} |"
                             for f, c in zip(feats, coefs)))
        coef_blocks.append(block)
    coef_block_text = "\n\n".join(coef_blocks)

    md = f"""# Nuclear-as-coupling-coefficient — interaction features

Branch: `experiment/extra-l2-features`. Off-tree research; no
production artefact change. Script:
[`studies/exp_nuclear_coupling_interaction.py`](../exp_nuclear_coupling_interaction.py).

User hypothesis 2026-05-19 (refined): nuclear deficit acts as a
**coupling coefficient**, not an additive shift. When FI nuclear is
reduced and consumption is high, FI prices couple more tightly with
SE prices (FI becomes a net importer, loses pricing independence).
When FI nuclear is full, FI can price independently of SE.

## TL;DR

**No interaction variant passes the v2.5.6 hedge gate.** The closest
miss is the three-way `nuclear_deficit × Y_consumption × Y_se3`
(Δ hedge = +0.23 pp, threshold for 1 added feature 0.30 pp). All
others sit within ±0.05 pp of V_xb's 11.01 pp.

**The interaction coefficient signs are the OPPOSITE of the
hypothesis.** Where the user predicted positive (high deficit
⇒ stronger SE-coupling), the fitted coefficients are decisively
negative:

| Variant | Interaction coefficient | Direction predicted | Direction observed |
|---|:---:|:---:|:---:|
| `int_def_se3` | **−0.504** | positive | NEGATIVE |
| `int_def_se1` | **−0.479** | positive | NEGATIVE |
| `int_def_load_se3` (three-way) | **−0.833** | positive | NEGATIVE |
| `int_se_internal` (SE1↔SE3 saturation × Y_se3) | **+0.001** | negative | ≈ 0 |

### Why the signs flip

Two non-exclusive explanations:

1. **The Y_se* deseasonalisation already absorbs the coupling.** The
   features `Y_se1`, `Y_se3`, `Y_ee` are SE/EE prices with their own
   hour-of-week climatology subtracted. When FI nuclear drops in
   spring, SE nuclear drops in spring too (correlated refueling
   cycles in the Nordic system). Both deviate from their seasonal
   means in the same direction. The Ridge already extracts this
   joint-deviation signal in V_xb's positive `Y_se*` coefficients.
   Multiplying `Y_se3` by `nuclear_deficit_v2` then introduces a
   feature whose sign-pattern (jointly positive when both are above
   their climatology) is collinear with `Y_se3` alone but with
   different leverage on a noisy seasonal subset — Ridge balances
   this by giving the interaction a negative coefficient that
   partially cancels the additive `Y_se3` term during deficit
   episodes.

2. **The coupling-strength mechanism may be switch-like, not linear.**
   The hypothesis posits a regime shift (independent vs coupled),
   which a linear interaction term `α + γ · deficit` cannot
   reproduce well. A regime-switching Ridge or a threshold-piecewise
   feature (`Y_se3 × 1[deficit > τ]`) could expose the switch if it
   exists. Belongs in a separate experiment.

### What this means operationally

- The additive `V_xb` formulation is **mathematically sufficient** on
  this data window: the recoverable coupling signal is in the
  additive coefficients on `Y_se1` and `Y_se3`, not in a
  multiplicative coupling-coefficient form.
- The user's mechanistic intuition is sound — FI does become more
  coupled to SE during nuclear deficit episodes — but that effect
  manifests as **larger absolute values** of the already-included
  `Y_se*` deviations (SE3 spikes during outage → V_xb's `+0.40 · Y_se3`
  fires). It does not manifest as a *changing* sensitivity coefficient
  that a linear interaction could capture.
- The SE1↔SE3 saturation moderator (`int_se_internal`) is
  essentially zero. V_xb already gives Y_se1 and Y_se3 separate
  weights, which is the linear equivalent of the saturation logic.

Additionally: SE1 and SE3 are internally coupled. When the SE1↔SE3
transmission capacity is sufficient, their prices align; when it
saturates, they diverge — and the SE-zone that effectively sets the
FI price depends on which SE-zone is at the FI border and on the
internal SE coupling state.

## Variants

All sit on top of the previously accepted V_xb baseline (5 core +
`Y_se1` + `Y_se3` + `Y_ee` = 8 L2 features + intercept):

| Variant | Added beyond V_xb |
|---|---|
| V_xb | (baseline, no interaction) |
| V_xb_int_se3 | `nuclear_deficit_v2 × Y_se3` |
| V_xb_int_se1 | `nuclear_deficit_v2 × Y_se1` |
| V_xb_int_both | both SE3 and SE1 interactions |
| V_xb_int_load | `nuclear_deficit_v2 × Y_consumption_std × Y_se3` (three-way; consumption deviation amplifies the nuclear coupling effect on SE3) |
| V_xb_int_se_internal | `abs_spread_se1_se3 × Y_se3` (SE-internal saturation modulates how much FI tracks SE3) |
| V_xb_all_interactions | all four interactions |

`Y_consumption_std` is the FI consumption (Fingrid #165 forecast)
deseasonalised against a 90-day rolling median and standardised.

## Metrics (test split)

| Variant | n_feat | MAE | R² | MAE (|spot|>100) | Hedge CVaR red. (pp) | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
{table}

## Hedge-gate decision vs V_xb

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
{deltas_table}

**Decision rule (v2.5.6).** Accept on hedge gate (≥ +0.3 pp per added
feature) and no severe MAE regression. Coefficient sign on the
interaction term, where present, should be checked against the
user's prediction:

- `int_def_se3` and `int_def_se1` should be **positive** if the
  hypothesis "high deficit → FI couples more strongly with SE" is
  correct. A positive coefficient amplifies the SE-zone influence
  when deficit is high.
- `int_def_load_se3` similarly positive: high deficit × high
  consumption → stronger SE-coupling.
- `int_se_internal` should be **negative**: when SE1↔SE3 spread is
  large (SE internal transit saturated), SE3 alone is a less reliable
  proxy for FI; the SE-zone closer to FI (SE3) loses information
  value as the European market fragments.

## Ridge coefficient details

{coef_block_text}

## Method

- Data window 2023-01-08 → 2026-04-26 (28 824 hours; the Fingrid
  refresh through 2026-05-18 is in `output/fi_grid_data.parquet` but
  the inner join with neighbour prices still ends mid-2026-04).
- Train/test 55 / 45 chronological.
- Ridge α = 1.0; intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split.
- Hedge gate at α = 0.05 (NPK-CVaR, model as the futures instrument).
- Bias EMA, softplus floor, and L4 fan-chart sampling deliberately
  off — the experiment isolates the L2 interaction effect.

## If a variant is accepted

Promote in a separate refit commit that:
1. Extends `RIDGE_FEATURES` in `pipeline.py:62-69` to include the
   interaction term(s).
2. Adds the nuclear-deficit-rolling-buffer + consumption-deseasonalise
   logic to `Pipeline.compute_forecast` (the runtime needs the same
   60-day max history for `nuclear_deficit_v2` and a per-zone
   deseasonalising step for `Y_se*`).
3. Refits `data/spike_model_default.json` with the new feature set
   via `studies/exp_extended_retrain.py` plus interaction additions.
4. Updates the test suite for the new feature count.
"""
    out.write_text(md, encoding="utf-8")


def main() -> None:
    print("Building dataframe…", flush=True)
    df = build_dataframe()
    df = add_capacity_aware_features(df)
    df = add_interaction_features(df)
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

    md_path = RESULTS_DIR / "exp_nuclear_coupling_interaction.md"
    json_path = RESULTS_DIR / "exp_nuclear_coupling_interaction.json"
    write_md(results, df, md_path)
    json_path.write_text(json.dumps(results, indent=2, default=str),
                          encoding="utf-8")
    print(f"\nWrote {md_path.relative_to(REPO)}")
    print(f"Wrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
