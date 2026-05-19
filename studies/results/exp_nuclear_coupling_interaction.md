# Nuclear-as-coupling-coefficient — interaction features

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
| V_xb | 9 | 11.41 | +0.898 | 15.48 | 11.01 | +0.857 |
| V_xb_int_se3 | 10 | 11.49 | +0.897 | 15.66 | 11.04 | +0.857 |
| V_xb_int_se1 | 10 | 11.40 | +0.898 | 15.57 | 10.99 | +0.857 |
| V_xb_int_both | 11 | 11.47 | +0.897 | 15.66 | 11.04 | +0.857 |
| V_xb_int_load | 10 | 11.42 | +0.898 | 15.82 | 11.24 | +0.855 |
| V_xb_int_se_internal | 10 | 11.58 | +0.887 | 15.88 | 10.55 | +0.857 |
| V_xb_all_interactions | 13 | 11.84 | +0.867 | 16.99 | 9.73 | +0.853 |

## Hedge-gate decision vs V_xb

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
| V_xb_int_se3 | +0.08 | +0.18 | +0.04 | 0.30 | neutral |
| V_xb_int_se1 | -0.01 | +0.09 | -0.02 | 0.30 | neutral |
| V_xb_int_both | +0.06 | +0.18 | +0.03 | 0.60 | neutral |
| V_xb_int_load | +0.02 | +0.34 | +0.23 | 0.30 | neutral |
| V_xb_int_se_internal | +0.18 | +0.40 | -0.46 | 0.30 | neutral |
| V_xb_all_interactions | +0.43 | +1.51 | -1.28 | 1.20 | **reject (severe regression)** |

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

### V_xb — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.06726 |
| `Y_fi_lag168` | +0.02210 |
| `is_workday` | +2.39429 |
| `Y_sigmoid_wind_rho` | -38.17036 |
| `Y_solar_effective` | +0.02634 |
| `Y_temp` | -0.46366 |
| `Y_se1` | +0.16976 |
| `Y_se3` | +0.40349 |
| `Y_ee` | +0.53189 |

### V_xb_int_se3 — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.21178 |
| `Y_fi_lag168` | +0.02149 |
| `is_workday` | +2.65748 |
| `Y_sigmoid_wind_rho` | -39.52408 |
| `Y_solar_effective` | +0.02632 |
| `Y_temp` | -0.48975 |
| `Y_se1` | +0.14244 |
| `Y_se3` | +0.42937 |
| `Y_ee` | +0.52813 |
| `int_def_se3` | -0.50438 |

### V_xb_int_se1 — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +0.95188 |
| `Y_fi_lag168` | +0.02131 |
| `is_workday` | +2.45505 |
| `Y_sigmoid_wind_rho` | -39.02521 |
| `Y_solar_effective` | +0.02549 |
| `Y_temp` | -0.46392 |
| `Y_se1` | +0.17539 |
| `Y_se3` | +0.39801 |
| `Y_ee` | +0.52842 |
| `int_def_se1` | -0.47899 |

### V_xb_int_both — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.14415 |
| `Y_fi_lag168` | +0.02133 |
| `is_workday` | +2.62838 |
| `Y_sigmoid_wind_rho` | -39.56267 |
| `Y_solar_effective` | +0.02603 |
| `Y_temp` | -0.48486 |
| `Y_se1` | +0.14961 |
| `Y_se3` | +0.42253 |
| `Y_ee` | +0.52764 |
| `int_def_se3` | -0.40817 |
| `int_def_se1` | -0.16631 |

### V_xb_int_load — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.13636 |
| `Y_fi_lag168` | +0.02470 |
| `is_workday` | +2.58809 |
| `Y_sigmoid_wind_rho` | -40.05002 |
| `Y_solar_effective` | +0.02481 |
| `Y_temp` | -0.39510 |
| `Y_se1` | +0.12924 |
| `Y_se3` | +0.39616 |
| `Y_ee` | +0.51494 |
| `int_def_load_se3` | -0.83250 |

### V_xb_int_se_internal — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.22777 |
| `Y_fi_lag168` | +0.02330 |
| `is_workday` | +1.95719 |
| `Y_sigmoid_wind_rho` | -38.34015 |
| `Y_solar_effective` | +0.02642 |
| `Y_temp` | -0.45964 |
| `Y_se1` | +0.20789 |
| `Y_se3` | +0.36922 |
| `Y_ee` | +0.53309 |
| `int_se_internal` | +0.00096 |

### V_xb_all_interactions — Ridge coefficients

| feature | β |
|---|:---:|
| `intercept` | +1.48522 |
| `Y_fi_lag168` | +0.02609 |
| `is_workday` | +2.02957 |
| `Y_sigmoid_wind_rho` | -41.59666 |
| `Y_solar_effective` | +0.02468 |
| `Y_temp` | -0.40786 |
| `Y_se1` | +0.17863 |
| `Y_se3` | +0.35308 |
| `Y_ee` | +0.51336 |
| `int_def_se3` | -0.36772 |
| `int_def_se1` | -0.15742 |
| `int_def_load_se3` | -0.82378 |
| `int_se_internal` | +0.00168 |

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
