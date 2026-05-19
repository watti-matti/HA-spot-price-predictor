# v2.10.0 — Cross-border features (Y_se1, Y_se3, Y_ee) promoted to production

This release extends the L2 non-seasonal Ridge from 5 to 8 features by
adding deseasonalised Swedish (SE1, SE3) and Estonian (EE) spot prices.
The change is the outcome of the `experiment/extra-l2-features` branch
and is gated by the v2.5.6 NPK-CVaR hedge test.

## Highlights

- **8-feature L2 Ridge.** `RIDGE_FEATURES` extended with `Y_se1`,
  `Y_se3`, `Y_ee` — neighbour spot prices deseasonalised against the
  per-zone L1 components already shipped in
  `seasonal_components_default.json`.
- **Hedge-gate result that justified the promote.** On a held-out
  2024-10 → 2026-04 FI test window the new 8-feature design improves
  the NPK-CVaR hedge reduction from **2.0 pp → 11.0 pp** (+9.0 pp)
  and the extreme-price-hour (`|spot| > 100`) MAE from **20.14 → 15.48
  EUR/MWh** (−23 %). Overall MAE drifts up by 1.1 EUR/MWh on calm
  hours, well within the hedge-primary regression guard of 2.0
  EUR/MWh that v2.5.6 established. Full numbers in
  [`studies/results/exp_extended_retrain.md`](exp_extended_retrain.md).
- **Pipeline API extended.** `Pipeline.compute_forecast` now accepts
  an optional `recent_neighbour_prices={"se1": …, "se3": …, "ee": …}`
  argument. When the caller supplies it (raw EUR/MWh prices aligned
  with the forecast timestamps), the pipeline deseasonalises and
  scores the new features. When omitted — or when individual zones
  are missing / NaN — the corresponding columns are zero, equivalent
  to the v2.8.x behaviour (graceful fallback).
- **Coordinator wiring.** The coordinator now passes the existing
  `fetch_neighbor_prices()` results into the pipeline through a new
  `_align_neighbour_prices()` helper. No new fetcher, no new API key.
- **Feature-list-driven pipeline.** `pipeline.py` now reads the
  `ridge_features` list from the artifact and builds the design
  matrix in that exact order. Future feature additions / removals
  only need an artifact refit; no code change.

## What was rejected

The branch also tested — and rejected — every variant that added
nuclear-deficit information:

- Additive `nuclear_deficit` (`max(0, 1 − nuclear_mw)`): neutral on
  the 2023-2026 window because the post-OL3 fleet rarely runs below
  capacity.
- Capacity-aware `nuclear_deficit_v2`
  (`rolling_60d_max − nuclear_mw`, activates during the real spring
  refueling outages): also neutral.
- Interaction forms (`nuclear_deficit × Y_se3`, `nuclear_deficit ×
  Y_consumption × Y_se3`, etc., per the user's "nuclear as coupling
  coefficient" hypothesis): all sit ±0.5 pp of V_xb's hedge metric;
  none clears the +0.3 pp / added-feature threshold. Interaction
  coefficients also came out in the opposite sign from the hypothesis,
  suggesting the coupling mechanism — while real — is already absorbed
  by the additive `Y_se*` deviations (FI nuclear outages coincide with
  SE refueling outages; both deviate above seasonal climatology
  together, and the additive Ridge term picks that up).

Detailed analysis in
[`studies/results/exp_nuclear_coupling_interaction.md`](exp_nuclear_coupling_interaction.md)
and
[`studies/results/experiment_se1_and_nuclear_capacity.md`](experiment_se1_and_nuclear_capacity.md).

## Breaking changes

- **`data/spike_model_default.json` artifact schema.** `ridge_coef` is
  now length 9 (was 6) and `ridge_features` lists 8 feature names
  (was 5; pipeline always prepends `intercept`). Any tool reading the
  artifact directly must handle the new length / names.
- **Pipeline runtime API.** `Pipeline._features` is a new public-ish
  attribute carrying the ordered feature list from the artifact. Old
  test code that hardcoded a 6-coefficient assertion fails (the
  in-tree pytest suite is updated).
- The sensor schema is **unchanged**. All forecast-row keys
  (`spot_eur_mwh`, `consumer_eur_kwh`, `P5..P95_eur_mwh`, etc.) and
  all duration-forecast keys (`dk_cheap_eur_mwh[24]` etc.) keep their
  names and meanings.

## Migration / no-op for downstream consumers

- **Home Assistant users**: HACS auto-updates the integration. After
  the next coordinator cycle the spot/consumer forecasts and the
  fan-chart bands reflect the new 8-feature model.
- **EMHASS / template automations** that read sensor attributes:
  unchanged. The schema is the same — only the numerical values shift.
- **HACS dashboards (`ha_dashboard.yaml`)**: unchanged.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/pipeline.py` | `_build_features` is now feature-list-driven; `compute_forecast` takes `recent_neighbour_prices`. `RIDGE_FEATURES` expanded to 9 names. |
| `custom_components/spot_price_predictor/coordinator.py` | `_apply_pipeline_pre_dk` accepts `neighbor` dict and aligns it via the new `_align_neighbour_prices()` helper before calling the pipeline. |
| `custom_components/spot_price_predictor/data/spike_model_default.json` | Refit end-to-end with the 8-feature design on the 2023-2026 cached parquets. `ridge_coef` length 9; `gpd_right` re-fit on the new (smaller) post-AR residual. |
| `custom_components/spot_price_predictor/manifest.json` | `version: 2.10.0`. |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version: 2.10.0`. |
| `tests/test_pipeline.py` | Updated for the 9-coef artifact + new tests for the neighbour-price path (full series, partial / NaN, missing zones). |

## Test status

`python -m pytest tests/` — **409 passed, 4 warnings, 0 failed** at
the v2.10.0 commit.

## Acceptance criteria (v2.5.6 hedge gate)

The acceptance was decided by the NPK-CVaR hedge test, in line with
the v2.5.6 release policy. Required:

- **Primary**: hedged-portfolio CVaR reduction ≥ +0.3 pp per added
  feature. **Achieved: +9.02 pp for 3 added features (≈ 3 pp /
  feature, 10× the threshold).**
- **Secondary**: extreme-price-hour MAE drop ≥ 1 EUR/MWh.
  **Achieved: −4.66 EUR/MWh.**
- **Regression guard**: overall MAE drift ≤ 2 EUR/MWh.
  **Drift: +1.11 EUR/MWh.**

All three pass. The cross-border features are also intuitive and
mechanistically well-understood: in Nord Pool's coupled market, FI
prices track neighbouring zones during constraint hours (FI imports
from SE3 via the FennoSkan cables, from SE1 via the AC interconnects,
from EE via Estlink). The new features extract that signal directly
rather than relying on the L1 seasonal layer to absorb it.
