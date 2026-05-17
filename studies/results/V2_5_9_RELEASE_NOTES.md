# v2.5.9 — Input/output mapping verification + refreshed artifact figures

## TL;DR

**No coordinator behaviour change.** v2.5.9 ships two visual-inspection tools that answer the user's 2026-05-17 questions:

> *"can you update outdated per_sensor_components_cloud.png. Can you provide me normalized input and outputs for visual inspection because I am not sure if all parameters are properly mapped to the model?"*

Two new scripts:

1. **`studies/refresh_artifact_figures.py`** — Re-renders `per_sensor_components_<NAME>.png` for all 9 inputs using the SHIPPED v2.5.8 artifact (so the figures match what's deployed, rather than the v2.5.4 audit-time fits). Each panel now consistently shows components as deviations from mean for direct visual comparison across inputs and depths.

2. **`studies/input_output_mapping.py`** — End-to-end pipeline verification:
   - For every input, plots raw `X(t)`, seasonal reconstruction, and residual `Y_X(t)` on both z-score and native-unit axes over a representative 2-week window.
   - Verifies `X = seasonal + Y` to machine precision (1e-9) for every input. **All 9 pass.**
   - Renders `fi_prediction_decomposition.png` showing the v2.5.6 winner (`Y_fi_lag168 + is_workday`) trained Ridge model decomposing the FI prediction into Layer-1 seasonal + Layer-2 Ridge contribution.

## Mathematical sanity-check result

All 9 inputs round-trip correctly through the v2.5.8 artifact:

```
fi        OK    se3       OK    se1       OK
ee        OK    wind      OK    solar     OK
temp      OK    cloud     OK    ghi_cs    OK
```

`max(|X - (seasonal + Y)|) < 1e-9` for every input. The parameter mapping in the deployed artifact is structurally correct.

## What the figures reveal — visually

### `input_output_mapping_native.png` — the smoking gun

Looking at the FI panel (top-left) for August 2025:
- **Actual price (black)** spikes to **220 EUR/MWh**.
- **Seasonal reconstruction (green)** stays near 50 EUR/MWh (the annual mean) with a diurnal swing.
- **Residual Y_fi (blue)** carries the entire 150+ EUR/MWh spike on top of the mean-removed signal.

The seasonal layer is doing its job — recovering the deterministic structure — but the residual carries ALL the price-spike risk. That's exactly the architecture the user described: deterministic seasonal at Layer 1, stochastic dynamics at Layers 2/3/4.

### `fi_prediction_decomposition.png` — and why the current model isn't enough

Fitting the v2.5.6 winner (`Y_fi_lag168 + is_workday`) on the v2.5.8 residuals:

| Metric | Value |
|---|---:|
| Ridge intercept | +4.4 EUR/MWh |
| β · Y_fi_lag168 | +0.083 (last week's same-hour residual contributes 8 %) |
| γ · is_workday  | −2.0 EUR/MWh (workday/weekend residual split) |
| Test MAE | 39.3 EUR/MWh |
| **Test R²** | **0.266** |

R² of 0.27 — the predicted FI price tracks only weakly. The figure makes this visceral: the prediction line (blue) sits near the seasonal baseline (green) while the actual line (black) swings far above with the spikes. **Layers 1 + 2 alone cannot deliver good 7-day price forecasts.**

This confirms the user's intuition from the previous message:
> *"Ridge regression can be applied with seasonal data which should be quite deterministic, but time series models and stochastic peak model needs to be added after this."*

The v2.5.2 study already proved that GPD POT (Layer 4) is feasible for cross-border zones. The next two patches need to:
- **v2.5.10**: Layer 3 — AR(1) / OU on the Ridge residual `ε(t)` = `actual − seasonal − Ridge·Y`. Expected to capture the autocorrelated portion of the residual.
- **v2.5.11**: Layer 4 — GPD POT spike model on the post-AR noise. Mirrors v2.5.2 methodology directly.

### `per_sensor_components_<NAME>.png` — refreshed to match deployment

Every per-input panel now shows the SHIPPED components (v2.5.8 with smoothing). Cloud's P_week is the smooth 7-bin envelope; wind's P_hour and P_week reflect the smoothed values; solar's huge diurnal cycle is unchanged (signal-dominated). What you see is what runs.

Plus `seasonal_artifact_overview.png` puts all inputs on one log-scale page so you can verify σ-reduction per input at a glance.

## Per-input stats (full 8.3 y window via deployed artifact)

| Input | mean | σ raw | σ Y | Variance reduction |
|---|---:|---:|---:|---:|
| fi    | 50.90 EUR/MWh | 64.64 | 57.06 | 22.1 % |
| se3   | 47.84 EUR/MWh | 44.84 | 37.07 | 31.6 % |
| se1   | 30.11 EUR/MWh | 34.91 | 31.86 | 16.7 % |
| ee    | 88.11 EUR/MWh | 72.82 | 62.27 | 26.9 % |
| wind  | 6.16 m/s      | 2.32  | 2.21  |  9.6 % |
| solar | 117.84 W/m²   | 201.12| 123.22| 62.5 % |
| temp  | 5.57 °C       | 9.80  | 4.48  | 79.1 % |
| cloud | 70.24 %       | 27.44 | 25.94 | 10.7 % |
| ghi_cs| 153.06 W/m²   | 219.78| 101.47| 78.7 % |

## Files

- **New**: `studies/refresh_artifact_figures.py` (~280 LOC) — overwrites per-input audit figures from shipped artifact + renders overview
- **New**: `studies/input_output_mapping.py` (~290 LOC) — end-to-end mapping verification + FI prediction decomposition
- **New**: `studies/results/V2_5_9_RELEASE_NOTES.md` — this document
- **Refreshed**: `studies/results/figures/per_sensor_components_{fi,se3,se1,ee,wind,solar,temp,cloud,ghi_cs}.png` (now match deployment)
- **New**: `studies/results/figures/seasonal_artifact_overview.png` (raw vs residual σ per input, log y)
- **New**: `studies/results/figures/input_output_mapping_zscore.png` (8-panel grid, z-score)
- **New**: `studies/results/figures/input_output_mapping_native.png` (8-panel grid, native units)
- **New**: `studies/results/figures/fi_prediction_decomposition.png` (deployed-model prediction vs actual + residual contribution)
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.8 → 2.5.9`)
- **Modified**: `README.md` release-notes index

## Tests

**369 / 369 passing** (no new tests; the new scripts are diagnostic visualizations only).

## Reproducibility

```bash
python studies/refresh_artifact_figures.py     # refresh per-input audit figures
python studies/input_output_mapping.py         # render normalized I/O grid + FI decomp
```

Both scripts are offline; they read the shipped artifact + cached weather + price parquets.

## What changes next

The v2.5.9 visualizations confirm what the architecture diagram said in v2.5.7:

- **Layer 1 (seasonal decomposition)**: correctly fit, properly shipped, mathematically verified — *done*.
- **Layer 2 (Ridge on residual)**: works mechanically, but R² = 0.27 alone is far below what 7-day CVaR needs.
- **Layer 3 (AR/OU on Ridge residual)**: *not built* — captures the autocorrelated noise structure.
- **Layer 4 (GPD POT spike model)**: *not built* — captures the heavy-tail risk that drives CVaR at low α.

Each layer is independent. Adding them is straightforward — Layer 3 reuses `npk_cvar_hedge.fit_ou_ar1()`, Layer 4 reuses the v2.5.2 GPD POT methodology. No data needs re-fetching. The user is the gate on starting v2.5.10.
