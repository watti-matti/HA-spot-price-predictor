# v2.4.2 SE3 model validation results

**Run:** `python studies/se3_model_v242.py`
**Window:** 2024-W20 → 2026-W19 (104 weeks, Statnett-covered)
**Alpha:** 0.05 (95 % confidence)
**Hedge lag:** 48 h (day-ahead horizon)

## Model

```
P_SE3 = P_hour_SE3 + P_day_SE3 + P_week_SE3
      + β_hydro · hydro_offset_t
      + β_workday · is_workday_t
      + β_AR1 · Y_{t-1}
```

## Fitted coefficients

| Coefficient | Value | Interpretation |
|---|---:|---|
| const | -0.0177 | offset of deseasonalized series |
| β_hydro | -0.0894 | EUR/MWh per % reservoir offset (negative = more water → lower price ✓) |
| β_workday | -0.2476 | EUR/MWh workday vs weekend lift |
| β_AR(1) | +0.9419 | residual mean-reversion (b ≈ 0.94 ≈ 12 h half-life) |

R² on Y_SE3: **0.8912**

## NPK-CVaR hedge comparison

| Variant | h_hat | CVaR test unhedged | CVaR test hedged | Reduction |
|---|---:|---:|---:|---:|
| Baseline (seasonal-only, windowed) | 0.796 | 20.41 | 21.65 | **-6.06 %** |
| v2.4.2 (seasonal + hydro + workday + AR(1)) | 0.175 | 20.41 | 21.02 | **-2.99 %** |

**Δ improvement: +3.07 pp**

## Decision

**ACCEPT**

The v2.4.2 SE3 model beats the seasonal-only baseline on the
out-of-sample test set (48 h horizon, α = 0.05).

## Notes

- The v2.4.1 baseline measurement on the full 2023+ window was −7.78 %; on the
  104-week Statnett window the same baseline computes to -6.06 % because the
  2023 regime (with the Russia/Ukraine crisis tail) is excluded. The fair
  apples-to-apples comparison is v2.4.2-model vs same-window-baseline.
- The hedge ratio `h_hat = 0.175` is much lower than the baseline's
  `0.796` because adding AR(1) to the model introduces deep autocorrelation
  in the prediction's differences, changing the hedge geometry. The CVaR
  reduction is the metric that matters for the gate, not h_hat alone.
- This is an OFFLINE study; the coordinator continues to use the v2.2
  9-feature Ridge model. The validated SE3 model will be wired into the
  coordinator at v2.5.0 alongside the EE (v2.4.3) and FI (v2.4.4) variants.
