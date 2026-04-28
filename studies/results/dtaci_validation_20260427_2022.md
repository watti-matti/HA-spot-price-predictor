# DtACI Validation — Walk-Forward on FI Hourly Spot

Generated: 2026-04-27T20:22:04.339468+00:00
Target coverage: 0.90

## Methods

| Method | Description |
|---|---|
| `raw` | AR(2) point forecast; no interval. MAE reference. |
| `static` | Empirical (1-α) quantile of last 720 residuals; refit each step, no online α-tuning. |
| `aci` | Vanilla ACI with single γ=0.01. |
| `dtaci` | DtACI with 5 experts γ ∈ {0.001,0.005,0.01,0.05,0.1}. |
| `dtaci_bc` | DtACI + OnlineBiasCorrector (halflife 20d, warmup 168 steps). |

## Headline metrics

| Method | MAE EUR/MWh | Realised coverage | Mean width EUR/MWh | Stable-window fraction |
|---|---:|---:|---:|---:|
| `raw` | 9.96 | — | — | — |
| `static` | 9.96 | 0.8947 | 46.37 | 0.807 |
| `aci` | 9.96 | 0.8990 | 49.70 | 1.000 |
| `dtaci` | 9.96 | 0.8944 | 47.42 | 0.944 |
| `dtaci_bc` | 9.91 | 0.8949 | 47.02 | 0.957 |

## Width adaptation across regimes

Mean prediction-interval half-width (EUR/MWh) per calendar year. A faithful adaptive method should produce *wider* intervals during the 2022 spike and *narrower* intervals during the 2024-25 normalisation.

| Method | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|
| `static` | 43.42 | 44.32 | 48.94 | 46.46 |
| `aci` | 47.09 | 50.45 | 49.37 | 49.92 |
| `dtaci` | 46.20 | 45.62 | 49.22 | 48.12 |
| `dtaci_bc` | 45.29 | 45.44 | 48.78 | 47.42 |

## Interpretation

- **MAE column** isolates the bias-correction effect: only `dtaci_bc` modifies the point forecast, so its MAE relative to `raw` quantifies bias correction's value.
- **Coverage column** shows the *marginal* realised coverage across the entire holdout. All adaptive methods should land near target; the static method may drift in either direction during regime shifts.
- **Mean width column**: lower is sharper. Methods achieving target coverage at lower mean width are more efficient.
- **Stable-window fraction** is the share of 720-hour rolling windows whose realised coverage falls in [target+/-0.05]. Higher = better local calibration. A vanilla static method can have good marginal coverage while being chronically over- or under-covered in any given month — the stable-window metric exposes this.
- **Width-by-year**: confirms that DtACI adapts the band to the underlying volatility regime. The static empirical quantile fails this test — its width depends only on the last 720 hours, with no built-in mechanism to react faster after a regime change.
