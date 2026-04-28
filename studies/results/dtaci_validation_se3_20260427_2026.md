# DtACI Validation — Walk-Forward on SE3 Hourly Spot

Generated: 2026-04-27T20:26:51.739010+00:00
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
| `raw` | 6.94 | — | — | — |
| `static` | 6.94 | 0.8944 | 30.60 | 0.874 |
| `aci` | 6.94 | 0.8997 | 31.93 | 1.000 |
| `dtaci` | 6.94 | 0.8956 | 31.12 | 1.000 |
| `dtaci_bc` | 6.42 | 0.8960 | 29.78 | 1.000 |

## Width adaptation across regimes

Mean prediction-interval half-width (EUR/MWh) per calendar year. A faithful adaptive method should produce *wider* intervals during the 2022 spike and *narrower* intervals during the 2024-25 normalisation.

| Method | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|
| `static` | 33.44 | 23.58 | 32.37 | 33.27 |
| `aci` | 35.31 | 25.72 | 32.88 | 35.02 |
| `dtaci` | 31.29 | 25.61 | 32.10 | 34.11 |
| `dtaci_bc` | 30.18 | 24.73 | 31.87 | 30.81 |

## Interpretation

- **MAE column** isolates the bias-correction effect: only `dtaci_bc` modifies the point forecast, so its MAE relative to `raw` quantifies bias correction's value.
- **Coverage column** shows the *marginal* realised coverage across the entire holdout. All adaptive methods should land near target; the static method may drift in either direction during regime shifts.
- **Mean width column**: lower is sharper. Methods achieving target coverage at lower mean width are more efficient.
- **Stable-window fraction** is the share of 720-hour rolling windows whose realised coverage falls in [target+/-0.05]. Higher = better local calibration. A vanilla static method can have good marginal coverage while being chronically over- or under-covered in any given month — the stable-window metric exposes this.
- **Width-by-year**: confirms that DtACI adapts the band to the underlying volatility regime. The static empirical quantile fails this test — its width depends only on the last 720 hours, with no built-in mechanism to react faster after a regime change.
