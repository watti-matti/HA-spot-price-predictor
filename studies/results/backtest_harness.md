# Backtest harness — frozen day-ahead walk-forward

Data snapshot: `85fe2b59dbb7f630` | eval 2025-07-01 → 2026-08-12 (9,757 h)

Regime: day-ahead (no one-step AR). PRODUCTION is the frozen shipped artifact evaluated exactly as `Pipeline.compute_forecast` does — frozen L1, frozen coefficients, neighbours lagged 168 h, physics deseasonalised with the artifact's `physics_seasonal` block, local-calendar workday and holiday flags, `Y_fi_lag168` zeroed, softplus floor. Parity is pinned by tests/test_harness_production_parity.py. FRESH configs refit L1 and the ridge monthly and are for model experiments only — do NOT read them as production behaviour (backlog D6). Weather inputs are identical across configs, so the deltas are the honest signal.

| segment | n | mean € | PRODUCTION MAE (bias) | FRESH MAE (bias) | FRESH_CONS MAE (bias) |
|---|---:|---:|---:|---:|---:|
| ALL | 9,757 | 52 | 23.92 (-1.2) | 34.18 (-4.6) | 26.22 (-2.8) |
| WINTER Dec-Feb | 2,160 | 95 | 31.97 (-1.6) | 44.82 (-19.0) | 34.14 (-6.3) |
| SUMMER May-Jul | 2,942 | 34 | 20.00 (-4.6) | 25.72 (-5.4) | 22.18 (-5.2) |
| midday 8-12 UTC | 1,628 | 53 | 23.35 (+1.7) | 34.01 (-3.2) | 25.55 (-0.1) |
| evening 15-19 UTC | 1,625 | 71 | 31.73 (-5.9) | 44.87 (-7.8) | 34.89 (-7.7) |
| tail p95 price | 488 | 225 | 84.30 (-83.9) | 123.78 (-123.7) | 96.27 (-96.1) |

## Weekday bias by month (PRODUCTION)

| month | n | mean € | bias | MAE |
|---|---:|---:|---:|---:|
| 2025-07 | 549 | 27 | -0.5 | 13.9 |
| 2025-08 | 503 | 65 | -7.3 | 31.4 |
| 2025-09 | 528 | 48 | +0.8 | 23.7 |
| 2025-10 | 552 | 63 | -12.2 | 28.2 |
| 2025-11 | 480 | 58 | +13.3 | 23.2 |
| 2025-12 | 552 | 41 | +22.5 | 28.7 |
| 2026-01 | 528 | 126 | -4.0 | 36.7 |
| 2026-02 | 480 | 155 | -28.9 | 44.0 |
| 2026-03 | 528 | 29 | +10.1 | 19.3 |
| 2026-04 | 528 | 55 | -0.3 | 25.8 |
| 2026-05 | 504 | 60 | -20.0 | 27.6 |
| 2026-06 | 528 | 53 | -7.9 | 21.8 |
| 2026-07 | 549 | 17 | +17.8 | 20.7 |
| 2026-08 | 171 | 14 | +4.5 | 17.7 |

Isolated deltas: FRESH − DEPLOYED = value of retraining on fresh data; FRESH_CONS − FRESH = value of the physics-deseasonalisation consistency fix.
