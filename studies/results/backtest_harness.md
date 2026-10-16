# Backtest harness — frozen day-ahead walk-forward

Data snapshot: `22a27347590b74d6` | eval 2025-07-01 → 2026-10-16 (11,319 h)

Regime: day-ahead (no one-step AR). PRODUCTION is the frozen shipped artifact evaluated exactly as `Pipeline.compute_forecast` does — frozen L1, frozen coefficients, neighbours lagged 168 h, physics deseasonalised with the artifact's `physics_seasonal` block, local-calendar workday and holiday flags, `Y_fi_lag168` zeroed, softplus floor. Parity is pinned by tests/test_harness_production_parity.py. FRESH configs refit L1 and the ridge monthly and are for model experiments only — do NOT read them as production behaviour (backlog D6). Weather inputs are identical across configs, so the deltas are the honest signal.

| segment | n | mean € | PRODUCTION MAE (bias) | FRESH MAE (bias) | FRESH_CONS MAE (bias) |
|---|---:|---:|---:|---:|---:|
| ALL | 11,319 | 52 | 24.53 (-0.6) | 34.20 (-3.9) | 26.60 (-2.0) |
| WINTER Dec-Feb | 2,160 | 95 | 31.97 (-1.6) | 44.81 (-19.0) | 34.14 (-6.3) |
| SUMMER May-Jul | 2,946 | 34 | 19.98 (-4.6) | 25.71 (-5.4) | 22.16 (-5.2) |
| midday 8-12 UTC | 1,888 | 53 | 24.41 (+2.6) | 34.27 (-2.3) | 26.34 (+0.9) |
| evening 15-19 UTC | 1,886 | 72 | 32.31 (-5.2) | 45.00 (-6.8) | 35.17 (-6.6) |
| tail p95 price | 566 | 221 | 86.96 (-86.7) | 123.33 (-123.3) | 97.27 (-97.1) |

## Weekday bias by month (PRODUCTION)

| month | n | mean € | bias | MAE |
|---|---:|---:|---:|---:|
| 2025-07 | 549 | 27 | -0.5 | 14.0 |
| 2025-08 | 504 | 65 | -7.3 | 31.3 |
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
| 2026-07 | 550 | 17 | +17.8 | 20.7 |
| 2026-08 | 502 | 53 | +9.7 | 25.0 |
| 2026-09 | 528 | 69 | -6.1 | 38.0 |
| 2026-10 | 266 | 43 | +0.8 | 25.1 |

Isolated deltas: FRESH − DEPLOYED = value of retraining on fresh data; FRESH_CONS − FRESH = value of the physics-deseasonalisation consistency fix.
