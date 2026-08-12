# Backtest harness — frozen day-ahead walk-forward

Data snapshot: `c8a5feece6ecdcc3` | eval 2025-07-01 → 2026-06-26 (8,658 h)

Regime: day-ahead (no one-step AR), monthly walk-forward refit for FRESH configs, per-day physics centring for production-faithful configs, weather-oracle inputs (identical across configs — deltas are the honest signal).

| segment | n | mean € | PRODUCTION MAE (bias) | FRESH MAE (bias) | FRESH_CONS MAE (bias) |
|---|---:|---:|---:|---:|---:|
| ALL | 8,658 | 57 | 24.81 (-2.3) | 35.90 (-6.9) | 27.17 (-4.5) |
| WINTER Dec-Feb | 2,160 | 95 | 31.97 (-1.6) | 44.82 (-19.0) | 34.14 (-6.3) |
| SUMMER May-Jul | 2,107 | 41 | 20.92 (-10.4) | 27.76 (-12.8) | 23.00 (-12.3) |
| midday 8-12 UTC | 1,444 | 59 | 24.39 (+0.7) | 36.19 (-5.2) | 26.74 (-1.6) |
| evening 15-19 UTC | 1,441 | 78 | 32.93 (-8.4) | 47.17 (-11.5) | 36.24 (-10.8) |
| tail p95 price | 433 | 233 | 89.56 (-89.3) | 130.29 (-130.3) | 102.02 (-102.0) |

Isolated deltas: FRESH − DEPLOYED = value of retraining on fresh data; FRESH_CONS − FRESH = value of the physics-deseasonalisation consistency fix.
