# Backtest harness — frozen day-ahead walk-forward

Data snapshot: `c8a5feece6ecdcc3` | eval 2025-07-01 → 2026-06-26 (8,656 h)

Regime: day-ahead (no one-step AR), monthly walk-forward refit for FRESH configs, per-day physics centring for production-faithful configs, weather-oracle inputs (identical across configs — deltas are the honest signal).

| segment | n | mean € | DEPLOYED MAE (bias) | FRESH MAE (bias) | FRESH_CONS MAE (bias) |
|---|---:|---:|---:|---:|---:|
| ALL | 8,656 | 57 | 22.15 (+2.0) | 21.75 (+1.1) | 19.81 (+2.8) |
| WINTER Dec-Feb | 2,160 | 95 | 28.77 (-1.0) | 27.65 (+3.2) | 26.21 (+8.9) |
| SUMMER May-Jul | 2,105 | 41 | 14.82 (-4.3) | 15.30 (-6.1) | 14.48 (-5.6) |
| midday 8-12 UTC | 1,444 | 59 | 20.22 (+11.4) | 22.02 (+11.0) | 19.41 (+7.2) |
| evening 15-19 UTC | 1,441 | 78 | 32.70 (-2.6) | 30.66 (-1.7) | 28.51 (+1.3) |
| tail p95 price | 433 | 233 | 52.71 (-40.0) | 56.00 (-41.3) | 49.97 (-29.2) |

Isolated deltas: FRESH − DEPLOYED = value of retraining on fresh data; FRESH_CONS − FRESH = value of the physics-deseasonalisation consistency fix.
