# Phase 2 — post-EMA bias decomposition

Snapshot `c8a5feece6ecdcc3`, eval 2025-07-01 → 2026-06-26, 8,656 h. Point forecast: v2.12.0 walk-forward (frozen harness). Correctors simulated with the production OnlineBiasCorrector under day-ahead information structure.

| segment | RAW MAE (bias) | GLOBAL(prod) MAE (bias) | PER_HOUR MAE (bias) |
|---|---:|---:|---:|
| ALL | 19.81 (+2.8) | 19.40 (+0.2) | 18.72 (+0.1) |
| WINTER Dec-Feb | 26.21 (+8.9) | 24.87 (-3.1) | 23.49 (-3.0) |
| SUMMER May-Jul | 14.48 (-5.6) | 12.70 (-2.0) | 12.06 (-2.4) |
| midday 8-12 UTC | 19.41 (+7.2) | 18.25 (+4.6) | 17.69 (+0.4) |
| evening 15-19 UTC | 28.51 (+1.3) | 27.29 (-1.3) | 27.50 (-0.4) |
| tail p95 price | 49.97 (-29.2) | 52.54 (-36.5) | 52.36 (-37.8) |

Post-GLOBAL residual bias by hour-of-day:

| hour | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bias | -1.9 | -1.9 | -3.2 | -4.4 | -4.3 | -0.3 | +4.2 | +6.9 | +6.7 | +5.2 | +3.7 | +2.8 | +1.4 | -1.5 | -2.3 | -3.6 | -2.3 | +0.2 | +0.4 | +0.9 | +0.1 | -0.3 | -0.6 | -2.0 |
