# Seasonal components build — v2.5.5

**Window:** 2023-01-01 → 2026-04-27 (29,112 aligned hourly rows)
**Artifact:** `custom_components/spot_price_predictor/data/seasonal_components_default.json` (23,743 bytes)

## Per-input fit results

| Input | Depth | σ_raw | σ_Y | Var reduction | E[Y] |
|---|---|---:|---:|---:|---:|
| `fi` | P_hour + P_day + P_week | 64.63 | 57.06 | 22.1% | -2.50e-16 |
| `se3` | P_hour + P_day + P_week | 44.84 | 37.07 | 31.6% | -2.60e+00 |
| `se1` | P_day + P_week | 34.91 | 31.86 | 16.7% | -1.74e+00 |
| `ee` | P_hour + P_day + P_week | 72.82 | 62.27 | 26.9% | +1.30e-03 |
| `wind` | P_hour + P_week | 2.32 | 2.21 | 9.6% | -4.62e-02 |
| `solar` | P_hour + P_week | 201.12 | 123.22 | 62.5% | -8.39e+00 |
| `temp` | P_hour + P_week | 9.80 | 4.48 | 79.1% | +2.22e-01 |
| `cloud` | P_week | 27.44 | 25.94 | 10.7% | +5.58e-01 |
| `ghi_cs` | P_hour + P_week | 219.78 | 101.46 | 78.7% | -3.18e-01 |

## Deployment story

- Artifact ships in `data/seasonal_components_default.json` —
  the integration loads it at startup; no fit at runtime.
- `seasonal_decomposition.compute_residual(X, ts, components)`
  is the inference entry point; pure-numpy, deterministic.
- E[Y] ≈ 0 by construction (sequential subtraction); the table
  above confirms this numerically (residual mean ~ 1e-13).
- Refresh quarterly via `python studies/build_seasonal_components.py`
  + commit the regenerated JSON. The integration picks it up on
  the next coordinator restart.

## Reproducibility

```bash
python studies/build_seasonal_components.py
```

Reads only `output/*.parquet` and `studies/.cache/openmeteo_cloud_*.json`
(populated by the v2.5.3 solar study). No API call.