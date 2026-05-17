# Seasonal components build — v2.5.5

**Window:** 2023-01-01 → 2026-04-27 (29,112 aligned hourly rows)
**Artifact:** `custom_components/spot_price_predictor/data/seasonal_components_default.json` (22,655 bytes)

## Per-input fit results

| Input | Depth | σ_raw | σ_Y | Var reduction | E[Y] |
|---|---|---:|---:|---:|---:|
| `fi` | P_hour + P_day + P_week | 64.63 | 57.06 | 22.1% | -2.50e-16 |
| `se3` | P_hour + P_day + P_week | 44.84 | 36.22 | 34.8% | +2.50e-16 |
| `se1` | P_day + P_week | 34.91 | 30.37 | 24.3% | +0.00e+00 |
| `ee` | P_hour + P_day + P_week | 72.82 | 62.27 | 26.9% | +2.50e-16 |
| `wind` | P_hour + P_week | 2.32 | 2.16 | 13.8% | +4.69e-17 |
| `solar` | P_hour + P_week | 201.12 | 121.23 | 63.7% | -1.75e-15 |
| `temp` | P_hour + P_week | 9.80 | 4.00 | 83.4% | -9.37e-17 |
| `cloud` | P_week | 27.44 | 24.69 | 19.0% | -1.47e-15 |
| `ghi_cs` | P_hour + P_week | 219.78 | 101.43 | 78.7% | -2.56e-15 |

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