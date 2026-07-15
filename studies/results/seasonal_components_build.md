# Seasonal components build — v2.5.5

**Window:** 2023-01-01 → 2026-07-15 (30,985 aligned hourly rows)
**Artifact:** `custom_components/spot_price_predictor/data/seasonal_components_default.json` (23,693 bytes)

## Per-input fit results

| Input | Depth | σ_raw | σ_Y | Var reduction | E[Y] |
|---|---|---:|---:|---:|---:|
| `fi` | P_hour + P_day + P_week | 63.40 | 56.19 | 21.5% | -1.41e-15 |
| `se3` | P_hour + P_day + P_week | 44.91 | 38.32 | 27.2% | -5.33e+00 |
| `se1` | P_day + P_week | 35.13 | 32.67 | 13.5% | -3.01e+00 |
| `ee` | P_hour + P_day + P_week | 72.01 | 61.56 | 26.9% | -4.70e-16 |
| `wind` | P_hour + P_week | 2.30 | 2.18 | 9.9% | -3.87e-02 |
| `solar` | P_hour + P_week | 204.56 | 123.82 | 63.4% | -8.94e+00 |
| `temp` | P_hour + P_week | 9.80 | 4.39 | 79.9% | +2.38e-01 |
| `cloud` | P_week | 27.25 | 25.95 | 9.3% | +9.44e-01 |
| `ghi_cs` | P_hour + P_week | 227.31 | 102.97 | 79.5% | -3.61e-01 |

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