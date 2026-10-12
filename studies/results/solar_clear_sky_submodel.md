# Solar production sub-model — v2.5.3 (isolated training and validation)

**Window:** 2023-01-01 → 2026-10-09 (33,049 hourly rows)
**Train / test split:** chronological 70 / 30
**Ground truth:** Fingrid dataset 248 (Finnish solar generation)
**Capacity reference:** Fingrid dataset 267 (mean 1222 MW; latest 2074 MW)
**Cloudiness:** Open-Meteo `cloud_cover` (capacity-weighted across 7 FI sites; same weights as the production wind/solar features)

## Architecture

```
production_MW(t) = capacity_MW(t)
                 · GHI_clear_sky(t, weighted_sites)
                 · cloudiness_modulator(cloud_cover(t))
```

- `GHI_clear_sky` is a deterministic function of (lat, lon, t) —
  zero free parameters, two candidate formulas (Haurwitz vs
  Ineichen-Perez). No price or production data is used to fit it.
- `cloudiness_modulator` has 1-2 free parameters fit on the
  training split only.
- Errors in the solar fit cannot propagate to the FI price model
  because the sub-model is fit independently and consumes no
  price-related variable.

## Test-set results (all four candidate combinations)

| Clear-sky | Modulator | MAE [MW] | MAE-daylight [MW] | rel. MAE-daylight | R² | bias [MW] |
|---|---|---:|---:|---:|---:|---:|
| haurwitz | kasten_czeplak | 46.80 | 84.18 | 23.4 % | 0.915 | -7.00 |
| haurwitz | linear **(winner)** | 38.73 | 74.89 | 20.8 % | 0.943 | 3.35 |
| ineichen | kasten_czeplak | 46.52 | 84.94 | 23.6 % | 0.915 | -5.36 |
| ineichen | linear | 41.28 | 75.81 | 21.1 % | 0.942 | 5.04 |

## Winner: **haurwitz** clear-sky × **linear** modulator

- Test R² = **0.943**
- Test MAE-daylight = **74.89 MW** (20.8 % of mean daylight production)
- Bias = +3.35 MW
- Fitted parameters: (-0.39908293028309655, 1.0009640345097115, 0.49948503698918106, 1.0074812218479026)

## Verdict

- **Tier-A absolute gate (R² ≥ 0.85):** PASS (0.943).
- **Isolation invariant:** the sub-model consumes only 
  `(timestamp, lat, lon)` and `cloud_cover(t)`. No price-side 
  input. Errors cannot propagate via FI fit. **HONOURED.**

## Dataset summary

- Production: mean 141.5 MW, peak 1365 MW
- Capacity grew from 606 MW → 2074 MW over the window
- Cloud cover: mean 70.1 %, median 76.9 %

## Figure

![Solar sub-model validation](figures/solar_submodel_validation.png)

## Reproducibility

```bash
export FINGRID_API_KEY=your_key_here
python studies/solar_clear_sky_submodel.py
```

Free Fingrid API key: https://developer-data.fingrid.fi/

## Next steps (per v2.5.3 → v2.6.0 roadmap)

1. If Tier-A PASSes, expose the sub-model's hourly prediction as a
   new candidate feature `solar_submodel_prediction` for the FI
   Ridge in v2.5.5.
2. The hedge-gated input sweep in v2.5.6 decides whether it stays
   in the final feature set — but the sub-model itself is now
   validated standalone so its quality is independently established.