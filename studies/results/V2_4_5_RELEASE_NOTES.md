# v2.4.5 — Alternative solar model (DEFERRED pending cloudiness data)

## TL;DR

**DEFERRED.** Building the user's proposed alternative solar model (clear-sky max scaled by installed Finnish PV capacity × Open-Meteo cloudiness forecast) requires an Open-Meteo input variable (`cloud_cover` or `cloud_cover_total`) that is **not currently fetched** by the integration. v2.4.5 ships the clear-sky envelope analysis (derivable from existing `solar_irradiance_weighted` data) and specifies the cloudiness integration work for a future v2.5.x patch. The current `solar_irradiance_weighted` Open-Meteo `global_tilted_irradiance_instant` feature stays in production.

## What we have today

Our cached `output/fi_weather.parquet` contains only three weather variables:

- `wind_speed_weighted` (m/s, 7-location-weighted)
- `solar_irradiance_weighted` (W/m², Open-Meteo `global_tilted_irradiance_instant` at 45° tilt)
- `temperature_weighted` (°C)

The current v2.2 9-feature Ridge consumes `solar_irradiance_weighted` directly — Open-Meteo has already done the cloud-cover-to-tilted-irradiance physics conversion server-side.

## What the user proposed

```
P_solar_predicted(t) = clear_sky_peak(t, lat, lon)
                     × installed_PV_capacity_FI(t)
                     × (1 - cloud_cover_open_meteo(t))
```

- `clear_sky_peak` ≈ maximum production achievable under clear skies, varies by hour-of-day and day-of-year
- `installed_PV_capacity_FI(t)` ≈ scaling factor that grows with the Finnish PV fleet (Motiva / Energiavirasto publishes annual values)
- `cloud_cover_open_meteo(t)` ≈ separate Open-Meteo variable (0–100 %)

This is a **structural / parametric** model. The current production feature is **empirical** (uses Open-Meteo's pre-computed irradiance). Both ultimately feed the FI Ridge model as a single `solar_*_weighted` feature.

## What v2.4.5 ships (the partial work)

### 1. Clear-sky envelope from existing data

Computed the 95th-percentile of `solar_irradiance_weighted` for each (month, hour-of-day) cell — represents "typical maximum under clear skies":

```
Clear-sky 95th-pct envelope (W/m², Finnish capacity-weighted)
         0     3     6     9    12    15    18    21
M1       0     0     0     0   272   242     0     0   ← deep winter peak ~270 W/m² at noon
M3       0     0     0   183   706   705   230     0   ← spring 700+ W/m² at noon
M6       1     1    78   401   863   872   449    65   ← summer peak 870 W/m² at noon
M12      0     0     0     0   137   107     0     0   ← darkest month, peak only 137 W/m²
```

Global stats:
- Peak: **948 W/m²** (clear summer noon)
- Mean envelope: 201 W/m²
- Mean cloud-deficit (envelope − actual): **78.5 W/m²** — typical cloudiness loss
- Fraction of observations above envelope: 3.3 % (should be ~5 % for 95th percentile, indicates outliers fall above too)

This envelope IS the implicit "clear-sky max" the user's alternative would multiply. It's derivable today; the only missing piece is the cloudiness multiplier.

### 2. What's missing — Open-Meteo cloudiness fetch

To wire up the user's alternative model in production, the integration would need:

1. Add `cloud_cover` (or `cloud_cover_low` + `_mid` + `_high`) to the Open-Meteo URL in `custom_components/spot_price_predictor/api_client.py`.
2. Add `cloud_cover_weighted` to the historical weather collection script (`src/data_sources.py`).
3. Re-fetch 2023+ historical weather data.
4. Implement `predicted_solar_alt = clear_sky_envelope(t) × (1 - cloud_cover/100)` in the coordinator.
5. Run the same NPK-CVaR hedge methodology comparing `solar_irradiance_weighted` (current) vs `predicted_solar_alt` (alternative) as the feature for FI Ridge.
6. Accept the winner per the v2.4.x gate.

### 3. Why this is DEFERRED, not REJECTED

The methodology requires data-driven empirical comparison. Without cloudiness data we cannot run the gate fairly. **Speculative adoption** of either approach would violate the v2.4.x principle that *"if test CVaR drops, the feature captures real signal; if unchanged, it's noise — discard"*.

Per v2.4.4's REJECT, the FI Ridge architecture **stays as v2.2** in v2.5.0 anyway. The `solar_irradiance_weighted` feature inside it remains unchanged. So v2.4.5 has no impact on the v2.5.0 milestone — it's clean to defer.

## Decision for v2.5.0

| Item | Status | Source |
|---|---|---|
| FI Ridge model | v2.2 9-feature, unchanged | v2.4.4 REJECT |
| `solar_irradiance_weighted` feature inside FI Ridge | unchanged (Open-Meteo `global_tilted_irradiance_instant`) | v2.4.5 DEFER |
| SE3 cross-border model | NEW: seasonal + hydro + workday + AR(1) | v2.4.2 ACCEPT |
| EE `ar_ee` feature | unchanged (v2.2 AR(2)) | v2.4.3 REJECT |
| Statnett hydro client | NEW, weekly refresh, no auth | v2.4.1 |
| NPK-CVaR validation tool | NEW in `studies/` | v2.4.1 |

## What v2.4.5 ships (files)

- **`studies/results/V2_4_5_RELEASE_NOTES.md`** — this document (specification + deferred-rationale)
- **`manifest.json`** — bumped `2.4.4` → `2.4.5`

No new scripts, no new tests, no coordinator changes. The clear-sky envelope analysis is documented inline above for future v2.5.x reference.

Test suite: **309 / 309 passing** (unchanged from v2.4.2).

## Specification for a future v2.5.x cloudiness integration

When the cloudiness work is taken on:

1. `api_client.py` Open-Meteo fetch URL: add `&hourly=cloud_cover` to current list (`wind_speed_120m`, `solar_radiation`, `temperature_2m`).
2. `data_sources.py` historical fetch: same addition.
3. Re-build `output/fi_weather.parquet` to include `cloud_cover_weighted`.
4. New module: `studies/solar_alt_model.py` implementing the user's proposed clear-sky × cloudiness model.
5. NPK-CVaR hedge gate: alternative solar feature for FI Ridge must beat the existing `solar_irradiance_weighted` feature on out-of-sample test set (apples-to-apples — replace just the solar feature, keep the other 8 in v2.2 Ridge).
6. Patch release: e.g. v2.5.1 if it ships post-v2.5.0, or v2.4.6 if the cloudiness work happens before v2.5.0 consolidation.
