# v2.8.1 — Documentation refresh + canonical sensor schema

Patch release on top of v2.8.0. No prediction-model change; this release cleans up the user-facing surface so the integration is easier to consume and to maintain going forward.

## Highlights

- **Documentation aligned with the shipping pipeline.** README, TECHNICAL_GUIDE, TEKNINEN_TOTEUTUS, INSTALLATION, [`docs/dk_cheap_peak_migration.md`](../../docs/dk_cheap_peak_migration.md) and [`docs/dtaci_layer.md`](../../docs/dtaci_layer.md) describe the four-layer prediction pipeline (L1 seasonal + L2 physics-feature Ridge + L3 AR(1) + L4 GPD POT) in the present tense, with no archeology of earlier model versions.
- **Single canonical sensor schema.** The duration-forecast sensor now exposes one 0-indexed 24-entry shape per direction per unit:
  - `dk_cheap_eur_mwh[24]` / `dk_peak_eur_mwh[24]` (spot)
  - `dk_cheap_eur_kwh[24]` / `dk_peak_eur_kwh[24]` (consumer)

  The previous 12-entry dual schema, the legacy cumulative arrays, and any version-tagged parallel attributes are removed.
- **Fan-chart attributes lose the version tag.** Forecast rows now carry `P5_eur_mwh` … `P95_eur_mwh`. The redundant `*_mean_eur_mwh` key (which duplicated `spot_eur_mwh`) is dropped.
- **Calibrator extended to the full curve.** The DtACI bundle now runs 48 instances per zone (cheap and peak × k = 1..24) instead of 24 (k = 1..12), so calibrated bands cover the entire cheap/peak surface.
- **Module rename.** The Python module is now `custom_components/spot_price_predictor/pipeline.py` exporting `Pipeline` (was `v26_pipeline.py` / `V26Pipeline`). The Ridge feature constant is now `RIDGE_FEATURES` (was `V26_FEATURES`). The calibrator-state directory under `.storage/` is now `spot_price_predictor_pipeline/` (was `spot_price_predictor_v26/`); the integration migrates the legacy directory automatically on first start so accumulated bias-corrector history is preserved.
- **Consolidated `retrain_models` service** introduced in v2.8.0 is unchanged: refits the three artifacts under `data/` atomically and reloads on the next coordinator cycle. Fires `spot_price_predictor_models_retrained` on completion.

## Sensor schema

```yaml
sensor.spot_price_predictor_price_forecast:
  forecast:
    - timestamp: "2026-05-20T10:00:00+00:00"
      spot_eur_mwh:    72.1            # point forecast (EUR/MWh)
      consumer_eur_kwh: 0.1198         # consumer price (EUR/kWh)
      wind:  5.8
      solar: 412.0
      temp:  12.3
      # Fan-chart percentiles from the L4 GPD POT layer
      P5_eur_mwh:   18.0
      P25_eur_mwh:  48.0
      P50_eur_mwh:  72.0
      P75_eur_mwh:  98.0
      P95_eur_mwh:  155.0

sensor.spot_price_predictor_duration_forecast:
  daily_forecast:
    - date: "2026-05-21"
      weekday: "Thu"
      source: "forecast"               # or "actual" for past days
      dk_cheap_eur_mwh: [c0, c1, ..., c22, c23]   # 24-entry, 0-indexed, EUR/MWh
      dk_peak_eur_mwh:  [p0, p1, ..., p22, p23]
      dk_cheap_eur_kwh: [c0, c1, ..., c22, c23]   # 24-entry, 0-indexed, EUR/kWh
      dk_peak_eur_kwh:  [p0, p1, ..., p22, p23]
      # PV-aware D(k) appear only when pv_capacity_kwp > 0
      dk_cheap_pv_eur_kwh: [...]                  # 24-entry
      dk_peak_pv_eur_kwh:  [...]                  # 24-entry
```

Coordinator-level diagnostics surface under `pipeline_diagnostics`:

```yaml
pipeline_diagnostics:
  pipeline_bias_eur_mwh: 0.0     # HourlyBiasCorrector EMA (warms over ~7 days)
  pipeline_ar1_phi: 0.904        # L3 AR(1) coefficient
  pipeline_n_features: 6         # L2 Ridge feature count
  pipeline_floor_eur_mwh: -5.0   # softplus floor level
  refit_recommended: false       # raised when RefitMonitor detects >5 pp coverage drift for 14 days
```

## Breaking changes

Downstream consumers (templates, automations, EMHASS bridges) that read any of the following attribute names need updating:

| Removed | Replacement |
|---|---|
| `v26_mean_eur_mwh` (forecast row) | `spot_eur_mwh` (same value) |
| `v26_P5_eur_mwh` … `v26_P95_eur_mwh` (forecast row) | `P5_eur_mwh` … `P95_eur_mwh` |
| `dk_cheap_eur_kwh[12]`, `dk_peak_eur_kwh[12]` | `dk_cheap_eur_kwh[24]`, `dk_peak_eur_kwh[24]` (0-indexed; `[k-1]` still gives the cheapest/priciest-k mean) |
| `dk_cheap_spot_eur_mwh[12]`, `dk_peak_spot_eur_mwh[12]` | `dk_cheap_eur_mwh[24]`, `dk_peak_eur_mwh[24]` |
| `dk_consumer_eur_kwh[24]`, `dk_spot_eur_mwh[24]` | replaced by the cheap/peak split above |
| `dk_cheap_v26_eur_mwh`, `dk_peak_v26_eur_mwh` | `dk_cheap_eur_mwh`, `dk_peak_eur_mwh` |
| `v26_diagnostics` (coordinator data) | `pipeline_diagnostics` |

The most-used convenience scalars on the duration sensor (`today_cheap_4h_eur_kwh`, `today_peak_1h_eur_kwh`, etc.) keep their names and semantics.

## Notes

- The bundled tests under `tests/` still reference the old schema and need regeneration in a follow-up.
- See [`docs/dk_cheap_peak_migration.md`](../../docs/dk_cheap_peak_migration.md) for the canonical schema reference and access patterns.
