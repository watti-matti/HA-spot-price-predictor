# v2.6.0 — Production coordinator wiring of the L1+L2+L3+L4+floor pipeline

## TL;DR

**HA users on v2.6.0 see new sensor attributes** alongside everything they already have. No breaking changes; no dashboard migration required.

Per user direction 2026-05-17 ("implement the plan for HA integration ... commit changes to git and GitHub"), v2.6.0 wires the four-layer architecture, softplus floor, and v2.5.15 hourly DtACI calibrators into the production coordinator as an **additive layer**:

- The existing v2.2 9-feature Ridge prediction continues to populate `sensor.spot_price_predictor_price_forecast` and `sensor.spot_price_predictor_duration_forecast` exactly as before.
- New keys appear on each forecast row and duration_forecast entry — `v26_mean_eur_mwh`, `v26_P5..P95_eur_mwh`, `dk_cheap_v26_eur_mwh[24]`, `dk_peak_v26_eur_mwh[24]`.
- New top-level `v26_diagnostics` attribute on the coordinator data carries bias estimate, AR(1) φ, refit-recommended flag.

If a downstream consumer (template, automation, EMHASS) wants the richer signal it reads the `v26_*` keys; if it doesn't, nothing changes.

## What landed in code

### `custom_components/spot_price_predictor/v26_pipeline.py` (~400 LOC)

New `V26Pipeline` class encapsulating the four-layer pipeline:

| Layer | Source | What it produces |
|---|---|---|
| **L1** seasonal | `data/seasonal_components_default.json` (v2.5.8 artifact) | `seasonal_fi(t)` for every forecast hour |
| **L2** Ridge on `Y_features` | `data/spike_model_default.json::ridge_coef` (v2.5.13) | `β · [Y_fi_lag168, is_workday, Y_sigmoid_wind_rho, Y_solar_effective, Y_temp]` |
| **L3** AR(1) | `spike_model_default.json::ar1_phi` (φ ≈ 0.904) | `φ^h · ε(t0-1)` decay per horizon |
| **softplus floor** | `price_floor.py` (v2.5.14) | smooth clip at −5 EUR/MWh on the mean |
| **L4** GPD POT fan-chart | `spike_model_default.json::gpd_right/left` (v2.5.13) | 500-path Monte Carlo → P5/P25/P50/P75/P95 per hour |
| **HourlyBiasCorrector** | persistent EMA (v2.5.15) | small constant offset correction |
| **RefitMonitor** | persistent drift trigger (v2.5.15) | `refit_recommended` flag after 14-day coverage drift |

Persistent state under `<config>/.storage/spot_price_predictor_v26/`:

- `hourly_bias.json` — bias EMA state
- `hourly_fan_chart.json` — DtACI bundles per coverage target
- `refit_monitor.json` — drift trigger state

### `coordinator.py` changes (~120 LOC of additions)

- `__init__`: instantiates `V26Pipeline`, logs Ridge coef shape + AR(1) φ on startup. Defensive — if artifact loading fails, the v26 path is disabled and the v2.2 path keeps working alone.
- `_async_update_data`: after the v2.2 pipeline produces `forecast` and `duration_forecast`, calls `_apply_v26_pipeline()` which:
  - Computes v26 mean prediction + fan bands for every hour in `forecast`
  - Injects `v26_mean_eur_mwh`, `v26_P5_eur_mwh`..`v26_P95_eur_mwh` per row
  - Computes 24-entry D(k) per day from the v26 mean
  - Injects `dk_cheap_v26_eur_mwh[24]` and `dk_peak_v26_eur_mwh[24]` onto matching `duration_forecast` entries
  - Persists calibrator state
- `result["v26_diagnostics"]` exposes: `v26_bias_eur_mwh`, `v26_phi`, `v26_n_features`, `v26_floor_eur_mwh`, `v26_pipeline_version`.

### `tests/test_v26_pipeline.py` (~280 LOC, 12 tests)

- Construction loads shipped artifacts; calibrators cold on first run
- `compute_forecast` returns expected shapes; fan bands monotone (P5 ≤ P25 ≤ P50 ≤ P75 ≤ P95)
- Softplus floor respected (no predictions below the floor)
- `compute_duration_curves` returns 24-entry vectors per day, non-decreasing cheap / non-increasing peak, `cheap[23] = peak[23]` (daily mean)
- `update_with_actuals` warms calibrators, returns `refit_recommended` flag
- State roundtrips through storage directory
- End-to-end smoke test on synthetic 168-hour horizon

## Sensor schema after v2.6.0

### `sensor.spot_price_predictor_price_forecast`

Every row in the `forecast` attribute array now carries:

```yaml
- timestamp: "2026-05-20T10:00:00+00:00"
  spot_eur_mwh: 75.4               # v2.2 (UNCHANGED)
  consumer_eur_kwh: 0.1240         # v2.2 (UNCHANGED)
  wind: 5.8                        # v2.2 (UNCHANGED)
  solar: 412.0                     # v2.2 (UNCHANGED)
  temp: 12.3                       # v2.2 (UNCHANGED)
  # ── NEW v2.6.0 keys ──
  v26_mean_eur_mwh: 72.1           # L1+L2+L3+floor+bias point forecast
  v26_P5_eur_mwh:   18.0           # 95% lower confidence
  v26_P25_eur_mwh:  48.0           # 50% lower confidence
  v26_P50_eur_mwh:  72.0           # median (≈ mean by sampling)
  v26_P75_eur_mwh:  98.0           # 50% upper confidence
  v26_P95_eur_mwh:  155.0          # 95% upper confidence
```

### `sensor.spot_price_predictor_duration_forecast`

Every day in `daily_forecast` now carries (in addition to the existing 12-entry vectors):

```yaml
- date: "2026-05-21"
  dk_cheap_eur_kwh: [...]          # v2.2 (UNCHANGED, 12 entries)
  dk_peak_eur_kwh:  [...]          # v2.2 (UNCHANGED, 12 entries)
  # ── NEW v2.6.0 keys ──
  dk_cheap_v26_eur_mwh: [c00, c01, ..., c22, c23]   # 24 entries, EUR/MWh
  dk_peak_v26_eur_mwh:  [p00, p01, ..., p22, p23]   # 24 entries, EUR/MWh
  hours_in_day: 24
```

### Top-level coordinator data

```yaml
v26_diagnostics:
  v26_bias_eur_mwh: 0.0            # bias EMA (warms over 7 days)
  v26_phi: 0.904                   # AR(1) coefficient
  v26_n_features: 6                # Ridge feature count
  v26_floor_eur_mwh: -5.0          # softplus floor level
  v26_pipeline_version: "2.6.0"
```

When the refit monitor fires (after 14 consecutive days of >5pp coverage drift), `refit_recommended: true` appears in this block.

### Unchanged sensors

- `sensor.spot_price_predictor_spot_electricity_price` — completely unchanged (pure Nordpool transformation, orthogonal to v26)
- `sensor.spot_price_predictor_spot_electricity_selling_price` — completely unchanged (same reason)

## Why additive and not opt-in

Two reasons we chose additive integration over an opt-in config flag:

1. **Zero risk to existing dashboards.** All current attribute names and types are preserved. Templates referencing `state_attr('sensor.spot_price_predictor_duration_forecast', 'dk_cheap_eur_kwh')` continue to work unchanged.

2. **Field validation by parallel-run.** v26 outputs appear alongside v2.2 outputs in the same coordinator data. Users can compare `v26_mean_eur_mwh` vs `spot_eur_mwh` in their own dashboards to validate before any future switchover. Operationally this is much safer than a single-shot replacement.

A future v2.7.0 (or later) can flip the default after positive field reports, but that's a separate decision.

## EMHASS / downstream consumer integration

Adopt the new fields when ready:

```yaml
# EMHASS configuration example
mlforecaster_attribute:
  forecast_sensor: sensor.spot_price_predictor_price_forecast
  point_attribute: v26_P50_eur_mwh        # median forecast
  lower_attribute: v26_P25_eur_mwh        # 50% lower
  upper_attribute: v26_P75_eur_mwh        # 50% upper
  fan_p95_attribute: v26_P95_eur_mwh      # tail risk

optimisation_objective: mean_cvar
cvar_alpha: 0.10
```

Without changes to EMHASS, the existing `consumer_eur_kwh` path continues to work for point-forecast scheduling.

## Files

- **New**: `custom_components/spot_price_predictor/v26_pipeline.py` (~400 LOC)
- **New**: `tests/test_v26_pipeline.py` (12 tests)
- **New**: `studies/results/V2_6_0_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/coordinator.py` (~120 LOC additions)
- **Modified**: `manifest.json` `2.5.17 → 2.6.0`, `README.md` release-notes index

## Tests

**403 / 403 passing** (391 prior + 12 new v26 pipeline tests).

## Runtime cost

- Startup: ~100 ms to load three artifacts (~30 KB total) + initialise calibrators.
- Per coordinator cycle (every 6h): ~50 ms for the v26 path (compute 168 hourly predictions + 500-path Monte Carlo for fan bands + 7-day D(k) aggregation + state persistence).
- Storage: ~20 KB of JSON per zone for calibrator state.

Negligible on a Raspberry Pi 4. Zero new external API calls.

## What's next

The integration ships with parallel-run validation enabled by default. Recommended monitoring period: 2-4 weeks of comparing `v26_mean_eur_mwh` vs `consumer_eur_kwh` in production before deciding whether to make v26 the primary path in v2.7.0.

If you want to consume v26 outputs immediately:
- Dashboards: add chart series reading `v26_P25_eur_mwh` / `v26_P75_eur_mwh` as fan bands
- Automations: trigger on `v26_diagnostics.refit_recommended == true` for refit notifications
- EMHASS: configure as shown above for CVaR-aware optimisation

## Reproducibility

The integration runs the v26 pipeline automatically on every coordinator update — no operator action required. To run the pipeline standalone offline:

```python
from custom_components.spot_price_predictor.v26_pipeline import V26Pipeline
p = V26Pipeline(data_dir="custom_components/spot_price_predictor/data",
                storage_dir="/tmp/v26_state")
out = p.compute_forecast(timestamps, wind, solar, temp,
                          enable_fan_chart=True)
```
