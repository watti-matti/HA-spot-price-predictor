# Technical Guide — HA Spot Price Predictor (v2.8.1)

Finnish consumer-electricity price and D(k) duration cost forecasting for Home Assistant. Produces a 170-hour spot/consumer point forecast, P5/P25/P50/P75/P95 fan-chart bands, and 7-day cheap/peak duration curves driven by a four-layer prediction pipeline. This guide describes only what the shipping code actually does.

## Architecture

The integration has two phases:

- **Inference** runs inside Home Assistant. The `SpotPriceCoordinator` ([`coordinator.py`](custom_components/spot_price_predictor/coordinator.py)) drives a periodic update cycle that fetches weather and price data, calls the `Pipeline` ([`pipeline.py`](custom_components/spot_price_predictor/pipeline.py)) for the four-layer forecast, builds per-hour and per-day attributes, and pushes them to the sensor entities ([`sensor.py`](custom_components/spot_price_predictor/sensor.py)).
- **Retraining** is an on-demand refit of the three JSON artifacts under `data/`, exposed as the `spot_price_predictor.retrain_models` service. The refit scripts (`studies/build_seasonal_components.py`, `studies/v2513_layer4_spike_model.py`, `studies/v253_solar_submodel.py`) read cached parquets and write the artifacts atomically; the coordinator auto-reloads them on the next update.

```
Open-Meteo  ──┐
Sahkotin    ──┼──> SpotPriceCoordinator
Elpriset    ──┤      ├── builds 170 forecast rows
Elering     ──┤      ├── calls Pipeline.compute_forecast → spot, consumer, P5..P95 per row
Fingrid     ──┤      ├── computes per-day D(k) curves (4 × 24 arrays per day)
Nord Pool UMM ┘      ├── (optional) PV-aware augmentation per row + PV D(k) per day
                     └── (optional) DtACI calibrator wraps the D(k) curves with bands
                            │
                            ▼
                  sensor.spot_price_predictor_*
```

The pipeline reads three frozen artifacts under `custom_components/spot_price_predictor/data/`:

| Artifact | Loaded into | Contents |
|---|---|---|
| `seasonal_components_default.json` | `Pipeline._seasonal_artifact` | Per-series additive components (hour + day + week) for FI spot and temperature. |
| `spike_model_default.json` | `Pipeline._spike_artifact` | L2 `ridge_coef`, L3 `ar1_phi`, L4 `gpd_left` / `gpd_right` tail parameters, Normal-body `stats.eta_train_mean` / `stats.eta_train_sigma`. |
| `solar_submodel_default.json` | (PV-aware path only) | Clear-sky × cloudiness solar production sub-model. |

Persistent calibrator state lives under `<config>/.storage/spot_price_predictor_pipeline/` (`hourly_bias.json`, `hourly_fan_chart.json`, `refit_monitor.json`). On first start after upgrading from an older version, the coordinator automatically renames the legacy `.storage/spot_price_predictor_v26/` directory so accumulated bias-corrector history is preserved.

## The four-layer prediction pipeline

Public entry point: `Pipeline.compute_forecast(timestamps, wind, solar, temp, recent_fi_residuals=None, enable_fan_chart=True)` ([`pipeline.py:318`](custom_components/spot_price_predictor/pipeline.py:318)). At each forecast hour h:

```
mean(h)  = L1 seasonal_fi(h)
         + L2 ridge(h)
         + L3 φ^h · η(t₀−1)
mean(h)  = softplus_floor(mean(h), floor = −5 EUR/MWh)
mean(h) -= hourly_bias_corrector.bias_estimate   # only when warmed up

P{5,25,50,75,95}_eur_mwh(h) ← 500-sample mixture (Normal body + GPD tails)
                              centered on mean(h)
```

### L1 — Seasonal decomposition

`Pipeline._seasonal_fi` and `Pipeline._deseasonalize_input` ([`pipeline.py:200-216`](custom_components/spot_price_predictor/pipeline.py:200)) read additive hour + day + week components from `seasonal_components_default.json` and subtract them from FI price and temperature to produce deseasonalized residuals. Wind and solar are not deseasonalized via L1 — they are locally centered (mean-subtracted) before entering the Ridge.

### L2 — Non-seasonal Ridge regression (six features)

The feature ordering is fixed in `RIDGE_FEATURES` ([`pipeline.py:62-69`](custom_components/spot_price_predictor/pipeline.py:62)):

| # | Feature | Built in `_build_features` ([`pipeline.py:220-242`](custom_components/spot_price_predictor/pipeline.py:220)) | Definition |
|---|---|---|---|
| 1 | `intercept` | `np.ones(n)` | constant 1 |
| 2 | `Y_fi_lag168` | passed by caller via `recent_fi_residuals["lag168"]` | Deseasonalized FI residual 7 days prior. Coordinator currently passes zeros (cold-start prior) because the rolling forecast history is shorter than 7 days. |
| 3 | `is_workday` | `Pipeline._is_workday` ([`pipeline.py:244-250`](custom_components/spot_price_predictor/pipeline.py:244)) — `weekday < 5` | binary {0, 1} |
| 4 | `Y_sigmoid_wind_rho` | `_sigmoid_turbine_rho` ([`pipeline.py:81-87`](custom_components/spot_price_predictor/pipeline.py:81)), then locally centered | `σ((wind − 7.5) / 1.5) × ρ(T) / 1.225` |
| 5 | `Y_solar_effective` | `_solar_effective` ([`pipeline.py:90-96`](custom_components/spot_price_predictor/pipeline.py:90)), then locally centered | `GHI × (1 − 0.004 · max(0, T_cell − 25))`, `T_cell = T + 0.03 · GHI` |
| 6 | `Y_temp` | `_deseasonalize_input("temp", …)` | Deseasonalized temperature |

The Ridge prediction is `ridge = X @ self._ridge_coef` ([`pipeline.py:361`](custom_components/spot_price_predictor/pipeline.py:361)).

### L3 — AR(1) momentum

`Pipeline._ar_contribution` ([`pipeline.py:254-266`](custom_components/spot_price_predictor/pipeline.py:254)) returns `φ^h · last_eta` for h = 1..n. `φ` is loaded from `spike_model_default.json` (`ar1_phi`); `last_eta` is the most-recent observed post-AR residual, updated via `Pipeline.update_with_actuals` ([`pipeline.py:423`](custom_components/spot_price_predictor/pipeline.py:423)). When `last_eta` is unknown (cold start), L3 contributes zero.

### Softplus floor and hourly bias EMA

Before the fan-chart is sampled, the mean is floored at −5 EUR/MWh via a softplus (`price_floor.apply_floor`, default `_pf.DEFAULT_FLOOR_EUR_MWH`). A slowly-moving bias estimate is then subtracted by the `HourlyBiasCorrector` (`hourly_calibration.py`, half-life 14 days, warmup 168 hours). The corrected mean is the `spot_eur_mwh` value on each forecast row.

### L4 — GPD POT spike model

`Pipeline._sample_fan_chart` (called from `compute_forecast` when `enable_fan_chart=True`) draws 500 paths from a mixture of a Normal body (μ, σ from `stats.eta_train_mean` / `stats.eta_train_sigma`) and Generalized Pareto right / left tails (`gpd_right`, `gpd_left` parameter blocks: `threshold`, `shape`, `scale`, `p_exceed`). The empirical 5th / 25th / 50th / 75th / 95th percentiles per hour become `P5_eur_mwh` … `P95_eur_mwh` on each forecast row.

### Persistent calibrators

Three calibrators serialise their state on every coordinator cycle (`Pipeline.save_state`):

| State file | Class | Configured defaults |
|---|---|---|
| `hourly_bias.json` | `HourlyBiasCorrector` | halflife_days = 14, warmup_hours = 168 |
| `hourly_fan_chart.json` | `HourlyFanChartCalibrator` | target_coverages = (0.5, 0.9), window = 720, min_warmup = 24 |
| `refit_monitor.json` | `RefitMonitor` | target_coverage = 0.9, drift_pp = 0.05, persistence_steps = 14 × 24 |

The refit monitor raises `refit_recommended` when realised coverage drifts more than 5 pp from the target for 14 consecutive days.

## Inputs evaluated but not currently consumed

The coordinator fetches several streams that no longer feed the user-facing spot forecast:

- **Sahkotin neighbour and historical prices, Elpriset SE1/SE3, Elering EE.** Fed into the legacy duration model and the actual-D(k) prepend; not used by `RIDGE_FEATURES`.
- **Fingrid #188** (nuclear production). Currently feeds the legacy duration model's `nuclear_deficit` segment feature; not in `RIDGE_FEATURES`.
- **Fingrid #165 / #246 / #247** (consumption / wind / solar forecasts). Auxiliary streams; not in `RIDGE_FEATURES`.
- **Nord Pool UMM** outage schedule. Fetched and combined with `nuclear_mw` to produce `nuclear_hourly` for the duration model; not in `RIDGE_FEATURES`.

The canonical D(k) attributes on the duration sensor are built directly from the pipeline's hourly spot forecasts (see "Duration model" below), so the legacy duration model's per-segment outputs do not currently surface on user-facing attributes. Re-introducing nuclear-deficit or cross-border transit-capacity features is treated as experimental work, evaluated on a separate branch, and only merged based on measured performance against `main`.

## Duration model — the four D(k) arrays per day

Implemented in `SpotPriceCoordinator._apply_pipeline_pre_dk` and `_compute_duration_forecast` ([`coordinator.py`](custom_components/spot_price_predictor/coordinator.py)).

For each local day with 24 complete hours in the forecast window:

1. **Spot side** — `Pipeline.compute_duration_curves` ([`pipeline.py:388-421`](custom_components/spot_price_predictor/pipeline.py:388)) sorts the 24 hourly `mean_eur_mwh` values ascending and descending, then takes cumulative means. Result: `dk_cheap_eur_mwh[24]` (mean of (i+1) cheapest hours) and `dk_peak_eur_mwh[24]` (mean of (i+1) priciest hours), each monotone in i.
2. **Consumer side** — each hourly spot is first converted to consumer EUR/kWh using the per-hour tariff (`day_rate` for hours 07–22, `night_rate` for hours 22–07), then the same sort + cumulative-mean step yields `dk_cheap_eur_kwh[24]` / `dk_peak_eur_kwh[24]`.

All four arrays are 0-indexed; `dk_cheap_eur_mwh[3]` is the mean of the four cheapest hours of that day (and similarly for the consumer arrays). The full-day mean is recovered at index 23 in either direction: `dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == daily_average_spot`.

For past days, `_compute_actual_duration_curves` produces the same four arrays from observed Sahkotin spot prices and stamps the entry `source: "actual"`.

## PV-aware effective pricing (optional)

When `pv_capacity_kwp > 0` (or a `pv_external_entity` is configured), the coordinator augments every forecast hour with an `effective_eur_kwh` — the marginal cost of running one extra kWh of flexible load given the household's PV and typical baseload. Implementation: `marginal_effective_eur_kwh` and `net_household_cost_eur` in [`pv_estimate.py`](custom_components/spot_price_predictor/pv_estimate.py).

Per hour h with consumer buy price `b_h`, sell price `s_h = max(0, spot − pv_sell_commission − pv_export_grid_fee)`, PV production `p_h`, and baseload `c_h`:

```
pv_avail_h = max(0, p_h − c_h)
from_pv    = min(1, pv_avail_h)
from_grid  = 1 − from_pv
m_h        = from_pv · s_h + from_grid · b_h
```

`m_h` is bounded analytically in `[s_h, b_h]`, captures self-consumption gain when surplus ≥ 1 kWh, and propagates negative spot prices through to `effective_eur_kwh` (rare but real).

The duration sensor then computes parallel `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` arrays from the sorted hourly `effective_eur_kwh` per day.

### PV input modes

- **Internal estimator (default)** — uses Open-Meteo `global_tilted_irradiance_instant` × `pv_capacity_kwp` × tilt/azimuth correction × `pv_system_efficiency`. Free, 7-day horizon.
- **External entity (override)** — set `pv_external_entity` to any HA sensor whose attributes match one of the supported conventions (`forecast` list-of-dict in kWh, `wh_hours` dict in Wh, `watts` dict in W, or `irradiance` list auto-detected). Forecast.Solar, EMHASS, and custom Open-Meteo templates all work.

### Baseload

Two configuration fields drive the baseload `c_h` used in the PV calculation:

- **`annual_consumption_kwh`** (default 12 000) — typical TOTAL annual household demand from the bill. Multiplied by a 12-element Finnish residential monthly seasonal profile (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS` in [`const.py`](custom_components/spot_price_predictor/const.py); sum of factors = 12.00 exactly, range ≈ ±19 %).
- **`consumption_entity`** (optional) — any HA consumption sensor (cumulative-kWh counter, daily/monthly `utility_meter`, or instantaneous-power sensor). The integration auto-detects the sensor type, smooths it on a 14-day rolling window with a 5 % hysteresis dead-band, and caches the result under `.storage/spot_price_predictor_consumption_cache.json` (recomputed at most once per day).

### Stability invariant

`_resolve_baseload(ts)` MUST NOT call `hass.states.get` or any HA entity-read API — enforced by a grep test in `tests/test_coordinator_pv.py`. With `consumption_entity = ""` the baseload is a deterministic function of `(annual_consumption_kwh, hour-of-year)` and the forecast is fully open-loop wrt the optimizer. With `consumption_entity` set, the 14-day rolling smoothing + 5 % hysteresis keeps the closed-loop gain well below 1, so EMHASS's daily rescheduling cannot create oscillation.

## DtACI calibration (optional)

When `enable_dtaci_dk = true` the coordinator wraps the D(k) curves with adaptive conformal bands. Implementation: [`dk_dtaci.py`](custom_components/spot_price_predictor/dk_dtaci.py) (`DkDtACIBundle`) + [`dtaci_integration.py`](custom_components/spot_price_predictor/dtaci_integration.py).

- **48 DtACI instances per zone** — one per `(direction, k)` for direction ∈ {cheap, peak} and k = 1..24. Each instance tracks its own residual distribution, alpha, dominant gamma, weight entropy, and per-instance bias EMA.
- **Bands written back** as 24-element arrays `dk_cheap_lower_eur_kwh`, `dk_cheap_upper_eur_kwh`, `dk_peak_lower_eur_kwh`, `dk_peak_upper_eur_kwh` per day entry once instances have warmed up. Before warmup, bands collapse to the point forecast (deliberate — no spurious confidence).
- **Diagnostics** surface on the duration sensor as `dtaci_diagnostics`, `dtaci_warmup_status`, `dtaci_target_coverage`, `dtaci_fi_mean_coverage`, `dtaci_fi_mean_width_eur_kwh`, `dtaci_fi_warm_instances`, `dtaci_fi_total_instances`, `dtaci_min_n_updates`.

State is persisted to `<config>/.storage/spot_price_predictor_dtaci_dk_fi.json`. Warmup completes after ≈ 5 days of reconciled daily updates.

See [docs/dtaci_layer.md](docs/dtaci_layer.md) for the algorithm details (Gibbs & Candès JMLR 2024) and troubleshooting.

## Sensor attribute reference

### Price forecast sensor — forecast row keys

Each entry in `forecast[]` (length 170):

| Key | Type | Source |
|---|---|---|
| `timestamp` | string (ISO 8601 UTC) | hour index from coordinator |
| `spot_eur_mwh` | float | `Pipeline.compute_forecast` mean (with floor + bias correction) |
| `consumer_eur_kwh` | float | `spot_eur_mwh / 1000 + seller_margin + transfer + energy_tax) × VAT`, with per-hour day/night transfer rate |
| `wind` | float (m/s) | Open-Meteo capacity-weighted 120 m wind |
| `solar` | float (W/m²) | Open-Meteo GHI capacity-weighted |
| `temp` | float (°C) | Open-Meteo temperature capacity-weighted |
| `P5_eur_mwh`, `P25_eur_mwh`, `P50_eur_mwh`, `P75_eur_mwh`, `P95_eur_mwh` | float | L4 GPD POT fan-chart percentiles |
| `pv_production_kwh`, `baseload_kwh`, `effective_eur_kwh`, `net_household_cost_eur`, `is_export_hour`, `sell_eur_kwh` | varied | Present only when PV is enabled |

### Duration forecast sensor — day entry keys

Each entry in `daily_forecast[]` (up to 7):

| Key | Type | Notes |
|---|---|---|
| `date`, `weekday`, `source` | string | `source ∈ {"forecast", "actual"}` |
| `dk_cheap_eur_mwh`, `dk_peak_eur_mwh` | float[24] | Spot EUR/MWh, 0-indexed, monotone in i |
| `dk_cheap_eur_kwh`, `dk_peak_eur_kwh` | float[24] | Consumer EUR/kWh, per-hour tariff applied |
| `dk_cheap_pv_eur_kwh`, `dk_peak_pv_eur_kwh` | float[24] | PV-aware variants. Present only when PV is enabled. |
| `dk_cheap_lower_eur_kwh`, `dk_cheap_upper_eur_kwh`, `dk_peak_lower_eur_kwh`, `dk_peak_upper_eur_kwh` | float[24] | DtACI bands. Present only when DtACI is enabled and instances are warm. |

### Diagnostics on the coordinator result

| Key | Source |
|---|---|
| `pipeline_diagnostics` | `pipeline_bias_eur_mwh`, `pipeline_ar1_phi`, `pipeline_n_features`, `pipeline_floor_eur_mwh` |
| `dtaci_diagnostics` | Returned by `DkDtACIBundle.diagnostics()` when DtACI is enabled |
| `data_sources_active` | Plain-text summary of fetched streams |
| `last_update`, `stale`, `data_age_minutes` | Standard status block |
| `pv_enabled`, `pv_capacity_kwp`, `pv_source`, `baseload_kwh_per_hour`, `current_effective_eur_kwh` | PV metadata (always emitted) |

## Home Assistant services

Registered in [`__init__.py`](custom_components/spot_price_predictor/__init__.py); schemas declared in [`services.yaml`](custom_components/spot_price_predictor/services.yaml):

| Service | Arguments | Effect |
|---|---|---|
| `spot_price_predictor.retrain_models` | `layers` (optional list, subset of `{seasonal, spike, solar}`); `fingrid_api_key` (optional, also read from env) | Refits the listed (or all) artifacts atomically, reloads the `Pipeline` on every active coordinator, fires `spot_price_predictor_models_retrained` on completion. |
| `spot_price_predictor.force_refresh` | none | Triggers an immediate coordinator update. |
| `spot_price_predictor.model_info` | none | Posts a persistent notification with current artifact metadata. |
| `spot_price_predictor.upload_coefficients` | `file_path` (optional) or `json_data` (optional) | Replaces the legacy v2.2 user-coefficient file. |
| `spot_price_predictor.reset_coefficients` | none | Reverts to the bundled default coefficients. |

The integration emits the event `spot_price_predictor_models_retrained` after a successful retrain, with `{"result": …, "reloaded_coordinators": …}` payload.

## Update cadence

Defined in [`const.py:155-161`](custom_components/spot_price_predictor/const.py:155):

| Constant | Value | Used for |
|---|---|---|
| `UPDATE_INTERVAL_WEATHER` | 21 600 s (6 h) | Coordinator periodic refresh after success. |
| `UPDATE_INTERVAL_FINGRID` | 3 600 s (1 h) | Fingrid datasets refresh interval. |
| `FORECAST_HOURS` | 170 | Length of the per-hour forecast array. |

## Localization

The system is driven by `config/regions/finland.yaml`. To support a new region:

1. Identify 5–8 weather measurement locations, weighted by installed wind / solar capacity and population.
2. Create a new YAML file (e.g. `sweden.yaml`) with the local price API, holiday rules, consumer pricing, and (optionally) neighbour-zone price sources.
3. Refit via the `spot_price_predictor.retrain_models` service with regional cached data.

The pipeline architecture (L1 seasonal + L2 physics-feature Ridge + L3 AR(1) + L4 GPD POT) is applicable to any electricity market with weather-dependent generation.

## Project Structure

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md
├── TEKNINEN_TOTEUTUS.md
├── INSTALLATION.md
├── config/regions/finland.yaml
├── custom_components/spot_price_predictor/
│   ├── __init__.py                 # entry point + service registration
│   ├── coordinator.py              # DataUpdateCoordinator + pipeline orchestration
│   ├── pipeline.py                 # Pipeline (L1+L2+L3+L4+floor+bias EMA)
│   ├── seasonal_decomposition.py   # L1 component fitter / lookup
│   ├── hourly_calibration.py       # HourlyBiasCorrector / HourlyFanChartCalibrator / RefitMonitor
│   ├── dk_dtaci.py                 # DkDtACIBundle (48 instances per zone)
│   ├── dtaci_integration.py        # bundle ↔ duration_forecast wiring
│   ├── price_floor.py              # softplus floor at −5 EUR/MWh
│   ├── solar_clear_sky.py          # clear-sky × cloudiness solar sub-model
│   ├── pv_estimate.py              # internal PV estimator + marginal effective price
│   ├── retrain.py                  # retrain_models orchestrator
│   ├── sensor.py                   # sensor entities
│   ├── api_client.py               # async API clients
│   ├── const.py                    # constants, operators, config keys
│   ├── config_flow.py              # HA setup wizard
│   └── data/
│       ├── seasonal_components_default.json
│       ├── spike_model_default.json
│       ├── solar_submodel_default.json
│       └── finland.yaml
├── studies/                        # retrain scripts + historical analyses
└── tests/                          # pytest suite (402 passed at v2.8.1)
```
