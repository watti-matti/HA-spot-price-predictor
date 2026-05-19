# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.10.0-blue.svg)](https://github.com/watti-matti/HA-spot-price-predictor/releases/tag/v2.10.0)

**Forecast Finnish electricity prices for the next 170 hours**, in both spot (EUR/MWh) and consumer (EUR/kWh) terms, with calibrated probabilistic bands and 7-day duration curves for cost-aware load scheduling.

[Suomenkieliset ohjeet (Finnish)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour point forecast** in spot EUR/MWh and consumer EUR/kWh (transfer tariff, energy tax, seller margin, and VAT applied per hour).
- **Probabilistic fan chart** — per-hour P5 / P25 / P50 / P75 / P95 bands sampled from a Normal-body + Generalized Pareto tail mixture (heavy-tail spike model).
- **D(k) cheap/peak duration curves** — 7 days × 4 arrays per day, each 24-entry and 0-indexed: `dk_cheap_eur_mwh[i]` / `dk_peak_eur_mwh[i]` (spot) and `dk_cheap_eur_kwh[i]` / `dk_peak_eur_kwh[i]` (consumer). Each `[i]` is the mean of the (i+1) cheapest / priciest hours of the day. Equivalent to CVaR at α=(i+1)/24 in both tails.
- **Optional PV-aware effective price** — when a `pv_capacity_kwp > 0` (or an external PV-forecast entity) is configured, each forecast hour gains `effective_eur_kwh` (marginal cost of running one extra kWh given PV self-consumption) and parallel `dk_cheap_pv_eur_kwh[24]` / `dk_peak_pv_eur_kwh[24]` curves.
- **Optional online calibration (DtACI)** — adaptive conformal prediction intervals on the D(k) curves, with per-(direction, k) bias correction. Targets 90 % marginal coverage; warms up over the first few days.
- **Refit on demand** — the `spot_price_predictor.retrain_models` Home Assistant service refits the L1 seasonal, L2/L3/L4 spike, and (optionally) solar sub-model artifacts and reloads the pipeline without a Home Assistant restart.
- **All data sources are free.** The integration ships pre-trained artifacts and works out of the box after picking your distribution operator.

## How It Works

The point forecast is produced by a four-layer pipeline implemented in [`custom_components/spot_price_predictor/pipeline.py`](custom_components/spot_price_predictor/pipeline.py) (class `Pipeline`). At each forecast hour h the pipeline computes:

```
spot_eur_mwh(h) =  L1 seasonal_fi(h)         # additive hour+day+week pattern
                 + L2 ridge(h)                # 8-feature physics + cross-border Ridge
                 + L3 φ^h · η(t₀−1)           # AR(1) momentum on the last residual
                 - softplus_floor(−5)         # clamps deep negatives
                 - hourly_bias_ema(h)         # DtACI bias corrector
```

The same point forecast is then passed through the **L4 GPD POT spike model**, which samples 500 paths from a Normal body + Generalized Pareto right/left tails to produce the per-hour `P5_eur_mwh` … `P95_eur_mwh` fan-chart bands.

### L2 Ridge — the eight features

Defined as `RIDGE_FEATURES` in [`pipeline.py:62-77`](custom_components/spot_price_predictor/pipeline.py:62). The shipped `data/spike_model_default.json` lists the same order in its `ridge_features` field; the pipeline is feature-list-driven and always reads from the artifact.

| # | Feature | Definition |
|---|---|---|
| 1 | `intercept` | constant 1 |
| 2 | `Y_fi_lag168` | deseasonalized FI spot residual 7 days ago — own-lag memory of the local market regime |
| 3 | `is_workday` | `weekday < 5` (calendar) |
| 4 | `Y_sigmoid_wind_rho` | sigmoid wind-power curve scaled by relative air density: `σ((wind − 7.5) / 1.5) × ρ(T) / 1.225` |
| 5 | `Y_solar_effective` | temperature-derated GHI: `GHI × (1 − 0.004 · max(0, T_cell − 25))` with `T_cell = T + 0.03 · GHI` |
| 6 | `Y_temp` | deseasonalized temperature |
| 7 | `Y_se1` | deseasonalized SE1 spot — Sweden zone 1 (added in v2.10.0; passes the v2.5.6 hedge gate) |
| 8 | `Y_se3` | deseasonalized SE3 spot — Sweden zone 3, the FennoSkan cable terminus |
| 9 | `Y_ee` | deseasonalized EE spot — Estonia, the Estlink terminus |

Coefficients live in `data/spike_model_default.json` under `ridge_coef`.

### Inputs the pipeline consumes

| Input | Provided by |
|---|---|
| `wind` (m/s at 120 m), `solar` (GHI W/m²), `temp` (°C) | Open-Meteo, 7 capacity-weighted Finnish sites |
| `Y_fi_lag168` | Recent FI spot history (cold-start = zero until the rolling history is 7 days deep) |
| Neighbour spot prices (SE1, SE3, EE) | elprisetjustnu.se + Elering, via `fetch_neighbor_prices()` |
| Hourly seasonal components for FI / SE1 / SE3 / EE / temperature | `data/seasonal_components_default.json` |
| Ridge β, AR(1) φ, GPD tail params, Normal-body μ/σ | `data/spike_model_default.json` |

### Inputs fetched but not currently consumed by the spot model

The integration also fetches Fingrid datasets #188 / #165 / #246 / #247 and Nord Pool UMM nuclear-outage schedules. These streams feed the legacy duration model and some auxiliary diagnostics, but the canonical user-facing forecast is driven by the eight L2 features above. Re-introducing nuclear-deficit-related signals is treated as experimental work; the `experiment/extra-l2-features` branch documents the test results (nuclear features, even capacity-aware and as multiplicative coupling coefficients, did not pass the hedge gate on the 2023-2026 data window).

## Sensors Created

All sensors share the device "Spot Price Predictor" and the domain prefix `spot_price_predictor`.

### Always created

| Sensor entity ID (suffix) | State | Unit |
|---|---|---|
| `sensor.spot_price_predictor_price_forecast` | Current hour consumer price | EUR/kWh |
| `sensor.spot_price_predictor_duration_forecast` | Today's `dk_cheap_eur_kwh[3]` (mean of cheapest 4 hours) | EUR/kWh |

### Conditional

| Sensor | Created when |
|---|---|
| `sensor.spot_price_predictor_spot_electricity_price` | A `nordpool_entity` is configured |
| `sensor.spot_price_predictor_spot_electricity_selling_price` | A `nordpool_entity` is configured **and** `enable_pv_selling` is on |

### Price Forecast — attributes

| Attribute | Type | Description |
|---|---|---|
| `forecast` | array[170] | One entry per hour. Keys: `timestamp`, `spot_eur_mwh`, `consumer_eur_kwh`, `wind`, `solar`, `temp`, `P5_eur_mwh`, `P25_eur_mwh`, `P50_eur_mwh`, `P75_eur_mwh`, `P95_eur_mwh`. PV-aware extra keys when enabled: `pv_production_kwh`, `baseload_kwh`, `effective_eur_kwh`, `net_household_cost_eur`, `is_export_hour`, `sell_eur_kwh`. |
| `current_spot_eur_mwh` | float | Spot price for the current hour. |
| `forecast_hours` | int | Length of the `forecast` array. |
| `operator` | string | Configured distribution operator. |
| `week_min_eur_kwh` / `week_avg_eur_kwh` / `week_max_eur_kwh` | float | Statistics over the consumer prices in the forecast window. |
| `last_update`, `data_sources_active`, `stale`, `data_age_minutes` | — | Standard status block. |
| `pv_capacity_kwp`, `pv_source`, `baseload_kwh_per_hour`, `current_effective_eur_kwh`, `week_min/avg/max_effective_eur_kwh` | — | Present only when PV is enabled. |

### Duration Forecast — attributes

| Attribute | Shape | Unit | Description |
|---|---|---|---|
| `daily_forecast` | array[≤7] | — | One entry per day. See "day entry" table below. |
| `forecast_days` | int | — | Length of `daily_forecast`. |
| `today_cheap_1h_eur_kwh` / `today_cheap_4h_eur_kwh` / `today_cheap_8h_eur_kwh` / `today_cheap_12h_eur_kwh` | float | EUR/kWh | Convenience scalars — `dk_cheap_eur_kwh[0/3/7/11]` for day 0. |
| `today_peak_1h_eur_kwh` / `today_peak_4h_eur_kwh` / `today_peak_8h_eur_kwh` / `today_peak_12h_eur_kwh` | float | EUR/kWh | Convenience scalars — `dk_peak_eur_kwh[0/3/7/11]` for day 0. |
| `today_cheap_pv_*h_eur_kwh` / `today_peak_pv_*h_eur_kwh` | float | EUR/kWh | PV-aware versions of the above. Present only when PV is enabled. |
| `today_cheap_4h_lower_eur_kwh` / `today_cheap_4h_upper_eur_kwh` / `today_peak_1h_lower_eur_kwh` / `today_peak_1h_upper_eur_kwh` | float | EUR/kWh | DtACI calibrated band endpoints. Present only when DtACI is enabled and warmed up. |
| `dtaci_diagnostics`, `dtaci_warmup_status`, `dtaci_target_coverage`, `dtaci_fi_mean_coverage`, `dtaci_fi_mean_width_eur_kwh`, `dtaci_fi_warm_instances`, `dtaci_fi_total_instances`, `dtaci_min_n_updates` | — | DtACI calibrator diagnostics. Present only when DtACI is enabled. |
| `pv_capacity_kwp`, `pv_source` | — | Mirrored from the price-forecast sensor for dashboard convenience when PV is enabled. |

#### `daily_forecast[i]` — keys

| Key | Shape | Unit | Description |
|---|---|---|---|
| `date` | string | — | ISO date (YYYY-MM-DD) |
| `weekday` | string | — | `Mon` … `Sun` |
| `source` | string | — | `forecast` for future days, `actual` for past days reconciled from Sahkotin |
| `dk_cheap_eur_mwh` | float[24] | EUR/MWh | `[i]` = mean spot price of the (i+1) cheapest hours of the day, i = 0..23 (monotone non-decreasing) |
| `dk_peak_eur_mwh` | float[24] | EUR/MWh | `[i]` = mean spot price of the (i+1) priciest hours of the day, i = 0..23 (monotone non-increasing) |
| `dk_cheap_eur_kwh` | float[24] | EUR/kWh | Same cheapest-end curve in consumer price (per-hour day/night tariff applied) |
| `dk_peak_eur_kwh` | float[24] | EUR/kWh | Same priciest-end curve in consumer price |
| `dk_cheap_pv_eur_kwh`, `dk_peak_pv_eur_kwh` | float[24] | EUR/kWh | PV-aware variants. Present only when PV is enabled. |
| `dk_cheap_lower_eur_kwh`, `dk_cheap_upper_eur_kwh`, `dk_peak_lower_eur_kwh`, `dk_peak_upper_eur_kwh` | float[24] | EUR/kWh | DtACI band endpoints. Present only when DtACI is enabled and warmed up. |

**Identity** — the full-day mean is direction-invariant: `dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == daily_average_spot` (and the same for the `_eur_kwh` arrays in consumer space).

### Spot Electricity Price / Selling Price (Nordpool, optional)

When a Nordpool entity is configured, the integration also exposes the actual current price (and, if `enable_pv_selling` is on, the selling price = spot − configured commission). These sensors apply the same overhead as the forecast (`spot + seller_margin + transfer + energy_tax) × VAT`) so dashboards can compare forecast vs actual in identical units.

## Supported Operators (Finland)

Defined in [`const.py:169-195`](custom_components/spot_price_predictor/const.py:169):

| Operator | Day rate (07–22) | Night rate (22–07) |
|---|:---:|:---:|
| Elenia | 3.61 c/kWh | 2.20 c/kWh |
| Caruna Espoo | 2.21 c/kWh | 2.21 c/kWh |
| Caruna North | 4.07 c/kWh | 2.49 c/kWh |
| Helen | 3.54 c/kWh | 3.54 c/kWh |
| Custom | user-defined | user-defined |

VAT defaults to 25.5 % (1.255 multiplier); energy tax defaults to 0.02325 EUR/kWh (class I, 2026); seller margin defaults to 0 and is meant to be set from your contract.

## Data Sources

All sources are free; the Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

| Source | API client method | Role in current system |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) | `fetch_weather()` | Wind (120 m), solar (GHI), temperature — **driving inputs of the L2 Ridge.** |
| [Open-Meteo Historical](https://historical-forecast-api.open-meteo.com) | — | Used by the retraining scripts (not by inference). |
| [Sahkotin](https://sahkotin.fi) | `fetch_spot_prices()`, `fetch_spot_prices_historical()` | FI Nord Pool spot (current + last 2 days). Feeds the actual-D(k) prepend for past days; feeds the legacy duration model. |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | `fetch_neighbor_prices()` | SE1, SE3 day-ahead spot — duration model input only, not in the spot-pipeline features. |
| [Elering](https://dashboard.elering.ee) | `fetch_neighbor_prices()` | EE day-ahead spot — same as SE1/SE3. |
| [Fingrid #188](https://data.fingrid.fi) | `fetch_fingrid_data()` | Real-time nuclear production. Fed into the duration model as `nuclear_deficit`; not in the spot-pipeline features. |
| [Fingrid #165 / #246 / #247](https://data.fingrid.fi) | `fetch_fingrid_forecasts()` | Day-ahead consumption / wind / solar forecasts. Auxiliary streams; not in the spot-pipeline features. |
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com) | `fetch_nuclear_outage_schedule()` | Planned nuclear outage schedule for forward-looking nuclear capacity; not in the spot-pipeline features today. |

## Configuration

The HA config flow is a four-step wizard (region → operator and tariffs → optional APIs → PV system).

Key options:

| Option (key) | Default | Notes |
|---|---|---|
| `region` | `finland` | Only Finland is currently supported. |
| `operator` | `elenia` | One of `elenia`, `caruna_espoo`, `caruna_north`, `helen`, `custom`. |
| `custom_day_rate`, `custom_night_rate` | from operator | Override per-tariff if `operator = custom`. |
| `custom_vat` | 1.255 | VAT multiplier. |
| `custom_energy_tax` | 0.02325 EUR/kWh | Class I (2026). |
| `seller_margin` | 0.0 EUR/kWh | Set from your electricity contract. |
| `nordpool_entity` | "" | Linking your Nordpool integration enables the optional actual-price sensors. |
| `enable_pv_selling` | false | Enables the selling-price sensor when a Nordpool entity is linked. |
| `pv_sell_commission` | 0.002 EUR/kWh | Used by the selling-price sensor and by the PV-aware effective price. |
| `fingrid_api_key` | "" | Free key from data.fingrid.fi. Enables Fingrid data fetches for the duration model and the solar sub-model retraining. |
| `enable_neighbor_prices` | true | Toggles SE1/SE3/EE fetches. |
| `enable_dtaci_dk` | false | Enables the per-(direction, k) DtACI calibration on the D(k) curves. |
| `pv_capacity_kwp` | 0.0 | 0 disables PV-aware outputs. |
| `pv_tilt_deg` / `pv_azimuth_deg` / `pv_system_efficiency` | 45 / 180 / 0.85 | Internal PV estimator parameters. |
| `pv_external_entity` | "" | Optional HA sensor to override the internal PV estimate. Supported attribute conventions: `forecast` (list[dict]), `wh_hours` (dict), `watts` (dict), `irradiance` (list). |
| `pv_export_grid_fee` | 0 EUR/kWh | Extra fee on exported energy. |
| `annual_consumption_kwh` | 12 000 | Typical total annual household demand from the bill. Drives the baseload via a Finnish monthly seasonal profile. |
| `consumption_entity` | "" | Optional HA consumption sensor for adaptive baseload (14-day rolling smoothing + 5 % hysteresis). |

## Installation

See [INSTALLATION.md](INSTALLATION.md) for the step-by-step setup with screenshots. Short version:

1. Add this repo as a HACS custom repository (type: Integration), download, and restart Home Assistant.
2. **Settings → Devices & Services → Add Integration → Spot Price Predictor**.
3. Follow the four-step wizard.

## Retraining the bundled models

Use the `spot_price_predictor.retrain_models` Home Assistant service to refit the artifacts in place from cached data — no PC-side training or coefficient upload required.

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike", "solar"]   # omit to refit all three
  # fingrid_api_key: "..."                  # only needed for the solar layer
```

| Artifact | Produced by | Layers it carries |
|---|---|---|
| `data/seasonal_components_default.json` | `studies/build_seasonal_components.py` | L1 — hour / day / week components for FI price and temperature |
| `data/spike_model_default.json` | `studies/v2513_layer4_spike_model.py` | L2 Ridge `ridge_coef`, L3 `ar1_phi`, L4 `gpd_left` / `gpd_right` + Normal-body stats |
| `data/solar_submodel_default.json` | `studies/v253_solar_submodel.py` | Clear-sky × cloudiness solar production sub-model (used by PV-aware path) |

On completion the service fires the `spot_price_predictor_models_retrained` Home Assistant event so automations can react.

Other services available on the integration: `spot_price_predictor.force_refresh` (re-runs a coordinator cycle), `spot_price_predictor.model_info` (persistent notification with current artifact metadata), `spot_price_predictor.upload_coefficients` and `spot_price_predictor.reset_coefficients` (manage the legacy v2.2 user-coefficient file).

## Localization

The system is driven by `config/regions/finland.yaml`. To support a new region, create a new YAML file (`sweden.yaml`, etc.) covering price API endpoints, weather stations, holidays, consumer pricing, and neighbour price sources, then refit via the retrain service with regional cached data.

## Technical Documentation

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) — Architecture, the four-layer pipeline, PV-aware pricing, DtACI calibration, retraining (English).
- [TEKNINEN_TOTEUTUS.md](TEKNINEN_TOTEUTUS.md) — Sama suomeksi.
- [INSTALLATION.md](INSTALLATION.md) — Step-by-step setup with screenshots.
- [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md) — Canonical duration-sensor schema reference.
- [docs/dtaci_layer.md](docs/dtaci_layer.md) — DtACI online calibration: algorithm details, state persistence, troubleshooting.
- `studies/results/` — Supporting analyses and the v2.8.1 release notes.

## License

[MIT](LICENSE)
