# Documentation: HA-spot-price-predictor (v2.8.1)

Consumer electricity price and D(k) = CVaR duration cost forecasting for Home Assistant. Produces 170-hour consumer price forecasts (EUR/kWh) and 7-day D(k) cheap/peak duration curves for cost-optimal load scheduling, using a four-layer pipeline — L1 seasonal decomposition, L2 physics-feature Ridge, L3 AR(1) momentum, L4 GPD POT spike model — together with a softplus negative-price floor and an hourly DtACI calibrator. Optionally augments every forecast hour with a PV-aware marginal effective price `m_h` and parallel PV-aware D(k) curves when the user configures household solar.

## Architecture

The system has two phases: **inference** (Home Assistant custom integration, always-on) and **retraining** (refits the bundled artifacts on demand via a Home Assistant service). Both phases share the same code paths under `custom_components/spot_price_predictor/`.

### Retraining flow (Home Assistant service `spot_price_predictor.retrain_models`)

```
Cached price + weather   ──> studies/build_seasonal_components.py  ──> data/seasonal_components_default.json
parquets / Sahkotin         studies/v2513_layer4_spike_model.py     ──> data/spike_model_default.json
                            studies/v253_solar_submodel.py          ──> data/solar_submodel_default.json
                                                                          │
                                                                          v
                                                        Atomic JSON writes + Pipeline auto-reload
                                                        (fires spot_price_predictor_models_retrained)
```

### Home Assistant Deployment

```
Open-Meteo  ──┐
Elpriset    ──┼──> Coordinator ──> Pipeline (L1 seasonal + L2 Ridge + L3 AR(1) + L4 GPD POT)
Elering     ──┤    + Data fetch    + Softplus floor + Hourly DtACI bias/fan calibrators
Fingrid     ──┤    + Tariff        + Duration model (segment-hierarchical Ridge + PAVA)
Sahkotin    ──┤    conversion              │
Nord Pool UMM ┘                            v
                                  Spot/Consumer Forecast (170h, with P5/P25/P50/P75/P95 fan)
                                  + D(k) cheap/peak (7 days)
                                  + (optional) PV-aware effective price and PV-aware D(k)
                                  ↑
                                  └── Open-Meteo irradiance (internal)
                                      OR pv_external_entity (Forecast.Solar / EMHASS / template)
```

### Dashboards

Two visualization dashboards are available:

| Dashboard | Script | Purpose |
|-----------|--------|---------|
| `model_dashboard.html` | `model_dashboard.py` | Model monitoring: D(k) accuracy, feature importance, rolling Spearman, lambda sweep |
| `forecast.html` | `forecast_dashboard.py` | Live 7-day forecast: D(k) duration curves, hourly prices, weather context |

---

## Data Sources

All data sources are configured in `config/regions/finland.yaml`.

### Required (free, no authentication)

| Source | Purpose | Rate Limit |
|--------|---------|------------|
| [Sahkotin API](https://sahkotin.fi/prices) | FI Nord Pool spot prices (EUR/MWh) | Unlimited |
| [Open-Meteo API](https://api.open-meteo.com) | Wind (120m), solar (45° tilt), temperature | 10,000/day |
| [Open-Meteo Historical Forecast](https://historical-forecast-api.open-meteo.com) | Historical weather for training | 10,000/day |

### Cross-border price sources (free, no authentication)

| Source | Zones fetched | Consumed by the current non-seasonal model? |
|--------|---------------|---------------------------------------------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE3 | No — used by the duration model and dashboards |
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1 | No — auxiliary spread / historical context |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | No — used by the duration model and dashboards |

Cross-border prices remain part of the data layer for the duration model and historical-context attributes but are not features of the non-seasonal price model.

### Optional grid data (free API key)

| Source | Purpose |
|--------|---------|
| [Fingrid Open Data](https://data.fingrid.fi) | Nuclear production (#188), consumption / wind / solar forecasts (#165 / #246 / #247) — used by the duration model and auxiliary streams |

Register for free at data.fingrid.fi. Without the Fingrid key the non-seasonal price model is unaffected; the duration model falls back to its weather-only feature subset.

### Nuclear outage schedule (free, no key)

| Source | Purpose |
|--------|---------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Planned nuclear outages — auxiliary forward-looking stream; not consumed by the non-seasonal price model |

---

## Feature Engineering

The non-seasonal price model is deliberately compact: six Ridge features composed with an AR(1) momentum term and a heavy-tail spike layer. The complete feature ordering is defined in `custom_components/spot_price_predictor/pipeline.py:68-75` (constant `RIDGE_FEATURES`) — the documentation table below reflects that exact ordering.

### L1 — Seasonal decomposition

`custom_components/spot_price_predictor/seasonal_decomposition.py` implements an additive Moazeni–Powell decomposition: each input series is split into hourly, daily, and weekly components, fitted on the training history and shipped in `data/seasonal_components_default.json`. At inference time the pipeline subtracts the matching components to produce deseasonalized residuals (`Y_*`). FI price uses the full hour+day+week depth; temperature uses hour+week; wind / solar are not deseasonalized by L1 (they are locally centered before the Ridge step instead).

### L2 — Non-seasonal Ridge regression

Six features are stacked into the design matrix in this order (matches `RIDGE_FEATURES`):

| # | Feature (code name) | Built at | Definition |
|---|---|---|---|
| 1 | `intercept` | `pipeline.py:241` | Constant 1. Captures the deseasonalized mean. |
| 2 | `Y_fi_lag168` | `pipeline.py:242` | Deseasonalized FI residual 7 days prior — own-lag memory of the local market regime. Falls back to zero during cold-start. |
| 3 | `is_workday` | `pipeline.py:243`, computed at `:251-256` | `weekday < 5`. Captures industrial demand pattern. |
| 4 | `Y_sigmoid_wind_rho` | `pipeline.py:244`, helper `_sigmoid_turbine_rho` at `:87-93` | `σ((wind − 7.5) / 1.5) × ρ(T) / 1.225`. Sigmoid wind-power curve scaled by relative air density; physics-realistic supply driver. Locally centered before entering the Ridge. |
| 5 | `Y_solar_effective` | `pipeline.py:245`, helper `_solar_effective` at `:96-102` | `GHI × (1 − 0.004 · max(0, T_cell − 25))` with `T_cell = T + 0.03 · GHI`. Temperature-derated effective irradiance. Locally centered before the Ridge. |
| 6 | `Y_temp` | `pipeline.py:246`, deseasonalized at `:239` | Deseasonalized temperature — residual heating-load signal. |

The Ridge coefficient vector lives in `data/spike_model_default.json` under `ridge_coef`; it is applied at `pipeline.py:367` as `ridge = X @ self._ridge_coef`.

### L3 — AR(1) momentum

At forecast horizon `h`, the AR(1) term contributes `φ^h · η(t₀−1)` where `η(t₀−1)` is the most-recent observed deseasonalized FI residual and `φ` is the AR(1) coefficient loaded from `spike_model_default.json` (typically `φ ≈ 0.904`). Implementation: `pipeline.py:260-266`; the residual state is refreshed via `update_with_actuals()` at `:429-446`.

### L4 — GPD POT spike model

A Normal-body + Generalized Pareto tail mixture is sampled 500 times around the L1+L2+L3 point forecast to produce the P5 / P25 / P50 / P75 / P95 fan bands. Parameters live in `spike_model_default.json`:

| Parameter | Role |
|---|---|
| `stats.eta_train_mean`, `stats.eta_train_sigma` | Normal-body mean and scale for the residual `η` |
| `gpd_right.{threshold, shape, scale, p_exceed}` | Right-tail exceedance model |
| `gpd_left.{threshold, shape, scale, p_exceed}` | Left-tail exceedance model |

Sampler implementation: `_sample_fan_chart` at `pipeline.py:270-320`. The fan bands surface on the `forecast` sensor as `P5_eur_mwh` … `P95_eur_mwh` per row.

### Softplus floor and hourly DtACI calibrators

Before the fan-chart is sampled, the L1+L2+L3 mean is floored at −5 EUR/MWh via a softplus (`price_floor.py`). An hourly DtACI bias corrector (`hourly_calibration.HourlyBiasCorrector`, half-life 14 days, 168-hour warm-up) subtracts a slowly-moving systematic-bias estimate; a parallel `HourlyFanChartCalibrator` adapts the per-hour fan widths to track the 0.5 and 0.9 marginal coverage targets. A `RefitMonitor` flags persistent drift over a 14-day window (`spot_price_predictor_pipeline/refit_monitor.json`).

### Inputs evaluated and not currently active

Earlier feasibility work explored using a nuclear-deficit signal `nuclear_deficit ∈ [0, 1]` (Fingrid dataset #188) and an SE3 cross-border transit-capacity / export-spread proxy as features. The current non-seasonal model does not consume either:

- Fingrid nuclear data is still fetched (`coordinator.py:871`) and surfaces in the duration model and diagnostics, but is not in `RIDGE_FEATURES`.
- SE3 / SE1 / EE prices are still fetched (`coordinator.py:852`) and feed the duration model; the pipeline call site (`coordinator.py:1210-1215`) passes only weather and the lag168 residual.

Re-introducing either signal would require a fresh ablation against the current coefficient vector and is gated on a separate experimental branch — see "Open question — per-input accuracy contribution" below.

### Open question — per-input accuracy contribution

The only per-input ablation currently published in this repository is [studies/results/v2511_physics_features.md](studies/results/v2511_physics_features.md), which evaluated the sigmoid-wind / solar / raw-wind variants against a base configuration. It is useful as background reading for the physics-feature design but does not pin down the marginal contribution of each L2 feature in the shipping coefficient set. A leave-one-out study over the current `ridge_coef` vector is the prerequisite for any change to the input list.

---

## Model Architecture

### Hourly model: Four-layer pipeline

The hourly point forecast is the sum of three additive contributions, softplus-floored and bias-corrected:

```
pipeline_mean(h) = L1_seasonal_fi(h)
            + L2_ridge(h)          # six features above
            + L3_ar(h)              # φ^h · η(t₀−1)
            - hourly_bias_ema(h)    # DtACI bias corrector
            (softplus floor at −5 EUR/MWh)
```

The full sampler then produces the fan-chart bands `P5_eur_mwh` … `P95_eur_mwh` around the point forecast (`spot_eur_mwh`). Public entry point: `Pipeline.compute_forecast` at `pipeline.py:324`.

Artifacts loaded at construction time:

- `data/seasonal_components_default.json` — L1 components for FI price and weather inputs.
- `data/spike_model_default.json` — L2 Ridge coefficients (`ridge_coef`), L3 AR(1) (`ar1_phi`), L4 Normal-body + GPD-tail parameters (`stats`, `gpd_left`, `gpd_right`).
- `data/solar_submodel_default.json` — clear-sky × cloudiness solar production sub-model used by the PV-aware path.

Persistent calibrator state under `<config>/.storage/spot_price_predictor_pipeline/`:

- `hourly_bias.json` — DtACI bias corrector state.
- `hourly_fan_chart.json` — fan-chart DtACI bundle per coverage target.
- `refit_monitor.json` — drift-trigger state for the refit monitor.

### Duration model: Segment-hierarchical Ridge + PAVA (cheap/peak split)

Predicts two complementary duration curves per day:

- **`dk_cheap_eur_mwh[i]`** = mean spot price of the **(i+1) cheapest hours**, i = 0..23 (monotone non-decreasing). CVaR at α=(i+1)/24 in the lower tail. Best achievable cost for a deferrable load that can choose its k cheapest slots.
- **`dk_peak_eur_mwh[i]`** = mean spot price of the **(i+1) priciest hours**, i = 0..23 (monotone non-increasing). CVaR at α=(i+1)/24 in the upper tail. Worst-case cost if the load is forced into peak hours (storage-depletion / risk-aware planning).

The full-day mean is recovered at i = 23: `dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == daily_average_spot`. The consumer-side `dk_cheap_eur_kwh` / `dk_peak_eur_kwh` arrays have the same shape; each hour's spot is converted to consumer EUR/kWh with the corresponding day/night tariff before sorting.

**PAVA** (Pool Adjacent Violators Algorithm) is an isotonic regression method that enforces monotonicity. The cheap end requires non-decreasing PAVA; the peak end requires non-increasing PAVA (mirrored). Both are applied independently per direction after the per-segment Ridge predictions.

**Architecture (dual cheap/peak training):**
- 4 day segments aligned with day/night tariff boundaries: night (22-07, 9 levels), morning (07-12, 5 levels), midday (12-18, 6 levels), evening (18-22, 4 levels). Total = 24 hourly slots.
- Per `(segment, direction, k)`: independent Ridge model. Each segment carries `cheap_models` (k = 1..n_levels) and `peak_models` (k = 1..n_levels). Total bundled Ridge fits = 2 × (9 + 5 + 6 + 4) = **48 small models**.
- Per-segment **12 features** (segment-level aggregates over the segment's hours):
  `wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`, `net_load_mean`, `net_load_squared_mean`. The last two are zero-padded when Fingrid net-load forecasts are unavailable, matching the training-side fallback.
- Log-linear target: `log(D(k) + 100)`
- Forgetting factor λ = 0.960 (half-life 17 days, optimized via sweep)
- PAVA isotonic post-processing per direction:
  - Cheap end: enforces `dk_cheap[0] ≤ dk_cheap[1] ≤ … ≤ dk_cheap[11]`
  - Peak end:  enforces `dk_peak[0]  ≥ dk_peak[1]  ≥ … ≥ dk_peak[11]`
- Segment-to-day reconstruction: each segment yields its own sorted-price vector; segments merge into 24 hourly forecasts; `compute_dk_cheap_peak()` produces the two 12-element arrays exposed at the sensor.

**Performance (Spearman rank correlation, last 365 days):**

| Duration level | Use case | ρ |
|:-:|:-:|:-:|
| D(1) | Cheapest 1h | 0.898 |
| D(4) | Cheapest 4h | 0.930 |
| D(8) | Cheapest 8h | 0.937 |
| D(24) | Daily average | 0.940 |

**Output artifacts:** the four-layer hourly model parameters live in `data/spike_model_default.json` and `data/seasonal_components_default.json`; the duration model carries the AR(2) parameters for `ar_se3` and `ar_ee` and the dual cheap/peak coefficients (48 segment-direction-k Ridge fits).

---

## Configuration

All tunable parameters are centralized in `config/regions/finland.yaml`. The config covers:

| Section | Parameters |
|---------|-----------|
| `region` | Name, timezone, latitude, currency, bidding zone |
| `price_source` | Sahkotin API URL, unit conversion |
| `weather_source` | Open-Meteo URLs, 7 location definitions with capacity weights |
| `neighbor_price_sources` | Elpriset (SE1, SE3), Elering (EE) APIs |
| `grid_sources` | Fingrid nuclear dataset |
| `demand` | HDD threshold, peak hours, sauna hours, wind rated speed |
| `holidays` | Fixed, Easter-based, and special rule holidays |
| `consumer_pricing` | VAT, energy tax, seller margin, operator tariffs |
| `features` | Wind thresholds, AR normalization, AR stability bounds |
| `training` | Years, test split, half-life, Ridge alpha, power stretch bounds |
| `duration_model` | Lambda, segments, features, log offset, exp cap |

---

## Consumer Price Calculation

**Formula:** `(max(0, spot_EUR_MWh) / 1000 + seller_margin + transfer_rate + energy_tax) × VAT` [EUR/kWh]

Configurable per operator in `finland.yaml`. Default: Elenia (day 3.61, night 2.20 c/kWh), VAT 25.5%, energy tax 2.325 c/kWh, seller margin 0.00 c/kWh (set from your electricity contract).

### Forecast sensors (always created)

| Sensor | State | Unit | Description |
|--------|-------|------|-------------|
| Price Forecast | Current consumer price | EUR/kWh | 170h hourly forecast with spot, consumer, weather per hour |
| Duration Forecast | Today's `dk_cheap_eur_kwh[3]` (cheapest 4h) | EUR/kWh | 7-day × 24-level cheap/peak duration curves in both EUR/MWh (spot) and EUR/kWh (consumer) |

#### Price Forecast attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | array[170] | Per-hour record: `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp, P5_eur_mwh, P25_eur_mwh, P50_eur_mwh, P75_eur_mwh, P95_eur_mwh}`. `spot_eur_mwh` is the pipeline point forecast (EUR/MWh) and the `P*_eur_mwh` keys are the fan-chart percentiles from the L4 GPD POT layer. |
| `current_spot_eur_mwh` | float | Current hour spot price (EUR/MWh) |
| `week_min_eur_kwh` | float | Minimum consumer price in forecast window |
| `week_avg_eur_kwh` | float | Average consumer price in forecast window |
| `week_max_eur_kwh` | float | Maximum consumer price in forecast window |
| `operator` | string | Configured distribution operator |
| `last_update` | datetime | Last successful data refresh |
| `data_sources_active` | string | Currently active data sources |
| `stale` | bool | True if data is older than threshold |
| `data_age_minutes` | int | Minutes since last successful fetch |

#### Duration Forecast attributes — D(k) cheap/peak

The `daily_forecast` attribute provides up to 7 days. Each day exposes the canonical cheap/peak split as four 24-entry arrays (spot in EUR/MWh and consumer in EUR/kWh, cheap and peak).

| Attribute | Shape | Unit | Description |
|-----------|-------|------|-------------|
| `daily_forecast` | array[≤7] | — | One entry per day |
| `daily_forecast[i].date` | string | — | ISO date (YYYY-MM-DD) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` for future days, `actual` for past days reconciled from Sahkotin |
| `daily_forecast[i].dk_cheap_eur_mwh` | float[24] | EUR/MWh | Mean spot price of the (i+1) cheapest hours of the day, i = 0..23 (monotone non-decreasing) |
| `daily_forecast[i].dk_peak_eur_mwh`  | float[24] | EUR/MWh | Mean spot price of the (i+1) priciest hours of the day, i = 0..23 (monotone non-increasing) |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[24] | EUR/kWh | Same cheapest-end curve in consumer price (per-hour tariff applied) |
| `daily_forecast[i].dk_peak_eur_kwh`  | float[24] | EUR/kWh | Same priciest-end curve in consumer price |
| **Convenience scalars (today only):** | | | |
| `today_cheap_4h_eur_kwh`, `today_cheap_8h_eur_kwh` | float | EUR/kWh | Today's cheapest 4h/8h indicators |
| `today_peak_4h_eur_kwh`,  `today_peak_1h_eur_kwh` | float | EUR/kWh | Today's worst-case 4h/1h indicators |
| `forecast_days` | int | — | Number of days emitted (up to 7) |

**Access patterns:**
- Cheapest k hours of day d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` for k in 1..24 (use this for deferrable-load scheduling)
- Priciest k hours of day d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` for k in 1..24 (use this for worst-case / storage planning)
- Cross-check identity: `cheap[11] + peak[11] = 2 × daily_avg` (always holds to numerical noise)
- Consumer prices include per-segment tariff conversion: night hours use night transfer rate, day hours use day rate, then merged and re-sorted

### Actual price sensors (optional, Nordpool)

| Sensor | Unit | Description |
|--------|------|-------------|
| Spot Electricity Price | EUR/kWh | Actual consumer price from Nordpool with continuous timeline |
| Spot Electricity Selling Price | EUR/kWh | Spot minus PV selling commission (for solar panel owners) |

### Design principle

This integration provides **forecasts only**. The cheap/peak duration curves are the primary API for downstream systems:
- Thermal optimization / load scheduling reads `dk_cheap[k-1]` to answer "what does it cost per kWh to run for k hours today, scheduled into the cheapest slots?"
- Risk-aware planning (storage depletion, capacity reservation) reads `dk_peak[k-1]` to answer "what's the worst case if we have to run during k peak hours?"

This clean separation allows either component to be replaced independently.

The **Price Forecast** sensor provides a unified 170-hour forecast array for visualization and hourly price display. The state is the current hour's consumer price in EUR/kWh.

The **Duration Forecast** sensor exposes the cheap/peak curves for optimization decisions. The state is today's `dk_cheap_eur_kwh[3]` — the average cost of running during the cheapest 4 hours — serving as a quick-glance cost indicator. See [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md) for the migration guide for downstream consumers.

---

## PV-aware pricing

When the user configures a non-zero `pv_capacity_kwp` (or supplies an external PV forecast entity) the coordinator augments every forecast hour with a **marginal effective price** representing the cost of running 1 additional kWh of flexible load given the household's PV production and the typical-total baseload.

### Notation

For each hour `h`:

| Symbol | Meaning |
|--------|---------|
| `b_h` | Consumer buy price (EUR/kWh) = `(spot/1000 + margin + transfer + tax) × VAT`. Always > 0 in practice. |
| `s_h` | Sell price (EUR/kWh) = `spot/1000 − pv_sell_commission − pv_export_grid_fee`. NOT clipped at zero — can be negative during deep oversupply. |
| `c_h` | Configured baseload (kWh) = `annual_consumption_kwh / 8760 × monthly_factor[month_of_h]`. Resolved deterministically from configuration. |
| `p_h` | Hourly PV production (kWh), from internal estimator or external entity. Bounded in `[0, capacity_kwp · efficiency]`. |

### Marginal effective price (the metric that feeds D(k))

```
pv_avail_h = max(0, p_h − c_h)            # PV surplus available to extra load
from_pv    = min(1, pv_avail_h)           # fraction of new 1 kWh covered by PV
from_grid  = 1 − from_pv
m_h        = from_pv · s_h + from_grid · b_h
```

Properties:

- **Bounded analytically**: `m_h ∈ [s_h, b_h]` for all (b, s, p, c). No baseload-divisor pathology.
- **Captures self-consumption gain**: when PV surplus ≥ 1 kWh, `m_h = s_h` (you forgo export revenue, not retail spend).
- **Captures partial cover**: linear interpolation between sell and buy prices.
- **Captures negative-spot liability**: `s_h < 0` propagates to `m_h < 0` (rare but real).
- **Captures finite capacity**: `p_h ≤ capacity_kwp · efficiency` is a hard ceiling enforced in `pv_estimate.py`.

### PV-aware D(k) cheap/peak

Computed directly from the 24 hourly `effective_eur_kwh` values per local day, using the same `compute_dk_cheap_peak` utility as the spot-only path. Sorted ascending → `dk_cheap_pv[12]` (monotone non-decreasing); sorted descending → `dk_peak_pv[12]` (monotone non-increasing).

**Validation on 4 years of real data** (1,460 days, 5 kWp configuration): zero monotonicity violations, mean D(1) = 6.90 c/kWh (p01 = −0.7 c/kWh, p99 = 30.8 c/kWh) — bounded, realistic, optimization-ready.

### Stability invariant — open-loop wrt the optimizer

The forecast must remain a deterministic function of `(spot, weather, PV config, baseload config)` so the downstream optimizer's flexible-load decisions cannot feed back into next cycle's price forecast. Concretely:

- **`_resolve_baseload(ts)` MUST NOT call `hass.states.get` or any HA entity-read API.** Enforced by a grep test in `tests/test_coordinator_pv.py`.
- **The configured `annual_consumption_kwh` should represent the user's typical TOTAL annual consumption** — bill-derived total demand including all loads (heat pump, EV, sauna, water heater, etc.). Static configuration cannot create optimizer feedback because it doesn't depend on observed consumption; the actual stability requirement is only about what the predictor reads from HA.
- **`_read_external_pv_forecast()` MAY read an HA entity** because the PV forecast is weather-driven and independent of optimizer decisions — no feedback loop is created.

#### Why "typical total" not "non-flexible only" — worked example

Sunny noon, 4 kWh PV, heat-pump household with 16 000 kWh/yr typical demand:

| Scenario | baseload | pv_avail | m_h | Behaviour |
|---|---|---|---|---|
| **A: non-flex only (~0.5 kWh/h)** | 0.5 | 3.5 kWh | ≈ 4 c/kWh | Over-optimistic. Forecast claims all PV is free for extra load. EMHASS schedules heat pump there + further loads on top → second load actually pulls 16 c/kWh from grid. **Systematic optimism bias.** |
| **B: typical total (~1.83 kWh/h × seasonal)** | ~1.83 | 2.17 kWh | ≈ 4 c/kWh | Self-consistent. Forecast assumes typical demand (heat pump etc.) is happening; EMHASS plans around that; reality matches assumption; equilibrium. |

With PV at only 2 kWh, Case B correctly returns m_h ≈ 14 c/kWh (PV mostly absorbed by typical demand, only 0.17 kWh headroom). Case A would have returned ~10 c/kWh — still optimistic. Both cases satisfy the stability invariant because both are static config; Case B is **more accurate** because the PV/grid ratio that drives the marginal cost is genuinely a function of total demand, not non-flex demand.

#### Baseload schema

The PV-aware path uses two configuration fields to derive per-hour baseload:

- **`annual_consumption_kwh`** (default 12 000) — the user's typical TOTAL annual household demand from the electricity bill. Single user-friendly number. Internally:

  ```
  baseload(h) = annual_consumption_kwh / 8760 × monthly_factor[month_of_h]
              ≡ annual_consumption_kwh / 365 / 24 × monthly_factor[month_of_h]
  ```

  where `monthly_factor` is the 12-element Finnish residential non-electric-heating seasonal profile defined in `const.py` (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS`). Sum of factors = 12.00 exactly (normalization invariant); range ≈ ±19 % around the mean (Finnish 60°N latitude pattern: lighting-driven winter peak Dec/Jan, vacation/long-day trough Jul). Source: literature-derived from VTT Publications 289, Adato Energia DSO standard load profiles ("tyyppikäyrät"), Statistics Finland "Energy consumption in households" survey 2024. **TODO**: replace with verbatim values from Fingrid Open Data dataset #360 (BE03 typing curve).

- **`consumption_entity`** (optional) — any HA consumption sensor; the integration auto-detects type and smooths internally:

  | Detected type | Detection (HA attrs) | Smoothing strategy |
  |---|---|---|
  | Cumulative-kWh counter | `unit = kWh`, `state_class = total_increasing` | 14-day delta divided by 14 → daily kWh |
  | Daily/monthly `utility_meter` | `state_class = total` with cycle attribute | History-window average of daily totals |
  | Instantaneous power | `unit = W` or `kW`, `device_class = power` | `statistics_during_period(28 d, mean)` × 24 |
  | Unknown | (fallback) | Silent fallback to `annual_consumption_kwh` config; log warning |

  Smoothed value cached in `.storage/spot_price_predictor_consumption_cache.json`, recomputed at most once per day (not every coordinator cycle). 5 % hysteresis dead-band on the cached value prevents minor sensor noise from re-triggering coordinator updates.

**Stability re-check**:

- **Default mode** (`consumption_entity = ""`): `baseload(h)` is a deterministic function of `(annual_consumption_kwh, h)` only — no HA entity reads, fully open-loop.
- **HA-sensor mode**: 14-day rolling average dampens a single-day perturbation to `1/14 ≈ 7 %`. Combined with the 5 % hysteresis dead-band, EMHASS rescheduling a 5 kWh load between days produces `5/14 ≈ 0.36 kWh` rolling change, only ~3 % of a 12 kWh/day baseline — within the dead-band, so the cached baseload value doesn't move and EMHASS sees a stable forecast.

### External PV forecast — supported attribute conventions

`_read_external_pv_forecast()` is source-agnostic. Auto-detected attribute conventions, in priority order:

| Convention | Attribute | Shape | Unit | Conversion |
|---|---|---|---|---|
| 1. Generic forecast list | `forecast` | list[dict] | kWh | direct (keys: `pv_kwh`, `kwh`, `energy`, `value`) |
| 2. Forecast.Solar Wh dict | `wh_hours` | dict {ISO ts → number} | Wh | `/ 1000` |
| 3. Forecast.Solar W dict | `watts` | dict {ISO ts → number} | W | `/ 1000` (1-hour granularity) |
| 4. EMHASS template list | `irradiance` | list[number] | W or kWh | magnitude > 50 → assume W and `/ 1000`; else kWh |

All paths return up to 168 hourly kWh values clamped to `[0, capacity_kwp · efficiency]`. Silent fallback to internal estimator if the entity is missing or none of the conventions match.

### Configuration

PV system parameters live in `consumer_pricing` adjacent fields and the optional "PV system" step of the HA config flow:

| Field | Default | Description |
|-------|---------|-------------|
| `pv_capacity_kwp` | 0 (disabled) | Installed PV peak power |
| `pv_tilt_deg` | 45 | Panel tilt; matches Open-Meteo's fetch tilt so default = no correction |
| `pv_azimuth_deg` | 180 (south) | 0=N, 90=E, 180=S, 270=W |
| `pv_system_efficiency` | 0.85 | Lumped DC/AC + soiling + losses |
| `pv_external_entity` | "" | Optional HA sensor that overrides internal estimator |
| `pv_export_grid_fee` | 0 | Extra EUR/kWh fee on exported energy (above seller commission) |
| `annual_consumption_kwh` | 12 000 | Typical TOTAL annual household demand from the bill, including PV self-consumption AND optimizer-controlled loads. Multiplied by the built-in Finnish residential monthly seasonal profile to get per-hour baseload. |
| `consumption_entity` (optional) | "" | Any HA consumption sensor (cumulative-kWh counter, daily/monthly utility_meter, instantaneous power). The integration auto-detects type and smooths internally over 14 days with 5 % hysteresis. Recommended placeholder: `sensor.energy_yesterday`. |

Setting `pv_capacity_kwp = 0` (the default) and leaving `pv_external_entity` empty disables all PV-aware outputs cleanly — the integration produces byte-identical no-PV outputs.

### Out of scope

- **Battery storage** — adds a temporal state variable; deferred.
- **Capacitated water-filling D(k)** — for very large flexible loads relative to PV capacity, the marginal-1-kWh model under-counts by ~`(load − pv_avail) × s_h` per hour. Acceptable for typical residential loads (heat pump, EV).
- **Per-tilt second Open-Meteo fetch** — current implementation reuses the integration's existing 45°-S irradiance fetch with scalar correction.
- **Spot-model retraining with PV** — the non-seasonal price model is PV-agnostic; PV is a post-prediction transform.

---

## Home Assistant Integration

### Holiday and workday detection

The model uses workday/holiday status to select demand patterns. Two options are supported via the `holidays.ha_workday_integration` setting.

#### Option A: HA Workday integration (recommended)

Uses Home Assistant's built-in [Workday integration](https://www.home-assistant.io/integrations/workday/) which automatically resolves public holidays.

**Setup:**
1. Go to **Settings** → **Devices & Services** → **Add Integration** → search **Workday**
2. Set **Country** to `FI` (Finland)
3. The integration creates `binary_sensor.workday_sensor`

#### Option B: Built-in holiday calculator

Uses the holiday rules defined in `finland.yaml` with a built-in Easter algorithm and special date rules. Used by the training pipeline and when the Workday integration is not available.

---

## Regional Localization

The system is driven by a single region config file (`config/regions/finland.yaml`). To support a new region:

1. **Identify weather measurement locations** — find 5-8 geographical locations representing wind, solar, and consumption centers. Weight by installed capacity and population.
2. **Create a new YAML file** (e.g., `sweden.yaml`)
3. **Define local price API** — free API providing day-ahead spot prices
4. **Configure holidays** — fixed dates, Easter-relative dates, and special rules
5. **Add neighboring price sources** for the duration model
6. **Set consumer pricing** — VAT rate, energy tax, and distribution operator tariffs
7. **Refit** via the `spot_price_predictor.retrain_models` service with regional data cached locally

See the AI prompt template in the expanded section below for identifying weather locations.

<details>
<summary><b>AI prompt template for weather locations (click to expand)</b></summary>

```
I am building an electricity spot price prediction model for [COUNTRY] that uses
weather data (wind speed, solar irradiance, temperature) from multiple locations
weighted by installed generation capacity and population.

Please identify 5-8 representative weather measurement locations for [COUNTRY] with:
- Name, latitude, longitude
- Wind weight (proportional to installed wind capacity)
- Solar weight (proportional to installed solar PV capacity)
- Temperature weight (proportional to population density)

Weights should sum to approximately 0.8-1.0 each. Include the representative
latitude for daylight calculation and HDD base temperature for the country's
building stock.
```

</details>

---

## Accuracy and Retraining

### Current end-to-end performance

Numbers below characterise the configuration that ships today on a held-out FI test window.

**Hourly model (spot point forecast):**

| Metric | Value |
|---|:---:|
| MAE (h = 24 … 168) | ≈ 10 EUR/MWh |
| R² | ≈ 0.93 |
| Threshold hit rate (high-price hours) | 98 % |

**Duration model (R², per D(k) index):**

- `dk_cheap[i]` and `dk_peak[i]` both achieve R² ≥ 0.95 at every i = 0 … 23 ([studies/results/V2_5_17_DK_FULL_RANGE.md](studies/results/V2_5_17_DK_FULL_RANGE.md)).
- cheap_4 MAE 4.4 EUR/MWh, peak_4 MAE 6.9 EUR/MWh on the held-out window.

**PV-aware D(k) validation** (5 kWp reference, 4-year backtest on 1,460 complete days): zero PAVA-monotonicity violations, PV-aware D(1) mean 6.90 c/kWh (std 6.0), bounded analytically in `[s_h, b_h]` per hour. Estimated annual savings vs grid-only D(4) ≈ 600 EUR/yr.

### Recommended retraining frequency

Quarterly is a reasonable default; the `RefitMonitor` calibrator also flags persistent drift over a 14-day window so users can refit on demand when the production environment changes (new wind capacity, regime shift, tariff change).

### How to retrain

The `spot_price_predictor.retrain_models` Home Assistant service refits the bundled artifacts in place. From Developer Tools → Services or an automation:

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike", "solar"]   # omit to refit all three
  # fingrid_api_key: "..."                  # only needed for the solar layer
```

The service atomically rewrites the three JSON artifacts under `custom_components/spot_price_predictor/data/` and the coordinators auto-reload them on the next update cycle. On completion it fires the `spot_price_predictor_models_retrained` event so automations can react.

### Open question — per-input accuracy contribution

A leave-one-out / SHAP-style ablation over the current Ridge coefficient set would let us pin down the marginal contribution of each L2 feature and re-evaluate inputs that were validated in earlier feasibility work (nuclear deficit, SE3 transit-capacity proxy). That study is not in scope of this documentation pass — it belongs on a separate experimental branch and would only be merged based on evidence against `main`.

---

## Project Structure

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # This document
├── TEKNINEN_TOTEUTUS.md         # Finnish translation
├── INSTALLATION.md              # Step-by-step setup guide
├── config/regions/
│   └── finland.yaml             # Central configuration (all parameters)
├── custom_components/
│   └── spot_price_predictor/    # HA HACS integration
│       ├── __init__.py          # Entry point + service registration (incl. retrain_models)
│       ├── coordinator.py       # Data fetch + Pipeline orchestration
│       ├── pipeline.py      # L1+L2+L3+L4 pipeline + softplus floor + DtACI
│       ├── seasonal_decomposition.py  # L1 component fitter / lookup
│       ├── hourly_calibration.py      # DtACI bias / fan-chart / refit-monitor
│       ├── price_floor.py             # Softplus floor
│       ├── solar_clear_sky.py         # Clear-sky × cloudiness solar sub-model
│       ├── retrain.py                 # Retraining orchestrator (HA service backend)
│       ├── sensor.py                  # HA sensor entities
│       ├── api_client.py              # Async API clients
│       ├── const.py                   # Constants and defaults
│       └── data/
│           ├── seasonal_components_default.json
│           ├── spike_model_default.json
│           ├── solar_submodel_default.json
│           └── finland.yaml
├── ha_dashboard.yaml            # Home Assistant Lovelace dashboard (ApexCharts + Mushroom)
├── studies/                     # Refit scripts (build_seasonal_components, v2513_layer4_spike_model, v253_solar_submodel) and historical analyses
└── tests/                       # Unit + integration tests
```
