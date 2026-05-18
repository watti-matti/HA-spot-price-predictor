# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.8.0-blue.svg)](https://github.com/watti-matti/HA-spot-price-predictor/releases/tag/v2.8.0)

**Forecast consumer electricity costs and D(k) duration curves up to 7 days ahead** using machine learning with physics-based weather features, cross-border trade analysis, and nuclear outage awareness.

[Suomenkieliset ohjeet (Finnish)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour consumer price forecast** — predict your electricity cost (EUR/kWh) for the next 7 days, including spot price, transfer tariff, seller margin, energy tax, and VAT
- **D(k) cheap/peak duration curves** — 7-day forecast of two complementary 24-level curves (0-indexed): `dk_cheap_eur_kwh[i]` = mean consumer price of the (i+1) cheapest hours (best achievable cost for deferrable load) and `dk_peak_eur_kwh[i]` = mean consumer price of the (i+1) priciest hours (worst-case planning cost). Parallel `dk_cheap_eur_mwh` / `dk_peak_eur_mwh` arrays carry the same shape in spot EUR/MWh. Equivalent to CVaR at α=(i+1)/24 in both tails of the daily price distribution.
- **Optional PV-aware effective price** — configure your rooftop solar (kWp, tilt, azimuth, efficiency) and the integration produces an `effective_eur_kwh` per hour and PV-aware D(k) cheap/peak curves. Captures both self-consumption savings and export revenue, including the rare "negative spot + high PV" liability case. Uses Open-Meteo irradiance internally; can also read an external PV forecast entity (e.g. Forecast.Solar, custom templates) for users with multi-array setups or shading.
- **Works out-of-the-box** — pre-trained model included, no setup beyond choosing your distribution operator
- **Physics-grounded features** — sigmoid wind power curve scaled by air density, temperature-derated solar effective irradiance, and deseasonalized own-lag residual; all coefficients respect their expected sign
- **Probabilistic fan chart** — every forecast hour carries P5/P25/P50/P75/P95 bands sampled from a Normal-body + GPD-tail mixture (heavy-tail spike model)
- **Online calibration (DtACI)** — optional adaptive conformal prediction intervals with per-D(i) online bias correction; warms up in ~5 days
- **Clean API for optimization** — forecast-only sensor interface with structured D(k) matrix, ready for downstream thermal optimization
- **Retrainable in place** — the `spot_price_predictor.retrain_models` Home Assistant service refits the L1 seasonal, L2 Ridge / L3 AR(1) / L4 spike, and solar sub-model artifacts and reloads them without a restart
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The forecast is produced by a four-layer pipeline. Each layer adds a specific kind of structure to the spot-price estimate; the layers are then composed additively per forecast hour.

**L1 — Seasonal decomposition.** A Moazeni–Powell additive decomposition (hourly + daily + weekly components) lifts out the deterministic seasonal pattern from the FI spot history. The same decomposition is applied to temperature so the downstream Ridge model sees deseasonalized signals only. Components live in `data/seasonal_components_default.json`.

**L2 — Non-seasonal Ridge regression.** A six-feature linear model captures the physics-driven and calendar-driven residual:

| # | Feature | What it represents |
|---|---|---|
| 1 | `intercept` | constant offset |
| 2 | `Y_fi_lag168` | deseasonalized FI residual 7 days ago — own-lag memory of the local market regime |
| 3 | `is_workday` | weekday flag — captures industrial demand pattern |
| 4 | `Y_sigmoid_wind_rho` | sigmoid wind-power curve `σ((wind − 7.5) / 1.5)` scaled by relative air density `ρ(T) / 1.225` — physics-realistic supply driver |
| 5 | `Y_solar_effective` | global horizontal irradiance derated for cell temperature `1 − 0.004·max(0, T_cell − 25)` with `T_cell = T + 0.03·GHI` |
| 6 | `Y_temp` | deseasonalized temperature — residual heating-load signal |

Coefficients live in `data/spike_model_default.json` (key `ridge_coef`).

**L3 — AR(1) momentum.** At forecast horizon `h`, the most-recent observed deseasonalized FI residual `η(t₀−1)` is decayed as `φ^h · η(t₀−1)` with `φ ≈ 0.904`. This carries short-term market state without re-introducing the seasonal pattern that L1 already handles.

**Softplus floor and hourly EMA bias.** The L1+L2+L3 mean is floored at −5 EUR/MWh via a softplus and corrected by an exponentially-weighted hourly bias estimate (half-life 14 days). The corrected point forecast is the `spot_eur_mwh` value on each forecast row.

**L4 — GPD POT spike model.** A heavy-tail mixture (Normal body + Generalized Pareto right and left tails) is sampled 500 times around the point forecast to produce the P5 / P25 / P50 / P75 / P95 fan bands that are exposed as `P5_eur_mwh` … `P95_eur_mwh` per forecast hour. Tail parameters live in `spike_model_default.json` under `gpd_left` / `gpd_right`.

The coordinator converts each predicted spot price to consumer price (EUR/kWh) using your configured tariffs (transfer, margin, tax, VAT).

**Duration model (D(k) cheap/peak)** — a segment-hierarchical Ridge model predicts the daily duration curve for each of the next 7 days, exposed as two 12-element arrays:

- `dk_cheap_eur_kwh[i]` = mean consumer price of the **(i+1) cheapest** hours, i = 0..23 (monotone non-decreasing). Use this for deferrable-load scheduling: "run k hours into the cheapest slots, expected cost = `dk_cheap_eur_kwh[k-1]` × kWh."
- `dk_peak_eur_kwh[i]` = mean consumer price of the **(i+1) priciest** hours, i = 0..23 (monotone non-increasing). Use this for storage-depletion / worst-case planning: "if I'm forced to run during k peak hours, expected cost = `dk_peak_eur_kwh[k-1]` × kWh."

Both arrays are CVaR-equivalent — `dk_cheap_eur_kwh[i]` = CVaR at α=(i+1)/24 in the lower tail, `dk_peak_eur_kwh[i]` = CVaR at α=(i+1)/24 in the upper tail. The full-day mean is recovered at i = 23: `dk_cheap_eur_kwh[23] == dk_peak_eur_kwh[23] == daily_average`.

The duration model splits each day into 4 tariff-aligned segments (night 22–07, morning 07–12, midday 12–18, evening 18–22), predicts D(k) independently per segment using Ridge regression, enforces monotonicity via PAVA (Pool Adjacent Violators Algorithm), then merges all segments into the cheap/peak curves.

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

### Forecast Sensors (always created)

| Sensor | State | Unit | Description |
|--------|-------|------|-------------|
| `sensor.price_forecast` | Current consumer price | EUR/kWh | 170h hourly forecast with spot, consumer, weather per hour |
| `sensor.duration_forecast` | Today's `dk_cheap_eur_kwh[3]` (cheapest 4h) | EUR/kWh | 7-day × 24-level cheap/peak duration curves in both EUR/MWh (spot) and EUR/kWh (consumer) |

#### Price Forecast attributes

The **Price Forecast** sensor provides the complete hourly forecast as its `forecast` attribute. Each entry contains `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp}`. The state is the current hour's consumer price in EUR/kWh.

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | array[170] | Hourly entries: `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp, P5_eur_mwh, P25_eur_mwh, P50_eur_mwh, P75_eur_mwh, P95_eur_mwh}`. `spot_eur_mwh` is the pipeline mean (point forecast); the `P*_eur_mwh` keys are the fan-chart percentiles. |
| `current_spot_eur_mwh` | float | Current hour spot price (EUR/MWh) |
| `week_min_eur_kwh` | float | Minimum consumer price in forecast window |
| `week_avg_eur_kwh` | float | Average consumer price in forecast window |
| `week_max_eur_kwh` | float | Maximum consumer price in forecast window |
| `operator` | string | Configured distribution operator |

#### Duration Forecast attributes — D(k) cheap/peak curves

The **Duration Forecast** sensor provides daily duration curves split into a cheap end and a peak end. The `daily_forecast` attribute contains 7 days, each carrying four 24-level arrays (0-indexed). The state is today's `dk_cheap_eur_kwh[3]` — the average consumer price of the cheapest 4 hours, the most-used scheduling indicator.

| Attribute | Shape | Unit | Description |
|-----------|-------|------|-------------|
| `daily_forecast` | array[7] | — | One entry per day |
| `daily_forecast[i].date` | string | — | ISO date (YYYY-MM-DD) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` or `actual` (past days from Sahkotin) |
| `daily_forecast[i].dk_cheap_eur_mwh` | float[24] | EUR/MWh | Mean spot price of the (i+1) cheapest hours of the day, i=0..23 (monotone non-decreasing) |
| `daily_forecast[i].dk_peak_eur_mwh`  | float[24] | EUR/MWh | Mean spot price of the (i+1) priciest hours of the day, i=0..23 (monotone non-increasing) |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[24] | EUR/kWh | Same cheapest-end curve in consumer price (per-hour tariff applied) |
| `daily_forecast[i].dk_peak_eur_kwh`  | float[24] | EUR/kWh | Same priciest-end curve in consumer price |
| **Convenience scalars (today only):** | | | |
| `today_cheap_4h_eur_kwh` | float | EUR/kWh | Today's `dk_cheap_eur_kwh[3]` (cheapest 4h) |
| `today_cheap_8h_eur_kwh` | float | EUR/kWh | Today's `dk_cheap_eur_kwh[7]` (cheapest 8h) |
| `today_peak_4h_eur_kwh` | float | EUR/kWh | Today's `dk_peak_eur_kwh[3]` (priciest 4h) |
| `today_peak_1h_eur_kwh` | float | EUR/kWh | Today's `dk_peak_eur_kwh[0]` (single priciest hour) |

**Access patterns:**
- Cheapest k hours of day d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` for k = 1..24
- Priciest k hours of day d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` for k = 1..24
- Cross-check identity: `dk_cheap_eur_mwh[23] == dk_peak_eur_mwh[23] == daily_average_spot` (the 24-hour mean is direction-invariant)
- Consumer prices include configured tariffs (day/night transfer rate, seller margin, energy tax, VAT)

**Optional online calibration (DtACI).** Enable the `enable_dtaci_dk` option to wrap the duration forecast with adaptive conformal prediction intervals (Gibbs & Candes, JMLR 2024) plus per-D(i) online bias correction. When on, each daily forecast entry gains four 12-element band attributes — `dk_cheap_lower/upper_eur_kwh` and `dk_peak_lower/upper_eur_kwh` — that achieve 90% marginal coverage and adapt to regime shifts. A `dtaci_diagnostics` attribute exposes per-(direction, k) coverage / bias EMA / dominant gamma / weight entropy for monitoring; see [docs/yaml_examples/dtaci_diagnostics_card.yaml](docs/yaml_examples/dtaci_diagnostics_card.yaml) for a Lovelace card. Calibrated intervals appear after approximately 5 days of reconciled actuals.

**Design principle:** This integration provides *forecasts only*. The cheap/peak curves are the primary API for downstream systems — thermal optimization, load scheduling, and heat pump control consume `dk_cheap` for cost-minimization and `dk_peak` for risk-aware planning.

#### Optional PV-aware effective price (configure in setup UI)

When a non-zero `pv_capacity_kwp` is set in the optional "PV system" config step, every `forecast[i]` entry gains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `pv_production_kwh` | float | Estimated PV output for the hour (kWh) |
| `baseload_kwh` | float | Configured typical total household consumption for that hour |
| `effective_eur_kwh` | float | **Marginal cost of running 1 additional kWh of flexible load** at this hour given PV. Bounded in `[s_h, b_h]` (sell ≤ effective ≤ buy). |
| `net_household_cost_eur` | float | Informational raw EUR/h flow (export revenue ↔ grid import). Not used for D(k). |
| `is_export_hour` | bool | True when PV exceeds baseload |
| `sell_eur_kwh` | float | Sell price = spot − commission − optional grid fee. Can be negative during deep oversupply. |

The duration sensor gains parallel `dk_cheap_pv_eur_kwh[12]` / `dk_peak_pv_eur_kwh[12]` curves and matching `today_cheap_pv_{1,4,8,12}h_eur_kwh` / `today_peak_pv_{1,4,8,12}h_eur_kwh` scalars derived from the sorted hourly `effective_eur_kwh` per day.

**PV input modes:**
- **Internal estimator (default)** — uses Open-Meteo `global_tilted_irradiance_instant` × your configured `kwp` × tilt/azimuth correction × `efficiency`. Free, 7-day horizon, no rate limit.
- **External entity (override)** — set `pv_external_entity` to any HA sensor whose attributes match one of: `forecast` list-of-dict (kWh), `wh_hours` dict (Wh), `watts` dict (W), or `irradiance` list (auto-detected W/kWh). Forecast.Solar, custom Open-Meteo templates, and similar publishers all just work. Use this if you have multi-array setups or want shading-aware values.

**Configuration:** the PV system step accepts two baseload-related fields:

- **`annual_consumption_kwh`** (default 12 000) — the user's typical TOTAL annual household demand from the electricity bill, including PV self-consumption and optimizer-controlled loads (heat pump, EV, sauna, water heater). The integration multiplies by a built-in Finnish residential monthly seasonal profile (Fingrid Datahub BE03 typing curve, range ±19 % around the mean) to get per-hour baseload. No day/night split — that's the optimizer's domain.
- **`consumption_entity`** (optional, e.g. `sensor.energy_yesterday`) — any HA consumption sensor: cumulative-kWh counter (smart meter, `utility_meter`), HA Energy Dashboard's daily/yearly counters, or instantaneous-power sensor (W / kW). The integration auto-detects the sensor type and applies internal long-window smoothing (14-day rolling average + 5 % hysteresis), so EMHASS's daily scheduling decisions don't propagate back into the forecast. No `filter:` template required.

**Stability invariant.** Baseload is a deterministic function of `(config + long-window EMA, time)`. With `consumption_entity` empty, baseload comes purely from the static `annual_consumption_kwh` × seasonal profile — fully open-loop wrt the optimizer. With `consumption_entity` set, the 14-day smoothing window plus 5 % hysteresis keep the closed-loop gain well below 1, so EMHASS's daily decisions cannot create oscillation. See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md#pv-aware-pricing) for the worked Case A vs Case B example and the cross-system contract.

### Actual Price Sensors (optional, when Nordpool entity is configured)

If you have a Nordpool integration installed (e.g., [custom-components/nordpool](https://github.com/custom-components/nordpool)), you can link it to get actual price sensors for comparison:

| Sensor | Description |
|--------|-------------|
| `sensor.spot_electricity_price` | Actual consumer price from Nordpool with continuous timeline attribute |
| `sensor.spot_electricity_selling_price` | Spot price minus PV selling commission (for solar panel owners) |

**Setup:** Enter your Nordpool sensor entity ID (e.g., `sensor.nordpool_kwh_fi_eur_3_10_0`) in the operator configuration step.

### Dashboard

A complete [Lovelace dashboard](ha_dashboard.yaml) is included (ApexCharts + Mushroom cards) with:
- **48h consumer price bar chart** — color-coded hourly bars with extrema markers
- **7-day hourly price trend** — area chart of consumer prices across the full forecast window
- **D(k) cheap/peak duration curves** — two-group multi-line chart: cool colors show `D_cheap(1)`, `D_cheap(4)`, `D_cheap(8)`, `D_cheap(12)` (best achievable cost); warm dashed lines show `D_peak(12)`, `D_peak(8)`, `D_peak(4)`, `D_peak(1)` (worst-case cost)
- **Current price + D(4) chips** — at-a-glance status with color-coded icons
- **Week statistics** — min/avg/max consumer price summary
- **Data status** — active sources, forecast horizon, staleness indicator

An additional [ApexCharts-only dashboard](docs/yaml_examples/apexcharts_dashboard.yaml) provides actual vs forecast comparison with wind overlay. Both require the [apexcharts-card](https://github.com/RomRider/apexcharts-card) custom card (install via HACS Frontend).

## Data Sources & Features

The non-seasonal price model consumes a deliberately small set of inputs — the six features tabulated in [How It Works](#how-it-works) — and is driven from a single external dependency at inference time: **Open-Meteo** (wind at 120 m, global horizontal irradiance, temperature) for the seven Finnish weather points. The own-lag feature `Y_fi_lag168` is read from the cached spot-price history. No cross-border, nuclear, or net-load data is required for the non-seasonal price forecast itself.

Additional data sources participate in the surrounding system:

| Source | Role in the current system |
|---|---|
| Open-Meteo (forecast + historical-forecast) | Wind, solar irradiance, temperature for the non-seasonal model **and** the duration model |
| Sahkotin (FI Nord Pool) | Recent and historical FI spot prices — feeds the lag168 cache, the actual-D(k) prepend, and DtACI reconciliation |
| elprisetjustnu.se (SE3, SE1) + Elering (EE) | Cross-border spot prices — used by the duration model and historical-context attributes; not consumed by the non-seasonal price model |
| Fingrid dataset #188 (nuclear production) | Used by the duration model when a Fingrid key is configured; not consumed by the non-seasonal price model |
| Fingrid datasets #165 / #246 / #247 (consumption / wind / solar forecasts) | Fetched as auxiliary forecast streams; not consumed by the non-seasonal price model |
| Nord Pool UMM (planned nuclear outages) | Auxiliary signal; not consumed by the non-seasonal price model |

All sources are free; the Fingrid key (free email registration at [data.fingrid.fi](https://data.fingrid.fi)) is only needed if you want the nuclear-aware duration model.

## Model Performance

End-to-end accuracy on a held-out FI test window (the most recent v26 benchmark documented in [studies/results/V2_6_1_BENCHMARK.md](studies/results/V2_6_1_BENCHMARK.md)):

| Metric | Value |
|---|:---:|
| Spot-price MAE (h = 24 … 168) | ≈ 10 EUR/MWh |
| Spot-price R² | ≈ 0.93 |
| Threshold hit rate (high-price hours) | 98 % |
| D(k) R² across k = 1 … 23 (cheap and peak) | ≥ 0.95 at every index |

The fan-chart bands (P5/P25/P50/P75/P95) are calibrated online by an hourly DtACI loop; realised marginal coverage tracks the nominal 0.5 and 0.9 targets within a couple of percentage points once the calibrator warms up (≈ 720 hours of data per `coverage × hour-of-day` cell).

**PV-aware D(k) validation** (5 kWp configuration, 4-year backtest on 1,460 complete days): zero PAVA-monotonicity violations, mean PV-aware D(1) = 6.90 c/kWh (std 6.0), bounded analytically in `[s_h, b_h]` per hour. Estimated annual savings vs grid-only D(4) ≈ 600 EUR/yr.

## Supported Operators (Finland) (check your contract)

| Operator (1.4.2026) | Day rate (07-22) | Night rate (22-07) |
|----------|:---:|:---:|
| Elenia | 3.61 c/kWh | 2.20 c/kWh |
| Caruna Espoo | 2.21 c/kWh | 2.21 c/kWh |
| Caruna North | 4.07 c/kWh | 2.49 c/kWh |
| Helen | 3.54 c/kWh | 3.54 c/kWh |
| Custom | User-defined | User-defined |

Excl. VAT: 25.5% · Energy tax: 2.325 c/kWh (class I, 2026) · Seller margin: configurable (from your contract)
For yleissiirto (general transfer), set day and night rates equal.

## Installation

**[Step-by-step installation guide with screenshots](INSTALLATION.md)**

### HACS (Recommended)

1. Open **HACS** in Home Assistant
2. Click **Integrations** → menu → **Custom repositories**
3. Add `https://github.com/watti-matti/HA-spot-price-predictor` as **Integration**
4. Search for "Spot Price Predictor" and **Download**
5. **Restart** Home Assistant
6. Go to **Settings** → **Devices & Services** → **Add Integration** → **Spot Price Predictor**
7. Follow the setup wizard

### Manual Installation

Copy `custom_components/spot_price_predictor/` to your Home Assistant `custom_components/` directory and restart.

## Optional: Retraining the Bundled Models

The `spot_price_predictor.retrain_models` Home Assistant service refits the pipeline artifacts in place from cached data — no PC-side training or coefficient upload required.

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike", "solar"]   # omit to refit all three
  # fingrid_api_key: "..."                  # only needed for the solar layer
```

The service rewrites the three artifacts atomically and the coordinators auto-reload them on the next update cycle:

| Artifact | Produced by | Layers it carries |
|---|---|---|
| `data/seasonal_components_default.json` | `studies/build_seasonal_components.py` | L1 — hourly / daily / weekly components for FI price and weather inputs |
| `data/spike_model_default.json` | `studies/v2513_layer4_spike_model.py` | L2 Ridge coefficients (six features), L3 AR(1) φ, L4 Normal-body + GPD-tail parameters |
| `data/solar_submodel_default.json` | `studies/v253_solar_submodel.py` | Clear-sky × cloudiness solar production sub-model |

On completion the service fires the `spot_price_predictor_models_retrained` event so automations can react.

## Localization to Other Countries

The system is designed around a **region configuration file** (`config/regions/finland.yaml`) that defines:
- Price API endpoints and format
- Weather station locations with capacity-based weights
- Holiday rules (fixed dates, Easter-relative, special rules)
- Demand modeling parameters (peak hours, heating thresholds)
- Consumer pricing (VAT, energy tax, operator tariffs)
- Neighboring country price sources for the duration model
- Optional grid data sources and normalization values

To adapt for another country, create a new region YAML file and retrain via the service above. The pipeline architecture (seasonal decomposition + physics-feature Ridge + AR(1) + GPD POT) is applicable to any electricity market with weather-dependent generation.

## Data Sources

| Source | Purpose | Free | Auth |
|--------|---------|:---:|:---:|
| [Sahkotin](https://sahkotin.fi) | FI Nord Pool spot prices (historical + current) | Yes | None |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts (7 locations, 120m wind) | Yes | None |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | Swedish spot prices (SE3) | Yes | None |
| [Elering](https://dashboard.elering.ee) | Estonian spot prices (EE) | Yes | None |
| [Fingrid #188](https://data.fingrid.fi) | Nuclear production (real-time) — used by the duration model | Yes | API key (free) |
| [Fingrid #165/246/247](https://data.fingrid.fi) | Consumption / wind / solar forecasts — auxiliary streams, not consumed by the non-seasonal price model | Yes | API key (free) |
| [Nord Pool UMM](https://umm.nordpoolgroup.com) | Planned nuclear outage schedules — auxiliary stream | Yes | None |

## Technical Documentation

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) — Architecture, feature engineering, model details (English)
- [TEKNINEN_TOTEUTUS.md](TEKNINEN_TOTEUTUS.md) — Arkkitehtuuri, piirre-engineering, mallin kuvaus (suomeksi)
- [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md) — D(k) schema migration guide for downstream consumers
- [docs/dtaci_layer.md](docs/dtaci_layer.md) — DtACI online calibration: algorithm details, state persistence, troubleshooting
- [studies/results/V2_8_0_RELEASE_NOTES.md](studies/results/V2_8_0_RELEASE_NOTES.md) — Current release notes: consolidated `spot_price_predictor.retrain_models` service, atomic artifact writes, V26Pipeline auto-reload, `spot_price_predictor_models_retrained` event.
- [studies/results/V2_6_1_BENCHMARK.md](studies/results/V2_6_1_BENCHMARK.md) — Head-to-head benchmark on real FI test data backing the accuracy numbers quoted above (spot MAE ≈ 10 EUR/MWh, R² ≈ 0.93, 98 % high-price hit rate, D(k) R² ≥ 0.95 at every index).
- `studies/results/` — full archive of prior release notes and supporting analyses for readers who want the design rationale behind individual layers.

## License

[MIT](LICENSE)
