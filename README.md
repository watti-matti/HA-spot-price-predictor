# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/watti-matti/HA-spot-price-predictor/releases/tag/v2.2.0)

**Forecast consumer electricity costs and D(k) duration curves up to 7 days ahead** using machine learning with physics-based weather features, cross-border trade analysis, and nuclear outage awareness.

[Suomenkieliset ohjeet (Finnish)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour consumer price forecast** — predict your electricity cost (EUR/kWh) for the next 7 days, including spot price, transfer tariff, seller margin, energy tax, and VAT
- **D(k) cheap/peak duration curves** — 7-day forecast of two complementary 12-level curves: `dk_cheap[k]` = mean price of the cheapest k hours (best achievable cost for deferrable load) and `dk_peak[k]` = mean price of the priciest k hours (worst-case planning cost). Equivalent to CVaR at α=k/24 in both tails of the daily price distribution.
- **Optional PV-aware effective price** — configure your rooftop solar (kWp, tilt, azimuth, efficiency) and the integration produces an `effective_eur_kwh` per hour and PV-aware D(k) cheap/peak curves. Captures both self-consumption savings and export revenue, including the rare "negative spot + high PV" liability case. Uses Open-Meteo irradiance internally; can also read an external PV forecast entity (e.g. Forecast.Solar, custom templates) for users with multi-array setups or shading.
- **Works out-of-the-box** — pre-trained model included, no setup beyond choosing your distribution operator
- **Modular data sources** — starts with free weather + cross-border data, optionally adds Fingrid nuclear data for the full 9-feature model
- **Sign-validated features** — all model coefficients match economic theory (more wind = lower price, more scarcity = higher price)
- **Nuclear outage awareness** — planned outage schedules from Nord Pool UMM enable forward-looking nuclear capacity prediction
- **Online calibration (DtACI)** — optional adaptive conformal prediction intervals with per-D(i) online bias correction; warms up in ~5 days
- **Clean API for optimization** — forecast-only sensor interface with structured D(k) matrix, ready for downstream thermal optimization
- **Retrainable** — advanced users can retrain the model with local data for better personalization
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The system produces two complementary forecasts from a single model coefficients file:

**Hourly price model** — a log-linear Ridge regression (log_offset=100, α=50) trained on 4+ years of historical data predicts spot price (EUR/MWh) for each of the next 170 hours. The coordinator converts each hour to consumer price (EUR/kWh) using your configured tariffs (transfer, margin, tax, VAT). The log transform naturally handles the nonlinear price-scarcity relationship: nearly linear at low prices, exponential amplification at high prices.

The v2.2 bundled model uses **9 sign-validated features** selected via a leave-one-out redundancy sweep — 8 features from v2.1 were identified as redundant or harmful (collinear with the remaining 9) and pruned:

| Feature | Category | Economic role |
|---------|----------|---------------|
| `wind_speed_weighted` | Weather | Primary price driver — more wind = lower price |
| `wind_log_scarcity` | Weather | Nonlinear scarcity signal in low-wind regimes |
| `hdd_sq` | Weather | Squared heating degree days — nonlinear cold amplification |
| `month_cos` | Calendar | Seasonal cycle (peak winter / low summer) |
| `is_holiday` | Calendar | Demand reduction on public holidays |
| `ar_se3` | Cross-border | AR(2) day-ahead forecast for Sweden SE3 |
| `ar_ee` | Cross-border | AR(2) day-ahead forecast for Estonia |
| `export_potential_se3` | Cross-border | SE3 price spread — FI→SE export capacity signal |
| `nuclear_x_scarcity` | Fingrid | Nuclear availability × weather-stress interaction |

**Duration model (D(k) cheap/peak)** — a segment-hierarchical Ridge model predicts the daily duration curve for each of the next 7 days, exposed as two 12-element arrays:

- `dk_cheap[k-1]` = mean consumer price of the **cheapest** k hours, k=1..12 (monotone non-decreasing). Use this for deferrable-load scheduling: "run k hours into the cheapest slots, expected cost = `dk_cheap[k-1]` × kWh."
- `dk_peak[k-1]` = mean consumer price of the **priciest** k hours, k=1..12 (monotone non-increasing). Use this for storage-depletion / worst-case planning: "if I'm forced to run during k peak hours, expected cost = `dk_peak[k-1]` × kWh."

Both arrays are CVaR-equivalent — `dk_cheap[k-1]` = CVaR at α=k/24 in the lower tail, `dk_peak[k-1]` = CVaR at α=k/24 in the upper tail. Together they cover the entire decision-relevant span (the legacy 24-level cumulative D(k) can be exactly reconstructed via the sum identity `dk_cheap[11] + dk_peak[11] = 2 × daily_avg`).

The duration model splits each day into 4 tariff-aligned segments (night 22–07, morning 07–12, midday 12–18, evening 18–22), predicts D(k) independently per segment using Ridge regression, enforces monotonicity via PAVA (Pool Adjacent Violators Algorithm), then merges all segments into the cheap/peak curves.

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

### Forecast Sensors (always created)

| Sensor | State | Unit | Description |
|--------|-------|------|-------------|
| `sensor.price_forecast` | Current consumer price | EUR/kWh | 170h hourly forecast with spot, consumer, weather per hour |
| `sensor.duration_forecast` | Today's `dk_cheap[3]` (cheapest 4h) | EUR/kWh | 7-day × (12 cheap + 12 peak) duration curves |

#### Price Forecast attributes

The **Price Forecast** sensor provides the complete hourly forecast as its `forecast` attribute. Each entry contains `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp}`. The state is the current hour's consumer price in EUR/kWh.

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | array[170] | Hourly entries: `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp}` |
| `current_spot_eur_mwh` | float | Current hour spot price (EUR/MWh) |
| `week_min_eur_kwh` | float | Minimum consumer price in forecast window |
| `week_avg_eur_kwh` | float | Average consumer price in forecast window |
| `week_max_eur_kwh` | float | Maximum consumer price in forecast window |
| `operator` | string | Configured distribution operator |

#### Duration Forecast attributes — D(k) cheap/peak curves

The **Duration Forecast** sensor provides daily duration curves split into a cheap end and a peak end. The `daily_forecast` attribute contains 7 days, each with both new and legacy attributes for backward compatibility. The state is today's `dk_cheap_eur_kwh[3]` — the average consumer price of the cheapest 4 hours, the most-used scheduling indicator.

| Attribute | Shape | Unit | Description |
|-----------|-------|------|-------------|
| `daily_forecast` | array[7] | — | One entry per day |
| `daily_forecast[i].date` | string | — | ISO date (YYYY-MM-DD) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` or `actual` (past days from Sahkotin) |
| **Preferred:** | | | |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[12] | EUR/kWh | Mean price of cheapest k hours, k=1..12 (non-decreasing) |
| `daily_forecast[i].dk_peak_eur_kwh` | float[12] | EUR/kWh | Mean price of priciest k hours, k=1..12 (non-increasing) |
| `daily_forecast[i].dk_cheap_spot_eur_mwh` | float[12] | EUR/MWh | Same in spot price |
| `daily_forecast[i].dk_peak_spot_eur_mwh` | float[12] | EUR/MWh | Same in spot price |
| **Convenience scalars:** | | | |
| `today_cheap_4h_eur_kwh` | float | EUR/kWh | Today's `dk_cheap_eur_kwh[3]` (cheapest 4h) |
| `today_cheap_8h_eur_kwh` | float | EUR/kWh | Today's `dk_cheap_eur_kwh[7]` (cheapest 8h) |
| `today_peak_4h_eur_kwh` | float | EUR/kWh | Today's `dk_peak_eur_kwh[3]` (priciest 4h) |
| `today_peak_1h_eur_kwh` | float | EUR/kWh | Today's `dk_peak_eur_kwh[0]` (single priciest hour) |
| **Legacy (deprecated):** | | | |
| `daily_forecast[i].dk_consumer_eur_kwh` | float[24] | EUR/kWh | Legacy cumulative D(k), k=1..24 |
| `daily_forecast[i].dk_spot_eur_mwh` | float[24] | EUR/MWh | Legacy cumulative spot D(k) |

**Access patterns:**
- Cheapest k hours of day d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` for k in 1..12
- Priciest k hours of day d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` for k in 1..12
- Cross-check identity: `cheap[11] + peak[11] = 2 × daily_average` (always holds to numerical noise)
- Consumer prices include configured tariffs (day/night transfer rate, seller margin, energy tax, VAT)

**Migration:** the legacy `dk_consumer_eur_kwh[24]` array is still emitted for one transition release. New consumers should read the cheap/peak split; see [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md).

**Optional online calibration (DtACI).** Enable the `enable_dtaci_dk` option to wrap the duration forecast with adaptive conformal prediction intervals (Gibbs & Candes, JMLR 2024) plus per-D(i) online bias correction. When on, each daily forecast entry gains four 12-element band attributes — `dk_cheap_lower/upper_eur_kwh` and `dk_peak_lower/upper_eur_kwh` — that achieve 90% marginal coverage and adapt to regime shifts. A `dtaci_diagnostics` attribute exposes per-(direction, k) coverage / bias EMA / dominant gamma / weight entropy for monitoring; see [docs/yaml_examples/dtaci_diagnostics_card.yaml](docs/yaml_examples/dtaci_diagnostics_card.yaml) for a Lovelace card. Calibrated intervals appear after approximately 5 days of reconciled actuals (v2.2: warmup lowered from ~14 days).

**Design principle:** This integration provides *forecasts only*. The cheap/peak curves are the primary API for downstream systems — thermal optimization, load scheduling, and heat pump control consume `dk_cheap` for cost-minimization and `dk_peak` for risk-aware planning.

#### Optional PV-aware effective price (configure in setup UI)

When a non-zero `pv_capacity_kwp` is set in the optional "PV system" config step, every `forecast[i]` entry gains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `pv_production_kwh` | float | Estimated PV output for the hour (kWh) |
| `baseload_kwh` | float | Configured non-flexible household consumption for that hour |
| `effective_eur_kwh` | float | **Marginal cost of running 1 additional kWh of flexible load** at this hour given PV. Bounded in `[s_h, b_h]` (sell ≤ effective ≤ buy). |
| `net_household_cost_eur` | float | Informational raw EUR/h flow (export revenue ↔ grid import). Not used for D(k). |
| `is_export_hour` | bool | True when PV exceeds baseload |
| `sell_eur_kwh` | float | Sell price = spot − commission − optional grid fee. Can be negative during deep oversupply. |

The duration sensor gains parallel `dk_cheap_pv_eur_kwh[12]` / `dk_peak_pv_eur_kwh[12]` curves and matching `today_cheap_pv_{1,4,8,12}h_eur_kwh` / `today_peak_pv_{1,4,8,12}h_eur_kwh` scalars derived from the sorted hourly `effective_eur_kwh` per day.

**PV input modes:**
- **Internal estimator (default)** — uses Open-Meteo `global_tilted_irradiance_instant` × your configured `kwp` × tilt/azimuth correction × `efficiency`. Free, 7-day horizon, no rate limit.
- **External entity (override)** — set `pv_external_entity` to any HA sensor whose attributes match one of: `forecast` list-of-dict (kWh), `wh_hours` dict (Wh), `watts` dict (W), or `irradiance` list (auto-detected W/kWh). Forecast.Solar, custom Open-Meteo templates, and similar publishers all just work. Use this if you have multi-array setups or want shading-aware values.

**Stability invariant.** The `baseload_kwh_per_hour` configuration value represents *non-flexible* consumption only (lighting, fridge, base appliances). **Do NOT** include heat pump, EV, sauna, or any other load that a downstream optimizer schedules — the integration would otherwise see its own decisions reflected back in baseload, breaking the open-loop guarantee and potentially causing schedule oscillation. See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md#pv-aware-pricing) for the full cross-system contract.

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

The v2.2 bundled model uses **9 features** selected by leave-one-out redundancy analysis — a pruning that improved walk-forward MAE by 16% over the previous 17-feature v2.1 model. The feature set scales with available data sources:

| Data Sources | Features | API Keys Required |
|-------------|:---:|:---:|
| Weather (Open-Meteo) + Cross-border (elprisetjustnu.se + Elering) | 8 | None |
| + Nuclear (Fingrid dataset #188) | **9** | 1 (free) |

**Weather features** — wind speed at 120m weighted by installed wind capacity (dominant price driver), logarithmic wind scarcity term, squared heating degree days for nonlinear cold amplification, and seasonal month cycle.

**Calendar features** — holiday flag (Finnish public holidays reduce demand and prices).

**Cross-border features** — AR(2) autoregressive day-ahead price models for Sweden (SE3) and Estonia (EE), each with separate workday/weekend hourly profiles capturing European market coupling. SE3 export potential from 7-day price spreads signals FI→SE transmission headroom.

**Nuclear features** — nuclear availability × scarcity interaction (amplified price impact during weather stress). Planned outage schedules from [Nord Pool UMM](https://umm.nordpoolgroup.com/) (public API, no key required) provide forward-looking nuclear capacity prediction. Fingrid dataset #188 provides real-time nuclear production.

**Fingrid net-load infrastructure** (datasets #165 consumption, #246 wind forecast, #247 solar forecast) is present in the codebase for future experimentation but is not used by the bundled model — the existing 9 features already capture the same supply-pinch signal without multicollinearity.

## Model Performance

**v2.2.0 bundled model:**

| Metric | v2.1.0 | v2.2.0 | Change |
|--------|:---:|:---:|:---:|
| Hourly Ridge features | 17 | **9** | -8 redundant features pruned |
| MAE (training test split) | 23.94 EUR/MWh | **20.07 EUR/MWh** | -16% |
| R² | 0.515 | **0.719** | +40% |
| D(4) Spearman rho (last 365d) | 0.913 | **0.930** | +0.017 |
| Walk-forward MAE (180d holdout) | — | **20.99 EUR/MWh** | vs. AR(2) floor 37.82 |

The walk-forward evaluation (weekly refit on 540-day rolling window, tested on the most recent 180 days including the Jan-Mar 2026 Finnish price spike at 113 EUR/MWh mean) confirms that the 9-feature pruned model outperforms both the v2.1 baseline and the AR(2) neighbour-price floor across all tracked metrics.

**Duration model D(k) ranking accuracy (Spearman rho):**

| Metric | Value |
|--------|:---:|
| D(4) cheap-end rho | 0.930 |
| Training data | 4+ years (2022-2026) |
| Walk-forward holdout | 180 days (most recent) |

The D(k) rho measures whether the relative ranking of days by cheap-hour price is correct — the metric that matters for thermal scheduling decisions.

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

## Optional: Custom Training

Advanced users can retrain the model with their own historical data. This is useful for personalizing to local conditions or incorporating new data sources.

```bash
git clone https://github.com/watti-matti/HA-spot-price-predictor.git
cd HA-spot-price-predictor
pip install -r requirements.txt

# Train with weather + cross-border data only (no API key needed)
python -m src.train_model --region finland

# Train with Fingrid nuclear data (adds nuclear_x_scarcity feature)
python -m src.train_model --region finland --fingrid-key YOUR_KEY

# Walk-forward accuracy evaluation (180-day holdout)
python studies/validate_forecaster_performance.py --zone fi --test-days 180
```

Upload the resulting `output/model_coefs.json` to your Home Assistant to replace the bundled defaults.

**Hyperparameter note:** The bundled model uses `log_offset=100` and `alpha=50`, tuned via a 72-variant overnight sweep on the most recent 180-day holdout. If you retrain, use `--log-offset 100 --ridge-alpha 50` to match.

## Localization to Other Countries

The system is designed around a **region configuration file** (`config/regions/finland.yaml`) that defines:
- Price API endpoints and format
- Weather station locations with capacity-based weights
- Holiday rules (fixed dates, Easter-relative, special rules)
- Demand modeling parameters (peak hours, heating thresholds)
- Consumer pricing (VAT, energy tax, operator tariffs)
- Neighboring country price sources for cross-border features
- Optional grid data sources and normalization values

To adapt for another country, create a new region YAML file and retrain. The model architecture (Ridge regression with physics-based features) is applicable to any electricity market with weather-dependent generation.

## Data Sources

| Source | Purpose | Free | Auth |
|--------|---------|:---:|:---:|
| [Sahkotin](https://sahkotin.fi) | FI Nord Pool spot prices (historical + current) | Yes | None |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts (7 locations, 120m wind) | Yes | None |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | Swedish spot prices (SE3) | Yes | None |
| [Elering](https://dashboard.elering.ee) | Estonian spot prices (EE) | Yes | None |
| [Fingrid #188](https://data.fingrid.fi) | Nuclear production (real-time) | Yes | API key (free) |
| [Fingrid #165/246/247](https://data.fingrid.fi) | Consumption / wind / solar forecasts (infrastructure present, not used by bundled model) | Yes | API key (free) |
| [Nord Pool UMM](https://umm.nordpoolgroup.com) | Planned nuclear outage schedules | Yes | None |

## Technical Documentation

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) — Architecture, feature engineering, model details (English)
- [TEKNINEN_TOTEUTUS.md](TEKNINEN_TOTEUTUS.md) — Arkkitehtuuri, piirre-engineering, mallin kuvaus (suomeksi)
- [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md) — D(k) schema migration guide for downstream consumers
- [docs/dtaci_layer.md](docs/dtaci_layer.md) — DtACI online calibration: algorithm details, state persistence, troubleshooting
- [studies/results/V2_2_RELEASE_NOTES.md](studies/results/V2_2_RELEASE_NOTES.md) — v2.2.0 full release notes with sweep results

## License

[MIT](LICENSE)
