# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Forecast consumer electricity costs and D(k) duration curves up to 7 days ahead** using machine learning with physics-based weather features, cross-border trade analysis, and nuclear outage awareness.

[Suomenkieliset ohjeet (Finnish)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour consumer price forecast** — predict your electricity cost (EUR/kWh) for the next 7 days, including spot price, transfer tariff, seller margin, energy tax, and VAT
- **D(k) duration curves (CVaR)** — 7-day × 24-level matrix predicting cost-optimal load scheduling: D(k) = average consumer price for the cheapest k hours per day, mathematically equivalent to CVaR at α = k/24
- **Works out-of-the-box** — pre-trained model included, no setup beyond choosing your distribution operator
- **Modular data sources** — starts with free weather data, optionally adds cross-border prices and Fingrid nuclear data for improved accuracy
- **Sign-validated features** — all model coefficients match economic theory (more wind = lower price, more scarcity = higher price)
- **Nuclear outage awareness** — planned outage schedules from Nord Pool UMM enable forward-looking nuclear capacity prediction
- **Clean API for optimization** — forecast-only sensor interface with structured D(k) matrix, ready for downstream thermal optimization
- **Retrainable** — advanced users can retrain the model with local data for better personalization
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The system produces two complementary forecasts from a single model coefficients file:

**Hourly price model** — a log-linear Ridge regression trained on 4 years of historical data predicts spot price (EUR/MWh) for each of the next 170 hours. The coordinator converts each hour to consumer price (EUR/kWh) using your configured tariffs (transfer, margin, tax, VAT). The log transform naturally handles the nonlinear price-scarcity relationship: nearly linear at low prices, exponential amplification at high prices. Features are selected via greedy forward selection with sign constraints and bootstrap stability analysis. Key drivers:

- **Wind speed at 120m** from 7 Finnish locations weighted by installed wind capacity — the dominant price driver
- **Nonlinear wind scarcity** — logarithmic scarcity and calm-wind × demand-peak interactions
- **AR neighbor price models** — autoregressive forecasts for Sweden (SE1, SE3) and Estonia (EE) capturing European market coupling
- **Thermal demand** — squared heating degree days for nonlinear cold amplification
- **Nuclear deficit** (optional, Fingrid API) — nuclear availability + scarcity interaction

**Duration model (D(k) = CVaR)** — a segment-hierarchical Ridge model predicts D(k) for each of the next 7 days. D(k) is the average consumer price for the cheapest k hours in a day, mathematically equivalent to Conditional Value-at-Risk (CVaR) at level α = k/24. This makes D(k) the natural cost metric for load scheduling: "run your appliance during the cheapest k hours" costs exactly D(k) per kWh.

The duration model splits each day into 4 tariff-aligned segments (night 22–07, morning 07–12, midday 12–18, evening 18–22), predicts D(k) independently per segment using Ridge regression, enforces monotonicity via PAVA (Pool Adjacent Violators Algorithm), converts each extracted sorted price to consumer EUR/kWh using segment-appropriate transfer tariffs (day/night rate), then merges all segments into a full 24-level D(k) curve.

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

### Forecast Sensors (always created)

| Sensor | State | Unit | Description |
|--------|-------|------|-------------|
| `sensor.price_forecast` | Current consumer price | EUR/kWh | 170h hourly forecast with spot, consumer, weather per hour |
| `sensor.duration_forecast` | Today's D(4) | EUR/kWh | 7-day × 24-level D(k) = CVaR duration matrix |

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

#### Duration Forecast attributes — D(k) matrix

The **Duration Forecast** sensor provides D(k) = CVaR of the intra-day price distribution at level α = k/24. The `daily_forecast` attribute contains a 7-day × 24-level D(k) matrix in day-per-row orientation. The state is today's D(4) in EUR/kWh.

| Attribute | Shape | Unit | Description |
|-----------|-------|------|-------------|
| `daily_forecast` | array[7] | — | D(k) matrix, one entry per day |
| `daily_forecast[i].date` | string | — | ISO date (YYYY-MM-DD) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].dk_consumer_eur_kwh` | float[24] | EUR/kWh | D(1)…D(24) consumer price, index k−1 |
| `daily_forecast[i].dk_spot_eur_mwh` | float[24] | EUR/MWh | D(1)…D(24) spot price, index k−1 |

**Matrix access patterns:**
- Single value: `daily_forecast[day].dk_consumer_eur_kwh[k-1]` → D(k) for one day
- Consumer transposes to k-per-row: `dk_matrix[k-1][day]` for D(k) trajectory across days
- All vectors guaranteed length 24; only complete days included
- Consumer prices include configured tariffs (day/night transfer rate, seller margin, energy tax, VAT)

**Design principle:** This integration provides *forecasts only*. The D(k) matrix is the primary API for downstream systems — thermal optimization, load scheduling, and heat pump control consume D(k) to answer "what does it cost to run k hours today?" This clean separation means either component can be replaced independently.

### Actual Price Sensors (optional, when Nordpool entity is configured)

If you have a Nordpool integration installed (e.g., [custom-components/nordpool](https://github.com/custom-components/nordpool)), you can link it to get actual price sensors for comparison:

| Sensor | Description |
|--------|-------------|
| `sensor.spot_electricity_price` | Actual consumer price from Nordpool with continuous timeline attribute |
| `sensor.spot_electricity_selling_price` | Spot price minus PV selling commission (for solar panel owners) |

**Setup:** Enter your Nordpool sensor entity ID (e.g., `sensor.nordpool_kwh_fi_eur_3_10_0`) in the operator configuration step. These sensors enable side-by-side comparison of actual prices vs forecast in the dashboard.

### Dashboard

A complete [Lovelace dashboard](ha_dashboard.yaml) is included (ApexCharts + Mushroom cards) with:
- **48h consumer price bar chart** — color-coded hourly bars with extrema markers
- **7-day hourly price trend** — area chart of consumer prices across the full forecast window
- **D(k) duration curves** — multi-line chart showing D(1), D(4), D(8), D(24) trajectories across 7 days
- **Current price + D(4) chips** — at-a-glance status with color-coded icons
- **Week statistics** — min/avg/max consumer price summary
- **Data status** — active sources, forecast horizon, staleness indicator

An additional [ApexCharts-only dashboard](docs/yaml_examples/apexcharts_dashboard.yaml) provides actual vs forecast comparison with wind overlay. Both require the [apexcharts-card](https://github.com/RomRider/apexcharts-card) custom card (install via HACS Frontend).

## Data Sources & Features

The model supports up to 17 sign-validated features selected via greedy forward selection with bootstrap stability analysis. The bundled default model uses 15 features (weather + cross-border, no API keys required). Adding Fingrid nuclear data enables the full 17 features.

| Data Sources | Features | API Keys |
|-------------|:---:|:---:|
| Weather (Sahkotin + Open-Meteo) | 11 (weather + wind nonlinear) | None |
| + Cross-border (elprisetjustnu.se + Elering) | 15 (+AR neighbor prices) | None |
| + Nuclear (Fingrid) | 17 (+nuclear features) | 1 (free) |

**Weather features** include wind speed, solar irradiance, heating degree days, time cycles, holidays, and nonlinear wind features (log-scarcity, calm-wind x demand-peak interactions).

**Cross-border features** add AR(2) autoregressive neighbor price models for Sweden (SE1, SE3) and Estonia (EE), each with separate workday/weekend hourly profiles. The AR models capture European market coupling — when neighbor markets are expensive, Finnish prices follow. Also includes SE3 export potential from 7-day price spreads.

**Nuclear features** add nuclear deficit (fraction of nuclear capacity offline) and nuclear x scarcity interaction (amplified price impact during weather stress). Planned outage schedules from [Nord Pool UMM](https://umm.nordpoolgroup.com/) (public API, no key required) provide forward-looking nuclear awareness.

## Model Performance

**Bundled model (weather + cross-border, 15 features):**

| Metric | Value |
|--------|:---:|
| MAE | 24.7 EUR/MWh |
| R² | 0.39 |
| Features | 15 (sign-validated, bootstrap-stable) |
| Training data | 4 years (2022-2026) |

**Duration model (D(k) Spearman ρ, last 365 days):**

| D(k) | Use case | ρ |
|:---:|:-:|:---:|
| D(4) | Cheapest 4h | 0.908 |
| D(8) | Cheapest 8h | 0.921 |
| D(24) | Daily avg | 0.937 |

Retraining with Fingrid nuclear data (free API key) adds 2 features and improves accuracy.

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
2. Click **Integrations** → ⋮ menu → **Custom repositories**
3. Add `https://github.com/watti-matti/HA-spot-price-predictor` as **Integration**
4. Search for "Spot Price Predictor" and **Download**
5. **Restart** Home Assistant
6. Go to **Settings** → **Devices & Services** → **Add Integration** → **Spot Price Predictor**
7. Follow the setup wizard

### Manual Installation

Copy `custom_components/spot_price_predictor/` to your Home Assistant `custom_components/` directory and restart.

## Optional: Custom Training

For advanced users who want to retrain the model with their own historical data:

```bash
git clone https://github.com/watti-matti/HA-spot-price-predictor.git
cd HA-spot-price-predictor
pip install -r requirements.txt

# Train with available data (adapts automatically)
python -m src.train_model --region finland

# With Fingrid nuclear data (for nuclear x scarcity feature)
export FINGRID_API_KEY=your_key_here
python -m src.train_model --region finland

# Evaluate accuracy with interactive dashboard
python -m src.evaluate --region finland
```

Upload the resulting `output/model_coefs.json` to your Home Assistant to replace the bundled defaults.

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
| [Sahkotin](https://sahkotin.fi) | FI Nord Pool spot prices | Yes | None |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts (7 locations, 120m wind) | Yes | None |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | Swedish spot prices (SE3) | Yes | None |
| [Elering](https://dashboard.elering.ee) | Estonian spot prices (EE) | Yes | None |
| [Fingrid](https://data.fingrid.fi) | Nuclear production (#188) | Yes | API key (free) |
| [Nord Pool UMM](https://umm.nordpoolgroup.com) | Planned nuclear outage schedules | Yes | None |

## Technical Documentation

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) — Architecture, feature engineering, model details (English)
- [TEKNINEN_TOTEUTUS.md](TEKNINEN_TOTEUTUS.md) — Arkkitehtuuri, piirre-engineering, mallin kuvaus (suomeksi)

## License

[MIT](LICENSE)
