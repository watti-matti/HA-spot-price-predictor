# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Predict electricity spot prices up to 7 days ahead** using machine learning with physics-based weather features, cross-border trade analysis, and nuclear outage awareness.

[Suomenkieliset ohjeet (Finnish)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour price forecast** — predict Nord Pool day-ahead prices a full week into the future
- **Works out-of-the-box** — pre-trained model included, no setup beyond choosing your operator
- **3-tier data architecture** — starts with free weather data, optionally adds cross-border prices and Fingrid nuclear data for improved accuracy
- **Sign-validated features** — all model coefficients match economic theory (more wind = lower price, more scarcity = higher price)
- **Nuclear outage awareness** — planned outage schedules from Nord Pool UMM enable forward-looking nuclear impact
- **D(k) duration forecast** — predict daily cost by usage duration (CVaR of intra-day price distribution)
- **Consumer price calculation** — adds your energy seller's margin, operator's transfer tariff, energy tax, and VAT automatically
- **Clean API for optimization** — forecast-only sensor interface, ready for downstream thermal optimization
- **Retrainable** — advanced users can retrain the model with local data for better personalization
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The system uses a **log-linear Ridge regression** model trained on 4 years of historical data. The log transform naturally handles the nonlinear price-scarcity relationship: nearly linear at low prices, exponential amplification at high prices. Features are selected via greedy forward selection with sign constraints and bootstrap stability analysis. The model combines:

- **Wind speed at 120m** from 7 Finnish locations weighted by installed wind capacity — the dominant price driver
- **Nonlinear wind scarcity** — logarithmic scarcity and calm-wind x demand-peak interactions
- **AR neighbor price models** — autoregressive forecasts for Sweden (SE1, SE3) and Estonia (EE) with workday/weekend hourly profiles, capturing European market coupling
- **Thermal demand** — squared heating degree days for nonlinear cold amplification
- **Nuclear deficit** (optional, Fingrid API) — standalone nuclear availability + scarcity interaction

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

### Forecast Sensors (always created)

| Sensor | State | Description |
|--------|-------|-------------|
| `sensor.price_forecast` | Consumer price (c/kWh) | 170h forecast array with spot EUR/MWh, consumer c/kWh, wind, solar, temperature per hour |
| `sensor.duration_forecast` | D(4) (c/kWh) | 7-day D(k) duration curves — `dk_consumer_cent_kwh[24]` and `dk_spot_eur_mwh[24]` per day |

The **Price Forecast** sensor provides the complete hourly forecast as a `forecast` attribute array. Each entry contains `{timestamp, spot_eur_mwh, consumer_ckwh, wind, solar, temp}`. Week statistics (`week_min/avg/max_ckwh`) are included as attributes. This is the primary data interface — downstream systems (thermal optimization, load scheduling) consume this forecast to make control decisions.

The **Duration Forecast** sensor provides D(k) = CVaR of the intra-day price distribution at level k/24. The `daily_forecast` attribute contains 7 days, each with `dk_consumer_cent_kwh[24]` (c/kWh) and `dk_spot_eur_mwh[24]` (EUR/MWh). Access any level as `dk_consumer_cent_kwh[k-1]` for k=1..24. All vectors are guaranteed length 24 — only complete days are included.

**Design principle:** This integration provides *forecasts only*. Optimization functions (cheapest hours, load scheduling, heat pump control) belong in a separate thermal optimization layer that consumes these forecasts. This clean separation means either component can be replaced independently.

### Actual Price Sensors (optional, when Nordpool entity is configured)

If you have a Nordpool integration installed (e.g., [custom-components/nordpool](https://github.com/custom-components/nordpool)), you can link it to get actual price sensors for comparison:

| Sensor | Description |
|--------|-------------|
| `sensor.spot_electricity_price` | Actual consumer price from Nordpool with continuous timeline attribute |
| `sensor.spot_electricity_selling_price` | Spot price minus PV selling commission (for solar panel owners) |

**Setup:** Enter your Nordpool sensor entity ID (e.g., `sensor.nordpool_kwh_fi_eur_3_10_0`) in the operator configuration step. These sensors enable side-by-side comparison of actual prices vs forecast in the dashboard.

### Dashboard

An [ApexCharts dashboard example](docs/yaml_examples/apexcharts_dashboard.yaml) is included showing:
- **Actual consumer price** from Nordpool (step-line, color-coded) — ground truth
- **Forecast consumer price** from the predictor (smooth line) — prediction
- **Weekly average** reference line
- **Wind speed forecast** on secondary axis (key price driver)

This allows visual comparison of how well the forecast matches reality. Requires the [apexcharts-card](https://github.com/RomRider/apexcharts-card) custom card (install via HACS Frontend).

## Feature Tiers

The model uses 17 sign-validated features selected via greedy forward selection with bootstrap stability analysis.

| Tier | Features | Data Sources | API Keys |
|------|:---:|-------------|:---:|
| 1 | 11 (weather + wind nonlinear) | Sahkotin + Open-Meteo | None |
| 1+2 | 15 (+AR neighbor prices) | + elprisetjustnu.se + Elering | None |
| 1+2+3 | 17 (+nuclear features) | + Fingrid nuclear | 1 (free) |

**Tier 1** includes wind speed, solar irradiance, heating degree days, time cycles, holidays, and nonlinear wind features (log-scarcity, calm-wind x demand-peak interactions).

**Tier 2** adds AR(2) autoregressive neighbor price models for Sweden (SE1, SE3) and Estonia (EE), each with separate workday/weekend hourly profiles. The AR models capture European market coupling — when neighbor markets are expensive, Finnish prices follow. Also includes SE3 export potential from 7-day price spreads.

**Tier 3** adds nuclear deficit (fraction of nuclear capacity offline) and nuclear x scarcity interaction (amplified price impact during weather stress). Planned outage schedules from [Nord Pool UMM](https://umm.nordpoolgroup.com/) (public API, no key required) provide forward-looking nuclear awareness.

## Model Performance (v6.0)

| Metric | Value |
|--------|:---:|
| MAE | 2.34 EUR/MWh |
| R2 | 0.75 |
| Features | 17 (sign-validated, bootstrap-stable) |
| 4h block rank concordance | 97.1% (for >5 EUR/MWh differences) |
| 8h block rank concordance | 99.0% |
| EV savings captured | 94.2% of optimal |
| Training data | 4 years (2022-2026) |

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
