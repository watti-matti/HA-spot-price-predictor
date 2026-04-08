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
- **Cheapest hours detection** — find the optimal 1-8 hour windows for EV charging, water heating, and thermal storage
- **Consumer price calculation** — adds your energy seller's margin, operator's transfer tariff, energy tax, and VAT automatically
- **Retrainable** — advanced users can retrain the model with local data for better personalization
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The system uses a **two-stage Ridge regression** model trained on 4 years of historical data. Features are selected via greedy forward selection with sign constraints, ensuring every coefficient matches economic theory. The model combines:

- **Wind speed at 120m** from 7 Finnish locations weighted by installed wind capacity — the dominant price driver (more wind = lower price)
- **Solar irradiance** weighted by installed solar capacity
- **Demand patterns** — workday AM/PM peaks (Gaussian at 09:00/19:00), holidays
- **Thermal demand** — squared heating degree days for nonlinear cold amplification
- **Wind scarcity** — penalty for low wind (<4 m/s) on workdays
- **Cross-border export potential** — 7-day rolling price spreads with Sweden (SE3) and Estonia (EE)
- **Nuclear x scarcity interaction** (optional, Fingrid API) — amplified price impact when nuclear capacity is reduced during weather stress

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

### Forecast Sensors (always created)

| Sensor | Description |
|--------|-------------|
| `sensor.spot_price_forecast` | Current hour predicted price (EUR/MWh) with 170h forecast attribute |
| `sensor.consumer_price` | Total consumer price (EUR/kWh) including VAT, seller margin, transfer tariff, energy tax |
| `sensor.cheapest_hours` | Cheapest 1/2/3/4/6/8h blocks within configurable search window |
| `sensor.week_price_stats` | Weekly consumer price min/avg/max (EUR/kWh) |

The **Cheapest Hours** sensor is the primary automation tool — use its attributes to schedule EV charging, water heating, or heat pump operation during the cheapest N-hour windows. The search window start and duration are configurable in the integration settings (default: starting tomorrow, 48h window).

### Spot Price Sensors (optional, when Nordpool entity is configured)

If you have a Nordpool integration installed (e.g., [custom-components/nordpool](https://github.com/custom-components/nordpool)), you can link it in the configuration to get additional sensors:

| Sensor | Description |
|--------|-------------|
| `sensor.spot_electricity_price` | Actual spot buying price from Nordpool with continuous timeline attribute |
| `sensor.spot_electricity_selling_price` | Spot price minus PV selling commission (for solar panel owners) |

**Setup:** Enter your Nordpool sensor entity ID (e.g., `sensor.nordpool_kwh_fi_eur_3_10_0`) in the operator configuration step. Optionally enable PV selling price and set the commission (EUR/kWh).

These sensors provide a continuous timeline in their `timeline` attribute, combining Nordpool's today and tomorrow data into a single series. This enables side-by-side comparison of actual prices vs forecast in the dashboard.

### Dashboard

An [ApexCharts dashboard example](docs/yaml_examples/apexcharts_dashboard.yaml) is included showing:
- **Actual consumer price** from Nordpool (step-line, color-coded) — ground truth
- **Forecast consumer price** from the predictor (smooth line) — prediction
- **Weekly average** reference line
- **Wind speed forecast** on secondary axis (key price driver)

This allows visual comparison of how well the forecast matches reality. Requires the [apexcharts-card](https://github.com/RomRider/apexcharts-card) custom card (install via HACS Frontend).

## Feature Tiers

The model uses 14 sign-validated features selected via greedy forward selection. Only features whose learned coefficients match economic theory are included.

| Tier | Features | Data Sources | API Keys | MAE (EUR/MWh) |
|------|:---:|-------------|:---:|:---:|
| 1 | 12 (weather + demand) | Sahkotin + Open-Meteo | None | Baseline |
| 1+2 | 14 (+export potential) | + elprisetjustnu.se + Elering | None | 2.66 |
| 1+2+3 | 15 (+nuclear interaction) | + Fingrid nuclear | 1 (free) | 2.55 |

**Tier 1** includes wind speed, solar irradiance, demand peaks, heating degree days, wind scarcity, and time cycles.

**Tier 2** adds export potential to Sweden (SE3) and Estonia (EE) from 7-day rolling price spreads.

**Tier 3** adds nuclear x scarcity interaction: amplified price impact when nuclear production is reduced and weather conditions are stressed (low wind + cold + peak demand). Nuclear production data comes from Fingrid dataset #188. Planned outage schedules are fetched from [Nord Pool UMM](https://umm.nordpoolgroup.com/) (public API, no key required) for forward-looking awareness.

## Model Performance (v5.0)

| Metric | Value |
|--------|:---:|
| MAE | 2.55 EUR/MWh |
| R2 | 0.70 |
| Features | 15 (sign-validated) |
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
