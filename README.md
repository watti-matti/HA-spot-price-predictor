# Spot Price Predictor for Home Assistant

[![HACS Integration](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Predict electricity spot prices up to 7 days ahead** using machine learning with physics-based weather features, cross-border trade analysis, and grid infrastructure data. [(Suomenkieliset ohjeet)](TEKNINEN_TOTEUTUS.md)

## Key Features

- **170-hour price forecast** — predict Nord Pool day-ahead prices a full week into the future
- **Works out-of-the-box** — pre-trained model included, no setup beyond choosing your operator
- **3-tier data architecture** — starts with free weather data, optionally adds cross-border prices and Fingrid nuclear/grid data for improved accuracy
- **Cheapest hours detection** — find the optimal 1-8 hour windows for EV charging, water heating, and thermal storage
- **Consumer price calculation** — adds your energy seller's margin, operator's transfer tariff, energy tax, and VAT automatically
- **Retainable** — advanced users can retrain the model with local data for better personalization
- **Localizable** — region configuration files allow adaptation to other Nordic/European countries

## How It Works

The system uses a **two-stage Ridge regression** model trained on 4 years of historical data. It combines:

- **Weather forecasts** from 7 Finnish locations (wind speed, solar irradiance, temperature) weighted by installed generation capacity
- **Demand patterns** — time-of-day peaks (verified at 09:00 and 19:00), heating degree days, daylight deficit, Finnish holidays and cultural patterns
- **Cross-border trade signals** — 7-day rolling price spreads with Sweden (SE1, SE3) and Estonia calculate import/export potential
- **Grid infrastructure** (optional, Fingrid API) — real-time nuclear production and commercial cross-border power flows

All data sources are **free**. The optional Fingrid API key is also free (email registration at [data.fingrid.fi](https://data.fingrid.fi)).

## Sensors Created

| Sensor | Description |
|--------|-------------|
| `sensor.spot_price_forecast` | Current hour predicted price (EUR/MWh) with 170h forecast attribute |
| `sensor.consumer_price` | Total consumer price (EUR/kWh) including VAT, transfer tariff, energy tax |
| `sensor.cheapest_hours` | Cheapest 1/2/3/4/6/8h blocks with start times, avg prices, and all hours below average |
| `sensor.week_price_stats` | Weekly consumer price min/avg/max (EUR/kWh) |

The **Cheapest Hours** sensor is the primary automation tool — use its attributes to schedule EV charging, water heating, or heat pump operation during the cheapest N-hour windows. The forecast sensors carry full 170-hour prediction arrays including weather context (wind, solar, temperature) for dashboards.

### Dashboard

An [ApexCharts dashboard example](docs/yaml_examples/apexcharts_dashboard.yaml) is included showing consumer price forecast with color-coded price levels, spot price, weekly average, wind speed, and temperature. Requires the [apexcharts-card](https://github.com/RomRider/apexcharts-card) custom card (install via HACS Frontend).

## Feature Tiers

The model automatically adapts to available data sources:

| Tier | Features | Data Sources | API Keys | Accuracy |
|------|----------|-------------|----------|----------|
| 1 | 28 (weather + demand) | Sahkotin + Open-Meteo | None | Baseline |
| 1+2 | 34 (+cross-border trade) | + elprisetjustnu.se + Elering | None | ~14% better MAE |
| 1+2+3 | 38 (+grid infrastructure) | + Fingrid nuclear + flows | 1 (free) | Best (R² 0.55) |

**Tier 2 adds import/export potential** between Finland and neighboring countries by analyzing persistent price spread patterns (verified autocorrelation r=0.54-0.73 weekly).

**Tier 3 adds Fingrid real-time data**: nuclear power production (dataset #188) and commercial cross-border flows with Sweden and Estonia (datasets #31, #32, #140).

## Supported Operators (Finland)

| Operator | Day rate (07-22) | Night rate (22-07) |
|----------|:---:|:---:|
| Elenia | 3.61 c/kWh | 2.20 c/kWh |
| Caruna South | 5.90 c/kWh | 4.50 c/kWh |
| Caruna North | 5.20 c/kWh | 4.10 c/kWh |
| Helen (Helsinki) | 5.37 c/kWh | 4.03 c/kWh |
| Custom | User-defined | User-defined |

VAT: 25.5% · Energy tax: 2.325 c/kWh (class I, 2026) · Seller margin: configurable (from your contract)
For yleissiirto (general transfer), set day and night rates equal.

## Installation

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

# With Fingrid nuclear + grid data
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
|--------|---------|------|------|
| [Sahkotin](https://sahkotin.fi) | FI Nord Pool spot prices | Yes | None |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts (7 locations) | Yes | None |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | Swedish spot prices (SE1, SE3) | Yes | None |
| [Elering](https://dashboard.elering.ee) | Estonian spot prices (EE) | Yes | None |
| [Fingrid](https://data.fingrid.fi) | Nuclear production + cross-border flows | Yes | API key (free) |

## Technical Documentation

- [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) — Architecture, feature engineering, model details (English)
- [TEKNINEN_TOTEUTUS.md](TEKNINEN_TOTEUTUS.md) — Arkkitehtuuri, piirre-engineering, mallin kuvaus (suomeksi)

## License

[MIT](LICENSE)
