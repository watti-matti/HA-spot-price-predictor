# Documentation: HA-spot-price-predictor

Finnish electricity spot price forecasting for Home Assistant using log-linear Ridge regression with physics-based features, duration curve prediction, and multi-source data integration.

## Architecture

The system has two phases: **training** (Python, run periodically on PC) and **inference** (Home Assistant custom integration, always-on).

### Training Pipeline

```
Sahkotin API  ──┐
Open-Meteo API ─┼──> Feature Engineering ──> Log-linear Ridge ──> model_coefs.json
Elpriset API ───┤    (17 sign-validated       + Power stretch      (hourly + duration
Elering API ────┤     features)                + Duration model)     model coefficients)
Fingrid API ────┘ (optional)
```

### Home Assistant Deployment

```
Open-Meteo  ──┐
Elpriset    ──┼──> Feature Builder ──> Hourly Model  ──> Price Forecast (170h)
Elering     ──┤    (pure Python)       + Duration Model   + D(k) Duration Curves (7d)
Fingrid     ──┘                        (pure Python)      + Dashboard
Nord Pool UMM ─────────────────────────┘
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

| Source | Zones | Purpose |
|--------|-------|---------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1, SE3 | Swedish spot prices for AR models |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Estonian spot prices for AR models |

Used to fit AR(2) models on cross-border price deviations from hourly daytype profiles. Analysis confirmed strong autocorrelation (lag-1 weekly r=0.54-0.73, sign persistence 100%).

### Optional grid data (free API key)

| Source | Purpose |
|--------|---------|
| [Fingrid Open Data](https://data.fingrid.fi) | Nuclear production (#188) for nuclear deficit and scarcity features |

Register for free at data.fingrid.fi. Without this key, the model trains on Tier 1+2 features only (15 features).

### Nuclear outage schedule (free, no key)

| Source | Purpose |
|--------|---------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Planned nuclear outages for forward-looking capacity prediction |

---

## Feature Engineering

Model v2.0 uses 17 sign-validated features selected via greedy forward selection with sign constraints. All tunable parameters are in `config/regions/finland.yaml` under the `features` section.

### Tier 1: Base features (11) -- no API keys needed

| Category | Features | Coefficient sign |
|----------|----------|:---:|
| Supply | `wind_speed_weighted`, `solar_irradiance_weighted` | negative (more supply = lower price) |
| Time cycles | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | cyclic |
| Calendar | `is_holiday` | negative (lower demand) |
| Thermal demand | `hdd_sq` (squared heating degree days, threshold 17°C) | positive |
| Wind nonlinear | `wind_log_scarcity` = log1p(max(0, 8-wind)) | positive (low wind = higher price) |
| Wind x demand | `wind_calm_x_peak_am` = max(0, 6-wind) × AM peak (9h, σ=1.8) | positive |
| Wind x demand | `wind_calm_x_peak_pm` = max(0, 6-wind) × PM peak (19h, σ=2.0) | positive |

### Tier 2: AR neighbor prices + export potential (+4) -- no API keys needed

AR(2) models predict cross-border neighbor prices using workday/weekend hourly profiles with damped autoregressive deviation. This captures the European market coupling signal that drives Finnish prices.

| Feature | Source | Method |
|---------|--------|--------|
| `ar_se1` | Sweden SE1 | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `ar_se3` | Sweden SE3 | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `ar_ee` | Estonia | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `export_potential_se3` | SE3 spread | max(0, -spread_7d_fi_se3) |

The AR models decompose neighbor prices into deterministic daily profiles (workday vs weekend, 24 hours each) plus a stochastic deviation modeled by AR(2). The AR deviation is damped (max root < 0.95) so predictions converge to the daily profile within 24 hours, ensuring stability over the full 170-hour forecast window.

### Tier 3: Nuclear features (+0-2) -- requires Fingrid API key

| Feature | Formula | Meaning |
|---------|---------|---------|
| `nuclear_deficit` | max(0, 1 - nuclear_mw/4372) | Fraction of nuclear capacity offline |
| `nuclear_x_scarcity` | nuclear_deficit × scarcity_indicator | Nuclear outage amplifies weather-driven scarcity |

**Forward-looking outage data:** Planned outage schedules are fetched from the [Nord Pool UMM platform](https://umm.nordpoolgroup.com/) (public API, no key required). The coordinator computes per-hour nuclear availability for the forecast horizon.

### Feature count by configuration

| Configuration | Features | API keys |
|---------------|----------|----------|
| Tier 1 only | 11 | None |
| Tier 1 + 2 | 15 | None |
| Tier 1 + 2 + 3 | 17 | 1 (Fingrid, free) |

---

## Model Architecture

### Hourly model: Log-linear Ridge regression

**Prediction formula:** `price = scale × max(0, exp(Σ coef_i × feat_i + intercept) - 55) ^ power`

The log transform naturally handles the nonlinear price-scarcity relationship: nearly linear at low prices, exponential amplification at high prices.

- Ridge regression on log(price + 55) target
- 17 sign-validated features (all bootstrap-stable)
- Power stretch (scale, exponent) fitted via Nelder-Mead on test set
- Time-decay weighting: half-life 120 days
- Ridge alpha = 1.0, augmented matrix (no penalty on intercept)

**Training:** 4 years historical data, 85/15 time-ordered split, batch processing (512 rows).

### Duration model: Segment-hierarchical Ridge + PAVA

Predicts D(k) = average spot price for the cheapest k hours in a day. D(k) is mathematically equivalent to Conditional Value-at-Risk (CVaR) of the intra-day price distribution at level α = k/24, making it the natural cost metric for load scheduling: "schedule into the cheapest k hours" minimizes CVaR.

**PAVA** (Pool Adjacent Violators Algorithm) is an isotonic regression method that enforces monotonicity. Since D(k) must be non-decreasing by definition — adding more hours to the average can only include equal or more expensive hours — PAVA merges any violations from independent per-level Ridge predictions by averaging adjacent violating pairs until D(1) ≤ D(2) ≤ ... ≤ D(N) holds everywhere.

**Architecture:**
- 4 day segments aligned with day/night tariff: night (22-06, 9h), morning (07-11, 5h), midday (12-17, 6h), evening (18-21, 4h)
- Per (segment, duration level): independent Ridge model with 10 features
- Log-linear target: log(D(k) + 55)
- Forgetting factor λ = 0.960 (half-life 17 days, optimized via sweep)
- PAVA isotonic post-processing: enforces D(1) ≤ D(2) ≤ ... ≤ D(N)
- Segment-to-day reconstruction: extract sorted prices → merge → re-sort → full 24h D(k)

**Duration model features:**
`wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`

**Performance (Spearman rank correlation):**

| Duration level | Use case | ρ (all) | ρ (last 365d) |
|:-:|:-:|:-:|:-:|
| D(1) | Cheapest 1h | 0.895 | 0.898 |
| D(4) | Cheapest 4h | 0.904 | 0.906 |
| D(8) | Cheapest 8h | 0.929 | 0.921 |
| D(24) | Daily average | 0.935 | 0.937 |

**Output:** `model_coefs.json` containing hourly model coefficients, AR model parameters, and duration model coefficients.

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

**Formula:** `(max(0, spot_EUR_MWh) / 1000 + seller_margin + transfer_rate + energy_tax) × VAT × 100` [c/kWh]

Configurable per operator in `finland.yaml`. Default: Elenia (day 3.61, night 2.20 c/kWh), VAT 25.5%, energy tax 2.325 c/kWh, seller margin 0.00 c/kWh (set from your electricity contract).

### Forecast sensors (always created)

| Sensor | State | Attributes |
|--------|-------|------------|
| Price Forecast | Consumer c/kWh | `forecast` (170h array: spot_eur_mwh, consumer_ckwh, wind, solar, temp), `week_min/avg/max_ckwh` |
| Duration Forecast | D(4) c/kWh | `daily_forecast` (7-day array: dk_consumer_cent_kwh[24], dk_spot_eur_mwh[24] per day) |

### Actual price sensors (optional, Nordpool)

| Sensor | Unit | Description |
|--------|------|-------------|
| Spot Electricity Price | EUR/kWh | Actual consumer price from Nordpool with continuous timeline |
| Spot Electricity Selling Price | EUR/kWh | Spot minus PV selling commission (for solar panel owners) |

### Design principle

This integration provides **forecasts only**. Optimization functions (cheapest hours, load scheduling, heat pump control) belong in a separate thermal optimization layer that consumes the forecast data. This clean separation allows either component to be replaced independently.

The **Price Forecast** sensor provides a unified 170-hour forecast array in its `forecast` attribute. Each entry contains `{timestamp, spot_eur_mwh, consumer_ckwh, wind, solar, temp}`. Week statistics (`week_min/avg/max_ckwh`) are included as convenience attributes. The state is the current hour's consumer price in c/kWh.

The **Duration Forecast** sensor provides D(k) = CVaR of the intra-day price distribution. The `daily_forecast` attribute contains 7 days, each with `dk_consumer_cent_kwh[24]` (c/kWh) and `dk_spot_eur_mwh[24]` (EUR/MWh). Access any level as `dk_consumer_cent_kwh[k-1]` for k=1..24. All vectors are guaranteed length 24 — only complete days are included. All consumer prices use the configured tariffs — no hardcoded rates.

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
5. **Add neighboring price sources** for cross-border features
6. **Set consumer pricing** — VAT rate, energy tax, and distribution operator tariffs
7. **Run training** with `--region sweden`

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

### Current performance (v2.0, 4-year training, 120-day half-life)

**Hourly model:**

| Metric | Value |
|--------|:---:|
| MAE | 23.6 EUR/MWh |
| RMSE | 47.1 EUR/MWh |
| R² | 0.522 |

**Duration model (Spearman ρ, last 365 days):**

| D(k) | Use case | ρ |
|:---:|:-:|:---:|
| D(4) | Cheapest 4h | 0.906 |
| D(8) | Cheapest 8h | 0.921 |
| D(24) | Daily avg | 0.937 |

### Recommended retraining frequency

**Retrain every 3-4 months (quarterly).**

### How to retrain

```bash
cd HA-spot-price-predictor
pip install -r requirements.txt

# Retrain with latest data
export FINGRID_API_KEY=your_key_here  # optional
python -m src.train_model --region finland

# Generate monitoring dashboard
python model_dashboard.py

# Generate 7-day forecast preview
python forecast_dashboard.py

# Copy coefficients to HA integration
cp output/model_coefs.json custom_components/spot_price_predictor/data/model_coefs_default.json
```

### Half-life parameter

The `half_life_days` setting (default: 120) controls how the model weights historical data during training. Data from 120 days ago has 50% weight. Optimized for Finland's rapidly changing wind capacity.

The duration model uses a separate forgetting factor λ = 0.960 (half-life 17 days), optimized for tracking weather-regime changes.

---

## Project Structure

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # This document
├── TEKNINEN_TOTEUTUS.md         # Finnish translation
├── config/regions/
│   └── finland.yaml             # Central configuration (all parameters)
├── src/
│   ├── train_model.py           # Training pipeline
│   ├── features.py              # Feature engineering (training)
│   ├── data_sources.py          # API clients (training)
│   └���─ holidays.py              # Holiday calculator
├── custom_components/
│   └── spot_price_predictor/    # HA HACS integration
│       ├── model.py             # Pure Python inference (hourly + duration)
│       ├── features.py          # Pure Python feature builder
│       ├── coordinator.py       # HA data coordinator
│       ├── sensor.py            # HA sensor entities
│       ├── api_client.py        # Async API clients
│       ├── const.py             # Constants and defaults
│       └── data/
│           ├── model_coefs_default.json  # Bundled model
���           └── finland.yaml              # Bundled config
├── ha_dashboard.yaml            # Home Assistant Lovelace dashboard (ApexCharts + Mushroom)
├── model_dashboard.py           # Model monitoring dashboard generator
├── forecast_dashboard.py        # Live forecast dashboard generator
├── studies/                     # Archived analysis scripts
├���─ tests/                       # 98 unit tests
└── output/                      # Generated artifacts
    ├── model_coefs.json
    ├── model_dashboard.html
    └── forecast.html
```
