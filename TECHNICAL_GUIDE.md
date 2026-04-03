# Documentation: HA-spot-price-predictor

Finnish electricity spot price forecasting for Home Assistant using Ridge regression with physics-based features and multi-source data integration.

## Architecture

The system has two phases: **training** (Python, run periodically on PC) and **inference** (Home Assistant Jinja2 templates, always-on).

![Architecture Overview](docs/diagrams/architecture-overview.drawio.png)

*Source: [docs/diagrams/architecture-overview.drawio](docs/diagrams/architecture-overview.drawio)*

### Training Pipeline

```
Sahkotin API ──┐
Open-Meteo API ─┼──> Feature Engineering ──> Two-Stage Ridge ──> model_coefs.json
mgrey.se API ───┤    (28-38 features)        Regression
Elering API ────┤
Fingrid API ────┘ (optional)
```

### Home Assistant Deployment

```
REST Sensors ──> Weighted Average ──> Spot Price Forecast ──> Consumer Price
(7-11x)          Template Sensor      (Jinja2 inference)      + Control Signals
                                                               + Dashboard
```

![Data Flow](docs/diagrams/data-flow.drawio.png)

*Source: [docs/diagrams/data-flow.drawio](docs/diagrams/data-flow.drawio)*

---

## Data Sources

### Required (free, no authentication)

| Source | Purpose | Rate Limit |
|--------|---------|------------|
| [Sahkotin API](https://sahkotin.fi/prices) | FI Nord Pool spot prices (EUR/MWh) | Unlimited |
| [Open-Meteo API](https://api.open-meteo.com) | Wind (120m), solar (45 tilt), temperature | 10,000/day |
| [Open-Meteo Archive](https://archive-api.open-meteo.com) | Historical weather for training | 10,000/day |

### Cross-border price sources (free, no authentication)

| Source | Zones | Purpose |
|--------|-------|---------|
| [mgrey.se](https://mgrey.se/espot/api) | SE1, SE3 | Swedish spot prices for spread calculation |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Estonian spot prices for spread calculation |

Used to compute 7-day rolling price spreads and derive `import_potential_xx` / `export_potential_xx` features. Analysis confirmed strong autocorrelation (lag-1 weekly r=0.54-0.73, sign persistence 100%).

### Optional grid data (free API key)

| Source | Purpose |
|--------|---------|
| [Fingrid Open Data](https://data.fingrid.fi) | Nuclear production (#188), transmission capacity SE1-FI (#24), SE3-FI (#27), EE-FI (#115) |

Register for free at data.fingrid.fi. Without this key, the model trains on Tier 1+2 features only.

---

## Feature Engineering

### Tier 1: Base features (28) -- no API keys needed

| Category | Count | Features |
|----------|-------|----------|
| Supply-side | 3 | `wind_speed_weighted`, `solar_irradiance_weighted`, `temperature_weighted` |
| Time cycles | 4 | `hour_sin/cos`, `month_sin/cos` |
| Demand patterns | 8 | `double_peak_am/pm` (Gaussian, center 9h/19h), weekend variants, `sauna_hour`, `monday_ramp`, `is_holiday`, `is_weekend` |
| Thermal demand | 6 | `hdd`, `hdd_sq`, `daylight_deficit`, cross-terms (`wind_x_hdd`, `solar_x_deficit`, `temp_x_hdd`) |
| Physics supply | 3 | `wind_power_density` (density-corrected), `solar_power_temp` (NOCT model), `renewable_surplus` |
| Scarcity | 4 | `scarcity_indicator`, `wind_drought_penalty`, `cold_morning_stress`, `cold_calm_dark` (Dunkelflaute) |

### Tier 2: Cross-border trade features (6) -- no API keys needed

Derived from 7-day rolling average price spreads:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `import_potential_se1` | max(0, spread_7d_fi_se1) | Price incentive to import from SE1 |
| `import_potential_se3` | max(0, spread_7d_fi_se3) | Price incentive to import from SE3 |
| `import_potential_ee` | max(0, spread_7d_fi_ee) | Price incentive to import from EE |
| `export_potential_se1` | max(0, -spread_7d_fi_se1) | Price incentive to export to SE1 |
| `export_potential_se3` | max(0, -spread_7d_fi_se3) | Price incentive to export to SE3 |
| `export_potential_ee` | max(0, -spread_7d_fi_ee) | Price incentive to export to EE |

### Tier 3: Grid infrastructure features (0-4) -- requires Fingrid API key

| Feature | Source | Normalization |
|---------|--------|---------------|
| `nuclear_mw` | Fingrid #188 | 0-1 (0-4372 MW) |
| `import_capacity_se1` | Fingrid #24 | 0-1 (0-1500 MW) |
| `import_capacity_se3` | Fingrid #27 | 0-1 (0-1200 MW) |
| `import_capacity_ee` | Fingrid #115 | 0-1 (0-1016 MW) |

### Feature count by configuration

| Configuration | Features | API keys |
|---------------|----------|----------|
| Tier 1 only | 28 | None |
| Tier 1 + 2 | 34 | None |
| Tier 1 + 2 + 3 | 38 | 1 (Fingrid, free) |

---

## Model Architecture

### Two-stage Ridge regression with piecewise calibration

**Stage 1 (base model):**
- Linear polynomial (degree 1) on 28-38 features
- Weighted normal equations: beta = (X'WX + alpha*I)^(-1) X'Wy
- Time-decay weighting: w(t) = exp(-ln2 * age_hours / (365 * 24))
- Ridge alpha = 1.0

**Stage 2 (piecewise calibration):**
- Augmented features: stage1_prediction + 3 ReLU breakpoints
  - pw_relu_20 = max(0, s1 - 20)
  - pw_relu_40 = max(0, s1 - 40)
  - pw_relu_120 = max(0, s1 - 120)
- Corrects systematic bias at different price regimes

**Training:** 4 years historical data, 85/15 time-ordered split, batch processing (512 rows).

**Output:** `model_coefs.json` containing stage 1 + stage 2 coefficients, feature names, and tier info.

---

## Consumer Price & Control Signals

**Formula:** `(spot_EUR_MWh / 1000 + transfer_rate + energy_tax) x VAT`

Configurable per operator in `finland.yaml`. Default: Elenia (day 5.60, night 4.30 c/kWh), VAT 25.5%, energy tax 2.253 c/kWh.

**Output signals (170-hour lists):**

| Signal | Range | Use case |
|--------|-------|----------|
| `price_with_tariff_forecast` | EUR/kWh | Absolute consumer price |
| `power_control_factor_pm1` | [-1, +1] | Cheapest(+1) to most expensive(-1) |
| `power_control_factor_0_1` | [0, 1] | ON/OFF threshold control |
| `power_control_windowed_average_N_largest_0_1` | [0, 1] | Smoothed sliding window |
| `power_control_windowed_average_N_largest_pm1` | [-1, +1] | Smoothed, bipolar |

---

## Home Assistant Integration

### Holiday detection

**Option A (recommended):** HA [Workday integration](https://www.home-assistant.io/integrations/workday/) with `country: FI` -- zero maintenance, community-maintained.

**Option B:** Hardcoded holiday rules from `finland.yaml` -- more portable, requires manual updates.

### Sensors by tier

All sensors are generated by `generate_ha_yaml.py` based on the trained model's active tiers.

**Tier 1 (always present):** 7 Open-Meteo REST sensors, weighted average, spot price forecast, consumer price.

**Tier 2 (if active):** 3 cross-border price REST sensors (mgrey.se, Elering), spread calculation template.

**Tier 3 (if active):** 4 Fingrid REST sensors (nuclear, SE1, SE3, EE capacity).

---

## Regional Localization

The system is driven by a single region config file (`config/regions/finland.yaml`). To support a new region:

1. Create a new YAML file (e.g., `sweden.yaml`)
2. Define local price API, weather locations with weights, holidays, tariffs
3. Add neighboring price sources for cross-border features
4. Run training with `--region sweden`

Optional data sources are skipped gracefully if their API key is missing or if the region config omits them.

---

## Accuracy Targets

| Configuration | Expected MAE | Notes |
|---------------|-------------|-------|
| Tier 1 (28 features) | ~29-30 EUR/MWh | Matches existing v3 baseline |
| Tier 1+2 (34 features) | ~25-28 EUR/MWh | Cross-border spreads help |
| Tier 1+2+3 (38 features) | ~22-26 EUR/MWh | Nuclear + capacity for extremes |

Evaluation uses time-ordered 85/15 split with hourly, monthly, and segment-level (peak/off-peak, workday/weekend) breakdown.

---

## Project Structure

```
HA-spot-price-predictor/
├── README.md
├── TECHNICAL_GUIDE.md           # This document
├── docs/diagrams/               # draw.io architecture diagrams
├── requirements.txt
├── .env.example                 # FINGRID_API_KEY placeholder
├── .gitignore
├── src/
│   ├── train_model.py           # Main entry point
│   ├── features.py              # Dynamic feature engineering
│   ├── data_sources.py          # API clients (config-driven)
│   ├── holidays.py              # Holiday calculator
│   ├── generate_ha_yaml.py      # HA YAML generator
│   └── evaluate.py              # Metrics + visualization
├── config/regions/
│   └── finland.yaml             # Finnish region config
├── homeassistant/               # Generated HA sensor YAML
└── output/                      # Generated model artifacts
```
