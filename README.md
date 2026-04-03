# HA-spot-price-predictor

Finnish electricity spot price forecasting for Home Assistant. Predicts Nord Pool day-ahead prices up to 170 hours (7 days) ahead using Ridge regression with physics-based features.

## Quick Start

```bash
# 1. Clone and install
git clone git@github-watti:watti-matti/HA-spot-price-predictor.git
cd HA-spot-price-predictor
pip install -r requirements.txt

# 2. Train the model (Tier 1 only, no API keys needed)
python -m src.train_model --region finland

# 3. With cross-border prices (Tier 1+2, still no keys needed)
python -m src.train_model --region finland

# 4. With Fingrid grid data (Tier 1+2+3)
export FINGRID_API_KEY=your_key_here
python -m src.train_model --region finland
```

## Feature Tiers

| Tier | Features | Data Sources | Keys |
|------|----------|-------------|------|
| 1 | 28 (weather + demand) | Sahkotin + Open-Meteo | None |
| 1+2 | 34 (+cross-border trade) | + mgrey.se + Elering | None |
| 1+2+3 | 38 (+grid infrastructure) | + Fingrid | 1 (free) |

Start with Tier 1 only. Add tiers by making more data available -- the model adapts automatically.

## Data Sources

- **Sahkotin** (sahkotin.fi) -- FI spot prices, free
- **Open-Meteo** (open-meteo.com) -- Weather forecasts, free
- **mgrey.se** -- Swedish spot prices (SE1, SE3), free
- **Elering** (dashboard.elering.ee) -- Estonian spot prices (EE), free
- **Fingrid** (data.fingrid.fi) -- Nuclear + transmission capacity, free API key

## How It Works

1. **Training** (Python, run periodically): Fetches 4 years of historical data, engineers 28-38 physics-based features, trains a two-stage Ridge regression model, exports coefficients to JSON.

2. **Inference** (Home Assistant, always-on): REST sensors fetch live weather + grid data. Jinja2 templates evaluate the model using exported coefficients. Produces 170-hour price forecasts and 5 control signals for home automation.

## Project Structure

```
src/
  train_model.py     # Main training pipeline
  features.py        # Dynamic feature engineering (28-38 features)
  data_sources.py    # API clients for all data sources
  holidays.py        # Finnish holiday calculator
  evaluate.py        # Accuracy metrics + HTML report
config/regions/
  finland.yaml       # All settings: APIs, locations, tariffs, holidays
homeassistant/       # Generated HA sensor YAML (by generate_ha_yaml.py)
output/              # Model artifacts (model_coefs.json, parquet files)
```

## Documentation

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed technical documentation including architecture diagrams, feature formulas, and model description.

## License

MIT
