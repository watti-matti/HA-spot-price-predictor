# HA-spot-price-predictor

Finnish electricity spot price forecasting for Home Assistant. Predicts Nord Pool day-ahead prices up to 170 hours (7 days) ahead using Ridge regression with physics-based features.

## Installation (HACS)

1. Open **HACS** in Home Assistant
2. Click **Integrations** → three-dot menu → **Custom repositories**
3. Add `https://github.com/watti-matti/HA-spot-price-predictor` as **Integration**
4. Search for "Spot Price Predictor" and install
5. Restart Home Assistant
6. Go to **Settings** → **Devices & Services** → **Add Integration** → **Spot Price Predictor**
7. Follow the setup wizard:
   - Select your region (Finland)
   - Select your electricity operator (Elenia, Caruna, Helen...)
   - Optionally enter a Fingrid API key for enhanced predictions

### Manual Installation

Copy `custom_components/spot_price_predictor/` to your Home Assistant `custom_components/` directory and restart.

## What You Get

After installation, these sensors are created automatically:

| Sensor | Description |
|--------|-------------|
| `sensor.spot_price_forecast` | Current hour predicted price (EUR/MWh) with 170h forecast |
| `sensor.consumer_price` | Total consumer price (EUR/kWh) including VAT, tariff, energy tax |
| `sensor.power_control_factor` | [-1, +1] signal for smart automation (+1 = cheapest) |
| `sensor.cheapest_hours` | Timestamps of cheapest upcoming hours |

Each sensor carries a `forecast` attribute with the full 170-hour prediction array, useful for ApexCharts dashboards and automations.

## Feature Tiers

The prediction model uses three tiers of data, activating automatically based on available sources:

| Tier | Features | Data Sources | API Keys |
|------|----------|-------------|----------|
| 1 | 28 (weather + demand) | Sahkotin + Open-Meteo | None |
| 1+2 | 34 (+cross-border trade) | + elprisetjustnu.se + Elering | None |
| 1+2+3 | 38 (+grid infrastructure) | + Fingrid | 1 (free) |

**Works out-of-the-box** with pre-trained coefficients (Tier 1+2+3). No training needed to start.

## Optional: Custom Training

For advanced users who want to retrain the model with their own data:

```bash
# Clone and install training tools
git clone https://github.com/watti-matti/HA-spot-price-predictor.git
cd HA-spot-price-predictor
pip install -r requirements.txt

# Train (adapts to available data sources)
python -m src.train_model --region finland

# With Fingrid data
export FINGRID_API_KEY=your_key_here
python -m src.train_model --region finland

# Evaluate model accuracy
python -m src.evaluate --region finland
```

After training, copy `output/model_coefs.json` to Home Assistant and use the `spot_price_predictor.upload_coefficients` service to update the model.

## Data Sources

| Source | Purpose | Free |
|--------|---------|------|
| [Sahkotin](https://sahkotin.fi) | FI spot prices | Yes |
| [Open-Meteo](https://open-meteo.com) | Weather forecasts (7 Finnish locations) | Yes |
| [elprisetjustnu.se](https://www.elprisetjustnu.se) | Swedish spot prices (SE1, SE3) | Yes |
| [Elering](https://dashboard.elering.ee) | Estonian spot prices (EE) | Yes |
| [Fingrid](https://data.fingrid.fi) | Nuclear production + cross-border flows | Yes (API key) |

## Documentation

See [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for detailed technical documentation including architecture diagrams, feature formulas, and model description.

## License

MIT
