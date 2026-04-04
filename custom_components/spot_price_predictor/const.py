"""Constants for the Spot Price Predictor integration."""

from homeassistant.const import Platform

DOMAIN = "spot_price_predictor"
PLATFORMS: list[Platform] = [Platform.SENSOR]

# Configuration keys
CONF_REGION = "region"
CONF_OPERATOR = "operator"
CONF_FINGRID_API_KEY = "fingrid_api_key"
CONF_ENABLE_TIER2 = "enable_tier2"
CONF_CUSTOM_DAY_RATE = "custom_day_rate"
CONF_CUSTOM_NIGHT_RATE = "custom_night_rate"
CONF_CUSTOM_VAT = "custom_vat"
CONF_CUSTOM_ENERGY_TAX = "custom_energy_tax"
CONF_SELLER_MARGIN = "seller_margin"

# Default seller margin (EUR/kWh, excl. VAT)
DEFAULT_SELLER_MARGIN = 0.0
CONF_SEARCH_START_HOURS = "search_start_hours"
CONF_SEARCH_DURATION_HOURS = "search_duration_hours"

# Defaults for cheapest hour search window
DEFAULT_SEARCH_START_HOURS = 24    # 1d 0h = tomorrow midnight
DEFAULT_SEARCH_DURATION_HOURS = 48 # 2d 0h = 48 hour window

# Update intervals (seconds)
UPDATE_INTERVAL_WEATHER = 21600  # 6 hours
UPDATE_INTERVAL_FINGRID = 3600   # 1 hour

# Forecast window
FORECAST_HOURS = 170  # 7.08 days

# Available regions
REGIONS = {
    "finland": "Finland",
}

# Finnish electricity operators with day/night transfer tariffs (EUR/kWh)
OPERATORS = {
    "elenia": {
        "name": "Elenia",
        "day_rate": 0.0560,
        "night_rate": 0.0430,
    },
    "caruna_south": {
        "name": "Caruna South",
        "day_rate": 0.0590,
        "night_rate": 0.0450,
    },
    "caruna_north": {
        "name": "Caruna North",
        "day_rate": 0.0520,
        "night_rate": 0.0410,
    },
    "helen": {
        "name": "Helen (Helsinki)",
        "day_rate": 0.0537,
        "night_rate": 0.0403,
    },
    "custom": {
        "name": "Custom",
        "day_rate": 0.0500,
        "night_rate": 0.0400,
    },
}

# Finnish consumer pricing defaults
DEFAULT_VAT_MULTIPLIER = 1.255       # 25.5%
DEFAULT_ENERGY_TAX = 0.02253         # EUR/kWh, class I 2025

# Weather locations for Finland (capacity-weighted)
FINLAND_LOCATIONS = [
    {"name": "Raahe coast", "lat": 64.25, "lon": 24.40,
     "wind_weight": 0.22, "solar_weight": 0.08, "temp_weight": 0.05},
    {"name": "Rauma", "lat": 61.10, "lon": 21.50,
     "wind_weight": 0.12, "solar_weight": 0.22, "temp_weight": 0.08},
    {"name": "Ii-Simo", "lat": 65.30, "lon": 25.40,
     "wind_weight": 0.18, "solar_weight": 0.06, "temp_weight": 0.05},
    {"name": "Joroinen", "lat": 62.18, "lon": 27.83,
     "wind_weight": 0.05, "solar_weight": 0.20, "temp_weight": 0.10},
    {"name": "Helsinki", "lat": 60.17, "lon": 24.94,
     "wind_weight": 0.04, "solar_weight": 0.24, "temp_weight": 0.38},
    {"name": "Lapua", "lat": 62.97, "lon": 23.00,
     "wind_weight": 0.10, "solar_weight": 0.12, "temp_weight": 0.12},
    {"name": "Kolari", "lat": 67.85, "lon": 24.15,
     "wind_weight": 0.08, "solar_weight": 0.04, "temp_weight": 0.05},
]

# Demand modeling defaults
DEMAND_DEFAULTS = {
    "hdd_threshold": 17.0,
    "peak_am_center": 9,
    "peak_am_sigma": 1.8,
    "peak_pm_center": 19,
    "peak_pm_sigma": 2.0,
    "sauna_hours": [20, 21],
    "sauna_days": [4, 5],       # Friday=4, Saturday=5
    "monday_ramp_hours": [6, 7, 8, 9],
    "wind_rated_speed": 14,
    "latitude": 62.0,
}

# API endpoints
API_SAHKOTIN = "https://sahkotin.fi/prices"
API_OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
API_ELERING = "https://dashboard.elering.ee/api/nps/price"
API_ELPRISET = "https://www.elprisetjustnu.se/api/v1/prices"
API_FINGRID = "https://data.fingrid.fi/api/datasets"

# Fingrid dataset IDs
FINGRID_NUCLEAR = 188
FINGRID_FLOW_SE1 = 31
FINGRID_FLOW_SE3 = 32
FINGRID_FLOW_EE = 140

# Normalization values for Fingrid data
FINGRID_MAX_VALUES = {
    "nuclear_mw": 4372,
    "flow_fi_se1": 5500,
    "flow_fi_se3": 1200,
    "flow_fi_ee": 1016,
}
