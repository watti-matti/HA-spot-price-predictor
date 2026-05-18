"""Constants for the Spot Price Predictor integration."""

from __future__ import annotations

from homeassistant.const import Platform

DOMAIN = "spot_price_predictor"
PLATFORMS: list[Platform] = [Platform.SENSOR]

# Configuration keys
CONF_REGION = "region"
CONF_OPERATOR = "operator"
CONF_FINGRID_API_KEY = "fingrid_api_key"
CONF_ENABLE_NEIGHBOR_PRICES = "enable_neighbor_prices"
CONF_CUSTOM_DAY_RATE = "custom_day_rate"
CONF_CUSTOM_NIGHT_RATE = "custom_night_rate"
CONF_CUSTOM_VAT = "custom_vat"
CONF_CUSTOM_ENERGY_TAX = "custom_energy_tax"
CONF_SELLER_MARGIN = "seller_margin"

# Default seller margin (EUR/kWh, excl. VAT)
DEFAULT_SELLER_MARGIN = 0.0
CONF_NORDPOOL_ENTITY = "nordpool_entity"
CONF_ENABLE_PV_SELLING = "enable_pv_selling"
CONF_PV_SELL_COMMISSION = "pv_sell_commission"

DEFAULT_PV_SELL_COMMISSION = 0.002   # EUR/kWh (0.2 c/kWh)

# ---------------------------------------------------------------------------
# Household PV production forecast (Phase 1, post-prediction transform).
#
# When `pv_capacity_kwp > 0` the coordinator augments each forecast hour with
# a marginal effective price `m_h` representing the cost of running 1
# additional kWh of flexible load at that hour given (PV, baseload). PV-aware
# D(k) cheap/peak duration curves are computed directly from the sorted
# hourly `m_h` per day. See TECHNICAL_GUIDE.md for the theorem and stability
# invariant.
#
# IMPORTANT: in Phase 1 baseload is a constant (with optional day/night
# shape). The coordinator never reads HA energy entities for baseload —
# this guarantees the price forecast is open-loop with respect to the
# downstream optimizer (no schedule oscillation).
# ---------------------------------------------------------------------------
CONF_PV_CAPACITY_KWP        = "pv_capacity_kwp"          # 0 disables PV-aware outputs
CONF_PV_TILT_DEG            = "pv_tilt_deg"
CONF_PV_AZIMUTH_DEG         = "pv_azimuth_deg"
CONF_PV_SYSTEM_EFFICIENCY   = "pv_system_efficiency"
CONF_PV_EXTERNAL_ENTITY     = "pv_external_entity"       # optional, overrides internal
CONF_PV_EXPORT_GRID_FEE     = "pv_export_grid_fee"       # extra EUR/kWh fee on exported energy
CONF_BASELOAD_KWH_PER_HOUR  = "baseload_kwh_per_hour"
CONF_BASELOAD_DAY_FACTOR    = "baseload_day_factor"
CONF_BASELOAD_NIGHT_FACTOR  = "baseload_night_factor"

DEFAULT_PV_CAPACITY_KWP        = 0.0     # 0 = PV awareness disabled
DEFAULT_PV_TILT_DEG            = 45.0    # matches Open-Meteo fetch
DEFAULT_PV_AZIMUTH_DEG         = 180.0   # south
DEFAULT_PV_SYSTEM_EFFICIENCY   = 0.85
DEFAULT_PV_EXPORT_GRID_FEE     = 0.0     # EUR/kWh extra fee on export (above sell commission)

# ---------------------------------------------------------------------------
# v2.3 baseload schema (legacy, kept for backwards compatibility / migration)
# ---------------------------------------------------------------------------
# These three fields are superseded by `annual_consumption_kwh` /
# `consumption_entity` in v2.4. Migration is automatic in coordinator
# `__init__` — when a config entry only carries the legacy fields, they
# are converted to an inferred `annual_consumption_kwh` and an INFO log
# line is emitted. The legacy defaults are preserved unchanged so existing
# v2.3.x deployments continue to behave identically until the user
# explicitly updates their config via Options.
DEFAULT_BASELOAD_KWH_PER_HOUR  = 0.8     # legacy default — re-tune per bill
DEFAULT_BASELOAD_DAY_FACTOR    = 1.2
DEFAULT_BASELOAD_NIGHT_FACTOR  = 0.7

# ---------------------------------------------------------------------------
# v2.4 baseload schema — annual kWh + optional HA consumption entity
# ---------------------------------------------------------------------------
# `annual_consumption_kwh` represents the user's typical TOTAL annual
# household demand (the bill-derived total, including PV self-consumption
# and all optimizer-controlled loads). The integration computes per-hour
# baseload as:
#
#   baseload(h) = annual_consumption_kwh / 8760 × monthly_factor[month(h)]
#
# `consumption_entity` is an optional HA sensor (any cumulative-kWh counter,
# `utility_meter` daily/monthly counter, instantaneous power sensor, etc.).
# When set, the integration auto-detects the sensor type, queries HA's
# recorder/statistics API for a long-window (14 to 28 day) smoothed mean,
# applies 5 % hysteresis, and uses that as the daily-kWh baseline instead
# of the static config. The smoothed value persists across HA restarts.
#
# Stability invariant: baseload is a deterministic function of (config +
# long-window-EMA, time). 14-day smoothing + 5% hysteresis prevents
# EMHASS's daily decisions from propagating back into the forecast.

CONF_ANNUAL_CONSUMPTION_KWH = "annual_consumption_kwh"
CONF_CONSUMPTION_ENTITY     = "consumption_entity"

DEFAULT_ANNUAL_CONSUMPTION_KWH = 12000   # mid-range Finnish single-family with heat pump
DEFAULT_CONSUMPTION_ENTITY     = ""

# Smoothing window for `consumption_entity` (days). 14 days is long
# enough to wash out EMHASS's daily scheduling decisions while still
# tracking seasonal drift on a meaningful timescale.
CONSUMPTION_SMOOTHING_DAYS = 14
CONSUMPTION_HYSTERESIS_PCT = 0.05        # 5 % dead-band on cached value

# Monthly normalization factors for the Finnish residential non-electric-
# heating load profile (Fingrid Datahub category BE03 equivalent, "Type 1").
# Multiply annual_kwh / 12 by this factor to get the typical monthly kWh.
# Sum is exactly 12.00. Variation ±19 % around the mean — characteristic of
# Finnish 60°N latitude where lighting load drives strong winter peak and
# vacation/long-day combination drives July trough.
#
# Source: literature-derived from Finnish residential load profile research
# (VTT Publications 289 "Load research and load estimation in electricity
# distribution", Adato Energia DSO standard load profiles ["tyyppikäyrät"],
# Statistics Finland "Energy consumption in households" survey, 2024).
# These are NOT verbatim from a single publication — no Finnish source
# publishes a clean 12-element normalized array. They are calibrated to
# the published BE03 shape (winter 1.10–1.18, summer trough at 0.80 in
# July, shoulder months near 1.00).
#
# TODO (v2.4.x patch): replace with verbatim values from Fingrid Open Data
# dataset #360 ("Sähkönkulutus Suomen jakeluverkoissa käyttäjäryhmittäin",
# https://data.fingrid.fi/en/datasets/360) filtered to category BE03,
# averaged across the most recent 12 months. Requires Fingrid API key
# (the same key already used for nuclear data).
FINLAND_RESIDENTIAL_MONTHLY_FACTORS = [
    1.18,  # Jan — long nights, lighting peak
    1.12,  # Feb
    1.08,  # Mar
    1.00,  # Apr — shoulder
    0.92,  # May
    0.85,  # Jun — vacations begin, long days
    0.80,  # Jul — vacation trough
    0.85,  # Aug
    0.93,  # Sep — back-to-school
    1.02,  # Oct — nights getting longer
    1.10,  # Nov
    1.15,  # Dec — holidays + lighting
]
assert abs(sum(FINLAND_RESIDENTIAL_MONTHLY_FACTORS) - 12.0) < 1e-9, \
    "Monthly factors must sum to exactly 12 for normalization invariant"

# DtACI online calibration layer (Phase B v2). Off by default; enabling
# wires up per-(direction, k) DtACI on FI consumer-price D(i) statistics
# and runs neighbour-zone (SE1, SE3, EE) bundles for bias diagnostics.
CONF_ENABLE_DTACI_DK = "enable_dtaci_dk"
DEFAULT_ENABLE_DTACI_DK = False
DTACI_TARGET_COVERAGE = 0.9  # 90% prediction intervals
# Zones whose D(i) statistics get a DtACI bundle. FI drives sensor bands;
# the neighbour bundles produce bias estimates that can later be fed back
# into the AR(2) features (separate enhancement).
DTACI_ZONES = ("fi", "se1", "se3", "ee")

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
        "day_rate": 0.0361,
        "night_rate": 0.0220,
    },
    "caruna_espoo": {
        "name": "Caruna Espoo",
        "day_rate": 0.0221,
        "night_rate": 0.0221,
    },
    "caruna_north": {
        "name": "Caruna North",
        "day_rate": 0.0407,
        "night_rate": 0.0249,
    },
    "helen": {
        "name": "Helen",
        "day_rate": 0.0354,
        "night_rate": 0.0354,
    },
    "custom": {
        "name": "Custom",
        "day_rate": 0.0500,
        "night_rate": 0.0400,
    },
}

# Region timezone (from region config, handles DST automatically)
DEFAULT_TIMEZONE = "Europe/Helsinki"

# Finnish consumer pricing defaults
DEFAULT_VAT_MULTIPLIER = 1.255       # 25.5%
DEFAULT_ENERGY_TAX = 0.02325         # EUR/kWh, class I 2026

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
FINGRID_NUCLEAR = 188                    # Real-time nuclear production (3-min)
FINGRID_CONSUMPTION_FORECAST = 165       # Once-a-day consumption forecast (15-min)
FINGRID_WIND_FORECAST = 246              # Once-a-day wind generation forecast (15-min)
FINGRID_SOLAR_FORECAST = 247             # Once-a-day solar generation forecast (15-min)

# Normalization values for Fingrid data (max_value used for bounded
# fraction-of-capacity series like nuclear_mw)
FINGRID_MAX_VALUES = {
    "nuclear_mw": 4372,
}

# Nord Pool UMM (Urgent Market Messages) — public API, no key required
API_NORDPOOL_UMM = "https://ummapi.nordpoolgroup.com/messages"
UMM_FUEL_TYPE_NUCLEAR = 14
UMM_AREA_FINLAND = "FI"
FINNISH_NUCLEAR_CAPACITY_MW = 4394
