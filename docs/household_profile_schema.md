# Household profile schema

Defines the JSON contract used by the `PV_adjusted_price` study for
modelling a household's load and PV behaviour. The schema is
deliberately **shape-statistics only** — no raw timestamps, no
absolute kWh totals, no temperature traces. A profile produced from
real HA recorder data must not allow reconstruction of the underlying
schedule, occupancy, or absolute energy bills.

## Privacy boundary

A profile JSON lives at `studies/_private/household_profile.json`
(directory is gitignored). The committed code reads the path via a
CLI argument or environment variable — never hardcodes a profile.
If no profile is supplied, the simulators fall back to the synthetic
default at `studies/sim_household_default_profile.json`, which is
calibrated to public Finnish-household statistics and never derived
from any individual user.

The author's own profile is **never** the public default. Other users
who want better fidelity can run the same extraction script against
their own HA database; their profile stays local.

## What may go in a profile

| ✅ Allowed (shape statistics, dimensionless or aggregated) |
|---|
| Hour-of-day × weekday baseload **shape** (24 × 7 matrix normalised to mean 1.0) |
| Mean baseload kW (a single number) and its monthly multiplier (12 values, dimensionless, mean 1.0) |
| EV session energy histogram (10–20 bins, frequencies only) |
| EV session start-hour histogram (24 bins, frequencies only) |
| Heat-pump electrical-kWh-vs-outdoor-°C linear regression coefficients (slope + intercept + RMSE) |
| PV-power-vs-irradiance regression coefficients and residual std |
| Deferrable-zone duty-cycle fractions (% of hours active, by month) |
| Self-consumption fraction by month (12 numbers in [0, 1]) |

| ❌ Forbidden |
|---|
| Raw kWh per timestamp |
| Specific charge or trip times |
| Indoor temperature traces |
| PV production at specific timestamps |
| Boiler/heating on/off events |
| Geographic coordinates beyond climate-zone label |
| Any data tagged with a date the user could match back to a calendar event |

## JSON shape

```json
{
  "schema_version": "1.0",
  "source": "ha_recorder" | "synthetic" | "csv_export",
  "extraction_window_days": 72,
  "climate_zone": "FI_south" | "FI_central" | "FI_north" | "generic",
  "baseload": {
    "mean_kw": 0.42,
    "shape_hour_weekday": [[24 floats], … × 7],
    "monthly_multiplier": [12 floats, mean 1.0]
  },
  "deferrables": [
    {
      "name": "boiler" | "bathroom_floor" | "entrance_floor" |
              "garage_floor" | "workshop_thermal" | "ev",
      "nominal_kw": 3.0,
      "semi_continuous": true,
      "duty_cycle_monthly": [12 floats in [0,1]],
      "energy_per_event_kwh_histogram": {
        "bins": [floats, edges], "counts": [ints]
      },
      "start_hour_histogram_24": [24 ints],
      "deadline_hour_weekday": [24 bools] | null
    },
    …
  ],
  "heat_pumps": {
    "count": 2,
    "regression_kw_vs_outdoor_c": {
      "slope": -0.07,
      "intercept": 1.4,
      "rmse": 0.18,
      "r2": 0.72
    }
  },
  "pv": {
    "installed_kwp": 8.91,
    "tilt_deg": 45.0,
    "azimuth_deg": 160.0,
    "regression_kw_vs_irradiance_w_m2": {
      "slope": 0.0085,
      "intercept": 0.05,
      "rmse": 0.42,
      "r2": 0.91
    },
    "self_consumption_fraction_monthly": [12 floats]
  }
}
```

## How to produce one

### Option A — automated extraction from HA recorder DB

```
python studies/extract_household_profile.py \
    --db /path/to/home-assistant_v2.db \
    --out studies/_private/household_profile.json \
    --sensor-map studies/_private/sensor_map.json
```

`sensor_map.json` maps the profile's canonical names (`baseload`,
`ev_power`, `boiler_power`, …) to the entity IDs in your HA setup.
The first run prints the detected statistics entities and writes a
template you edit.

### Option B — manual editing

Copy `studies/sim_household_default_profile.json` to
`studies/_private/household_profile.json` and edit values to match
your household. All fields are dimensionless or in the units shown
above; no timestamps anywhere.

### Option C — CSV from HA Energy dashboard

Export hourly long-term statistics as CSV from HA's Energy dashboard
for each canonical sensor and feed them to:

```
python studies/extract_household_profile.py --csv-dir <dir> ...
```

Same output shape.

## What the study uses the profile for

The `PV_adjusted_price` simulators (`sim_household_load.py`,
`sim_pv_production.py`) read the profile and synthesise hourly load
+ PV time series over the historical cached weather window. The
profile is the *parameterisation*; the simulation produces the
time-series. No raw user data flows through the simulator — only
the regression coefficients and shape histograms.

## Validation

The extraction script verifies, before writing a profile:

1. No JSON value is a timestamp string (`re.match(r"\d{4}-\d{2}-\d{2}")`).
2. No top-level energy total exceeds 200 MWh/year (sanity check
   against an unintended raw kWh leak).
3. All histograms have ≥10 buckets and frequencies sum to 1.0
   (otherwise it's not really anonymised).

Failing any of those raises and refuses to write.
