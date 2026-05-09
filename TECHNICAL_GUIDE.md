# Documentation: HA-spot-price-predictor

Consumer electricity price and D(k) = CVaR duration cost forecasting for Home Assistant. Produces 170-hour consumer price forecasts (EUR/kWh) and 7-day × 24-level D(k) duration matrices for cost-optimal load scheduling, using log-linear Ridge regression with physics-based features and multi-source data integration.

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

Register for free at data.fingrid.fi. Without this key, the model trains on weather + cross-border features only (15 features).

### Nuclear outage schedule (free, no key)

| Source | Purpose |
|--------|---------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Planned nuclear outages for forward-looking capacity prediction |

---

## Feature Engineering

Model v2.0 supports up to 17 sign-validated features selected via greedy forward selection with sign constraints. The bundled default model uses weather + cross-border features (15 features). All tunable parameters are in `config/regions/finland.yaml` under the `features` section.

### Base features (11) — weather + demand, no API keys needed

| Category | Features | Coefficient sign |
|----------|----------|:---:|
| Supply | `wind_speed_weighted`, `solar_irradiance_weighted` | negative (more supply = lower price) |
| Time cycles | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | cyclic |
| Calendar | `is_holiday` | negative (lower demand) |
| Thermal demand | `hdd_sq` (squared heating degree days, threshold 17°C) | positive |
| Wind nonlinear | `wind_log_scarcity` = log1p(max(0, 8-wind)) | positive (low wind = higher price) |
| Wind x demand | `wind_calm_x_peak_am` = max(0, 6-wind) × AM peak (9h, σ=1.8) | positive |
| Wind x demand | `wind_calm_x_peak_pm` = max(0, 6-wind) × PM peak (19h, σ=2.0) | positive |

### Cross-border features (+4) — AR neighbor prices, no API keys needed

AR(2) models predict cross-border neighbor prices using workday/weekend hourly profiles with damped autoregressive deviation. This captures the European market coupling signal that drives Finnish prices.

| Feature | Source | Method |
|---------|--------|--------|
| `ar_se1` | Sweden SE1 | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `ar_se3` | Sweden SE3 | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `ar_ee` | Estonia | AR(2) on deviation from hourly daytype profile, normalized ÷100 |
| `export_potential_se3` | SE3 spread | max(0, -spread_7d_fi_se3) |

The AR models decompose neighbor prices into deterministic daily profiles (workday vs weekend, 24 hours each) plus a stochastic deviation modeled by AR(2). The AR deviation is damped (max root < 0.95) so predictions converge to the daily profile within 24 hours, ensuring stability over the full 170-hour forecast window.

### Nuclear features (+0-2) — requires Fingrid API key

| Feature | Formula | Meaning |
|---------|---------|---------|
| `nuclear_deficit` | max(0, 1 - nuclear_mw/4372) | Fraction of nuclear capacity offline |
| `nuclear_x_scarcity` | nuclear_deficit × scarcity_indicator | Nuclear outage amplifies weather-driven scarcity |

**Forward-looking outage data:** Planned outage schedules are fetched from the [Nord Pool UMM platform](https://umm.nordpoolgroup.com/) (public API, no key required). The coordinator computes per-hour nuclear availability for the forecast horizon.

### Feature count by configuration

| Configuration | Features | API keys |
|---------------|----------|----------|
| Weather only | 11 | None |
| Weather + cross-border | 15 | None |
| All sources | 17 | 1 (Fingrid, free) |

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

### Duration model: Segment-hierarchical Ridge + PAVA (cheap/peak split)

Predicts two complementary duration curves per day:

- **`dk_cheap[k-1]`** = mean spot price of the **cheapest k hours**, k=1..12 (monotone non-decreasing). CVaR at α=k/24 in the lower tail. Best achievable cost for a deferrable load that can choose its k cheapest slots.
- **`dk_peak[k-1]`** = mean spot price of the **priciest k hours**, k=1..12 (monotone non-increasing). CVaR at α=k/24 in the upper tail. Worst-case cost if the load is forced into peak hours (storage-depletion / risk-aware planning).

The legacy 24-element cumulative D(k) — useful for k=1..12, indistinguishable from random for k=13..24 — is exactly recoverable from cheap+peak via the sum identity `cheap[11] + peak[11] = 2 × daily_avg` (`src/dk_utils.py` round-trip is exact to numerical noise; tested in `tests/test_dk_consumers.py`).

**PAVA** (Pool Adjacent Violators Algorithm) is an isotonic regression method that enforces monotonicity. The cheap end requires non-decreasing PAVA; the peak end requires non-increasing PAVA (mirrored). Both are applied independently per direction after the per-segment Ridge predictions.

**Architecture:**
- 4 day segments aligned with day/night tariff: night (22-07, 9h), morning (07-12, 5h), midday (12-18, 6h), evening (18-22, 4h)
- Per `(segment, direction, duration level)`: independent Ridge model with 10 features (after Phase A retrain — currently the production model still uses single-direction Ridge per `(segment, k)` with cheap/peak derived from sorting hourly forecasts; see `src/train_model.py:train_duration_model` for the migration plan)
- Log-linear target: `log(D(k) + 55)`
- Forgetting factor λ = 0.960 (half-life 17 days, optimized via sweep)
- PAVA isotonic post-processing per direction:
  - Cheap end: enforces `dk_cheap[0] ≤ dk_cheap[1] ≤ ... ≤ dk_cheap[11]`
  - Peak end:  enforces `dk_peak[0]  ≥ dk_peak[1]  ≥ ... ≥ dk_peak[11]`
- Segment-to-day reconstruction: extract sorted prices from each segment → merge into 24 hourly forecasts → `compute_dk_cheap_peak()` produces the two 12-element arrays

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

**Formula:** `(max(0, spot_EUR_MWh) / 1000 + seller_margin + transfer_rate + energy_tax) × VAT` [EUR/kWh]

Configurable per operator in `finland.yaml`. Default: Elenia (day 3.61, night 2.20 c/kWh), VAT 25.5%, energy tax 2.325 c/kWh, seller margin 0.00 c/kWh (set from your electricity contract).

### Forecast sensors (always created)

| Sensor | State | Unit | Description |
|--------|-------|------|-------------|
| Price Forecast | Current consumer price | EUR/kWh | 170h hourly forecast with spot, consumer, weather per hour |
| Duration Forecast | Today's `dk_cheap[3]` (cheapest 4h) | EUR/kWh | 7-day × (12 cheap + 12 peak) duration curves; legacy `dk_consumer_eur_kwh[24]` retained for one transition release |

#### Price Forecast attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | array[170] | `{timestamp, spot_eur_mwh, consumer_eur_kwh, wind, solar, temp}` per hour |
| `current_spot_eur_mwh` | float | Current hour spot price (EUR/MWh) |
| `week_min_eur_kwh` | float | Minimum consumer price in forecast window |
| `week_avg_eur_kwh` | float | Average consumer price in forecast window |
| `week_max_eur_kwh` | float | Maximum consumer price in forecast window |
| `operator` | string | Configured distribution operator |
| `last_update` | datetime | Last successful data refresh |
| `data_sources_active` | string | Currently active data sources |
| `stale` | bool | True if data is older than threshold |
| `data_age_minutes` | int | Minutes since last successful fetch |

#### Duration Forecast attributes — D(k) cheap/peak

The `daily_forecast` attribute provides up to 7 days, each with both the new Phase A schema (preferred) and the legacy 24-array (deprecated, retained for one transition release).

| Attribute | Shape | Unit | Description |
|-----------|-------|------|-------------|
| `daily_forecast` | array[≤7] | — | One entry per day |
| `daily_forecast[i].date` | string | — | ISO date (YYYY-MM-DD) |
| `daily_forecast[i].weekday` | string | — | Mon–Sun |
| `daily_forecast[i].source` | string | — | `forecast` for future days, `actual` for past days reconciled from Sahkotin |
| **Phase A (preferred):** | | | |
| `daily_forecast[i].dk_cheap_eur_kwh` | float[12] | EUR/kWh | Mean of cheapest k hours, k=1..12 |
| `daily_forecast[i].dk_peak_eur_kwh`  | float[12] | EUR/kWh | Mean of priciest k hours, k=1..12 |
| `daily_forecast[i].dk_cheap_spot_eur_mwh` | float[12] | EUR/MWh | Same in spot price |
| `daily_forecast[i].dk_peak_spot_eur_mwh`  | float[12] | EUR/MWh | Same in spot price |
| **Convenience scalars (today only):** | | | |
| `today_cheap_4h_eur_kwh`, `today_cheap_8h_eur_kwh` | float | EUR/kWh | Today's cheapest 4h/8h indicators |
| `today_peak_4h_eur_kwh`,  `today_peak_1h_eur_kwh` | float | EUR/kWh | Today's worst-case 4h/1h indicators |
| **Legacy (deprecated):** | | | |
| `daily_forecast[i].dk_consumer_eur_kwh` | float[24] | EUR/kWh | Legacy cumulative ascending D(k); `dk_consumer_eur_kwh[k-1] == dk_cheap_eur_kwh[k-1]` for k=1..12 |
| `daily_forecast[i].dk_spot_eur_mwh`     | float[24] | EUR/MWh | Same in spot price |
| `forecast_days` | int | — | Number of days emitted (up to 7) |

**Access patterns:**
- Cheapest k hours of day d: `daily_forecast[d].dk_cheap_eur_kwh[k-1]` for k in 1..12 (use this for deferrable-load scheduling)
- Priciest k hours of day d: `daily_forecast[d].dk_peak_eur_kwh[k-1]` for k in 1..12 (use this for worst-case / storage planning)
- Cross-check identity: `cheap[11] + peak[11] = 2 × daily_avg` (always holds to numerical noise; foundation of the migration)
- Legacy reconstruction: any consumer still reading `dk_consumer_eur_kwh[k-1]` for k=13..24 can be served exactly from cheap+peak via `(12*(cheap[11]+peak[11]) - (24-k)*peak[24-k-1]) / k`. Implementation in `multi_load_ha_integration.py:fetch_dk_forecast` (thermal-energy-optimization repo).
- Consumer prices include per-segment tariff conversion: night hours use night transfer rate, day hours use day rate, then merged and re-sorted

### Actual price sensors (optional, Nordpool)

| Sensor | Unit | Description |
|--------|------|-------------|
| Spot Electricity Price | EUR/kWh | Actual consumer price from Nordpool with continuous timeline |
| Spot Electricity Selling Price | EUR/kWh | Spot minus PV selling commission (for solar panel owners) |

### Design principle

This integration provides **forecasts only**. The cheap/peak duration curves are the primary API for downstream systems:
- Thermal optimization / load scheduling reads `dk_cheap[k-1]` to answer "what does it cost per kWh to run for k hours today, scheduled into the cheapest slots?"
- Risk-aware planning (storage depletion, capacity reservation) reads `dk_peak[k-1]` to answer "what's the worst case if we have to run during k peak hours?"

This clean separation allows either component to be replaced independently.

The **Price Forecast** sensor provides a unified 170-hour forecast array for visualization and hourly price display. The state is the current hour's consumer price in EUR/kWh.

The **Duration Forecast** sensor exposes the cheap/peak curves for optimization decisions. The state is today's `dk_cheap_eur_kwh[3]` — the average cost of running during the cheapest 4 hours — serving as a quick-glance cost indicator. See [docs/dk_cheap_peak_migration.md](docs/dk_cheap_peak_migration.md) for the migration guide for downstream consumers.

---

## PV-aware pricing

When the user configures a non-zero `pv_capacity_kwp` (or supplies an external PV forecast entity) the coordinator augments every forecast hour with a **marginal effective price** representing the cost of running 1 additional kWh of flexible load given the household's PV production and non-flexible baseload.

### Notation

For each hour `h`:

| Symbol | Meaning |
|--------|---------|
| `b_h` | Consumer buy price (EUR/kWh) = `(spot/1000 + margin + transfer + tax) × VAT`. Always > 0 in practice. |
| `s_h` | Sell price (EUR/kWh) = `spot/1000 − pv_sell_commission − pv_export_grid_fee`. NOT clipped at zero — can be negative during deep oversupply. |
| `c_h` | Configured baseload (kWh) = `baseload_kwh_per_hour × {day_factor or night_factor}`. Constant in time per hour-of-day. |
| `p_h` | Hourly PV production (kWh), from internal estimator or external entity. Bounded in `[0, capacity_kwp · efficiency]`. |

### Marginal effective price (the metric that feeds D(k))

```
pv_avail_h = max(0, p_h − c_h)            # PV surplus available to extra load
from_pv    = min(1, pv_avail_h)           # fraction of new 1 kWh covered by PV
from_grid  = 1 − from_pv
m_h        = from_pv · s_h + from_grid · b_h
```

Properties:

- **Bounded analytically**: `m_h ∈ [s_h, b_h]` for all (b, s, p, c). No baseload-divisor pathology.
- **Captures self-consumption gain**: when PV surplus ≥ 1 kWh, `m_h = s_h` (you forgo export revenue, not retail spend).
- **Captures partial cover**: linear interpolation between sell and buy prices.
- **Captures negative-spot liability**: `s_h < 0` propagates to `m_h < 0` (rare but real).
- **Captures finite capacity**: `p_h ≤ capacity_kwp · efficiency` is a hard ceiling enforced in `pv_estimate.py`.

### PV-aware D(k) cheap/peak

Computed directly from the 24 hourly `effective_eur_kwh` values per local day, using the same `compute_dk_cheap_peak` utility as the spot-only path. Sorted ascending → `dk_cheap_pv[12]` (monotone non-decreasing); sorted descending → `dk_peak_pv[12]` (monotone non-increasing).

**Validation on 4 years of real data** (1,460 days, 5 kWp configuration): zero monotonicity violations, mean D(1) = 6.90 c/kWh (p01 = −0.7 c/kWh, p99 = 30.8 c/kWh) — bounded, realistic, optimization-ready.

### Stability invariant — open-loop wrt the optimizer

The forecast must remain a deterministic function of `(spot, weather, PV config, baseload config)` so the downstream optimizer's flexible-load decisions cannot feed back into next cycle's price forecast. Concretely:

- **`_resolve_baseload(ts)` MUST NOT call `hass.states.get` or any HA entity-read API.** Enforced by a grep test in `tests/test_coordinator_pv.py`.
- **The configured `baseload_kwh_per_hour` represents non-flexible consumption only.** Heat pump, EV, sauna, and any other optimizer-controlled load are excluded by contract.
- **`_read_external_pv_forecast()` MAY read an HA entity** because the PV forecast is weather-driven and independent of optimizer decisions — no feedback loop is created.

If a user includes a flexible-load sensor in baseload (Phase 2 will allow this opt-in with explicit warnings), the coupled forecast↔optimizer system can oscillate: optimizer schedules into cheap hours → those hours' baseload rises → next forecast shifts cheap hours → schedule chases. Phase 1 prevents this by construction.

### External PV forecast — supported attribute conventions

`_read_external_pv_forecast()` is source-agnostic. Auto-detected attribute conventions, in priority order:

| Convention | Attribute | Shape | Unit | Conversion |
|---|---|---|---|---|
| 1. Generic forecast list | `forecast` | list[dict] | kWh | direct (keys: `pv_kwh`, `kwh`, `energy`, `value`) |
| 2. Forecast.Solar Wh dict | `wh_hours` | dict {ISO ts → number} | Wh | `/ 1000` |
| 3. Forecast.Solar W dict | `watts` | dict {ISO ts → number} | W | `/ 1000` (1-hour granularity) |
| 4. EMHASS template list | `irradiance` | list[number] | W or kWh | magnitude > 50 → assume W and `/ 1000`; else kWh |

All paths return up to 168 hourly kWh values clamped to `[0, capacity_kwp · efficiency]`. Silent fallback to internal estimator if the entity is missing or none of the conventions match.

### Configuration

PV system parameters live in `consumer_pricing` adjacent fields and the optional "PV system" step of the HA config flow:

| Field | Default | Description |
|-------|---------|-------------|
| `pv_capacity_kwp` | 0 (disabled) | Installed PV peak power |
| `pv_tilt_deg` | 45 | Panel tilt; matches Open-Meteo's fetch tilt so default = no correction |
| `pv_azimuth_deg` | 180 (south) | 0=N, 90=E, 180=S, 270=W |
| `pv_system_efficiency` | 0.85 | Lumped DC/AC + soiling + losses |
| `pv_external_entity` | "" | Optional HA sensor that overrides internal estimator |
| `pv_export_grid_fee` | 0 | Extra EUR/kWh fee on exported energy (above seller commission) |
| `baseload_kwh_per_hour` | 0.8 | Constant non-flexible household consumption (~7000 kWh/yr typical) |
| `baseload_day_factor` | 1.2 | 07–22 multiplier |
| `baseload_night_factor` | 0.7 | 22–07 multiplier |

Setting `pv_capacity_kwp = 0` (the default) and leaving `pv_external_entity` empty disables all PV-aware outputs cleanly — the integration falls back to byte-identical v2.2 baseline behaviour.

### Out of scope (Phase 1)

- **Battery storage** — adds a temporal state variable; defer to Phase 2.
- **Capacitated water-filling D(k)** — for very large flexible loads relative to PV capacity, the marginal-1-kWh model under-counts by ~`(load − pv_avail) × s_h` per hour. Acceptable for typical residential loads (heat pump, EV).
- **HA energy entity for baseload** — Phase 2 only, opt-in with stability warnings, requires user-classified non-flexible sensor.
- **Per-tilt second Open-Meteo fetch** — current implementation reuses the integration's existing 45°-S irradiance fetch with scalar correction. Phase 2 may add a per-system fetch for sharper accuracy.
- **Spot-model retraining with PV** — model unchanged; PV is a post-prediction transform.

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

### Current performance (v2.0, weather + cross-border, 15 features, 4-year training, 120-day half-life)

**Hourly model:**

| Metric | Value |
|--------|:---:|
| MAE | 24.7 EUR/MWh |
| R² | 0.39 |

**Duration model (Spearman ρ, last 365 days):**

| D(k) | Use case | ρ |
|:---:|:-:|:---:|
| D(4) | Cheapest 4h | 0.908 |
| D(8) | Cheapest 8h | 0.921 |
| D(24) | Daily avg | 0.937 |

Retraining with Fingrid nuclear data (free API key) adds 2 features and improves hourly accuracy.

### Recommended retraining frequency

**Retrain every 3-4 months (quarterly).**

### How to retrain

```bash
cd HA-spot-price-predictor
pip install -r requirements.txt

# Retrain with latest data
export FINGRID_API_KEY=your_key_here  # optional
python -m src.train_model --region finland --fingrid-key YOUR_KEY

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
│   └── holidays.py              # Holiday calculator
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
│           └── finland.yaml              # Bundled config
├── ha_dashboard.yaml            # Home Assistant Lovelace dashboard (ApexCharts + Mushroom)
├── model_dashboard.py           # Model monitoring dashboard generator
├── forecast_dashboard.py        # Live forecast dashboard generator
├── studies/                     # Archived analysis scripts
├── tests/                       # 164 unit tests
└── output/                      # Generated artifacts
    ├── model_coefs.json
    ├── model_dashboard.html
    └── forecast.html
```
