# Documentation: HA-spot-price-predictor (v2.4.0)

Consumer electricity price and D(k) = CVaR duration cost forecasting for Home Assistant. Produces 170-hour consumer price forecasts (EUR/kWh) and 7-day D(k) cheap/peak duration curves for cost-optimal load scheduling, using log-linear Ridge regression with physics-based features and multi-source data integration. Optionally augments every forecast hour with a PV-aware marginal effective price `m_h` and parallel PV-aware D(k) curves when the user configures household solar.

## Architecture

The system has two phases: **training** (Python, run periodically on PC) and **inference** (Home Assistant custom integration, always-on). v2.3 adds an optional **post-prediction PV transform** that does not require retraining.

### Training Pipeline (unchanged from v2.2)

```
Sahkotin API  ──┐
Open-Meteo API ─┼──> Feature Engineering ──> Log-linear Ridge ──> model_coefs.json
Elpriset API ───┤    (9 sign-validated        + Power stretch      (hourly + duration
Elering API ────┤     features after v2.2     + Duration model)     model coefficients)
Fingrid API ────┘ pruning, optional)
```

### Home Assistant Deployment

```
Open-Meteo  ──┐
Elpriset    ──┼──> Feature Builder ──> Hourly Model  ──> Spot/Consumer Forecast (170h)
Elering     ──┤    (pure Python)       + Duration Model   + D(k) cheap/peak (7d)
Fingrid     ──┘                        (pure Python)      │
Nord Pool UMM ─────────────────────────┘                  │
                                                          │
                                                          v
                              (v2.3) PV-aware Transform ──┴─> + effective_eur_kwh per hour
                              [optional]                       + dk_cheap_pv / dk_peak_pv (7d)
                              ↑
                              └── Open-Meteo irradiance (internal)
                                  OR pv_external_entity (Forecast.Solar / EMHASS / template)
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

| Source | Zones fetched | Used in v2.2 model? |
|--------|---------------|---------------------|
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE3 | Yes — `ar_se3` + `export_potential_se3` |
| [Elpriset](https://www.elprisetjustnu.se/api/v1/prices) | SE1 | Fetched for spread/historical context only; `ar_se1` was pruned in v2.2 (collinear with `ar_se3`, since the FI↔SE transmission corridor terminates in SE3) |
| [Elering API](https://dashboard.elering.ee/api/nps/price) | EE | Yes — `ar_ee` |

Used to fit AR(2) models on cross-border price deviations from hourly daytype profiles. Analysis confirmed strong autocorrelation (lag-1 weekly r = 0.54–0.73, sign persistence 100%).

### Optional grid data (free API key)

| Source | Purpose |
|--------|---------|
| [Fingrid Open Data](https://data.fingrid.fi) | Nuclear production (#188) for the `nuclear_x_scarcity` interaction feature |

Register for free at data.fingrid.fi. Without the Fingrid key, training uses only weather + cross-border data and the resulting model omits the `nuclear_x_scarcity` feature (8 of the bundled 9). Inference can still load the bundled v2.2 9-feature model — `nuclear_x_scarcity` contributes 0 when no nuclear data is fed in.

### Nuclear outage schedule (free, no key)

| Source | Purpose |
|--------|---------|
| [Nord Pool UMM](https://ummapi.nordpoolgroup.com/messages) | Planned nuclear outages for forward-looking capacity prediction |

---

## Feature Engineering

The training pipeline can compute up to 17 sign-validated candidate features, but a v2.2 leave-one-out redundancy sweep showed that **only 9 of them carry independent signal** — the other 8 were either collinear with retained features or contributed nothing measurable to walk-forward MAE. The bundled v2.2 model (also shipped unchanged in v2.3) uses exactly these 9 features. All tunable parameters live in `config/regions/finland.yaml` under the `features` section.

### The bundled 9 features

| # | Feature | Category | Source | Sign | Role |
|---|---------|----------|--------|:---:|------|
| 1 | `wind_speed_weighted` | Supply | Open-Meteo (7 capacity-weighted FI sites) | − | Dominant price driver — more wind, lower spot |
| 2 | `month_cos` | Seasonality | Calendar | cyclic | Annual heating-load seasonality (winter peak) |
| 3 | `is_holiday` | Calendar | Holiday calculator / HA Workday | − | Public holidays → lower industrial demand |
| 4 | `hdd_sq` | Thermal demand | Open-Meteo temperature | + | `max(0, 17°C − T)²` — nonlinear cold amplification |
| 5 | `wind_log_scarcity` | Wind nonlinear | Open-Meteo | + | `log1p(max(0, 8 − wind))` — sharp price jump when wind drops below 8 m/s |
| 6 | `ar_se3` | Cross-border | elprisetjustnu.se SE3 | + | AR(2) deviation from SE3 hourly daytype profile, normalized ÷100 — captures FI↔SE3 transmission coupling |
| 7 | `ar_ee` | Cross-border | Elering EE | + | AR(2) deviation from EE hourly daytype profile, normalized ÷100 — captures FI↔EE coupling |
| 8 | `export_potential_se3` | Cross-border | SE3 spread | − | `max(0, −spread_7d_fi_se3)` — when FI is cheaper than SE3, FI→SE3 export pulls FI price up |
| 9 | `nuclear_x_scarcity` | Nuclear | Fingrid #188 + Nord Pool UMM | + | `nuclear_deficit × wind_log_scarcity` — outage amplifies weather scarcity |

The AR(2) cross-border models decompose neighbour prices into a deterministic daily profile (workday vs weekend, 24 hours each) plus a stochastic AR(2) deviation. The AR deviation is damped (max root < 0.95) so multi-step forecasts converge to the daily profile within ~24 hours, ensuring stability over the full 170-hour horizon.

`nuclear_x_scarcity` requires a free Fingrid API key for live nuclear production data; planned outage schedules come from the [Nord Pool UMM platform](https://umm.nordpoolgroup.com/) (public API, no key). Without the Fingrid key, training omits this one feature (model has 8 features) and inference can still load the bundled 9-feature model — the feature simply contributes 0 when no nuclear data is fed in.

### Features pruned in v2.2 (and why)

The leave-one-out sweep removed 8 candidates from the v2.0/v2.1 17-feature set:

| Pruned feature | Reason |
|---|---|
| `solar_irradiance_weighted` | Finland's solar share too small to move spot prices; coefficient was indistinguishable from zero. |
| `hour_sin`, `hour_cos` | Hour-of-day pattern is fully captured by the AR(2) daytype profiles in `ar_se3` / `ar_ee`. |
| `month_sin` | Month-of-year captured well enough by `month_cos` alone (heating peak ↔ cosine extremum). |
| `wind_calm_x_peak_am`, `wind_calm_x_peak_pm` | Collinear with `wind_log_scarcity`; their incremental MAE benefit was negative under the sweep. |
| `ar_se1` | Strongly collinear with `ar_se3`. The Fenno-Skan / FennoSkan-2 cables connect FI to SE3, not SE1, so SE3 dominates the transmission signal. SE1 is still fetched for spread context but no longer enters the model. |
| `nuclear_deficit` | Standalone nuclear deficit added little once `nuclear_x_scarcity` (the interaction term) was retained. Including both caused multicollinearity. |

Effect of pruning (training test split, 4-year history): MAE 23.94 → 20.07 EUR/MWh (−16%); R² 0.515 → 0.719 (+40%). Walk-forward MAE on a 180-day holdout is 20.99 EUR/MWh, well below the AR(2)-only neighbour-price floor of 37.82.

### Bundled vs retraining: feature count cheat sheet

The bundled `model_coefs_default.json` is fixed at the 9-feature v2.2 model regardless of which APIs the runtime can reach. If you retrain locally (`python -m src.train_model …`), the produced model uses the largest subset of the 9 that your data sources support:

| Available data | Trained features | Notes |
|---|:---:|---|
| Open-Meteo only | 5 | Drops `ar_se3`, `ar_ee`, `export_potential_se3`, `nuclear_x_scarcity` |
| + elprisetjustnu.se (SE3) + Elering (EE) | 8 | Drops `nuclear_x_scarcity` |
| + Fingrid (free key) | **9** | Full bundled-model feature set |

---

## Model Architecture

### Hourly model: Log-linear Ridge regression

**Prediction formula:** `price = scale × max(0, exp(Σ coef_i × feat_i + intercept) − log_offset) ^ power`

The log transform naturally handles the nonlinear price-scarcity relationship: nearly linear at low prices, exponential amplification at high prices.

- Ridge regression on `log(price + log_offset)` target (v2.2 retuned `log_offset` from 55 → 100 to better fit the 2025–2026 price regime)
- **9 sign-validated features** in the v2.2 bundled model (16 candidate features pruned by leave-one-out redundancy sweep)
- Power stretch (`scale`, `exponent`) fitted via Nelder-Mead on the test set
- Time-decay weighting: half-life 120 days (configurable per region)
- Ridge α = 50, augmented matrix (no penalty on intercept)

**Training:** 4+ years of historical data, 85/15 time-ordered split, batched normal-equation solve (512 rows). v2.3 ships the same coefficient file as v2.2 — PV-aware outputs are computed by the coordinator, not the trained model.

### Duration model: Segment-hierarchical Ridge + PAVA (cheap/peak split)

Predicts two complementary duration curves per day:

- **`dk_cheap[k-1]`** = mean spot price of the **cheapest k hours**, k=1..12 (monotone non-decreasing). CVaR at α=k/24 in the lower tail. Best achievable cost for a deferrable load that can choose its k cheapest slots.
- **`dk_peak[k-1]`** = mean spot price of the **priciest k hours**, k=1..12 (monotone non-increasing). CVaR at α=k/24 in the upper tail. Worst-case cost if the load is forced into peak hours (storage-depletion / risk-aware planning).

The legacy 24-element cumulative D(k) — useful for k=1..12, indistinguishable from random for k=13..24 — is exactly recoverable from cheap+peak via the sum identity `cheap[11] + peak[11] = 2 × daily_avg` (`src/dk_utils.py` round-trip is exact to numerical noise; tested in `tests/test_dk_consumers.py`).

**PAVA** (Pool Adjacent Violators Algorithm) is an isotonic regression method that enforces monotonicity. The cheap end requires non-decreasing PAVA; the peak end requires non-increasing PAVA (mirrored). Both are applied independently per direction after the per-segment Ridge predictions.

**Architecture (Phase A dual cheap/peak training):**
- 4 day segments aligned with day/night tariff boundaries: night (22-07, 9 levels), morning (07-12, 5 levels), midday (12-18, 6 levels), evening (18-22, 4 levels). Total = 24 hourly slots.
- Per `(segment, direction, k)`: independent Ridge model. Each segment carries `cheap_models` (k = 1..n_levels) and `peak_models` (k = 1..n_levels). Total bundled Ridge fits = 2 × (9 + 5 + 6 + 4) = **48 small models**.
- Per-segment **12 features** (segment-level aggregates over the segment's hours):
  `wind_mean`, `solar_mean`, `hdd_mean`, `se3_mean`, `se1_mean`, `nuclear_deficit`, `is_workday`, `month_sin`, `month_cos`, `wind_log_scarcity`, `net_load_mean`, `net_load_squared_mean`. The last two are zero-padded when Fingrid net-load forecasts are unavailable, matching the training-side fallback.
- Log-linear target with v2.2-retuned offset: `log(D(k) + 100)`
- Forgetting factor λ = 0.960 (half-life 17 days, optimized via sweep)
- PAVA isotonic post-processing per direction:
  - Cheap end: enforces `dk_cheap[0] ≤ dk_cheap[1] ≤ … ≤ dk_cheap[11]`
  - Peak end:  enforces `dk_peak[0]  ≥ dk_peak[1]  ≥ … ≥ dk_peak[11]`
- Segment-to-day reconstruction: each segment yields its own sorted-price vector; segments merge into 24 hourly forecasts; `compute_dk_cheap_peak()` produces the two 12-element arrays exposed at the sensor.

**Performance (Spearman rank correlation, last 365 days):**

| Duration level | Use case | ρ |
|:-:|:-:|:-:|
| D(1) | Cheapest 1h | 0.898 |
| D(4) | Cheapest 4h | 0.930 |
| D(8) | Cheapest 8h | 0.937 |
| D(24) | Daily average | 0.940 |

**Output:** `model_coefs.json` containing hourly Ridge coefficients (9-feature v2.2 bundled), AR(2) parameters for `ar_se3` and `ar_ee`, and the dual cheap/peak duration-model coefficients (48 segment-direction-k Ridge fits).

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

When the user configures a non-zero `pv_capacity_kwp` (or supplies an external PV forecast entity) the coordinator augments every forecast hour with a **marginal effective price** representing the cost of running 1 additional kWh of flexible load given the household's PV production and the typical-total baseload.

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
- **The configured `baseload_kwh_per_hour` should represent the user's typical TOTAL hourly consumption** — bill-derived total demand including all loads (heat pump, EV, sauna, water heater, etc.). Static configuration cannot create optimizer feedback because it doesn't depend on observed consumption; the actual stability requirement is only about what the predictor reads from HA.
- **`_read_external_pv_forecast()` MAY read an HA entity** because the PV forecast is weather-driven and independent of optimizer decisions — no feedback loop is created.

#### Why "typical total" not "non-flexible only" — worked example

Sunny noon, 4 kWh PV, heat-pump household with 16 000 kWh/yr typical demand:

| Scenario | baseload | pv_avail | m_h | Behaviour |
|---|---|---|---|---|
| **A: non-flex only (~0.5 kWh/h)** | 0.5 | 3.5 kWh | ≈ 4 c/kWh | Over-optimistic. Forecast claims all PV is free for extra load. EMHASS schedules heat pump there + further loads on top → second load actually pulls 16 c/kWh from grid. **Systematic optimism bias.** |
| **B: typical total (~1.83 kWh/h × seasonal)** | ~1.83 | 2.17 kWh | ≈ 4 c/kWh | Self-consistent. Forecast assumes typical demand (heat pump etc.) is happening; EMHASS plans around that; reality matches assumption; equilibrium. |

With PV at only 2 kWh, Case B correctly returns m_h ≈ 14 c/kWh (PV mostly absorbed by typical demand, only 0.17 kWh headroom). Case A would have returned ~10 c/kWh — still optimistic. Both cases satisfy the stability invariant because both are static config; Case B is **more accurate** because the PV/grid ratio that drives the marginal cost is genuinely a function of total demand, not non-flex demand.

#### v2.3 → v2.3.1 doc-fix note

The v2.3.0 release shipped with help text saying baseload should be "non-flexible only" and an emphatic warning to exclude heat pump, EV, etc. **That guidance was incorrect** — it conflated two separate stability concerns. The actual stability requirement is only about what the predictor READS (no optimizer-influenced HA entities), not what the static configured value REPRESENTS. Users following the old guidance get the Case-A optimism bias on heat-pump days. Please raise your `baseload_kwh_per_hour` to typical TOTAL hourly consumption (≈ annual_bill_kWh / 8760).

#### v2.4.0 baseload schema overhaul

v2.4.0 replaces the three v2.3 fields (`baseload_kwh_per_hour`, `baseload_day_factor`, `baseload_night_factor`) with two friendlier ones:

- **`annual_consumption_kwh`** (default 12 000) — the user's typical TOTAL annual household demand from the electricity bill. Single user-friendly number. Internally:

  ```
  baseload(h) = annual_consumption_kwh / 8760 × monthly_factor[month_of_h]
              ≡ annual_consumption_kwh / 365 / 24 × monthly_factor[month_of_h]
  ```

  where `monthly_factor` is the 12-element Finnish residential non-electric-heating seasonal profile defined in `const.py` (`FINLAND_RESIDENTIAL_MONTHLY_FACTORS`). Sum of factors = 12.00 exactly (normalization invariant); range ≈ ±19 % around the mean (Finnish 60°N latitude pattern: lighting-driven winter peak Dec/Jan, vacation/long-day trough Jul). Source: literature-derived from VTT Publications 289, Adato Energia DSO standard load profiles ("tyyppikäyrät"), Statistics Finland "Energy consumption in households" survey 2024. **TODO**: replace with verbatim values from Fingrid Open Data dataset #360 (BE03 typing curve) in a v2.4.x patch.

- **`consumption_entity`** (optional) — any HA consumption sensor; the integration auto-detects type and smooths internally:

  | Detected type | Detection (HA attrs) | Smoothing strategy |
  |---|---|---|
  | Cumulative-kWh counter | `unit = kWh`, `state_class = total_increasing` | 14-day delta divided by 14 → daily kWh |
  | Daily/monthly `utility_meter` | `state_class = total` with cycle attribute | History-window average of daily totals |
  | Instantaneous power | `unit = W` or `kW`, `device_class = power` | `statistics_during_period(28 d, mean)` × 24 |
  | Unknown | (fallback) | Silent fallback to `annual_consumption_kwh` config; log warning |

  Smoothed value cached in `.storage/spot_price_predictor_consumption_cache.json`, recomputed at most once per day (not every coordinator cycle). 5 % hysteresis dead-band on the cached value prevents minor sensor noise from re-triggering coordinator updates.

**Stability re-check under the v2.4 schema**:

- **Default mode** (`consumption_entity = ""`): `baseload(h)` is a deterministic function of `(annual_consumption_kwh, h)` only — no HA entity reads, fully open-loop. Identical safety property to Phase 1.
- **HA-sensor mode**: 14-day rolling average dampens a single-day perturbation to `1/14 ≈ 7 %`. Combined with the 5 % hysteresis dead-band, EMHASS rescheduling a 5 kWh load between days produces `5/14 ≈ 0.36 kWh` rolling change, only ~3 % of a 12 kWh/day baseline — within the dead-band, so the cached baseload value doesn't move and EMHASS sees a stable forecast.

**Migration from v2.3.x**: when a config entry only carries the legacy `baseload_kwh_per_hour` field, the coordinator's `__init__` infers the equivalent annual value:

```
inferred_annual_kwh = baseload_kwh_per_hour
                    × ((day_factor × 15 + night_factor × 9) / 24)
                    × 8760
```

and logs an INFO line. The legacy fields stay in `entry.data` untouched until the user opens the Options dialog and re-saves, at which point they are dropped cleanly. A user who configured the v2.3.0 default (`baseload_kwh_per_hour = 0.8`, day_factor = 1.2, night_factor = 0.7) gets `inferred_annual_kwh ≈ 7660 kWh/yr` — clearly low for a typical Finnish heat-pump house, prompting them to re-tune to their actual bill.

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
| `annual_consumption_kwh` (v2.4) | 12 000 | Typical TOTAL annual household demand from the bill, including PV self-consumption AND optimizer-controlled loads. Multiplied by the built-in Finnish residential monthly seasonal profile to get per-hour baseload. |
| `consumption_entity` (v2.4, optional) | "" | Any HA consumption sensor (cumulative-kWh counter, daily/monthly utility_meter, instantaneous power). The integration auto-detects type and smooths internally over 14 days with 5 % hysteresis. Recommended placeholder: `sensor.energy_yesterday`. |
| `baseload_kwh_per_hour` (v2.3, legacy) | 0.8 | Auto-migrated to `annual_consumption_kwh` on load. Kept for backwards compatibility; existing v2.3.x deployments continue to work unchanged until the user re-saves Options. |
| `baseload_day_factor` (v2.3, legacy) | 1.2 | Auto-migrated; ignored in v2.4. |
| `baseload_night_factor` (v2.3, legacy) | 0.7 | Auto-migrated; ignored in v2.4. |

Setting `pv_capacity_kwp = 0` (the default) and leaving `pv_external_entity` empty disables all PV-aware outputs cleanly — the integration produces byte-identical no-PV outputs, equivalent to v2.2 behaviour.

### Out of scope (Phase 1)

- **Battery storage** — adds a temporal state variable; defer to Phase 2.
- **Capacitated water-filling D(k)** — for very large flexible loads relative to PV capacity, the marginal-1-kWh model under-counts by ~`(load − pv_avail) × s_h` per hour. Acceptable for typical residential loads (heat pump, EV).
- **Battery storage** — adds a temporal state variable; defer to a future release.
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

### Current performance (v2.3.0 — bundled v2.2 9-feature pruned model, 4-year training)

**Hourly model:**

| Metric | v2.1 (17 features) | v2.2 / v2.3 (9 features) | Change |
|---|:---:|:---:|:---:|
| MAE (training test split) | 23.94 EUR/MWh | **20.07 EUR/MWh** | −16% |
| R² | 0.515 | **0.719** | +40% |
| Walk-forward MAE (180-day holdout) | — | **20.99 EUR/MWh** | vs. AR(2) floor 37.82 |

**Duration model (Spearman ρ, last 365 days):**

| D(k) | Use case | ρ |
|:---:|:-:|:---:|
| D(4) | Cheapest 4h | 0.930 |
| D(8) | Cheapest 8h | 0.937 |
| D(24) | Daily avg | 0.940 |

**v2.3 PV-aware D(k) validation** (post-prediction transform, no retraining; 5 kWp / 1 kWh-h baseload reference, 4-year backtest on 1,460 complete days): zero PAVA-monotonicity violations, PV-aware D(1) mean 6.90 c/kWh (std 6.0), bounded analytically in `[s_h, b_h]` per hour. Estimated annual savings vs grid-only D(4) ≈ 600 EUR/yr.

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
├── tests/                       # 267 unit tests (33 PV-aware in v2.3)
└── output/                      # Generated artifacts
    ├── model_coefs.json
    ├── model_dashboard.html
    └── forecast.html
```
