# v2.11.0 — Nordpool-compatible spot-forecast sensor + PV-aware risk fields

This release adds a **Nordpool-compatible spot-price-forecast time-series sensor** and a
small set of **PV-aware risk fields** on the duration-forecast sensor. It also
introduces a **clean integration point for an external EMA consumption profiler**
(separate sibling HA module), giving the predictor a path to evolve toward
runtime-learned consumption shapes without baking any individual user's data into
the public release.

The release is the result of the `PV_adjusted_price` branch — an architectural
study that produced the cost kernel, the PV-aware CVaR computation, the consumption-
profile loader, and supporting empirical validation. The full evidence trail lives
in `studies/results/`.

## Highlights

- **New sensor: `sensor.spot_price_forecast_fi`** — Nordpool-compatible spot-price-forecast
  time series. `state` (EUR/kWh) is the current-hour spot forecast; attributes mirror the
  Nordpool integration schema (`raw_today`, `raw_tomorrow`, today/tomorrow min/avg/max),
  plus a unique-to-this-integration `raw_extended` array covering the full 170-hour
  forecast horizon. The L4 fan-chart bands are exposed under `confidence_band` for
  risk-aware consumers.
- **PV-aware daily risk** on `duration_forecast.daily_forecast[i]`. Four new fields
  per day, present only when PV is enabled:
  - `pv_aware_cvar95_eur_kwh` — tail-mean of effective EUR/kWh in the worst 5 % of
    joint price+PV scenarios for this day. The headline risk number a downstream
    scheduler reads to decide which day this week is safest for a discretionary
    load.
  - `pv_aware_self_consumed_kwh` — expected PV used on-site this day (mean across
    scenarios).
  - `pv_aware_exported_kwh` — expected PV exported to grid this day (surplus
    available for diversion to deferrable loads).
  - `pv_aware_data_provenance` — confidence flag (`synthetic_cold_start`,
    `ema_blended`, `ema_warm`, `coordinator_baseload`) propagated from the
    consumption profile underlying the CVaR computation.
- **External EMA-profile integration point**: new config field
  `consumption_profile_entity` reads an external sibling HA module's published
  consumption-profile sensor when set. When unconfigured (default), the integration
  falls back to a synthetic Finnish-typical baseload calibrated to
  `annual_consumption_kwh`.
- **Shared cost kernel library** (`pv_cost_kernel.py`) — stateless function
  `cost_distribution(buy, sell, pv, consumption, alpha)` callable from both
  predictor and downstream thermal optimisers. Same math, different inputs:
  predictor calls it with the slow-EMA reference consumption; the planner can
  call the same library with its actual schedule for an achieved-CVaR / quality-gap
  metric. The kernel is the architectural single source of truth for joint
  price + PV cost realisation.

## Empirical validation summary

`sensor.spot_price_forecast_fi` is a presentation-layer sensor over the existing
L1+L2+L3+L4 pipeline — its accuracy is the pipeline's accuracy. The 12-month
held-out walk-forward back-test (see
[`studies/results/exp_spot_price_forecast_accuracy.md`](exp_spot_price_forecast_accuracy.md)):

| Statistic | Cold-start floor (this back-test) | Warm-state target (v2.10.1 release back-test) |
|---|:---:|:---:|
| MAE | 22.5 EUR/MWh | ~10 EUR/MWh |
| RMSE | 33.9 | — |
| R² | +0.71 | ~+0.91 |
| Extreme-hour MAE (|spot| > 100) | 34.2 | ~15.4 |
| 90 % band coverage | 74 % | ~92 % |
| 50 % band coverage | 49 % | — |

Cold-start floor = fresh HA install, no calibrator history. Warm-state target
= after 30–60 days of operation when `HourlyBiasCorrector` and
`HourlyFanChartCalibrator` are warm. Production sees the cold-start floor for
the first weeks and converges toward the warm-state target.

A sample-week illustration of forecast vs realised is at
[`studies/results/figures/spot_price_forecast_sample_week.png`](figures/spot_price_forecast_sample_week.png).

Additional supporting analyses on the branch:

- [`exp_pv_scenarios_backtest.md`](exp_pv_scenarios_backtest.md) — Phase A PV
  cloud-bootstrap validation. 91.7 % realised coverage at the 90 % band target on
  a 360-day held-out walk-forward.
- [`exp_pv_aware_cvar_backtest.md`](exp_pv_aware_cvar_backtest.md) — annual back-test
  on 1211 days using the user's real consumption shape; quantifies the per-day
  CVaR and the optimisation-yield gap between flat-baseload and shaped
  consumption.
- [`exp_share_by_rank.md`](exp_share_by_rank.md) — empirical share-by-rank
  validation. On 958 days of post-PV data the rank-shift concentration ratio is
  1.8× (top-4-cheapest receives 17.7 % of deferrable mass; bottom-4-most-expensive
  receives 10.0 %). Strongest in winter (2.0×); weakest in summer (0.7× reverse).
- [`exp_bootstrap_learning_curve.md`](exp_bootstrap_learning_curve.md) — bootstrap
  learning curve. The share-by-rank signal converges within 14 days; absolute
  baseline takes a full year (or HDH-regression bootstrap if a user is willing).

## Architectural plans landed on the branch

For future-self reference and for downstream consumers:

- [`pv_adjusted_price_plan.md`](pv_adjusted_price_plan.md) — parent plan.
- [`pv_adjusted_price_coupling_rules.md`](pv_adjusted_price_coupling_rules.md) —
  coupling rules R1–R7 that constrain how the predictor / planner / EMA module
  interact to avoid feedback oscillation.
- [`pv_adjusted_cvar_plan.md`](pv_adjusted_cvar_plan.md) — Phases A–E of the
  PV-aware CVaR architecture.
- [`pv_adjusted_buy_sell_duration_curves.md`](pv_adjusted_buy_sell_duration_curves.md) —
  audit table classifying the candidate PV-aware fields by genuine information
  content; rationale for the 4-field final surface.

## Breaking changes

**None.** This release is strictly additive:

- All v2.10.1 sensor entities continue to exist and publish the same attributes.
- All v2.10.1 config options keep their semantics.
- No artifact-schema changes (`spike_model_default.json`, `seasonal_components_default.json`,
  `solar_submodel_default.json` are unchanged).
- DtACI calibrator state files (`hourly_bias.json`, `hourly_fan_chart.json`,
  `refit_monitor.json`) carry over unchanged.

Consumers wired to Nordpool's `state` + `raw_today` / `raw_tomorrow` schema can
now optionally swap to `sensor.spot_price_forecast_fi` for an extended forecast
horizon (170 h vs Nordpool's 48 h), or run them side-by-side without
interference.

## Migration / no-op for downstream consumers

- **HACS users**: auto-update via HACS. After the next coordinator cycle (≤ 6 h
  by default) the new sensor appears in HA, and the four new daily-forecast
  attributes start appearing on PV-enabled installs.
- **EMHASS / template automations**: unchanged. The schema of existing sensors
  is preserved.
- **HACS dashboards (`ha_dashboard.yaml`)**: unchanged.
- **Sibling HA-consumption-profiler module**: when developed and configured
  via `consumption_profile_entity`, the `pv_aware_data_provenance` flag on
  duration_forecast rows transitions from `synthetic_cold_start` through
  `ema_blended` to `ema_warm` over the first 30–60 days of operation.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/pv_cost_kernel.py` | **NEW** — shared stateless cost-realisation + CVaR library. 50 statements, 100 % test coverage. |
| `custom_components/spot_price_predictor/pv_aware_cvar.py` | **NEW** — per-day PV-aware CVaR via parametric scenario sampler. 27 statements, 100 % test coverage. |
| `custom_components/spot_price_predictor/consumption_profile_loader.py` | **NEW** — external EMA-profile reader + synthetic fallback. 58 statements, 100 % test coverage. |
| `custom_components/spot_price_predictor/sensor.py` | Added `SpotPriceForecastSensor` class; wired into entity registration; `sw_version` 2.10.1 → 2.11.0. |
| `custom_components/spot_price_predictor/const.py` | Added `CONF_CONSUMPTION_PROFILE_ENTITY` + default. |
| `custom_components/spot_price_predictor/config_flow.py` | Added the new config field to both initial setup and Options flow. |
| `custom_components/spot_price_predictor/coordinator.py` | Wired the PV-aware CVaR computation into `_compute_daily_forecast` (strictly additive); reads `consumption_profile_entity` with synthetic fallback. |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.10.1 → 2.11.0. |
| `docs/household_profile_schema.md` | **NEW** — privacy contract for derived household profiles. |
| `scripts/check_no_private_data.py` | **NEW** — pre-commit guard against accidentally committing private user data. |
| `studies/_private/README.md` | **NEW** — explains the local-only directory for user-specific profiles. |
| `studies/lib_fingrid_csv.py`, `studies/lib_ha_db.py` | **NEW** — generic data loaders (Fingrid CSV, HA recorder DB) used by extraction scripts. |
| `studies/extract_household_profile.py`, `studies/extract_household_profile_from_fingrid.py` | **NEW** — profile extraction scripts. Public, generic; output stays under `studies/_private/`. |
| `studies/sim_pv_scenarios.py` | **NEW** — empirical PV cloud-bootstrap scenario generator (for validation studies). |
| `studies/exp_*.py`, `studies/results/exp_*.md`, `studies/results/figures/*.png` | **NEW** — empirical validation studies and their results. |
| `tests/test_pv_cost_kernel.py`, `tests/test_pv_aware_cvar.py`, `tests/test_consumption_profile_loader.py`, `tests/test_spot_price_forecast_fi.py` | **NEW** — 62 tests covering the new modules. |
| `README.md`, `TECHNICAL_GUIDE.md`, `TEKNINEN_TOTEUTUS.md` | Documentation updated for v2.11.0 — new sensor schema, new attribute reference, new config option, empirical accuracy section, project structure tree. |

## Test status

`python -m pytest tests/` — **471 passed, 4 warnings, 0 failed** at the v2.11.0
commit. New pure modules (`pv_cost_kernel.py`, `pv_aware_cvar.py`,
`consumption_profile_loader.py`) measured at **100 % statement coverage** with
135/135 statements covered.

## Acceptance criteria

The release is acceptable to merge to `main` because:

- **Strictly additive.** No v2.10.1 consumer breaks. The Nordpool-compatible
  sensor lives alongside the existing `sensor.price_forecast`; the four
  PV-aware fields are added to existing day-entries without removing anything.
- **Empirically validated.** Cold-start MAE 22.5 EUR/MWh on 12 months of held-out
  data; warm-state target 10 EUR/MWh confirmed on the v2.10.1 same-data
  back-test. PV scenario coverage 91.7 % vs 90 % target.
- **Privacy contract enforced.** A pre-commit hook
  (`scripts/check_no_private_data.py`) prevents accidentally committing private
  user data (`.db`, profile JSONs, `_raw/` CSVs). The synthetic profile shipped
  with the integration is derived only from public Finnish-household statistics.
- **100 % test coverage** on new pure modules. Coordinator wiring is tested at
  integration level (the daily-forecast emission path exercises the new code
  with PV configured).
- **Documentation in lockstep**: README / TECHNICAL_GUIDE / TEKNINEN_TOTEUTUS
  all updated to reflect the v2.11.0 interface.

## Honest caveats

1. **Cold-start coverage of the L4 fan-chart band is 74 %** vs target 90 %. The
   fan widens to target over the first 30–60 days of operation as
   `HourlyFanChartCalibrator` warms. Production users won't see "calibrated
   bands" on day 1; this is documented honestly in the empirical-accuracy
   results doc.
2. **`CONF_CONSUMPTION_PROFILE_ENTITY` is a forward-looking integration point.**
   The sibling `HA-consumption-profiler` module is not part of this release — it
   lives in a separate repo and will be a separate ship. Users who don't have it
   running see `pv_aware_data_provenance: synthetic_cold_start` and get a
   meaningful (if approximate) PV-aware CVaR until they (or a third party) ship
   the EMA module.
3. **Planner repo's architecture diagrams are stale** w.r.t. v2.11.0 interfaces.
   Recommended follow-up but not blocking — the planner's own next-release
   work will pick up the diagrams alongside the EMHASS 0.17.3 analysis
   (`HA-energy-needs-planner/studies/EMHASS_0173_integration_analysis.md`).
