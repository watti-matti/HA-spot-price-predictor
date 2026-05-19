# PV-adapted CVaR statistics with EMA consumption estimator

Branch: `PV_adjusted_price`. Companion to
[`pv_adjusted_price_plan.md`](pv_adjusted_price_plan.md) and
[`pv_adjusted_price_coupling_rules.md`](pv_adjusted_price_coupling_rules.md).

This plan covers the *forward-looking risk* arm of the study: a
PV-aware CVaR statistic for the household's effective electricity
cost, computable in the predictor without coupling to the planner,
made stable by a slow-EMA consumption profile.

The *backward-looking optimisation quality* arm (regret of realised
schedule vs hindsight-optimal) is covered in the main plan and is
strictly downstream of this one.

## Goal

Produce, for each rolling 7-day forecast horizon, a single number:

```
PV_aware_CVaR95_eur_kwh
```

defined as the 95 % expected-shortfall of the household's
effective per-kWh electricity cost, accounting for the joint
distribution of:

1. day-ahead spot price (existing P5..P95 fan + L4 GPD tails),
2. PV production (irradiance forecast + cloud-cover bootstrap),
3. household consumption (slow-EMA profile from the user's
   own observed consumption sensor).

It must be **independent of any planner decision** so it cannot
oscillate against the downstream thermal optimiser. It must be
**meaningful for users who do not run a planner** so it stands
alone as a risk thermometer.

## Why this needs a joint computation

CVaR is non-linear in its arguments. For a cost realisation

```
C_h = max(L_h − PV_h, 0) · buy_h  −  max(PV_h − L_h, 0) · sell_h
```

`CVaR(C)` cannot be derived from `CVaR(price)` and `CVaR(PV)`
separately because the worst-tail joint events — cold cloudy
spike-price days — are precisely where price and PV are *negatively
correlated*, and the linear combiners that produce marginal
CVaRs erase that correlation.

So we sample the joint distribution end-to-end and take CVaR on
the realised cost samples. There is no shortcut.

## Inputs

| Quantity | Owner | Distribution source | Shape |
|---|---|---|---|
| Spot price `buy_h` | predictor | L4 GPD POT sampler (already exists for the fan chart) | `[N_paths, 170 hours]` |
| Sell price `sell_h` | predictor | `_spot_to_sell_eur_kwh(buy_h)` — deterministic transform | `[N_paths, 170 hours]` |
| PV production `PV_h` | **NEW** — cloud-cover block bootstrap | resample weekly cloud sequences from the same calendar month across 2023–2026, run `pv_estimate.estimate_pv_kwh_per_hour` for each path | `[N_paths, 170 hours]` |
| Consumption `L_h` | **NEW** — slow-EMA forecaster | 24 × 7 × 12 profile (hour × weekday × month) updated daily with τ ≥ 14 days from the user's `CONF_CONSUMPTION_ENTITY` | `[170 hours]` — single deterministic path |

Note that `L_h` is treated as a **deterministic profile** in the
forward CVaR. Modelling consumption uncertainty as a fourth random
dimension is tempting but would expand the sample space and
add a noise channel that is not the user's question. The user's
question is "given my typical consumption shape, what is the cost
risk under price + PV uncertainty?" — three random dimensions,
not four.

`N_paths`: 500 by default (matches the existing fan-chart sampler).

## The shared cost kernel

A small stateless library that both predictor and planner can
call. Lives at `custom_components/spot_price_predictor/pv_cost_kernel.py`
(public, tested), copied to the planner repo's requirements or
vendored as a single file. One function:

```python
def cost_distribution(
    buy_eur_kwh:     np.ndarray,        # [N_paths, n_hours]
    sell_eur_kwh:    np.ndarray,        # [N_paths, n_hours]
    pv_kwh:          np.ndarray,        # [N_paths, n_hours]
    consumption_kwh: np.ndarray,        # [n_hours] OR [N_paths, n_hours]
    *, alpha: float = 0.05,
) -> CostDistribution:
    """Per-path realised cost + summary statistics.

    Returns:
        cost_per_path  [N_paths]              total horizon cost
        cost_per_kwh   [N_paths]              cost divided by sum(consumption)
        mean_eur_kwh                          expected cost
        cvar_eur_kwh                          CVaR_alpha of cost_per_kwh
        cvar_eur                              CVaR_alpha of cost_per_path
        var_eur_kwh                           VaR_alpha (the threshold)
    """
```

Implementation is straightforward — vectorised numpy, ~30 lines
of code. The kernel knows nothing about who is calling it or
what the consumption profile means. That separation is what makes
it safe to reuse across repos without import cycles or shared
state.

## What the predictor will publish

Three new attributes on `sensor.duration_forecast.daily_forecast[i]`
(one row per day in the next 7 days):

| Attribute | Meaning |
|---|---|
| `pv_aware_mean_eur_kwh` | expected effective per-kWh cost for the day under joint price+PV uncertainty, using the EMA consumption profile |
| `pv_aware_cvar95_eur_kwh` | CVaR_5% of effective per-kWh cost — the tail-mean over the worst 5 % of joint scenarios |
| `pv_aware_band_eur_kwh` | `[P5, P50, P95]` per kWh — for dashboard fan plotting |

Plus one top-level attribute on `sensor.price_forecast`:

| Attribute | Meaning |
|---|---|
| `pv_aware_scenarios_available` | bool; whether the predictor was able to generate joint scenarios this cycle (false on cold start or weather-API failure) |

The raw `forecast[i].consumer_eur_kwh` field stays canonical and
PV-unaware per coupling rule R1.

The full scenario tensor is too large to live on a sensor attribute
(170 × 500 = 85k floats per cycle), so it is held internally by
the coordinator. If the planner needs scenarios for its own
achieved-CVaR computation, the right plumbing is a HA service
call (`spot_price_predictor.get_scenarios`) that returns the
tensor on demand. Out of scope for this study; revisit in Phase 5.

## The consumption EMA forecaster

Module: `custom_components/spot_price_predictor/consumption_forecaster.py` (new).

State:

```python
profile[hour: 0..23, weekday: 0..6, month: 0..11]  -> kWh expected
```

Update rule (daily, at 04:00 local):

```python
for hour, kwh_observed in yesterday_consumption_by_hour:
    weekday, month = derived_from_yesterday_date
    cell = profile[hour, weekday, month]
    alpha = 1.0 / TAU_DAYS           # τ = 21 days by default
    deviation = kwh_observed - cell
    if abs(deviation) <= 3.0 * profile_sigma[hour, weekday, month]:
        profile[hour, weekday, month] = cell + alpha * deviation
        profile_sigma[hour, weekday, month] = update_sigma(...)
    # else: anomaly — sample is discarded from the EMA but logged.
```

Bootstrapping:

- **Days 0–14 since install**: profile = synthetic Finnish-household
  shape calibrated to `CONF_ANNUAL_CONSUMPTION_KWH`. CVaR sensor
  carries `data_provenance: "synthetic_cold_start"` and dashboards
  show a low-confidence flag.
- **Days 15–30**: blended profile, α scaled to give equal weight
  to synthetic and observed at day 22.
- **Day 30 onward**: pure EMA on observations.

Persistence: profile + sigma + last_update_ts written to
`output/consumption_profile.json` (gitignored), reloaded on
coordinator startup.

Anomaly guard:
- Per-cell σ tracked via the same EMA framework.
- Samples beyond 3σ are excluded from the update for that cell
  but counted in a coordinator diagnostic
  (`sensor.spot_predictor_anomalous_consumption_hours_24h`).

## Stability argument (load-bearing)

Closed-loop gain analysis for the predictor↔planner system:

| Signal | Symbol | Time scale |
|---|---|---|
| Planner re-plan period | T_p | ≈ 1 hour |
| Predictor cycle | T_pr | ≈ 1 hour |
| Consumption EMA time constant | τ | ≥ 14 days = 336 hours |

Per-cycle gain from a planner schedule change into the predictor's
consumption profile is bounded by `1/τ` (single-pole low-pass).
At τ = 21 days = 504 hours, a 100 % consumption swing in one cycle
propagates 0.2 % into the profile by the next cycle.

For closed-loop oscillation, the loop gain at the planner's
characteristic frequency would need to be ≥ 1. The transfer
function from planner output to predictor input is

```
H(s) = 1 / (1 + s·τ)
```

evaluated at ω_p = 2π / T_p ≈ 6.3 rad/hour, with `τ ω_p = 504 × 6.3 ≈ 3170`.
`|H(jω_p)| ≈ 1/3170 ≈ 0.0003`. Even with unity gain at every other
stage of the loop, the closed loop is stable by ~70 dB.

This is the same control-theory argument that makes the existing
HourlyBiasCorrector EMA safe.

The minimum-τ choice (14 days = 336 hours) gives `|H| ≈ 0.0005` —
still ≥ 60 dB margin. We could ship τ = 14 d as default; choosing
21 d gives a buffer against cases where the planner's `T_p` is
shorter than 1 hour.

## Phases

### Phase A — Joint PV scenario generator

`studies/sim_pv_scenarios.py`. Reads `pv_estimate` + cached
Open-Meteo irradiance. Implements block-bootstrap (weekly cloud
sequences) and produces an `[N_paths, n_hours]` tensor matching
the price-fan tensor's shape.

Validation:
- Mean across paths within 5 % of the point-forecast PV.
- Marginal P5 / P95 of PV across paths is consistent with the
  irradiance forecast's reasonable band.
- For a small historical sample, realised PV falls within the
  generated 90 % band ≥ 88 % of the time (target 90 %).

### Phase B — Consumption EMA forecaster

`consumption_forecaster.py`. As specified above. Tested in
isolation against the 72-day HA history (private path):

- Profile MAE < 15 % at hour level after 30 days of training.
- Profile RMSE < 25 % at hour level.
- Anomaly guard correctly catches injected outliers in unit tests.

### Phase C — Cost kernel

`pv_cost_kernel.py`. As specified above. Unit-tested with:

- Limit case `pv_kwh = 0` → reduces to `buy_eur_kwh · consumption`.
- Limit case `pv_kwh ≫ consumption` → reduces to net export
  `sell_eur_kwh · (pv − consumption)`.
- CVaR identity: `cvar(constant_cost_array) == constant`.
- Monotonicity: increasing PV uniformly should decrease mean cost.

### Phase D — Predictor integration

Coordinator wires Phases A–C into the existing cycle. Each
prediction run now also produces a `pv_aware_cvar95_eur_kwh` per
day and exposes it on `duration_forecast.daily_forecast[i]`. No
existing field is removed or has its semantics changed (R1).

Acceptance:
- Existing forecast metrics (MAE, R², fan-chart coverage) unchanged
  vs v2.10.1.
- New attributes are populated when scenarios available, `null`
  otherwise. No exceptions.
- Coordinator overhead < 0.5 s per cycle on the reference machine.

### Phase E — Stability test

`tests/test_pv_aware_cvar_stability.py`. Simulates 90 days of
predictor cycles with a synthetic planner that aggressively
reschedules in response to the published PV-aware D(k). Asserts:

- The slow-EMA consumption profile never moves more than
  `0.5 × 1/τ` per cycle (i.e. ≤ 0.25 % per day at τ = 21 d).
- The published CVaR series has bounded variation: total variation
  over the 90 days is < 5 × the mean CVaR (a "no perpetual
  oscillation" criterion).

This is the mechanical check on the analytical stability argument.

### Phase F — Reference vs achieved gap (downstream, planner-side)

Not in this study's scope; covered by the broader plan. Documented
here because it justifies the kernel-as-library design: the same
`pv_cost_kernel.cost_distribution()` call from the planner side,
with the planner's actual schedule as `consumption_kwh`, gives the
**achieved CVaR**. Subtracting this from the predictor's reference
CVaR yields the **optimisation quality gap** as a EUR/year number
the user can monitor.

## Sensor surface (Phase D output)

Already detailed above. Reiterated for the schema audit trail:

```yaml
sensor.duration_forecast.daily_forecast[i]:
  # existing fields …
  pv_aware_mean_eur_kwh:        float
  pv_aware_cvar95_eur_kwh:      float
  pv_aware_band_eur_kwh:        [P5, P50, P95]
  data_provenance:              "ema_warm" | "ema_blended" | "synthetic_cold_start"

sensor.price_forecast (top-level):
  pv_aware_scenarios_available: bool
  consumption_profile_age_days: float
```

All additive — coupling rule R5 holds.

## Acceptance criteria for the study

This plan ships when:

1. Phases A–E all pass their internal tests.
2. The Phase 0 + Phase 3 back-test from the parent plan shows that
   ranking days by `pv_aware_cvar95_eur_kwh` produces a meaningful
   classification — i.e. the top-1 cheapest-CVaR day in each week
   indeed gives ≥ 10 % cost reduction over the worst-CVaR day under
   a controlled deferrable-shift strategy.
3. The stability test (Phase E) passes with margin ≥ 10×.
4. No existing forecast metric regresses (overall MAE, extreme MAE,
   90 % band coverage all within 2 % of v2.10.1 numbers).

## Out of scope

- Stochastic consumption modelling (treating L_h as random).
  Possible future extension; not needed for the first version.
- Battery dispatch within the cost kernel. The reference setup
  has no battery; planner-side battery models would call the
  kernel with an effective `consumption_kwh − battery_discharge_kwh`
  vector, which is already supported by the function signature.
- Multi-household joint CVaR (community PV). Out of scope.
- CVaR at α other than 5 %. The kernel supports it; the published
  sensor fixes α = 0.05 for simplicity.

## Sequencing into the broader study

```
Phase 0 (parent plan): baseload extraction          ─┐
Phase A (this plan):   PV scenarios                  │  parallel
Phase B (this plan):   consumption EMA               ┘

Phase C (this plan):   cost kernel                    ↓ depends on A+B

Phase D (this plan):   predictor integration         ↓ depends on C
Phase E (this plan):   stability test                ↓

Phase 1–3 (parent):    load+PV simulators, schedulers, back-test
Phase 4 (parent):      planner-replay validation
Phase 5 (parent):      sensor-surface integration
```

Total to a complete first iteration of PV-aware CVaR on the
sensor: ~2 working weeks if uninterrupted.
