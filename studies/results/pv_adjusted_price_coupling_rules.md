# Predictor ↔ planner coupling — rules to prevent oscillation

Branch: `PV_adjusted_price`. Addendum to `pv_adjusted_price_plan.md`.

The user runs two systems in series:

```
[ HA-spot-price-predictor ]  →  sensor.price_forecast (hourly, 170h)
              │                  sensor.duration_forecast (daily D(k), 7d)
              ▼
[ HA-energy-needs-planner ]  →  sensor.enp_<load>_daily_kwh_target_7d
              │                  sensor.enp_<hp>_setpoint_trajectory_7d
              ▼
[ EMHASS ]                   →  15-min MILP dispatch
              │
              ▼
[ HA actuators ]             →  setpoints, switches, OCPP currents
```

The planner already implements **per-load PV-aware effective pricing
internally** (`enp/effective_price.py`): it reads the raw consumer
price, raw PV forecast, raw baseline load, and computes a different
`λ_eff(load, hour)` for each load based on its kW and the PV
coverage fraction `α(h) = min(1, PV_surplus(h) / load_kW)`.

This means our integration's existing per-hour `effective_eur_kwh`
field — and any new PV-adjusted derivative we add in this study —
**must not become the planner's price input**. Otherwise PV is
counted once by the predictor (with a single `baseload`) and again
by the planner (per-load α), and the planner loses the load-specific
fidelity it was built around.

It also means the predictor must never close the loop by reading
planner outputs back into its own baseload input. That's a feedback
edge between two control systems with different cadences, and the
common failure mode for that pattern is sustained oscillation around
the cheap-hour boundary.

## Hard rules

**R1. `consumer_eur_kwh` is the canonical price for downstream optimisation.**
It stays as a per-hour, **raw-tariff, PV-unaware** number. Never
deprecated, never changes semantics. The planner reads this field
and is entitled to assume it has not been adjusted.

**R2. `effective_eur_kwh` and any new PV-aware fields are diagnostic.**
For dashboards, the consumer-facing UI, and the integration's own
duration model — never for the planner or any other optimiser. This
must be stated explicitly in the schema doc the user-facing release
publishes.

**R3. The predictor does not read planner output.**
No coordinator code reads `sensor.enp_*` or any planner-derived
quantity. Baseload input to the integration comes from one of:
  - configured `annual_kwh / 8760` × monthly multiplier (current behaviour),
  - the user's `CONF_CONSUMPTION_ENTITY` (an HA sensor, calibrated
    independently of the planner — e.g. `power_load_no_var_loads`),
  - the static household profile (Phase 0 output of this study).
Never the planner's `daily_kwh_target_7d`. Enforced by a grep-based
test in CI.

**R4. The predictor does not read its own previous PV-aware output.**
Effective-price values used in the current cycle's duration model
already exist; no chain of derived → re-derived values that could
drift.

**R5. New sensor attributes are additive, not substitutive.**
Anything we add in this study (e.g. `regime_class`,
`pv_self_consumption_p50_kwh`, `forecast_savings_p50_eur`) appears
*alongside* the existing canonical fields. Schema-stable.

**R6. Cadence separation is preserved.**
Predictor runs hourly. Planner runs 1–6 hourly. EMHASS dispatches
15-min. Nothing we add changes any of these. New fields must be
safe to read at the slowest consumer cadence without harm.

## Oscillation analysis (what could go wrong)

The Finnish spot price is exogenous — one household's 11 kW EV does
not move the FI day-ahead clearing. So the *price* loop is open.

The risk is at the **effective-price loop**, which is closed inside
each household:

```
effective_eur_kwh(h)  =  f(buy(h), sell(h), pv(h), baseload(h))
```

Today, `baseload(h)` is a flat or weakly time-varying input
(monthly-scaled flat kWh, or an HA sensor with EMA smoothing). It
is **independent of the planner's decisions**. The loop is open.

The hazardous design — which we are explicitly *not* doing — would
be:

```
baseload_next(h)  =  flat_baseload(h)  +  planner_schedule(h)
```

That would close the loop. The signature pathology is:

1. Cycle N: predictor sees flat baseload, computes low effective
   price at 13:00 (PV peak).
2. Planner schedules EV at 13:00.
3. Cycle N+1: predictor sees planner's 11 kW added at 13:00. PV
   no longer covers baseload → effective price at 13:00 jumps.
4. Planner re-plans, moves EV to 14:00.
5. Cycle N+2: 13:00 is cheap again, 14:00 is now expensive.
6. EV oscillates between 13:00 and 14:00 indefinitely.

R3 prevents step 3. The integration never sees the planner's
schedule, so its `baseload(h)` is stable across predictor cycles.

## What this means for Phase 5 (sensor surface)

The candidate fields previously proposed:

| Field | Safe to add? | Rationale |
|---|---|---|
| `regime_class` (A/B/C/D) | ✅ | Diagnostic. Derived only from forecast + PV expectation. |
| `recommended_shift_priority` (rank 1..7) | ✅ | Diagnostic for the user's dashboard, not a control input. Planner must NOT use it. |
| `pv_self_consumption_p50_kwh` | ✅ | Derived from PV forecast and *static* baseload, not planner output. |
| `forecast_savings_p50_eur` | ✅ | Same. |
| `cvar95_regret_eur` | ✅ | Derived from forecast bands only. |
| `planner_aware_baseload_kwh` | ❌ | Would require reading `sensor.enp_*`. Violates R3. Don't build it. |

## Coupling validation tests

To be added in Phase 3 (`studies/exp_pv_self_consumption.py`):

**Test T1 — open-loop verification.**
With the predictor's current baseload input held constant, the
planner's allocation is a deterministic function of forecast +
PV. Two consecutive predictor runs over the same data must
produce the same allocation. Asserts no hidden state coupling
inside the predictor that depends on planner output.

**Test T2 — adversarial baseload sweep.**
Sweep the baseload from 0.2 to 1.5 kW; for each, run the
end-to-end forecast → planner → EMHASS pipeline and record the
EV charge hour. Plot. If the plot has hysteresis-like jumps or
multiple stable values for the same baseload, the loop has
non-monotonicity that could oscillate under noise. Should be
monotone or flat.

**Test T3 — feedback-edge audit.**
A small Python script that greps the integration source for any
import of `sensor.enp_*`, `enp_*`, or `planner_*`. CI test that
fails the build if any such reference appears.

## Seasonal representation note

As of v2.10.1 the predictor's improvements skew toward
extreme-hour accuracy (cross-border features) and tail-band
tightness (L4 σ shrinkage from 22.81 to 18.63). For the planner,
the practical effects per season:

- **Winter (Dec–Feb)**: heat-pump electrical demand is highest,
  COP is lowest, PV is negligible. The planner's daily kWh budget
  is dominated by heating. The cross-border features matter most
  here because cold snaps drive FI spikes. Phase 3 of this study
  will need to extrapolate from our spring data — flagged in the
  plan.
- **Spring (Mar–May)**: real data window. Heat-pump load tapers,
  PV ramps up from ~5% to ~30% self-coverage. Best window for
  the framework's first validation.
- **Summer (Jun–Aug)**: PV peak. The planner's per-load α is
  near 1 for daytime loads, so effective price ≈ feed-in tariff.
  Spot prices in FI are also lowest. Marginal value of scheduling
  flexibility is at its yearly minimum.
- **Autumn (Sep–Nov)**: heat-pump load rising, PV falling, spot
  rising. Asymmetric mirror of spring.

The seasonal interface concern is real but it does **not** require
a new contract — the planner already reads PV forecast and the
predictor publishes the cross-border-informed price. Both systems
handle seasonality through their own mechanisms.

What's worth adding to the study: a **per-season replay** of the
planner's allocations using the v2.10.1 forecast versus the v2.8.1
forecast on the same back-test window. Quantifies how much the
predictor upgrade helps the *planner's* decisions, not just the
forecast metric. This is a meaningful end-to-end validation that
the cross-border features delivered downstream value.

## Out of scope (oscillation control)

- Active damping (e.g. EMA on effective_eur_kwh between cycles).
  Not needed if R3 holds.
- Cross-system commit/release locks. Predictor and planner are
  separate processes with separate cadences; the architecture
  already gives us time-scale separation.
- Model-predictive control of the predictor itself. Out of scope
  for this study; would be a different project.
