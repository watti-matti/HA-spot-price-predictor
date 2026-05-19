# PV-adjusted effective price — study plan

Branch: `PV_adjusted_price`. Status: planning (no code yet).

The headline question is **how much of a real household's load can be
genuinely shifted to cheap hours, and how much PV can realistically
be self-consumed**, evaluated under forecast uncertainty rather than
perfect foresight.

The reference household is the integration author's own setup —
Tampere, Finland (61.3 °N, 23.75 °E), already running EMHASS. That
makes the load model concrete instead of generic.

---

## Reference household

### PV system

| Item | Value |
|---|---|
| Modules | 22 × LONGi LR5-54HPH-405M |
| Installed peak | **8.91 kWp** |
| Inverter | Growatt 8000TL US, grid-tied, no battery |
| Tilt | 45° |
| Azimuth | **160° (SW-facing, +20° from south)** |
| Production peak | shifted ≈80 min later than a 180° array — better evening overlap |

### Loads (the EMHASS deferrable basket — real numbers)

| # | Load | Nominal P (W) | Energy / day | Window today | Comfort cost if missed |
|---|---|---:|---|---|---|
| 0 | Hot-water boiler (resistive) | 3 000 | setpoint-driven, ≈8–15 kWh | 24h, deadline 07:00 | medium (cold water) |
| 1 | Bathroom floor | 1 134 | setpoint-driven | 24h, deadline 07:00 | low–medium |
| 2 | Hallway floor | 1 123 | setpoint-driven | 24h, deadline 07:00 | low |
| 3 | Garage floor | 1 688 | setpoint-driven | 24h, deadline 07:00 | low |
| 4 | Workshop thermal mass | 2 300 | setpoint-driven | 24h, deadline 07:00 | low |
| 5 | EV (OCPP, 3-phase) | up to 11 000 | trip-dependent | weekdays **hard 07:00**, weekends free | very high if missed |

Aggregate deferrable nameplate ≈ **20.2 kW**. Realistically the
thermal loads run intermittently to track setpoints; the EV is the
single biggest movable block.

### Heat pumps (continuous, NOT deferrable in the EMHASS sense)

| | Pump 1 (downstairs) | Pump 2 (upstairs) |
|---|---|---|
| Model | Mitsubishi MSZ-LN35VG | Mitsubishi MSZ-LN25VG |
| Thermal kW (nominal) | 3.5 | 2.5 |
| Electrical kW (peak) | 1.81 | 1.81 |
| COP @ 0 °C | 4.0 | 3.5 |
| COP slope | +0.07 / °C | +0.06 / °C |

These run continuously and modulate; from the integration's point of
view they look like a non-flat baseload that's **outdoor-temperature
dependent**. Critical: at −15 °C their COP halves, so the same
indoor comfort costs roughly **2× the kWh** vs a mild day.

### Baseload

No published constant-power sensor. We'll have to infer it from
either:

- HA's SQLite history (`home-assistant_v2.db`), aggregating
  `sensor.power_load_no_var_loads` over months, OR
- A simple monthly-scaled Finnish-household profile, calibrated to
  the user's annual kWh.

See Phase 0 below.

---

## What we're really measuring

Three quantities the integration cannot tell the user today:

1. **Realistic self-consumption fraction (SCF)** = PV_used_onsite / PV_generated. Bounded above by what no-battery + actual load timing physically allows.
2. **Realistic shift-yield (SY)** = (cost_naive − cost_scheduled) / cost_naive. How many EUR/year does following the v2.10.1 forecast actually save vs running everything at its "natural" hour?
3. **Forecast-driven regret** = cost(schedule built from v2.10.1 P50 prediction) − cost(schedule built from realised prices). The CVaR of this — across PV and price uncertainty — is the right risk object.

Self-consumption and shift-yield are partially the same lever (PV
midday + cheap-night = the same scheduling action), but they
trade off: a hot summer day with high midday PV has *expensive*
midday spot, so PV self-consumption is a different decision from
night-cheap-hours shifting. Quantifying that trade-off is one of
the study's outputs.

---

## Phase 0 — Consumption truth (do first)

Decision: do we read the user's actual HA SQLite history to derive
a real baseload profile, or do we use a synthetic Finnish profile
calibrated to annual kWh?

- **Reading the DB** gives a hourly load curve specific to this
  household, including weekday/weekend, occupancy, and seasonal
  drift. It's the honest baseline. Risk: the DB is ~hundreds of
  MB and contains every state change for every entity.
- **Synthetic profile** gets us moving immediately. It will miss
  the actual occupancy quirks (e.g. workshop usage on Saturdays).

Recommend: start with the synthetic profile to unblock Phases 1–3,
then upgrade to DB-derived after the framework works. Don't gate
the study on DB extraction.

Concrete output: `studies/exp_baseload_profile.py` that produces a
24×7 kW matrix `baseload[hour, weekday]`, calibrated either to
annual_kWh or to DB history.

---

## Phase 1 — Load and PV simulators

### Load simulator (`studies/sim_household_load.py`)

Mirrors the EMHASS formulation:

- **Non-deferrable**: heat-pump electrical demand as a function of
  outdoor temperature. Use the COP curves above; thermal demand is
  a linear function of (T_indoor_setpoint − T_outdoor) per the
  EMHASS-learned RC parameters when available, else a heating
  degree-hour model.
- **Baseload** from Phase 0.
- **Deferrable basket**: six loads with `(P, energy_today, window,
  semi_continuous_flag)` matching the EMHASS config table above.
  The simulator accepts a *schedule* (which hours each deferrable
  is on) and returns the realised hourly kWh.

### PV simulator (`studies/sim_pv_production.py`)

Wraps `pv_estimate.estimate_pv_kwh_per_hour` for the household's
8.91 kWp / 45° / 160° configuration, applied over the cached
Open-Meteo irradiance window. Adds:

- **Cloud-cover block bootstrap**: resample weekly cloud sequences
  from the same calendar month across 2023–2026 to generate
  alternate realisations of the same day pattern. 100 paths.
- **No new sensor**, no perturbation of the underlying model — we
  reuse what the integration already does for fairness.

The pair (load + PV simulator) defines a deterministic mapping
`(weather, schedule) → hourly kWh net of PV`.

---

## Phase 2 — Schedulers (the policies to compare)

Four policies, all evaluated on the same realised weather and
realised price sequence:

| Policy | Inputs | Realism |
|---|---|---|
| **P0 — Naive** | nothing; each load runs at its "natural" hour (EV plugged in 17:00 charges immediately; boiler heats at the legacy fixed times) | what most households do |
| **P1 — Greedy-PV** | realised PV, baseload curve | upper bound on self-consumption |
| **P2 — Perfect-price** | realised hourly spot, realised PV | upper bound on cost reduction |
| **P3 — Forecast P50** | v2.10.1 P50 spot + point-PV forecast | what the integration could actually deliver |
| **P4 — Robust P25** | v2.10.1 P25 spot + 25%-quantile PV (pessimistic) | risk-averse variant of P3 |

Each policy outputs a schedule for the six deferrable loads + a
heat-pump pre-heat decision (push thermal mass when cheap, coast on
inertia when expensive). The scheduler is a small MILP — six binary
streams of length 24 — solved per day. Existing libraries like
`pulp` are fine; runtime is sub-second.

---

## Phase 3 — Back-test on historical data

For every day in the cached 2023-01 → 2026-04 window:

1. Sample 100 cloud bootstrap paths for the day's PV.
2. Run each policy P0..P4. Score it on realised cost, SCF, SY.
3. Aggregate by month / season / weather regime.

**Deliverables**:

- `studies/results/exp_pv_self_consumption.md` — headline table
  averaged over the full window:
  - SCF (P10/P50/P90 across cloud paths), per policy
  - Annual € saved vs P0, per policy
  - Worst-month and best-month figures
- Per-season breakdown: winter (Dec–Feb), spring melt (Mar–May),
  summer (Jun–Aug), autumn (Sep–Nov)
- Sensitivity sweep: vary PV size (5 / 8.91 / 12 / 18 kWp) and
  annual kWh (10 / 15 / 20 / 30 MWh)

A separate doc — `studies/results/exp_cheap_day_classifier.md` —
classifies each forecast day into A/B/C/D regimes (see plan v1)
and reports realised savings by class.

---

## Phase 4 — Risk-aware acceptance: where CVaR earns its keep

Two distinct risk channels:

- **PV uncertainty**: the 100 cloud-bootstrap paths.
- **Price uncertainty**: the v2.10.1 P5/P25/P75/P95 bands.

Define **regret(policy, realised) = cost(policy) − cost(P2 perfect-foresight)**.

For each policy P3 / P4, compute regret over (100 PV paths) ×
(20 price paths sampled from the forecast distribution) = 2000
joint realisations per day. Then CVaR₀.₀₅ of regret is a single
number per policy per day, summable over the year.

Acceptance criterion for the study (analogous to v2.5.6 but at the
decision level): **P3 must beat P0 by ≥ X EUR/year in median
realised cost AND its CVaR₀.₀₅ regret must not exceed P0's by more
than Y EUR/year.** X and Y to be set after seeing Phase 3 numbers.

Open question: does CVaR-of-regret reveal anything beyond mean
realised cost on this kind of data? My honest suspicion is no for
the deferrable basket (it's bounded and discrete) but yes for the
EV (single big load, dominated by tail price events). We'll know
after Phase 3.

---

## Phase 5 — Integration surface (only if Phase 3 yields material savings)

Candidate new attributes on `sensor.duration_forecast.daily_forecast[i]`:

| Attribute | Meaning |
|---|---|
| `regime_class` | A / B / C / D (see plan v1) |
| `recommended_shift_priority` | rank 1..7 over the next week — which day to run discretionary loads |
| `pv_self_consumption_p50_kwh` | expected midday PV that the household will actually use |
| `forecast_savings_p50_eur` | realistic € saved vs naive that day |
| `cvar95_regret_eur` | downside if the schedule misfires |

No new sensor entity. No coordinator wiring changes until Phase 5.

---

## Sequencing and stopping rules

1. **Phase 0** — synthetic baseload calibrated to annual kWh.
   Stop at: 24×7 baseload matrix written. 1 day.
2. **Phase 1** — load + PV simulators. Stop at: end-to-end
   deterministic run reproduces a sample day. 2 days.
3. **Phase 2** — schedulers P0..P4. Stop at: each policy
   produces a valid 24h schedule honouring deadlines and
   energy budgets. 2 days.
4. **Phase 3** — back-test. Stop at: published headline
   table. 1 day.
5. **Decision gate** — review Phase 3 results. If savings >
   100 EUR/year for P3 vs P0 we proceed to Phase 4–5; otherwise
   we document the negative finding and pause.

Total to decision gate: ~1 working week if uninterrupted.

---

## Out of scope (this branch)

- Battery dispatch (no battery in the reference setup).
- Real-time MPC. EMHASS already does that — we're studying the
  *forecast quality contribution*, not replacing the optimiser.
- Dynamic grid tariffs / capacity-based pricing. Stays in the
  same flat distribution-fee assumption the integration already
  uses.
- Inverter-clipping non-linearities; the LONGi+Growatt combination
  rarely clips except on a clear May/June noon.
- HVAC comfort modelling beyond the COP curve. We treat heat-pump
  electrical kWh as a function of outdoor temp + setpoint;
  occupant comfort is not optimised.
