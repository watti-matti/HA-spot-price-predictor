# Per-load PV awareness — buy + sell duration curves vs PV-netted D(k)

Branch: `PV_adjusted_price`. Architectural analysis (no code yet).
Addendum to
[`pv_adjusted_price_coupling_rules.md`](pv_adjusted_price_coupling_rules.md)
and [`pv_adjusted_cvar_plan.md`](pv_adjusted_cvar_plan.md).

## The question

Should the integration publish PV-aware information as:

(A) **Composed**, today's approach:
```
dk_cheap_eur_kwh[24]      — buy-side duration curve (grid)
dk_peak_eur_kwh[24]
dk_cheap_pv_eur_kwh[24]   — single-baseload PV-netted duration curve
dk_peak_pv_eur_kwh[24]
```

(B) **Decomposed**, the proposed alternative:
```
dk_cheap_eur_kwh[24]            — buy-side duration curve (kept)
dk_peak_eur_kwh[24]
dk_cheap_sell_eur_kwh[24]       — sell-side duration curve (new)
dk_peak_sell_eur_kwh[24]
```

In (A) the integration does the PV netting up-front with a fixed
baseload assumption. In (B) the integration publishes the raw
buy/sell ingredients and the consumer (e.g. the thermal optimiser)
composes its own per-load effective price.

## The math each consumer cares about

Per-load effective price at hour `h` is

```
λ_eff(h, load) = (1 − α(h, load)) · buy(h) + α(h, load) · sell(h)
α(h, load)     = min(1, PV_surplus(h) / load_kW)
```

α depends on `load_kW`. For the reference household:

| Load | kW | α at midday PV peak (≈5 kW surplus) |
|---|:---:|:---:|
| Floor heating zone (bathroom) | 1.1 | **1.00** (fully PV-covered) |
| Workshop thermal mass | 2.3 | ~1.00 |
| Boiler (resistive) | 3.0 | ~1.00 |
| EV charger (three-phase) | 11.0 | **~0.45** (PV covers ≈ half) |

Schema (A) computes `λ_eff` *once*, using an integration-side average
α that doesn't know which load is asking. The result is correct for
the small loads but materially wrong for the EV. Schema (B) lets each
consumer compute its own α and therefore its own correct `λ_eff`.

## Pros for thermal optimisation (schema B)

1. **Per-load α correctness.** The whole point of the planner is
   load-aware scheduling. With (A) the EV sees a price that
   under-counts its grid-import cost by ≈ 50 % at PV-peak hours
   because the integration assumed the EV ran at average-baseload
   power. (B) gives the planner the ingredients to compute the EV's
   own α and price correctly.

2. **No baseload coordination.** Today the integration assumes a
   baseload to do PV netting. If the planner disagrees with that
   baseload — and it will, because the planner has time-varying
   per-load context the integration lacks — the two systems disagree
   on what hours are "cheap." (B) eliminates the disagreement by
   keeping the integration's contract limited to raw prices.

3. **Explicit export-decision support.** A thermal optimiser deciding
   whether to dump PV surplus into a hot-water tank (self-consume)
   vs export it needs `sell(h)` directly. Today the planner has to
   back-derive sell from tariff parameters published elsewhere in
   the integration's schema. (B) makes sell a first-class duration
   curve.

4. **Cleaner CVaR composition.** The PV-aware CVaR introduced in
   Phase D uses the integration's baseload assumption. The planner
   computing its own per-load CVaR via the shared `pv_cost_kernel`
   needs raw buy + sell scenarios, not an averaged composite. (B)
   aligns the published primitives with what the kernel actually
   wants.

5. **Coupling rule R2 alignment.** R2 declared `effective_eur_kwh`
   and PV-aware fields *diagnostic only*. (B) acts on that by
   actually making the *canonical* publication PV-unaware
   (buy + sell separately), with PV-netted fields surviving only
   as diagnostic conveniences for non-planner users.

6. **Better support for batteries and thermal storage.** The
   sell-side D(k) tells the planner which hours are *worst* to
   export (low `dk_cheap_sell`) and which are *best* to retain PV
   for self-consumption later. The integration's current single
   PV-netted D(k) collapses this into a scalar.

## Cons for thermal optimisation (schema B)

1. **More composition work in the planner.** ~10 lines of code to
   blend per load. Trivial.

2. **No headline PV-aware number for non-planner users.** The
   dashboard "PV-aware cheap price for today" loses its single
   sensor. *Mitigation*: keep `dk_*_pv_eur_kwh` alongside as
   explicitly-diagnostic, with a docstring clarifying it's a
   flexible-kWh approximation not suitable for per-load
   optimisation.

3. **Two CVaR figures can disagree on the same week.** The
   integration's `pv_aware_cvar95_eur_kwh` (Phase D, single-baseload)
   and the planner's per-load achieved CVaR can produce different
   numbers for the same week. *Mitigation*: name them clearly —
   "reference CVaR (typical flexible kWh)" vs "achieved CVaR
   (your scheduled load)" — and document the gap as the planner's
   value-add metric.

4. **Schema bloat.** 2 × 24 = 48 extra floats per day per
   forecast row. Negligible.

5. **Sensor consumers downstream of the integration that already
   parse the PV-aware fields will need a deprecation path** if we
   eventually retire (A). Stay additive for v2.11; deprecate (A)
   no earlier than v2.13 with clear release notes.

## Recommendation: additive — both schemas coexist

Keep all existing fields as the canonical contract. Add the
sell-side duration curves and per-hour sell field. Tag
`dk_*_pv_eur_kwh` and the new Phase D `pv_aware_*_eur_kwh` fields as
*diagnostic / flexible-kWh approximation*. The thermal optimiser
consumes the raw buy + sell, never the PV-netted variants.

### Concrete new attributes on `duration_forecast.daily_forecast[i]`

```
dk_cheap_buy_eur_kwh[24]     — same as dk_cheap_eur_kwh (alias for symmetry)
dk_peak_buy_eur_kwh[24]      — same as dk_peak_eur_kwh
dk_cheap_sell_eur_kwh[24]    — NEW: sell-side cheap duration curve
dk_peak_sell_eur_kwh[24]     — NEW: sell-side peak duration curve
```

### Concrete new attributes on `price_forecast.forecast[i]` (per-hour rows)

```
sell_eur_kwh                  — already exists in PV-enabled rows;
                                 promote to always-published when PV is
                                 enabled (it is today). No change.
```

The per-hour `sell_eur_kwh` is already on PV-enabled forecast rows.
The duration-curve aggregation of it is the new piece.

### Documentation contract

`docs/sensor_schema.md` (or the existing TECHNICAL_GUIDE) declares:

> **For per-load optimisation** (thermal optimiser, EMHASS, EV
> charge scheduling) consume `dk_cheap_buy_eur_kwh[24]`,
> `dk_peak_buy_eur_kwh[24]`, `dk_cheap_sell_eur_kwh[24]`,
> `dk_peak_sell_eur_kwh[24]`. Compose `λ_eff` per your load's kW
> using `α = min(1, PV_surplus/load_kW)`. Do NOT consume
> `dk_*_pv_eur_kwh` or the `pv_aware_*` Phase D fields — those are
> flexible-kWh approximations for dashboards only.
>
> **For dashboards and non-planner users**: `dk_cheap_pv_eur_kwh[24]`,
> `dk_peak_pv_eur_kwh[24]`, `pv_aware_cvar95_eur_kwh` give a single
> meaningful "PV-aware price" headline. They use a per-cycle
> single-baseload α and are accurate for the household average,
> approximate for any specific load.

### What stays exactly as today

- All buy-side D(k) (`dk_cheap_eur_kwh`, `dk_peak_eur_kwh` etc.) —
  canonical, never change semantics (R1).
- The Phase D PV-aware CVaR — survives as the diagnostic /
  reference number. Tag clearly in the schema doc.

### What changes

- Add four new per-day arrays as above.
- Update `TECHNICAL_GUIDE.md` and the per-zone equivalents to
  carry the contract clause above.
- Add a unit test that verifies `dk_cheap_buy_eur_kwh[24] ==
  dk_cheap_eur_kwh[24]` (the buy-side alias for symmetry must be a
  byte-identical alias, not a re-computation, so the two never
  drift apart).

## Implementation scope

If this is accepted:

1. Coordinator: compute sell-side D(k) by sorting the 24 hourly
   `sell_eur_kwh` values per day. ~10 lines, mirrors the existing
   buy-side D(k) loop.
2. Schema doc updates: one paragraph in each user-facing guide.
3. One unit test for the buy alias.
4. No breaking changes; production-safe for a v2.11 release.

The thermal optimiser's design (per EMHASS 0.17.3 analysis and
HA-energy-needs-planner intent) can target the new fields without
needing the existing PV-netted fields at all. Both representations
coexist; consumers pick the one matching their semantics.

## Honest limitations of (B)

1. **Sell-side spot prices are typically less volatile than buy-side**
   because feed-in tariffs are usually fixed or weakly time-varying
   in Finland. So `dk_cheap_sell_eur_kwh[24]` and
   `dk_peak_sell_eur_kwh[24]` may both be flat for many users —
   a sell-side duration curve degenerates into a single number.
   This is fine; the value of the schema is having the *channel*
   for time-varying feed-in tariffs when they exist (Octopus-style
   variable export, market-linked feed-in, etc.).

2. **Sensor consumers must understand the contract.** A naive
   consumer that adds buy and sell prices without computing α will
   produce nonsense. The schema doc must be explicit, and the
   thermal optimiser's tests must include a "scaled-α" coverage
   case to catch this.

3. **The (A) schema's PV-aware CVaR sensor was just shipped in
   Phase D.** Moving to (B) doesn't break that — the Phase D
   fields stay, the new buy/sell D(k) is purely additive. But the
   user-facing semantic shift ("compose your own per-load CVaR")
   is real and needs documentation.
