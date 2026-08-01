# Forecast model — backlog and open defects

Working document for the spot-price forecasting pipeline. Every entry is
backed by a measurement, with the producing script named so it can be
re-run. Statistics come from the frozen walk-forward harness
(`studies/backtest_harness.py`, day-ahead regime, snapshot-pinned) unless
stated otherwise.

Method: one isolated change per experiment, measured on the harness
before adoption. Aggregate numbers are reported alongside a per-segment
or per-lead-time breakdown — a single number has repeatedly hidden the
effect that mattered.

---

## Resolved

### R1 — Inverted solar sign (fixed in v2.16.0)

The shipped model priced irradiance with a **positive** coefficient
(`Y_solar_effective = +0.0236`): a sunnier forecast *raised* the
predicted price. Wind and PV are zero-marginal-cost, so more of either
can only lower the price or leave it unchanged.

Root cause is confounding, not signal. Left free, the ridge picks a
positive solar coefficient in *every* walk-forward refit
(+0.0196…+0.0266) because the solar residual is collinear with
temperature (clear Finnish winter skies are cold and expensive;
`corr(Y_solar, Y_temp) = +0.27` in summer while `Y_temp` carries −0.73)
and because the neighbour-price channel already transports the PV signal
(a sunny day here is usually sunny in Sweden). The marginal relationship
has the correct sign (`corr(Y_solar, Y_fi) = −0.14` in summer); the fit
reverses it.

Fixed in three layers so it cannot return: trainer sign constraint
(`fit_ridge_signed`), runtime clamp (`Pipeline._enforce_physics_signs`),
and tests — including two **behavioural** invariants asserting
`compute_forecast` never rises with more irradiance or more wind, which
survive refactors in a way a coefficient assertion does not.

Cost: harness MAE 19.81 → 19.86 overall; summer **improved** 14.47 →
14.29. The inverted term carried no predictive value.

### R2 — Dormant bias corrector (fixed in v2.15.0)

The coordinator never called `pipeline.update_with_actuals`, so the bias
corrector and the DtACI fan-chart calibrator never received realised
prices and never warmed up — dormant since v2.5.15. A reconciliation
loop now pairs raw forecasts with published prices each cycle.

### R3 — Train/inference deseasonalisation mismatch (fixed in v2.15.0)

The trainer deseasonalised the wind/solar physics terms with fitted
seasonal components while the pipeline mean-centred them per forecast
batch. Storing the components in the artifact (`physics_seasonal`) and
applying them at inference was worth −8.9% MAE — the largest single win
so far.

---

## Open defects

### D0 — Contemporaneous neighbour prices leak the target  *(supersedes D2; fix first)*

`Y_se1`, `Y_se3`, `Y_ee` are the **same-hour** prices of SE1, SE3 and EE.
Those zones clear in the **same day-ahead auction as FI, simultaneously**.
So the neighbour price for hour *t* is never known before the FI price
for hour *t* — the moment you can observe it, the answer is published
too. For every hour that genuinely needs forecasting (D+2 onward), the
feature is unavailable by construction.

`corr(FI(t), SE3(t)) = +0.82` contemporaneous, versus +0.66 at 24 h lag.
The model is therefore fitted against a near-proxy of its own target.

This is not only an evaluation problem — it distorts the trained
coefficients. Walk-forward refits, mean coefficient on the wind term:

| variant | wind coefficient | honest-task MAE | bias |
|---|--:|--:|--:|
| A — current (contemporaneous neighbours) | **−44.57** | 34.87 | −7.16 |
| B — neighbour features removed | **−92.98** | **26.81** | −4.36 |
| C — neighbour features lagged 168 h | −93.01 | 26.90 | −3.98 |
| D — B + net load (D1) | −5.79 † | **23.70** | **−1.81** |

"Honest task" = hours beyond the day-ahead auction, where neither the
neighbour price nor the FI price is known — i.e. the hours the forecast
exists to serve.

† In variant D the separate wind term nearly vanishes because net load
already contains wind generation (net load = consumption − wind − solar).
That is physically coherent: the model becomes a supply/demand balance
rather than a set of loosely related proxies.

Consequences:

1. **The physical driver was suppressed by more than half** (wind −44.6
   vs −93.0). The leaky feature absorbed explanatory power that belongs
   to weather. This is the mechanism behind the observation that weather
   dynamics barely move the forecast.
2. **The model is mis-specified at inference.** Coefficients were fitted
   with the neighbour block present; production zeroes it beyond
   day-ahead. The remaining coefficients are then too small for the
   information actually available — a systematic error, not just lost
   information.
3. **Removing the leak improves the honest task by 23%** (34.87 → 26.81
   MAE) and cuts bias by 39%. Adding net load reaches 23.70 / −1.81,
   **32% better than the current model with bias cut by 75%**.
4. Lagged neighbour prices (variant C) are legitimately knowable and
   perform the same as removing them entirely — so the neighbour channel
   can be retained in a leak-free form if desired.

**This invalidates two claims made earlier in this document** (now
corrected in D2): the "oracle" line in the lead-time study assumed
knowledge that never exists for the hours being forecast, and the
"neighbour block supplies 89% of dynamic variance" figure is an artifact
of the same leak.

Caveat on variant D: net load is itself published day-ahead only, and
the D row uses actual net load at all horizons, so it is also an upper
bound. The difference is that net load is a *physical* quantity
forecastable from weather + calendar (consumption ≈ f(temperature, hour,
day-type, holiday); wind/solar generation ≈ f(weather, installed
capacity)), whereas a coupled market price is another auction outcome
determined jointly with the target. The D gap is closable; the A gap is
not. **Next measurement: net load modelled from weather + calendar, so
the whole horizon is served by physically forecastable inputs.**

### D1 — The pipeline has no demand variable  *(highest value, ready to build)*

The L2 ridge sees `Y_fi_lag168`, `is_workday`, wind, solar, temperature
and three neighbour prices. **None of these is a demand signal.**
`net_load` exists in this repo, is fetched in production every cycle
(Fingrid 165/246/247 via `fetch_fingrid_forecasts`), and was documented
in the v2.2 work as *"the strongest single feature improvement"* — but
it only feeds the **base model**, whose output the pipeline overwrites.
The signal is fetched, computed, and discarded.

Evidence that this is the missing driver:

| summer, by local hour | price gap (weekday−weekend) | net-load gap |
|---|--:|--:|
| night 00–06 | +6.2 €/MWh | +120 MW |
| peak hours | +24.7 €/MWh | +573 MW |

`corr(price gap, net-load gap) = +0.87` across the 24 hours — one
physical variable with one coefficient reproduces a pattern that would
otherwise need a 24-parameter hour×day-type profile. It also explains
the July holiday effect that `is_workday` cannot see (weekday
consumption: May 8945 MW, July 8277 MW).

Why it hurts summer specifically — temperature is the model's only
demand proxy, and it goes silent when there is no heating load:

| | corr(temp, net load) | net-load variation |
|---|--:|--:|
| Winter | −0.698 | ±2248 MW |
| Summer | **+0.118** | ±1253 MW |

Measured effect of adding net load as a single 9th feature
(walk-forward, sign constraints retained):

| segment | v2.16 | +net load | Δ |
|---|--:|--:|--:|
| ALL | 19.86 | **18.84** | −5.1% |
| Summer | 14.29 | **13.56** | −5.1% |
| summer weekday peak | 19.49 | **17.93** | −8.0% |
| summer weekend peak | 15.73 | 14.99 | −4.7% |
| Winter | 26.54 | 25.33 | −4.6% |
| summer weekday night | 12.93 | 13.38 | +3.5% (only regression) |

Coefficient +7.5…+8.9 €/MWh per GW, stable across every refit.

Secondary benefit: net load = consumption − wind − **solar**, so PV
generation enters with the physically correct sign. As installed
capacity grows (e.g. the ~120 MWp Joroinen plant), Fingrid's solar
forecast reflects it, net load falls, and the price forecast falls —
automatically, with no capacity scaling. This is the PV channel the L2
residual approach could never provide (see D5).

Caveat: Fingrid publishes day-ahead only (~36 h of the 170 h horizon),
so this interacts with D2.

### D2 — Day-ahead data boundary: measured bias and discontinuity

> **Superseded in part by D0.** The study below treats the neighbour
> block as a legitimate input that merely runs out at the boundary. It
> does not: it is jointly determined with the target (D0), so the
> "oracle" column is unachievable in principle, not merely in practice,
> and the two figures below marked ⚠ are inflated by the same leak. The
> *shape* of the finding survives — assuming climatology beyond a data
> horizon does introduce bias and a discontinuity, and that lesson
> applies to any short-window input, including net load. But the fix is
> D0/D1 (replace the leaky driver with physically forecastable ones),
> not a crossfade of the neighbour feed. Retained for the method and for
> the crossfade statistics, which stay valid for genuinely forecastable
> short-window inputs.

⚠ The neighbour-price block supplies **89% of the forecast's dynamic
variance** but covers only ~28% of the horizon:

| driver block | std of contribution | horizon coverage |
|---|--:|---|
| Neighbour SE1+SE3+EE | 36.7 €/MWh | ~28% (day-ahead) |
| Weather — wind | 6.1 €/MWh | 100% (Open-Meteo 8 d) |
| Weather — temperature | 2.8 €/MWh | 100% |
| Weather — solar | 0.0 €/MWh | 100% (sign-clamped) |

Beyond the boundary the unknown tail is assumed to equal its
climatology. **This does introduce bias**, and it is not uniform —
producer: `studies/leadtime_fill_study.py`, 354 daily origins, error in
€/MWh, all inputs held fixed except the fill strategy:

**All seasons**

| lead | zero-fill (current) bias / MAE | AR(1) bias / MAE | oracle bias / MAE |
|---|--:|--:|--:|
| +1d | +0.1 / 21.5 | +0.1 / 21.5 | +0.1 / 21.5 |
| +2d | −1.7 / 30.0 | +0.4 / 26.4 | +0.1 / 21.4 |
| +3d | −3.6 / 35.0 | −2.4 / 33.7 | +0.2 / 21.4 |
| +4d | −3.5 / 35.0 | −3.2 / 34.8 | +0.2 / 21.4 |
| +5d | −3.7 / 35.0 | −3.5 / 34.9 | +0.2 / 21.4 |
| +6d | −3.7 / 35.1 | −3.7 / 35.0 | +0.3 / 21.5 |
| +7d | −3.7 / 35.0 | −3.7 / 35.0 | +0.4 / 21.5 |

**Winter (Nov–Feb)** — where the induced bias is worst

| lead | zero-fill bias / MAE | AR(1) bias / MAE | oracle bias / MAE |
|---|--:|--:|--:|
| +1d | +1.1 / 25.4 | +1.1 / 25.4 | +1.1 / 25.4 |
| +2d | −5.3 / 35.0 | −4.3 / 30.1 | +1.1 / 25.4 |
| +3d | −10.0 / 41.7 | −9.2 / 39.6 | +1.1 / 25.4 |
| +7d | −10.0 / 41.7 | −10.0 / 41.7 | +1.1 / 25.4 |

Three conclusions:

1. ⚠ **The oracle line is flat** (MAE 21.4–21.5, bias ≈ 0 from +1d to
   +7d) — but per D0 this is *not* an achievable bound: it requires
   knowing a price that clears simultaneously with the target. Read it as
   a diagnostic of how strongly the model leans on the leaked feature,
   not as headroom.
2. **Zero-fill introduces real bias**, −3.7 €/MWh all-season and −10.0
   €/MWh in winter, against an oracle bias of +0.2 / +1.1. The
   assumption is not neutral.
3. **The discontinuity is real**: at the +2d → +3d boundary, MAE steps
   30.0 → 35.0 (all seasons) and 35.0 → 41.7 (winter); bias steps −1.7 →
   −3.6 and −5.3 → −10.0.

Crossfade half-life scan (exponential blend from last observation to
climatology), MAE (bias), all seasons:

| strategy | +2d | +3d | +4d | +7d |
|---|--:|--:|--:|--:|
| zero (current) | 30.0 (−1.7) | 35.0 (−3.6) | 35.0 (−3.5) | 35.0 (−3.7) |
| hl = 24 h | 25.8 (+0.6) | 32.5 (−1.9) | 34.6 (−2.8) | 34.9 (−3.6) |
| **hl = 48 h** | **25.8 (+0.9)** | **32.5 (−1.0)** | 35.0 (−1.8) | 34.6 (−3.1) |
| hl = 96 h | 25.9 (+1.1) | 33.0 (−0.2) | 36.2 (−0.8) | 34.8 (−2.3) |
| persistence | 26.0 (+1.3) | 34.1 (+0.8) | 38.8 (+0.9) | 38.4 (+0.6) |

Bias step across the boundary (+2d → +3d): zero −1.9, hl=48h −1.9,
hl=96h −1.3, hl=192h −1.0, persistence −0.5.

Trade-off to decide: **longer half-life reduces the bias and the bias
step, but raises MAE at long lead** (classic bias/variance). `hl ≈ 48 h`
captures nearly all the available MAE gain (+2d 30.0 → 25.8, −14%; +3d
35.0 → 32.5, −7%) and halves the +3d bias (−3.6 → −1.0). Going longer
buys bias continuity at a growing MAE cost.

Honest limit: a crossfade **softens but does not remove** the step — the
MAE step is intrinsic because information genuinely ends at the boundary,
and no extension of a past observation recovers it. Winter bias only
improves −10.0 → −8.6 even with pure persistence. Removing the step
properly requires *forecasting* the neighbour prices (they are themselves
weather-driven) rather than extending the last observation, or reducing
the model's dependence on that block by giving it physical drivers with
full-horizon coverage (D1, D3).

Also note `_align_neighbour_prices`'s docstring claims missing hours
"contribute zero", but the pipeline actually flat-fills them with the
mean of the available hours. Measured at h=72 the three options are
close (flat-fill 42.12, AR(1) 42.10, zero 43.41), so this is a
documentation/behaviour mismatch rather than a large error source — but
it should be made explicit and deliberate.

### D3 — No hour×season interaction

`P_hour` is a single diurnal shape (44 €/MWh peak-to-peak) applied to
both January and July; season enters only as a level shift via `P_week`.
The L2 coefficients are just as season-dependent — per-season refits:

| coefficient | winter (Nov–Feb) | summer (May–Aug) |
|---|--:|--:|
| `Y_sigmoid_wind_rho` | −41.9 | −20.0 |
| `Y_solar_effective` | −0.052 | +0.0065 |

One global parameter set is a compromise between two regimes. Note D1
may absorb much of this, since load carries a large part of the seasonal
difference — **test D1 first and re-measure before adding interaction
terms**, to avoid adding parameters for an effect a physical variable
already explains.

### D4 — `Y_fi_lag168` is hard-coded to zero in production

`coordinator.py` sets `lag168 = np.zeros(len(forecast))` with a
"cold-start" comment. This is not a forecast quantity — it is the price
168 h ago, fully known — but only 2 days of price history are fetched.
The model was fitted with real values there (`std(Y_fi) = 84.1 €/MWh`,
coefficient +0.023, so ≈ ±1.9 €/MWh of signal). Fix is to retain 8 days
of price history; low effort, modest but free gain.

### D5 — PV capacity growth is not representable in the current design

Capacity-aware scaling (Fingrid 267/268) was tested and rejected in both
layers, and the reason is now understood:

- **L2**: the pipeline centres physics features per forecast batch, and
  `center(r·x) = r·center(x)` — scaling a centred feature is a pure
  coefficient rescale, provably unable to introduce a level shift. The
  PV effect is a level effect.
- **L1**: an uncentred `capacity_ratio × solar_effective` level term did
  not help either (γ < 0 as physics predicts, but MAE flat/worse) —
  irradiance already carries the PV signal, and the 2026 bias lives in
  the overall/winter level, not at midday.

Post-R1 the solar coefficient sits at exactly zero even after
orthogonalising against temperature: this feature set carries no
independent PV→price signal. The model is now *harmless* with respect to
PV but not *responsive* to it. **D1 is the route to responsiveness** —
PV enters as supply inside net load, at every horizon.

### D6 — Production behaviour is not reproduced offline

The field report (weekday morning/evening peaks over-predicted in July)
does not reproduce in the harness, which shows both the old and new
model *under*-predicting summer evening peaks (−15…−18 €/MWh). The
harness uses actual weather and actual neighbour prices; production uses
forecast weather and a truncated neighbour feed. Zeroing the neighbour
block flips summer midday bias from −1.2 to **+10.7**, so the truncation
path is the prime suspect (see D2), but this is **unverified against
production**.

Until it is verified, offline harness gains cannot be assumed to
transfer. This is the main reason not to promote a model release on
harness numbers alone.

---

## Testing gaps

1. **No production instrumentation.** Nothing logs per-hour ridge
   contributions, neighbour-price coverage, or the realised lead-time
   error profile. D6 cannot be settled without it. *Highest-value gap.*
2. **No lead-time regression test.** Nothing asserts the forecast error
   profile is smooth across the day-ahead boundary; D2's discontinuity
   went unnoticed until explicitly measured.
3. **Docstring/behaviour mismatch** in `_align_neighbour_prices`
   (documented "contribute zero" vs implemented flat-fill), untested
   either way.
4. **No local-time weekday/weekend assertion.** All harness segments are
   UTC; the reported symptom is in local time, where the weekday/weekend
   split actually lives.
5. **Spike-hour trade-off unguarded.** Bias correction worsens p95-price
   hours (50.0 → 52.4 MAE) because it learns on normal hours. Accepted
   deliberately, but no test pins the magnitude.
6. **Weather inputs are oracle in the harness.** Actual historical
   weather is used, not archived forecasts, so all harness numbers are
   optimistic in absolute terms. Deltas between configs remain valid;
   absolute levels do not transfer to production.

---

## Suggested order

1. **D0 + D1 together — rebuild the driver set.** Drop the
   contemporaneous neighbour features (or lag them, variant C), add net
   load. Measured on the honest task this is 34.87 → 23.70 MAE (−32%)
   with bias −7.16 → −1.81. These two must be done together: removing
   the leak alone re-scales the physical coefficients, and net load is
   the driver that should carry the load the leak was carrying.
2. **Net load from weather + calendar** — so the whole 170 h horizon is
   served by physically forecastable inputs rather than a 36 h feed.
   Closes the D0 caveat and gives PV capacity a path (D5).
3. **Production instrumentation** (testing gap 1) — settles D6, and
   should be in place before any release is promoted on harness numbers.
4. **D4 — price history buffer** — small, self-contained.
5. **D3** — only after re-measuring with D1 in place.
6. **D2 crossfade** — now a second-order concern, and applies to
   whatever short-window inputs remain after step 2.

Note: all harness statistics predating D0 that involve the neighbour
block should be re-derived on the honest task before being used to
justify a change.
