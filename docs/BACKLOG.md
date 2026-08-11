# Forecast model — backlog and open defects

## Current status (2026-08-11) — root cause found, and corrections below

The July weekday over-prediction is real and reproduces offline once the
harness is made to behave like production. Reproducing the deployed
pipeline against the data store (frozen L1 + L2, `Y_fi_lag168` zeroed,
neighbours lagged, no bias correction):

| 2026-07 weekday | actual | model | bias |
|---|--:|--:|--:|
| hourly mean | 21.56 | 41.07 | **+19.52 (+90 %)** |
| daily peak | 29.85 | 62.81 | **+32.96 (+110 %)** |

**Shipped in v2.18.0** — the bias corrector was mistuned. A 14-day
half-life behind a 14-update warm-up gate disabled the correction for
exactly one half-life, then applied it at 50 % strength (a
zero-initialised EMA reaches `1−(1−λ)ⁿ`). Retuned to a 3-day half-life
with a 2-observation guard and a CMA→EMA warm-up. Producer:
`studies/bias_corrector_warmup_study.py`.

| configuration | MAE | weekday | wd peak | \|mth bias\| | 2026-07 wd peak bias |
|---|--:|--:|--:|--:|--:|
| no correction | 25.76 | 28.22 | 35.55 | 9.54 | +29.36 |
| v2.17.3 | 24.91 | 26.83 | 33.91 | 7.21 | +27.17 |
| **v2.18.0** | **24.13** | **25.76** | **32.36** | **3.30** | **+13.66** |

Post-install (three weeks after a state wipe): bias −5.20 → −0.39,
MAE 22.41 → 21.00.

### Documentation debt found in the v2.18.0 doc audit

Tracked, not fixed in this release:

1. **Version archaeology removed from the docs.** References like "new in
   v2.11.0" / "added in v2.10.0" / "passes the v2.5.6 hedge gate" told a
   reader nothing about current behaviour and several had become false.
   Stripped from README.md and TECHNICAL_GUIDE.md outside the changelog.
2. **`TEKNINEN_TOTEUTUS.md` rewritten** as a current translation of
   TECHNICAL_GUIDE.md. It had documented the pre-v2.17 leaky model and
   cited a script that no longer exists. Pure attribute reference tables
   now point to the English document rather than being duplicated, so
   the two cannot drift apart again.
3. **All seven files in `docs/diagrams/` deleted.** None was referenced
   by any document, three had no rendered PNG so could not be viewed on
   GitHub at all, and every one described the pre-integration
   architecture: "Jinja2 inference", "Two-Stage Ridge Regression +
   Piecewise Calibration", "REST Sensors", `mgrey.se` (the code uses
   elprisetjustnu.se), Fingrid capacity `#24/27/115` (the config uses
   267/268), and sensor names `price_forecast` / `meteo_7day_forecast`
   that do not exist. Four of the five carried a feature-count claim of
   **28–38** against an actual **nine**; `data-flow.drawio` itemised
   them ("Supply (3), Time cycles (4), Demand patterns (8), Thermal
   demand (6), Physics supply (3), Scarcity indicators (4) = 28").
   Recoverable from git history. The accurate, compact data-flow diagram
   in TECHNICAL_GUIDE.md is now the single architecture picture — a
   second representation is what let these drift unnoticed.
4. **Six screenshots are unreferenced**: `evaluation-{features,full,
   metrics,timeseries}.png`, `install-05-operator_{1,2}.png`,
   `install-08-device-created.png`.

### Corrections to claims made earlier in this document

1. **D6 is the root cause of the offline/field contradiction.** The
   harness and `summer_weekday_status.py` refit L1 monthly; production
   runs the frozen artifact. Refitting averages the week-bin noise away;
   production applies one bin deterministically. Fix the harness before
   trusting any further offline number — the "most likely explanation:
   stale pre-v2.15 artifacts" hypothesis below is probably wrong.
2. **D1's diagnosis needs amending.** Net load's informative half is
   **wind, not consumption**. Measured with Fingrid day-ahead series:
   oracle wind/PV is worth +3.8 % summer, oracle consumption −1.0 %.
   Every consumption specification tried in the price mean (plain,
   ×peak-hour, ×workday, peak-only, ×wind-scarcity) scored −0.4 % to
   −1.3 %. Consumption does help the *daily-spread* model (+7 % on top
   of wind/PV/nuclear), where sensitivity saturates at ~3.6 % MAPE —
   so a focused consumption model has no headroom to buy.
3. **The nuclear result is not safe.** `nuclear_mw` (188) is *realised*
   production — the only realised series used in any of this work, and
   the only variable that failed every test. Nord Pool prices *planned*
   availability from the UMM outage schedule, which
   `api_client.fetch_nuclear_outage_schedule` already fetches but which
   feeds only the legacy base model. Re-run before concluding.
4. **Information-set principle.** Datasets 246/247/165 are Fingrid
   **day-ahead forecasts** — market-available at gate closure, and
   therefore the correct regressors. Realised production is endogenous
   (price and consumption are simultaneously determined) and must not
   become a fitting target for the price model.

### Largest unshipped opportunity (measured, not yet built)

Fingrid's day-ahead wind/PV forecasts are only 85 % / 75 % reconstructible
from our weather proxies. Adding the **orthogonal remainder** on top of
the retained proxies is worth **+8.5 % MAE, +8.0 % summer** — the largest
single effect measured. It must be *added to*, not substituted for, the
weather proxy (substituting regresses summer), and it covers D+1 only.

**Sequencing constraint:** the structural stack (capacity scaling,
Fingrid channel, wind nonlinearity, amplitude recalibration) improves
aggregate MAE 5.7 % but pushes the 2026-07 weekday-peak bias from +26.5
to **+37.2**. It must not ship before the bias-corrector fix, which
v2.18.0 delivers.

---

## Historical status (2026-08-01) — the July summer weekday/weekend report

**The original observation was correct.** Weekends *are* forecast more
accurately than weekdays in summer, and the gap is large. Producer:
`studies/summer_weekday_status.py` (honest task, local hours, summer
Jun–Aug):

| | weekday MAE | weekend MAE | gap |
|---|--:|--:|--:|
| v2.16 (as production behaved) | 27.35 | 18.02 | **+9.33** |
| **v2.17.1 (shipped)** | **24.26** | **17.42** | **+6.83** |

v2.17.1 narrows the gap by **27 %** and cuts weekday error by 11 %.
Per block (summer, local time, bias / MAE):

| block | actual | v2.16 | v2.17.1 |
|---|--:|--:|--:|
| weekday morning 07–11 | 62.3 | −7.3 / 31.8 | −4.8 / **25.8** |
| weekday evening 17–21 | 71.7 | −18.1 / 38.0 | −17.6 / **34.1** |
| weekday midday 12–16 | 40.4 | −0.1 / 21.1 | −3.2 / **18.6** |
| weekday night 00–06 | 29.5 | −8.5 / 19.3 | **−0.6** / 18.7 |
| weekend morning 07–11 | 22.1 | +9.5 / 18.6 | +8.3 / 17.4 |
| weekend evening 17–21 | 46.6 | −13.4 / 25.5 | −11.2 / 25.8 |

**But the direction still does not match the field report.** The report
was weekday morning/evening *over*-prediction; every leak-free
measurement shows weekday **under**-prediction (evening −17.6), while the
*over*-prediction sits on weekend mornings/middays (+8 … +13).

Most likely explanation, and the next thing to verify: the deployed
integration at the time of the report was running **pre-v2.15 artifacts**
whose training window ended **2024-11**. Measured early in this
investigation, that stale model showed **+11.4 EUR/MWh midday bias** —
over-prediction, matching the report. v2.15.0 refit L1 + L2/L3/L4 on
fresh data and cut that to +7.2; v2.17.0/v2.17.1 improved it further. So
the reported symptom is consistent with model staleness that has since
been fixed, but **this has not been confirmed against a running
instance** — see testing gap 1.

**Revised characterisation of the remaining problem.** The dominant
summer error is no longer a morning over-prediction; it is a **weekday
evening under-prediction (−17.6 EUR/MWh, MAE 34.1 at 17–21 local)**,
concentrated at 19:00–22:00 where actual prices peak at 74–87 EUR/MWh.
That is the next investigation, and it supersedes the original framing.

Overall accuracy against the model as production behaved: **35.46 →
27.06 EUR/MWh MAE (−24 %)**, winter −33 %.

---
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

### R5 — Lagged net load removed (v2.17.1)

Shipped in v2.17.0, removed one release later after a cleaner isolation
on the correct training window:

| variant | MAE | bias | holiday hours |
|---|--:|--:|--:|
| plain workday (v2.16 style) | 27.31 | −4.08 | 24.11 |
| + holiday-aware `is_workday` | 27.22 | −3.96 | 22.06 |
| **+ `is_holiday` dummy** | **27.06** | −3.94 | **17.65** |
| + `Y_netload_lag168` | 27.05 | −3.87 | 17.91 |

The holiday features carry the demand win (−0.9 % overall, **−27 % on
holiday hours**, coefficient −14.8 EUR/MWh). Lagged net load was worth
−0.04 % and its coefficient was **negative (−1.25)**, which is not a
demand relationship at all:

| model | net-load coefficient |
|---|--:|
| same-hour net load, no lagged price | **+18.92** |
| lagged net load, no lagged price | +1.58 |
| lagged net load + lagged price | −0.56 |
| + lagged neighbours (as shipped in v2.17.0) | −1.15 |

`corr(Y_netload_lag168, Y_fi_lag168) = +0.587` — last week's demand is
already embedded in last week's *price*, so once the lagged price is in
the model the demand term degenerates into a correction to it. A
suppressor coefficient is statistically legal but uninterpretable, and
this session's whole lesson is that confounded coefficients with
surprising signs are how the solar inversion survived for years. Removed.

The v2.17.0 figure that justified it (−2.6 %) had been measured on the
2022-contaminated window and did not survive the window correction.

**Still open, and the real opportunity**: SAME-HOUR net load has
coefficient **+18.9** (corr +0.587) — a strong, physically correct demand
driver. Fingrid publishes it day-ahead only, but that is exactly the
horizon most consumer decisions use. A hybrid that consumes the published
forecast for D+1 and falls back for later hours is the next demand
experiment. The pipeline still accepts `netload_lag168`, so no plumbing
is needed to try it.

### R4 — Contemporaneous neighbour-price leak + missing demand (fixed in v2.17.0)

Closes D0 and the actionable part of D1. Neighbour features are now built
from prices **lagged 168 h** (`NEIGHBOUR_LAG_HOURS`), and the pipeline
gained its first demand-side inputs: net load lagged 168 h and a public
holiday flag (`is_workday` now excludes holidays).

Leak-free evaluation (`studies/honest_horizon_study.py`):

| | all MAE | bias | winter MAE | winter bias | summer MAE |
|---|--:|--:|--:|--:|--:|
| v2.16 as production behaved | 35.46 | −7.29 | 46.57 | −12.52 | 27.69 |
| **v2.17.0** | **26.86** | **−3.92** | **31.08** | **−1.85** | **24.01** |

**−24% MAE overall, −33% winter, −13% summer.**

Side effects worth recording:

* **The solar sign fixed itself.** With the leak gone the ridge chooses
  `Y_solar_effective = −0.0242` unprompted — the R1 sign constraint is no
  longer binding. The leak *was* the cause of the inversion: the
  neighbour channel had been transporting the PV signal, leaving solar to
  fit confounding. The PV channel is now genuinely responsive, so growing
  capacity (Joroinen) moves the forecast the right way.
* **The physical drivers recovered.** Wind went −44.6 → −98.7.
* **The +2 d/+3 d discontinuity disappeared** — it was a symptom of the
  same defect, not an independent problem.
* Guards added: the shipped artifact may not declare an un-lagged
  neighbour feature; a behavioural test asserts a lagged input affects
  only its own hour; demand and holiday channels are asserted wired.

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

## Considered and rejected (with evidence)

* **Training from 2022** (requested; measured, declined). The store holds
  ~4 k extra hours from 2022-07, but that is the gas-crisis regime.
  Identical model, window varied: 2022-06 → MAE 32.40 / bias +11.67;
  **2023-01 → 26.86 / −3.92**; 2024-01 → 29.83 / −6.85. More data, worse
  model — the crisis period inflates the climatology and injects an
  over-prediction bias. Cutoff left at 2023-01; re-test when the crisis
  ages out of the rolling window.
* **Net load predicted from weather + calendar** — adds ~0.1%. A linear
  net-load model collapses into the weather features the ridge already
  has. The lagged *observed* net load now shipped is the version that
  pays, because it carries demand regime information weather cannot
  reconstruct.
* **24-parameter hour×day-type profile** — rejected in favour of one
  physical variable. The weekday−weekend price gap tracks the net-load
  gap at `corr = +0.87` across the 24 hours, so a single coefficient
  reproduces what 24 parameters were meant to model, without the
  overfitting risk.
* **Capacity-aware PV/wind scaling (Fingrid 267/268)** — see D5.
* **Crossfade at the day-ahead boundary** — see D2; became second-order
  once the leak was removed and the discontinuity vanished.

Watch item: the shipped `Y_netload_lag168` coefficient is **negative**
(−1.25). That is not a physical violation — it is a lagged regime signal,
not a contemporaneous causal driver, so mean-reversion can legitimately
produce this sign — but it should be re-examined when a contemporaneous
demand channel becomes available for the full horizon.

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

### D1 — The pipeline has no demand variable  *(gain does NOT survive a leak-free test)*

> **Corrected 2026-08-01.** The −5.1% headline below used *actual* net
> load at every horizon. Fingrid publishes it day-ahead only (~36 h), so
> that number was contaminated by the same class of error as D0. When net
> load is honestly forecastable — predicted from weather + calendar over
> the full horizon — it adds **≈ 0.1%** (27.30 → 27.28 MAE), because a
> linear net-load model collapses into the weather features the ridge
> already has. The diagnosis in this section stands (the pipeline has no
> demand signal, and load explains the weekday/weekend structure); the
> *remedy* does not pay unless the demand information is something
> weather cannot reconstruct — e.g. the published Fingrid forecast for
> D+1 only, or a holiday/industrial calendar. Do not spend effort here
> before D0.


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

## Leak-free reference result

`studies/honest_horizon_study.py` — origins 06:00 UTC daily (before the
day-ahead auction, so D+1…D+7 are all genuinely unknown); every feature
available for every forecast hour; walk-forward monthly refits; sign
constraints retained.

| variant | all MAE | bias | winter MAE | winter bias | summer MAE | summer bias |
|---|--:|--:|--:|--:|--:|--:|
| A — shipped v2.16, as it behaves in production | 35.46 | −7.29 | 46.57 | −12.52 | 27.69 | −13.27 |
| **B — leak-free (neighbours lagged 168 h)** | **27.30** | **−4.15** | **31.54** | **−1.85** | **24.58** | **−10.86** |
| C — B + net load modelled from weather | 27.28 | −4.14 | 31.52 | −1.84 | 24.55 | −10.83 |

B vs A: **−23.0% MAE overall, −32.3% winter, −11.2% summer**; bias
−7.29 → −4.15 overall and −12.52 → −1.85 in winter.

Two further findings:

* **The lead-time discontinuity disappears.** Both A and B are flat
  across +1d…+7d (A 35.28→35.58, B 27.19→27.40). The artificial +2d/+3d
  step documented in D2 was itself a symptom of the leak: the model was
  strong inside the auction window and collapsed outside it. With
  features that cover the whole horizon there is no boundary to cross.
* **The gain does not depend on D4.** With `Y_fi_lag168` left at zero as
  production does today, B still scores 27.71 (vs 27.33 with it working)
  — so fixing the history buffer is worth ~1.4%, independent and
  optional.

Note the absolute level: the shipped model's honest accuracy is ~35 MAE,
not the ~20 the old harness reported. That gap is the leak.

## Suggested order

1. **D0 — remove the leak.** Replace the same-hour neighbour features
   with their 168 h lags, retrain, ship. −23% MAE / −43% bias on the
   honest task, and it removes the D2 discontinuity as a side effect.
   Production already fetches 8 days of neighbour history, so no new
   data is required.
2. **Re-base the harness on the honest regime** — `backtest_harness.py`
   itself leaks, so it must not be used to justify further changes until
   corrected.
3. **Production instrumentation** (testing gap 1) — settles D6; should
   precede promoting any release on offline numbers.
4. **D4 — price history buffer** — small, independent, ~1.4%.
5. **Summer bias** (−10.9 after D0) — the largest remaining error, and
   still unexplained. Next investigation.
6. **D3 / D1 / D2** — re-measure only after D0, on the honest task.

Note: every statistic in this document predating D0 that involves the
neighbour block is inflated and must be re-derived before it is used to
justify a change.
