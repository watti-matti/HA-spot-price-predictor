# Modelling data assets, and the parked market-follower design

> **Two documents in one.** Sections 1–6 are the data-asset audit that
> produced the 4-year modelling dataset (done, shipped). The appendix is
> the PARKED architecture plan (decision 2026-10-13: not enough evidence
> to justify a rebuild over incremental work on #29).
>
> **Read `docs/SESSION_HANDOFF_2026-10-16.md` first.** It supersedes
> parts of the appendix — notably that the admissible fundamentals
> ceiling is R² 0.537 rather than ~0.30, that the `tail p95` figure is
> range compression rather than level bias (Theil: bias² is 0.03 % of
> MSE), and that a measured candidate now exists (`RL_PROXY`, +1.70 %
> MAE, −7.4 % on the price tail).

## Context

Concern raised: #29's baseline is said to be equivalent to production, yet
there was an indication that data access covered only ~9 of ~15 input
attributes. **Those two numbers are different things and neither is a
data-access failure** — but the audit they prompted did find a real
defect and three genuine gaps.

### Clearing up 9 vs 15

**9 is the model's design, not a data limit.** The L2 ridge has exactly
nine features (ten coefficients, intercept first):

```
intercept   -0.303   Y_temp          -2.412
Y_fi_lag168 +0.067   Y_se1_lag168    +0.046
is_workday  +3.289   Y_se3_lag168    +0.047
Y_sigmoid_wind_rho -98.821   Y_ee_lag168 -0.005
Y_solar_effective  -0.025   is_holiday -14.799
```

**15 is the number of Fingrid series in the grid parquet** — 6 original
plus the 9 cross-border series added on this branch. They are *data
sources*, and most are deliberately not model inputs: the capacity and
cross-border series were collected to be *tested* (and were, and were
rejected — see D5).

Every one of the nine model features has complete source data:

| feature | source | coverage |
|---|---|---|
| `Y_fi_lag168` | `fi_prices` | 35,040 rows, no gaps |
| `Y_se1/se3/ee_lag168` | `fi_neighbor_prices` | 62,270 rows to 2026-10-13 |
| `Y_sigmoid_wind_rho`, `Y_solar_effective`, `Y_temp` | `fi_weather` | 35,056 rows, no gaps |
| `is_workday`, `is_holiday` | calendar | n/a |

So the model is **not** data-starved, and #29 remains a structural no-op
on the forecast — re-verified 2026-10-16: SHA-256 over the 22 inference
modules plus the 4 shipped artifacts is `553f18914d189dd7…` on both
`origin/main` and the branch HEAD. The only component files #29 touches
are `manifest.json` (version string) and `retrain.py`, which is outside
the inference closure.

## Defect found: slow series are dropped instead of carried

`build_dataframe` inner-joins every input and then `dropna()`s. Four grid
columns have scattered reporting gaps, and **207 hours are being silently
discarded from the analysis window** as a result:

| column | NaN in window | sole cause of loss | longest run |
|---|--:|--:|--:|
| `solar_capacity_mw` | 155 | 125 | 63 h |
| `nuclear_mw` | 48 | 24 | 15 h |
| `consumption_mw` | 24 | 24 | 24 h |
| `wind_capacity_mw` | 10 | 4 | 4 h |

Installed capacity is a **slow step series** — a reporting gap carries no
missing information, so dropping the hour is wrong. Proper handling
recovers 159 rows:

```
as-is dropna                32,945
+ ffill capacity columns    33,080   (+135)
+ interpolate short gaps     33,104   (+159)
```

This is also the cause of the 126-row frame shrink observed after the
last data refresh, and it is the same class of silent-truncation bug as
the stale-snapshot incident that invalidated a batch of August results.

**Fix:** forward-fill `solar_capacity_mw` / `wind_capacity_mw` (step
series, `ffill().bfill()`), interpolate `nuclear_mw` / `consumption_mw`
with `limit=3`, and leave the forecast series alone (they have no gaps).
Apply inside `load_grid()` in `studies/exp_extra_features.py:97` so every
study benefits, next to the existing `GRID_COLUMNS` whitelist.

## Genuine gaps — data to load

All confirmed against the Fingrid portal, IDs verbatim, not guessed.

| what | dataset | why it is needed |
|---|---|---|
| **Realised wind production** | **75** "Wind power generation - 15 min data" | The store holds only the day-ahead forecast (246). Without the realised series, Fingrid's own wind forecast error is an unquantified row in the error budget — so a Stage-A miss cannot be attributed to our physics versus their forecast. |
| **Realised consumption** | **124** "Electricity consumption in Finland" | The store holds only the day-ahead forecast (165). Realised minus day-ahead is the clean measure of demand response / load shifting, and the only way to quantify Fingrid's consumption forecast error. |
| **FI–NO flow** | **57** / **405** | Finland–Norway is a real interconnector and is entirely uncollected. Needed for a complete import/export picture; the other three borders are collected. |

Resolution note: 75 and 124 are 15-minute from 2023-06-13 and hourly
before, so the store's hourly-mean resampling applies — the same path the
flow series already use.

## Confirmed NOT obtainable — do not chase

- **Realised national solar production does not exist.** Fingrid
  publishes only forecasts (247, 248) and installed capacity (267).
  Distributed PV is not TSO-metered, so every national figure is
  modelled. This closes the "calibrate the solar sub-model against
  realised production" idea permanently — 247 is the correct target, and
  on information-set grounds the better one for a price model anyway.
- **FI↔SE day-ahead transfer capacity ended 2024-10-29** (datasets
  24/25/26/27 and planned 142–145). Only intraday remains (38/44/39/45),
  and Fingrid defines it as *free capacity after the day-ahead market* —
  post-auction, so it must be lagged.
- **FI↔EE day-ahead capacity has now stopped too.** `ntc_ee_fi_da_mw` /
  `ntc_fi_ee_da_mw` end **2026-10-02**, matching the portal's
  "no updates after 29.9.2026" notice. The only market-available-at-gate
  capacity series we had is gone; anything built on it needs a successor
  or must fall back to the lagged intraday reconstruction.

## Also worth loading, lower priority

- **Planned nuclear availability.** The store holds *realised* nuclear
  (188) — the only realised series in the feature set, and the one
  variable that failed every test. The market prices *planned*
  availability. `api_client.fetch_nuclear_outage_schedule` already
  fetches the UMM schedule but feeds only the dead legacy model. (Note
  the UMM experiment separately found no day-ahead leverage, so this is
  for completeness, not expected gain.)

## Work items

1. **Gap handling in `load_grid()`** — step-series ffill, short-gap
   interpolation. Recovers 159 analysis hours and removes a silent
   truncation path. Touches `exp_extra_features.py`, which is in
   `HARNESS_MODULES`, so it costs one baseline re-record.
2. **Add datasets 75, 124, 57 to `config/regions/finland.yaml`**
   `grid_sources`, following the existing entry pattern
   (`name`/`env_key`/`url`/`feature_name`/`max_value: 1`). They land as
   new parquet columns and are **opt-in** via `load_grid_extra()`, so
   they cannot disturb the shared frame.
3. **Refresh the store** — needs `FINGRID_API_KEY`; the `.env` key is now
   the rotated one. The column-check fix in the pending
   `src/data_store.py` change is a prerequisite, because
   `_missing_ranges` compares time coverage only and would otherwise
   ignore the new columns entirely.
4. **Quantify the two forecast errors** the new series unlock: MAE/bias
   of Fingrid 246 vs 75 and 165 vs 124, by lead time. These are error-
   budget rows that are currently blank, not model changes.
5. **Re-record the baseline** — the store has moved (`7400c845…` →
   `a509dce1…`), so the recorded figures are already void; re-record once
   after 1–3 rather than between each.
6. **Document in `docs/BACKLOG.md`**: the EE day-ahead capacity series
   ending 2026-10-02, and that realised solar does not exist.

## Verification

```
python -m src.data_store update --years 4     # FINGRID_API_KEY from .env
python -m src.data_store status                # all sources end today
python studies/backtest_harness.py
python studies/verify_baseline.py --record
pytest tests -q                                # 690 pass today
```

Checks that matter:
- Grid parquet has 18 columns; the 3 new ones non-null over their
  published span.
- `build_dataframe` still returns **49 columns** with no `ntc_`/`flow_`/
  new columns leaked in — the whitelist is now load-bearing and was
  verified to hold at 15 columns (49 cols, zero leakage).
- Frame row count **rises** by ~159 relative to the as-is dropna, not
  falls.
- `verify_baseline.py` reports only `data_store_*` changed before the
  re-record, and all MATCH after.

## Appendix — parked architecture plan

**Activation evidence gathered since parking** (full detail in the
2026-10-16 handoff §5):

- Contemporaneous neighbour prices are worth **+19 pp** of R² (EE
  alone), against +0.59 pp for realised nuclear and ~0.6 pp for all four
  border flows combined. That is the quantitative case for the parked
  neighbour-forecasting work, and it dwarfs everything else tested.
- `RL_PROXY` already delivers part of the conditional-spread effect this
  plan predicted — p90+ MAE 68.24 → 63.19 with no transform — because
  residual load SQUARED captures merit-order convexity (marginal
  response +2.97 → +12.26 EUR/MWh per GW from the mean to +3 GW).
- Trigger (2), "the amplitude defect survives the VST", is now cheaper
  to test: a convex residual-load term attacks it directly and is
  already measured.


The market-follower rebuild is **parked** (decision 2026-10-13): there is
not yet evidence differentiating it from continued incremental work on
#29, which has a well-understood baseline against main.

Most of what I had framed as architecture is not: three production bugs
(`Y_fi_lag168` identically zero at `coordinator.py:1986`; L3 AR(1) fed a
floored forecast error from a possibly-future anchor; published prices
predicted rather than passed through), one missing measurement tool
(nothing can score the 1–36 h window — `backtest_harness` has no lead
structure, `honest_horizon_study` is deliberately pre-auction,
`leadtime_fill_study` raises `KeyError: 'Y_se1'` and its recorded D+2
numbers describe the v2.16 leaky model and must not be cited), and one
measured transform (asinh VST, −5.8 %). Only the neighbour forward-state
and the conditional spread are genuinely architectural.

Expected from the incremental queue: ~8–12 % MAE, concentrated in the
first 36 h. Not a route to LEAR's 0.516 — beyond 72 h no measured channel
beats climatology (origin-anchored FI AR: +69.7 % @1 h, +7.9 % @24 h,
~0 % @72 h; `phi = 0.9389`).

Activation triggers for the rebuild: per-lead rMAE beyond D+2 stays flat
near 0.82 after the incremental queue; and the amplitude defect survives
the VST (slope far below the actual 1.330, `tail p95` bias ≈ −MAE).
