# Session handoff — 2026-10-16

Read this before resuming. It supersedes
`docs/SESSION_HANDOFF_2026-08-12.md` where they disagree.

Branch `fix/retrain-pipeline`, PR #29, 16 commits, +6739/−180.
**The forecast is byte-identical to `origin/main`** — SHA-256 over the
22 inference modules plus the 4 shipped artifacts is
`553f18914d189dd7…` on both. Nothing here changes a prediction. Every
number that moved did so because the measurement path was repaired.

---

## 1. The headline result

**`RL_PROXY`: residual load, built from forecastable components, is
worth +1.70 % MAE and −7.4 % on the price tail.** The only change
measured this session that improves *every* lead day and the high-price
segment at once.

```
variant      MAE    bias    p90+   vs BASE
BASE       26.90   -1.81   68.24   +0.00%
RL_D1      27.79   -3.85   76.25   -3.31%
RL_PROXY   26.44   -0.84   63.19   +1.70%
FI_7D      26.82   -0.71   65.24   +0.28%
COMBO      26.72   +0.12   62.63   +0.67%

by lead day   D+1    D+2    D+3    D+4    D+5    D+6    D+7
BASE        26.70  26.76  26.80  26.80  26.92  27.02  27.03
RL_PROXY    26.26  26.31  26.35  26.35  26.47  26.56  26.58
```

`RL_PROXY` = `consumption − wind − solar`, where wind and solar MW are
projected from the weather proxies (fitted on train only) and
consumption comes from a seasonal + lag-168 offset model. It does not
decay with horizon *because* it avoids Fingrid's day-ahead feed.

`RL_D1`, which uses 165/246/247 directly, **loses 3.31 %** — winning at
D+1 (24.64) then collapsing past the publication boundary (31.64 at
D+7). Lesson, and it reproduces BACKLOG R5: a feature with a
publication boundary must have the boundary modelled at **fit** time,
not faded at inference.

Coefficients over 16 monthly refits, both now constrained `>= 0`:
`rl_linear` mean +2.974 (sd 0.802), `rl_squared` mean **+1.548 (sd
0.097)**, sign-stable 16/16. Marginal response accelerates +2.97 → +12.26
EUR/MWh per GW between the mean and +3 GW — merit-order convexity, and
the first term in this model able to express it. Caveat: the linear term
drifts (+1.54 → +3.85) while the wind coefficient decays (−92.9 → −83.7),
so the linear part absorbs something trending; the squared part does not.

**Not shipped.** It is a measured experiment. Shipping means changing
`pipeline.RIDGE_FEATURES` in five coordinated places, a retrain, and a
calibrator cold start.

## 2. Corrections to earlier claims — believe these, not the old ones

| old claim | correction |
|---|---|
| "69.5 % of price variance is unexplained by fundamentals" | Artifact of testing 14 raw columns. With 72 candidates the **admissible ceiling is R² 0.537**. |
| "The model under-predicts high prices (bias ≈ −MAE)" | Not a bias problem. Theil: bias² **0.03 %** of MSE, dispersion 15.07 %, imperfect correlation **84.90 %**; calibration slope 1.030, CI [0.859, 1.201]. It is **compression**, and 85 % of MSE is unreachable without better information. |
| "Per-site weather is collinear with its aggregate" | Untested assumption, and **false** — two site columns still enter after the national aggregates. |
| "sMAPE 79 % vs published ~5 % shows a gap" | Meaningless across regimes: the naive scores 84.3 % on our own data. Only rMAE carries signal. |
| "D+2 zero-fill costs 13 % (leadtime_fill_study)" | **Retracted.** That script raises `KeyError: 'Y_se1'` against the current artifact; its JSON describes the v2.16 *leaky* model. Since v2.17.0 neighbours are lagged 168 h and known for the whole horizon. |
| "#29's MAE moved because of the row set" | Wrong — it was a real lag-alignment bug (§4). |

## 3. The causal framing that reorganised everything

Nord Pool clears on **expectations**. Bidders submit curves built from
forecasts; the price is the outcome of those forecasts. A **realised**
value is therefore not a better version of the right input — it is an
input from the wrong side of the event, and regressing price on it is
anti-causal.

Of 18 Fingrid series collected, **only 7 are ex-ante** and 2 of those
have ended. Every column in the modelling dataset carries a `causality`
tag.

This retrospectively explains the failures: `nuclear_mw` (188) is the
only realised series in the feature set and the only variable that
failed every test; the cross-border capacity series are "free capacity
**after** the day-ahead market", i.e. post-auction, and fitted +1.397
where physics demands negative.

**Leak-freeness ≠ market-relevance.** A realised value lagged 168 h is
observable at the origin yet still is not what anyone bid on for the
target hour. That is why the lagged neighbour prices correlate only
0.39 — a past *outcome* is a poor proxy for a future *expectation*.

**Slow-variable exception (user's observation, measured):** nuclear
persists at corr 0.774 over 168 h, and a week-old value predicts the
current one within **3.9 %** of level in normal hours (corr 0.900)
versus 16.3 % when disturbed (corr 0.523), disturbed being 29.5 % of
hours. So for slow variables the lagged realised value *is* a good proxy
except during disturbances. Cross-border capacity does **not** share
this: free capacity and flow are both fast (0.385/0.396 at 168 h), so
the slow physical capacity beneath them is unobservable now that the
day-ahead capacity series are discontinued.

**Untested and reopened by this:** lagged realised nuclear as a level,
*gated* by the UMM planned-outage schedule as a disturbance indicator.
The earlier UMM experiment tested UMM alone as a feature (R² 0.001); as
a gate it is a different specification.

## 4. Three measurement defects fixed (the real work)

All three were silent, and all three made results non-reproducible
across data refreshes. Net effect on the headline: **+0.03 MAE**
(24.5145 → 24.5444). They barely moved aggregates while contaminating
every per-row and per-segment figure.

1. **Weather cache split.** The 28 per-site weather series existed only
   in gitignored, per-checkout `studies/.cache/wx_*.parquet`. Two
   clones held different vintages, which produced a false-positive
   branch-vs-production difference of 0.0154 EUR/MWh. Now consolidated
   into `data_store/fi_weather_sites.parquet` by
   `studies/consolidate_weather_store.py`, and the cache hash is part of
   the baseline fingerprint.
2. **Slow series dropped instead of carried.** `build_dataframe`
   inner-joins then `dropna()`s, so one NaN hour in one column deleted
   that hour for every study — 207 hours lost, 125 to
   `solar_capacity_mw` alone. Capacity is a step series; now
   forward-filled, nuclear/consumption short-gap interpolated.
   Recovered 159 hours.
3. **`.shift(168)` moves 168 ROWS, not 168 hours.** The frame is
   non-contiguous, so **4.3 % of rows read a lag from the wrong hour**,
   1–21 h off. Fixing (2) added rows, re-positioned everything after
   them, and moved predictions by up to 18.1 EUR/MWh. Replaced by
   `studies/lag_utils.lag_by_hours` at all 10 sites. Production was
   never affected — `coordinator.py` aligns by timestamp. The parity
   test could not catch it because its synthetic frame is *contiguous*;
   `tests/test_lag_alignment.py` now uses a holed frame.

## 5. What to do next, in order

1. **Ship `RL_PROXY`.** Best measured candidate. Needs the five-place
   feature pin, a retrain, and a cold start. On the stated merge
   criterion it is the first result that could justify unparking the
   architecture work.
2. **Forecast neighbour prices forward.** The oracle block says
   contemporaneous EE price alone is worth **+19 pp** of R², against
   +0.59 pp for realised nuclear and ~0.6 pp for all four border flows
   combined. Earlier work chased sub-1pp effects while this sat
   unexploited. SE1 `phi` 0.966 (half-life ~20 h) says a forward
   forecast is feasible at short horizon.
3. **Fix the three production defects** in
   `docs/MODELLING_DATA_AND_PARKED_DESIGN.md` §"three production bugs":
   `Y_fi_lag168` is multiplied by **zero** at `coordinator.py:1986`
   (shipped coefficient +0.06740, worth ~1.5 % rMAE); L3 AR(1) is inert
   (fed a floored forecast error from a possibly-future anchor); and
   published prices are predicted rather than passed through.
4. **Build a lead-time-resolved harness.** Nothing in the repo can
   score the 1–36 h window: `backtest_harness` has no lead structure,
   `honest_horizon_study` is deliberately pre-auction (06:00 UTC
   origin), `leadtime_fill_study` raises `KeyError`.
5. **asinh VST last** — measured −5.8 % MAE, but it compresses the error
   distribution and would mask structural gaps. Fix inputs first.

## 6. Reproducible producers

```bash
python scripts/reload_training_data.py        # update -> consolidate -> build -> verify
python studies/build_modelling_dataset.py     # 35,040 x 59, causality-tagged
python studies/backtest_harness.py            # then:
python studies/verify_baseline.py             # must say IDENTICAL
python studies/exp_hedge_comprehensive.py     # 72-candidate attribution, HAC
python studies/exp_hedge_to_forecast.py       # converts R^2 into forecast MAE
python studies/exp_statistical_analysis.py    # Theil, calibration, heteroskedasticity
python studies/exp_fresh_data_timeline.py     # FROZEN vs REFIT per month
```

Baseline: `n_eval` 11,303, PRODUCTION MAE **24.5444**, bias −0.6452,
`tail p95` 86.9318. Valid **only** for the recorded fingerprint;
`docs/TRAINING_BASELINE.md` has the protocol and is regenerated from
`BASELINE.json` by script.

## 7. Traps

- **Retraining makes it worse** — 14 of 16 months (−7.7 % to −9.9 % by
  segment, −12.0 % on the top decile). Staleness is not the problem.
- **Comparing across protocols is invalid.** `RL_PROXY`'s 26.44 and the
  baseline's 24.5444 differ because the *protocols* differ (origin-based
  170 h, all leads, 79,050 hours vs the harness's single-pass 11,303).
  Only within-run comparisons count.
- **The data store and weather cache are gitignored**, so a fresh clone
  cannot reproduce numbers without `reload_training_data.py`.
- **`scripts/retrain_model.bat` is kept locally modified with an API
  key — never commit it.** The old key *was* committed in `be35ab4` and
  reached GitHub; it has been rotated and the current `.env` holds the
  new one.
- **Fingrid has no realised national solar series.** Only forecasts
  (247/248) and capacity (267) — distributed PV is not TSO-metered.
  Dataset 267 is "Total production capacity used in the solar power
  forecast", explicitly *not* an official capacity figure.
- **FI↔EE day-ahead capacity ended 2026-10-02**; FI↔SE day-ahead
  capacity ended 2024-10-29. The builder auto-demotes any column whose
  last valid value is >7 days old to `ex-ante-ended`.
- **`sfit` now takes `lower=`** as well as `upper=`. Constrain any
  coefficient whose sign is known a priori.

## 8. Rejected with evidence — do not revisit

Capacity-scaled wind/PV proxies; cross-border transfer capacity as a
level or derate feature; consumption in the price mean (negative even
with perfect foresight); log-space target; `P_week` smoothing;
Norwegian hydro reservoir; UMM nuclear *as a standalone feature*; the
Fingrid day-ahead wind/PV channel (+0.9 %, costs +2.8 EUR/MWh bias);
`RL_D1` (−3.31 %); `FI_7D` as a flat origin-anchored hold (+0.28 %
overall: +6.9 % at D+1, −4.7 % at D+7 — it needs φ-decay, not a hold).

Details in `docs/BACKLOG.md` D5 and the "measured and rejected" blocks.
