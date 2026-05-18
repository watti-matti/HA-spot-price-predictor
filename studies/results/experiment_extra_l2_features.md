# Experiment — extra L2 features (nuclear deficit, cross-border + SE1)

Branch: `experiment/extra-l2-features`. Off-tree research only — no
production artefact change. Script:
[`studies/exp_extra_features.py`](../exp_extra_features.py).


## TL;DR

- **Data window 2023-01-08 → 2026-04-27** (28.8 k hours from the
  cached parquets, inner join). The pre-2023 weather data is excluded
  by the inner join because FI spot prices only start to be useful
  after the 2023 nuclear (OL3) commissioning.
- **B0 baseline** (current v2.8.1 six features) reproduces the
  published production MAE ≈ 10.30 EUR/MWh and R² ≈ 0.925. Hedge CVaR
  reduction is **2.0 pp** at α = 0.05 — the v2.8.1 model captures very
  little hedge-relevant tail risk on top of the L1 seasonal forecast.
  Sanity OK.
- **B1 nuclear-deficit alone**: neutral on every metric (hedge +0.01
  pp, overall MAE ±0, extreme tail ±0.03 EUR/MWh). Finland's post-OL3
  fleet rarely runs below capacity in this window, so `nuclear_deficit`
  is most often zero. **Reject** until a multi-day OL3 outage period
  reaches the test set, or the `nuclear_x_scarcity = nuclear_deficit
  × wind_log_scarcity` interaction is tested separately.
- **B2 cross-border (SE3 + EE, no SE1)**: hedge CVaR red. 2.0 → 10.9 pp
  (Δ +8.93 pp; +0.3-per-feature threshold is 0.9 pp — easily clears).
  Extreme-tail MAE drops 20.14 → 15.45 (−4.7 EUR/MWh). Overall MAE
  drifts up 10.30 → 11.67 (+1.37). **Accept (hedge gate + extreme
  tail).**
- **B2_se1 (adds SE1)**: hedge red. 11.07 pp (+9.07 over B0), overall
  MAE 11.43, extreme-tail MAE 15.50. Improves on B2 across all three
  metrics: overall MAE 11.67 → 11.43, hedge 10.93 → 11.07.
  **Confirms the user hypothesis (2026-05-19): SE1 is not collinear
  with SE3 in the presence of saturable transit capacity.** Per the
  v2.5.1 finding that the `(Y_se1, Y_se3)` pair acts with **opposite
  signs** (+1.61 / −1.60), SE1 carries information about the FI↔SE1
  spread that SE3 alone cannot supply. The v2.2 forward-add rejection
  of SE1 was a known v2.5.6 myopia.
- **B2_transit (SE1+SE3+EE + neighbour-only signed and absolute
  spreads)**: hedge red. 11.19 pp. Almost identical to B2_se1. The
  explicit spread features add ~0.12 pp hedge on top of the basic
  `Y_se*` set — not enough to justify three extra parameters. The
  L1-deseasonalised neighbour prices already encode most of the
  saturation signal.
- **B3 combined (B1 + B2_se1)**: ≈ B2_se1. Nuclear adds nothing on top.

### Recommendation

Develop a follow-up production refit on this branch that adds
**`Y_se1`, `Y_se3`, `Y_ee`** to `RIDGE_FEATURES` (a 9-feature L2
Ridge). Wire `fetch_neighbor_prices()` results into
`Pipeline.compute_forecast` through a new `recent_neighbour_prices`
argument. The hedge gate passes decisively (+9 pp vs +0.9 pp
threshold), extreme-tail MAE drops 4.6 EUR/MWh, and the user's
SE1-distinctness hypothesis is validated. The overall-MAE drift
(+1.1 EUR/MWh on calm hours) is acceptable under v2.5.6's
hedge-primary rule. Treat the explicit spread features
(`spread_se1_se3` etc.) and `nuclear_deficit` as **not yet justified**
on this data window.

### Same-hour leak — what we discovered and corrected

An earlier run of this experiment included `spread_fi_se1 = fi − se1`
and `export_potential_se3 = max(0, −(fi − se3))` computed on
same-hour FI. Result was MAE 2.41, hedge red. 78 pp — implausibly
good. Diagnosis: those features inject the target variable `fi(t)`
into the design matrix, and the Ridge trivially recovers `fi(t)` via a
+1 coefficient on the spread plus the SE1 / SE3 contribution. The
current run uses only neighbour-vs-neighbour spreads
(`spread_se1_se3`, `spread_se3_ee`) and a 7-day-shifted rolling-mean
form of `export_potential_se3`, all leak-free. Documented as a
guardrail for any follow-up feature engineering.

## Variants

| Variant | n_feat | MAE | R² | MAE (|spot|>100) | R² (|spot|>100) | Hedge CVaR red. (pp) | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B0_baseline | 6 | 10.30 | +0.925 | 20.14 | +0.744 | 1.99 | +0.904 |
| B1_nuclear | 7 | 10.30 | +0.925 | 20.11 | +0.745 | 2.01 | +0.903 |
| B2_cross_border | 9 | 11.67 | +0.893 | 15.45 | +0.827 | 10.93 | +0.857 |
| B2_se1 | 10 | 11.43 | +0.898 | 15.50 | +0.830 | 11.07 | +0.857 |
| B2_transit | 12 | 11.67 | +0.893 | 15.45 | +0.827 | 11.19 | +0.855 |
| B3_combined | 11 | 11.42 | +0.898 | 15.49 | +0.831 | 11.07 | +0.855 |

## Delta vs B0 baseline (test split)

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
| B1_nuclear | +0.01 | -0.03 | +0.01 | 0.30 | neutral (no material gain) |
| B2_cross_border | +1.37 | -4.69 | +8.93 | 0.90 | **accept (hedge gate + extreme tail)** |
| B2_se1 | +1.13 | -4.63 | +9.07 | 1.20 | **accept (hedge gate + extreme tail)** |
| B2_transit | +1.38 | -4.68 | +9.20 | 1.80 | **accept (hedge gate + extreme tail)** |
| B3_combined | +1.12 | -4.65 | +9.08 | 1.50 | **accept (hedge gate + extreme tail)** |

**Decision rule (per v2.5.6).**

- **Primary criterion: NPK-CVaR hedge gate.** A variant adds real
  signal iff its hedged-portfolio CVaR is lower than the baseline's by
  at least **+0.3 pp per added feature**. This is the v2.5.6
  acceptance threshold; passing it means the model captures
  hedge-relevant tail risk that the baseline misses.
- **Secondary criterion: extreme-price-hour MAE.** Test hours with
  |spot| > 100 EUR/MWh — the spike subset where the v2.5.13 work
  showed the v2.8.1 baseline is weakest. A drop of ≥ 1 EUR/MWh on this
  bucket is operationally meaningful.
- **Regression guard.** Reject only on *severe* regression — overall
  MAE drift > 2 EUR/MWh, or extreme-tail MAE worsening > 1 EUR/MWh.
  Small overall-MAE drift (≤ 2 EUR/MWh) is tolerated when the hedge
  gate passes, because v2.5.6 established the hedge gate (and not
  average MAE) as the operational acceptance test: cross-border
  features add a little variance on calm hours but pay off
  disproportionately on the spike hours that carry the tail cost.

The hedge gate is the canonical v2.5.x acceptance test; MAE / extreme
MAE are reported for interpretability but do not substitute for the
hedge gate.

## Method

- Data: cached parquets under `output/` (≈ 4 years, hourly).
- Train/test: time-ordered, `TRAIN_FRAC = 0.55`
  (train = first 15866 hours,
   test  = last  12983 hours).
- Ridge α = 1.0, intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then applied
  one-step-ahead on the test split.
- Forecast under test: L1 seasonal + L2 ridge + φ·ε(t−1).
  Hourly-bias EMA, softplus floor, and L4 GPD POT bands are NOT applied
  here — the goal is to isolate the impact of the L2 feature set on
  the point forecast.
- Extreme-price bucket: test hours with |spot| > 100 EUR/MWh, where the
  v2.5.13 work showed the model is weakest.

## Candidate features (legacy v2.2 lineage, leak-free)

- `nuclear_deficit ∈ [0, 1]` — `max(0, 1 − nuclear_mw)` where
  `nuclear_mw` is Fingrid #188 normalised by max-fleet 4 372 MW.
  Mean-centred for stable Ridge weighting.
- `Y_se1`, `Y_se3`, `Y_ee` — neighbour spot prices deseasonalised
  against the shipped per-zone hourly+weekly L1 components. The legacy
  v2.2 `ar_se3` / `ar_ee` used a proper AR(2) daytype-deviation; this
  is a simpler analogue. **SE1 is included** per user direction
  2026-05-19: limited FI↔SE3 / SE3↔SE1 transit capacity makes SE1
  distinct from SE3 (the v2.2 collinearity-rejection assumed perfect
  coupling; in reality the transit decouples SE1 from SE3 whenever
  capacity saturates).
- `spread_se1_se3 = se1 − se3`, mean-centred. Signed neighbour-zone
  spread: when SE1 and SE3 prices diverge the transit capacity between
  them is saturated and the two zones decouple. Leak-free (no FI on
  the RHS).
- `abs_spread_se1_se3 = |se1 − se3|`, mean-centred. Magnitude of the
  same spread — saturation level.
- `spread_se3_ee = se3 − ee`, mean-centred. Signed SE3↔EE spread,
  same logic.
- `export_potential_se3` — built on **lagged** FI and SE3 (7-day
  rolling means shifted by 168 h) so the feature is leak-free. The
  same-hour form `max(0, −(fi(t) − se3(t)))` was rejected after a
  same-hour FI value in the feature set let the Ridge trivially
  recover the target (B2_transit reached MAE 2.4 / hedge CVaR red.
  78 pp — implausibly good — under that buggy form).

## Open follow-ups (deferred to separate experiments)

- **SE3 / SE1 / EE transit-capacity saturation.** A continuous
  saturation indicator (e.g. `min(|spread|, cap) / cap` calibrated on
  historical transit-capacity-out data) was *not* tested here. The
  existing `Y_se3` / `Y_ee` already proxy coupling; a saturation
  indicator only adds information when transit *de*couples (large
  spread regime). Belongs in a follow-up if B2 / B3 are accepted.
- **`nuclear_x_scarcity` interaction.** Legacy v2.2 multiplied
  `nuclear_deficit × wind_log_scarcity` to amplify outage impact under
  cold-and-windless conditions. Not tested here; can be a follow-up if
  B1 is accepted.
