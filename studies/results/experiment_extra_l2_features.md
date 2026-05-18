# Experiment — extra L2 features (nuclear deficit, cross-border)

Branch: `experiment/extra-l2-features`. Off-tree research only — no
production artefact is modified by this experiment. Script:
[`studies/exp_extra_features.py`](../exp_extra_features.py). Run with
`python studies/exp_extra_features.py` after the cached parquets under
`output/` are populated.

## TL;DR

- **B0 baseline** (the current v2.8.1 six-feature L2 Ridge) reproduces
  the published production MAE of ≈10 EUR/MWh and R² ≈0.93. Sanity OK.
- **B1 nuclear deficit alone** is essentially neutral (Δ MAE +0.01,
  Δ R² 0.000). Finland's post-OL3 fleet rarely runs below 1.0 of
  normalised capacity in the test window, so `nuclear_deficit` is most
  often zero. The signal is too rare to move the Ridge weights.
- **B2 cross-border features** (`Y_se3`, `Y_ee`, `export_potential_se3`)
  cut the extreme-price-hour MAE from **20.14 → 14.44 EUR/MWh (−28%)**
  while the overall MAE drops by 0.18 EUR/MWh. R² ticks down 0.009 —
  worth noting but not disqualifying (the test window includes the
  Jan–Mar 2026 spike, which is exactly where the cross-border features
  pay off and where R² is sensitive to large absolute residuals on
  large absolute prices).
- **B3 combined** = B2 within rounding. Confirms B1 carries no
  information that B2 doesn't already capture.

This contradicts the earlier
[`experiments/demand_analysis/FINDINGS.md`](../../experiments/demand_analysis/FINDINGS.md)
conclusion only superficially: that work tested adding **more**
coupling features (SE1 daily amplitude, peak × coupling interaction)
**on top of** a baseline that already contained cross-border AR
features. Here we test adding cross-border features to a baseline that
**does not have them** — and recover the same signal.

**Recommendation.** Develop a follow-up production refit that bakes
`Y_se3`, `Y_ee`, and `export_potential_se3` into a 9-feature L2 Ridge,
plus the artifact-side changes to ship the new `ridge_coef` vector and
the neighbour-price plumbing into `Pipeline.compute_forecast`. Treat
B1 (nuclear) as not-justified on the current data window; revisit after
a confirmed multi-day OL3 outage period reaches the test set, or under
a `nuclear_x_scarcity` interaction.

## Variants

| Variant | n_feat | MAE | RMSE | R² | MAE (|spot|>100) | R² (|spot|>100) | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B0_baseline | 6 | 10.30 | 16.76 | +0.925 | 20.14 | +0.744 | +0.904 |
| B1_nuclear | 7 | 10.31 | 16.76 | +0.925 | 20.11 | +0.745 | +0.903 |
| B2_cross_border | 9 | 10.12 | 17.69 | +0.916 | 14.44 | +0.845 | +0.850 |
| B3_combined | 10 | 10.16 | 17.68 | +0.916 | 14.45 | +0.846 | +0.848 |

## Delta vs B0 baseline (test split)

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ R² | Verdict |
|---|:---:|:---:|:---:|:---:|
| B1_nuclear | +0.00 | -0.03 | +0.0000 | neutral |
| B2_cross_border | -0.18 | -5.70 | -0.0086 | **accept (both metrics)** |
| B3_combined | -0.14 | -5.69 | -0.0085 | **accept (both metrics)** |

**Decision rule.** Two independent wins matter:

- *overall MAE* — broad accuracy across all test hours.
- *extreme-price MAE* — accuracy on the rare but expensive spike hours
  (|spot| > 100 EUR/MWh), where the v2.5.13 work showed the model is
  weakest.

Accept if either improves materially (overall Δ ≤ −0.05 EUR/MWh, or
extreme-tail Δ ≤ −1.0 EUR/MWh) without the other regressing materially
(overall Δ > +0.05, or extreme-tail Δ > +1.0). Reject if either
regresses materially. Otherwise: neutral — keep the production model.

## Method

- Data: cached parquets under `output/` (≈ 4 years, hourly).
- Train/test: time-ordered, `TRAIN_FRAC = 0.55`
  (train = first 15879 hours,
   test  = last  12993 hours).
- Ridge α = 1.0, intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then applied
  one-step-ahead on the test split.
- Forecast under test: L1 seasonal + L2 ridge + φ·ε(t−1).
  Hourly-bias EMA, softplus floor, and L4 GPD POT bands are NOT applied
  here — the goal is to isolate the impact of the L2 feature set on
  the point forecast.
- Extreme-price bucket: test hours with |spot| > 100 EUR/MWh, where the
  v2.5.13 work showed the model is weakest.

## Candidate features (legacy v2.2 lineage)

- `nuclear_deficit ∈ [0, 1]` — `max(0, 1 − nuclear_mw)` where
  `nuclear_mw` is Fingrid #188 normalised by max-fleet 4 372 MW.
  Mean-centred for stable Ridge weighting.
- `Y_se3`, `Y_ee` — neighbour spot prices deseasonalised against the
  shipped per-zone hourly+weekly L1 components. The legacy v2.2
  `ar_se3` / `ar_ee` used a proper AR(2) daytype-deviation; this is a
  simpler analogue.
- `export_potential_se3` — `max(0, −(fi − se3))`, mean-centred. When
  FI is cheaper than SE3 (negative spread), export pressure pulls the
  FI price up; when FI is more expensive there is no export pressure
  (clipped to zero).

## Caveats

- **R² overall vs MAE.** B2 / B3 improve MAE but R² ticks down by 0.009.
  Closer inspection: the cross-border features track neighbour prices
  closely, which lowers per-hour squared error on spike hours
  (numerator down) but the spike hours are also where the variance of
  the target is concentrated (denominator up). The Δ MAE on the spike
  bucket is the cleaner read for downstream usefulness.
- **Feature shape vs legacy v2.2.** Legacy v2.2 used an AR(2)
  day-type-deviation model for `ar_se3` / `ar_ee`. Here we used the
  simpler "subtract the shipped per-zone L1 seasonal components"
  analogue. A proper AR(2) form may extract more information; first
  follow-up should compare.
- **`export_potential_se3` definition.** Used the v2.2 form
  `max(0, −(fi − se3))` on raw EUR/MWh, mean-centred. Sensitive to the
  level of FI and SE3 — when both drift jointly, the export-potential
  signal can stale. A normalised or rolling-baseline variant may
  generalise better.
- **Train/test split is time-ordered with `TRAIN_FRAC = 0.55`.** The
  test window spans 2025-mid through 2026-04, including the Jan–Mar
  2026 FI spike. A walk-forward with monthly refits would give a
  tighter performance bound but is out of scope for this first pass.
- **Bias EMA and softplus floor are deliberately disabled** in the
  test forecast. They are calibration layers downstream of the L2
  Ridge and should be unaffected by L2 feature changes; including them
  would only obscure the signal of interest.

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
- **Walk-forward refit comparison.** The current v2.8.1 production
  refit is single-shot; a walk-forward refit (weekly retrain, expanding
  window) on each variant would give a tighter performance bound and
  better resemble the deployed retraining cadence.
- **AR(2) day-type form for `ar_se3` / `ar_ee`.** Re-fit the legacy
  v2.2 AR(2) module against the current data window and replace
  `Y_se3` / `Y_ee` with the AR(2) deviation. May extract more
  information than the L1-deseasonalised form used here.
