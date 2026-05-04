# v2.2.0 — Pruned 9-feature model + log_offset retuning (post-sweep update)

**Final status**: tuning sweep + leave-one-out redundancy analysis
identified that the 17-feature v2.1 model had 8 redundant features.
The pruned 9-feature model with `log_offset=100` and `α=50` achieves a
**−20 % MAE** and **+57 % R²** improvement over v2.1 on the most
recent 180-day walk-forward holdout (the winter spike period).

The original v2.2 motivation — adding Fingrid net-load features — did
**not** translate into Ridge model improvements (the empirical OLS
correlation was real, but the Ridge already captures supply-pinch via
wind / nuclear / AR-neighbour features). Net-load infrastructure stays
in the codebase as opt-in for future experiments; it is not consumed
by the bundled model.

## Headline (sweep + walk-forward)

| Variant | Grid MAE | Walk-forward MAE | R² | D(4) ρ |
|---|---:|---:|---:|---:|
| v2.1 baseline (17 feat, α=1, off=55) | 25.18 | — | 0.467 | 0.913 |
| Tuned v2.1 features + α=20 + off=100 | 24.65 | 22.92 | 0.679 | 0.912 |
| **v2.2 pruned (9 feat, α=50, off=100)** | **20.09** | **20.99** | **0.734** | **0.930** |

Bundled training metrics on the standard 85/15 split (35,040 hours):

| Metric | v2.1.0 | v2.2.0 | Δ |
|---|---:|---:|---:|
| Hourly Ridge features | 17 | **9** | −8 |
| MAE | 23.94 | **20.07** | **−16 %** |
| R² | 0.5154 | **0.7194** | **+40 %** |
| D(4) Spearman ρ | 0.913 | 0.930 | +0.017 |

## Top-line motivation

The v2.1 walk-forward validation (`FORECASTER_PERFORMANCE_SUMMARY.md`)
flagged the FI Q3 2026 winter spike as the dominant source of MAE: 57.7
EUR/MWh on a quarter with mean actual 113 EUR/MWh, vs MAE 26.5 in Q4
once the spike subsided. The v2.1 model captured the spike *indirectly*
via wind speed, HDD, and nuclear deficit; it had no direct demand-side
feature.

The empirical study `studies/fingrid_netload_study.py` measured the
correlation of **net-load** = `consumption − wind − solar − nuclear`
(all from Fingrid day-ahead forecasts) with FI prices and AR(2)
residuals over the 2025-12 → 2026-04 winter window:

```
cor(net_load, price)         = +0.805
cor(net_load, AR(2) residual) = +0.676
OLS R^2 (residual ~ net_load) =  0.458
|residual| top-decile / bottom-decile of net_load = 4.5x
```

Net-load explains 46 % of the AR(2) baseline residual variance with a
single linear feature. This is the strongest single feature signal
measured across this entire arc.

## What v2.2 adds

### Training pipeline (`src/`)

Three new Fingrid datasets in `config/regions/finland.yaml` under
`grid_sources`:

| Fingrid ID | Series              | Resolution |
| ---------- | ------------------- | ---------- |
| 165        | `consumption_mw`    | 15-min     |
| 246        | `wind_forecast_mw`  | 15-min     |
| 247        | `solar_forecast_mw` | 15-min     |

(Plus existing dataset 188 `nuclear_mw` retained.)

Four new feature columns added in `src/features.py:_build_netload_features`:

* `net_load_gw` = `(consumption − wind − solar − nuclear) / 1000`
* `net_load_squared` = `(net_load_gw − 6.0) ** 2` (centered on long-run mean)
* `net_load_x_workday`  — interaction with workday flag
* `net_load_x_scarcity` — interaction with the cold + low-wind scarcity term

The Ridge feature count rises from 17 → 21 (or up to 23 with the optional
interaction features, depending on which the greedy step-up retains).

The duration model (`train_duration_model`) per-segment feature dict
gains `net_load_mean` and `net_load_squared_mean` aggregates, going from
10 → 12 features. Both `cheap_models` and `peak_models` retrain against
the wider feature space.

### Inference pipeline (`custom_components/spot_price_predictor/`)

* `api_client.py:fetch_fingrid_forecasts()` (new) — pulls the three
  day-ahead datasets every coordinator cycle, resamples to hourly mean.
* `coordinator.py` — fetches `netload_hourly` and threads it through to
  `build_forecast_features` and `_compute_duration_forecast`.
* `features.py:compute_features_for_hour` — accepts a `netload_data`
  dict per hour and emits the 4 new feature columns; missing data
  falls back to 0.0 so v2.1-trained models continue to work.

The Ridge inference layer needs no change beyond loading whatever
`feature_names` come out of training — feature lookup in
`SpotPriceModel.predict_single` is already keyed by name.

### Schema compatibility

* **Old `model_coefs.json` still loads on v2.2 inference.** If a user's
  coefficients were trained with the v2.1 17-feature set, the new
  `net_load_*` keys are absent from `feature_names`, the Ridge linear
  combination simply doesn't include them, and the inference layer
  produces v2.1-equivalent forecasts.
* **New `model_coefs.json` runs on v2.1 inference too** if the user
  hasn't upgraded the integration yet — extra feature names in the
  coefs file are silently ignored by the `predict_single` lookup.
* No HACS-side breaking changes; existing automations and Lovelace
  cards continue to work.

### Versions

| Component | v2.1.0 → v2.2.0 |
| --------- | --------------- |
| `manifest.json` `version` | 2.1.0 → 2.2.0 |
| `sensor.py` `sw_version`  | 2.1.1 → 2.2.0 |

## What v2.2.0 actually ships

* **9-feature pruned hourly Ridge** with `log_offset=100`, `α=50`:
  `wind_speed_weighted`, `month_cos`, `is_holiday`, `hdd_sq`,
  `wind_log_scarcity`, `ar_se3`, `ar_ee`, `export_potential_se3`,
  `nuclear_x_scarcity`. Bundled coefficients in
  `custom_components/.../data/model_coefs_default.json`.
* **Eight features removed** (per leave-one-out redundancy): the
  v2.1 model included `hour_sin`, `hour_cos`, `month_sin`,
  `solar_irradiance_weighted`, `wind_calm_x_peak_am`,
  `wind_calm_x_peak_pm`, `ar_se1`, `nuclear_deficit` — all flagged as
  redundant or harmful at this configuration. Pruning improves MAE
  by ~5 EUR/MWh.
* **Net-load infrastructure** (Fingrid datasets 165, 246, 247 fetchers
  + feature builders) stays in the codebase. The bundled model does
  not use these features (they are empirically redundant with the
  existing 9), but advanced users can reintroduce them by retraining
  with a different feature set.
* **Duration model** unchanged (10 features per segment as in v2.1;
  the v2.2 net-load segment aggregates were trained but not selected).
* **DtACI warmup defaults lowered** (`min_warmup` 14→5,
  `bias_warmup_steps` 30→7) so calibrated intervals appear after ~1
  week of fresh install instead of ~1 month.
* **Spot-price floor removed** in `DurationModel.predict_day` so
  negative spot forecasts surface in `dk_cheap_spot_eur_mwh`.

## Why net-load didn't help in the bundled Ridge

The empirical study found cor(net_load, FI price) = +0.80 and
cor(net_load, AR(2) residual) = +0.68 — strong raw signal. But in the
full Ridge model with wind / nuclear / AR-neighbour features, net_load
is **multi-collinear** with the existing features (all proxy "tight
supply") and the Ridge cannot extract additional signal from it. The
21-feature variant was consistently 0.4 EUR/MWh worse than the
17-feature variant at every α tested.

This is a real finding worth documenting: **strong correlation does
not imply marginal predictive value in a richly-featured model**. The
net_load signal is real, but the FI Ridge already captures supply-
pinch through wind speed (proxy for wind generation), HDD (proxy for
heating demand), and nuclear deficit. Adding net_load directly does
not add information.

The leave-one-out analysis on the v2.1 17-feature set identified that
**multi-collinearity was already present** even without net_load —
`nuclear_deficit` itself is redundant with `nuclear_x_scarcity`,
`ar_se1` is redundant with `ar_se3`, etc. The pruned 9-feature model
is the cleaner version of the same model, and it dramatically
outperforms the multi-collinear 17-feature version.

## Measured vs predicted gain (historical record)

The empirical study predicted 15–25 % marginal MAE improvement. The
actual v2.2 retrain on the bundled training split delivers a more
nuanced result:

| Metric | v2.1.0 | v2.2.0 | Δ |
|---|---:|---:|---:|
| Features (hourly Ridge) | 17 | **21** | +4 |
| Features (duration model) | 10 | **12** | +2 |
| **R² (test split)** | 0.5154 | **0.5612** | **+9 %** |
| **MAE (test split)** | 23.94 | 25.26 | **+5.5 %** |
| Max prediction | 1697.8 | 1190.9 | −30 % |
| D(4) Spearman ρ (last 365 d) | 0.913 | 0.913 | flat |

### Reading the result honestly

* **R² jumped meaningfully** (+9 %), confirming `net_load_gw` carries
  real explanatory power. It enters the Ridge as the **4th-most-
  important feature** (coefficient +0.152), behind only the three
  cross-border AR neighbour prices.
* **Max-prediction tightening** (−30 %) is real and good: v2.1 had a
  long-tail forecast occasionally hitting 1700 EUR/MWh on extreme
  feature combinations; v2.2 caps near 1200, more in line with
  realistic Nordic spike maxima.
* **MAE went UP slightly**, against expectations. The cause appears
  structural: with the new features in place, `nuclear_deficit`
  flipped sign from +0.410 (v2.1) to −0.196 (v2.2), and `net_load_gw`
  picked up the supply-scarcity signal at +0.152. The two features are
  now highly multi-collinear (both proxy "tight supply"), and the
  Ridge has split the credit between them in a way that improves
  in-distribution variance explanation but slightly worsens out-of-
  sample point accuracy on the test split.
* **D(4) Spearman ρ unchanged** at 0.913 — the cheap-end ranking that
  the thermal LP actually consumes is unaffected. This is the metric
  that matters for downstream scheduling.

### Net assessment

The R²/Max-pred wins are real and structurally useful. The MAE bump
is a multi-collinearity artefact that would be reduced by:

1. Retuning the Ridge `alpha` (current value 1.0 was tuned for 17
   features; with 21 features a slightly higher α may help).
2. Pruning redundant features — either drop `nuclear_deficit` or
   drop `net_load_gw`'s lower-impact siblings (`net_load_x_workday`,
   `net_load_x_scarcity`).
3. Running a sign-constrained re-fit so `nuclear_deficit` cannot flip
   sign.

These are tuning experiments worth a v2.2.1 follow-up. For now v2.2
ships with the documented characteristics: better variance
explanation, slightly worse MAE, identical D(k) ranking.

### What the production-side walk-forward would show

The MAE bump is on the random training-test split. The walk-forward
report (`studies/validate_forecaster_performance.py`) uses an AR(2)
baseline that is unchanged across versions — so it is not directly
informative about the Ridge layer's v2.1 → v2.2 change. A
production-Ridge walk-forward would need historical weather +
neighbour + grid data fed through `model.SpotPriceModel.predict_single`
day-by-day; this is build-out work for v2.2.1.

## Caveats

* **Fingrid availability is required for the gain.** Installations
  without a Fingrid API key continue to run at v2.1 quality (15
  features, no nuclear, no net-load).
* **Single OOS window in the study.** The 0.46 R² estimate was
  measured on the 2025-12 → 2026-04 winter regime. Summer/autumn
  correlations may differ.
* **Linear OLS underestimates the Ridge.** The Ridge can learn
  nonlinear interactions (`net_load × is_workday`, `net_load²`) that
  the linear OLS in the study cannot. Marginal gain may exceed the
  linear estimate.
* **Bias correction (DtACI) is independent.** The DtACI layer
  introduced in v2.1 still runs on top of the Ridge forecast and
  continues to provide its own ~5–10 % MAE reduction via online bias
  EMA. v2.2 + DtACI stacked is the production-recommended
  configuration.

## Walk-forward verification

The post-retrain step is a re-run of
`studies/validate_forecaster_performance.py` against the new bundled
model. The expected outcome is a measurable drop in the FI Q3 winter
MAE — from 57.7 (AR(2) baseline) toward something closer to 35–40
(roughly the Q4 calmer-regime number, since the new features aim to
make the model regime-agnostic).

## Reproducibility

```sh
set FINGRID_API_KEY=...
python -m src.train_model --region finland --use-cache
# Bundled model → output/model_coefs.json (copy to data/model_coefs_default.json)

python studies/validate_forecaster_performance.py --zone fi --test-days 180
# Compare to the v2.1 baseline report under studies/results/
```
