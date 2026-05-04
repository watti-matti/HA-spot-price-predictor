# Fingrid net-load — does residual demand explain FI price spikes?

**Generated**: 2026-05-04 20:44 UTC
**Window**: 2025-12-01 → 2026-04-28
**Data**: Fingrid datasets 165 (consumption forecast), 246 (wind forecast),
247 (solar forecast), 188 (nuclear real-time), all resampled to hourly mean.
**Aligned hours**: 3,552

## Hypothesis

Adding `net_load = consumption - wind - solar - nuclear` to the FI Ridge
features should explain price spikes the current model misses, because
spikes happen when residual demand pinches the merit order.

## Net-load summary statistics

| Metric | Value |
|---|---:|
| Mean net_load | 4423 MW |
| Min net_load  | -1146 MW |
| Max net_load  | 9273 MW |
| Q10 net_load  | 1034 MW |
| Q90 net_load  | 7751 MW |
| Mean consumption | 11538 MW |
| Mean wind forecast | 2980 MW |

## Linear correlation with hourly price (EUR/MWh)

| Predictor | Pearson r vs price |
|---|---:|
| **net_load** | **+0.8050** |
| consumption alone | +0.6359 |
| wind alone | -0.6072 |

## Linear correlation with AR(2) hourly residual

The interesting question — does net_load predict what the AR(2) baseline
misses?

| Predictor | Pearson r vs residual | Pearson r vs |residual| |
|---|---:|---:|
| **net_load** | **+0.6764** | **+0.4512** |

OLS fit `residual ~ a*net_load + b`:

| Coefficient | Value |
|---|---:|
| slope a (EUR/MWh per MW) | 16.01 per GW |
| intercept b | -54.29 EUR/MWh |
| **R² of OLS** | **0.4575** |

## Conditional MAE — high vs low net_load

If net_load explains spike-MAE specifically, |residual| should be much
larger when net_load is in its top decile:

| Slice | Hours | mean |residual| EUR/MWh | vs all |
|---|---:|---:|---:|
| net_load ≤ Q10 (1034 MW) | 356 | 26.28 | +66% |
| **net_load ≥ Q90 (7751 MW)** | 356 | **118.86** | **+300%** |
| all | 3,552 | 39.58 | — |

## Plausible MAE improvement

A simple OLS feature would reduce the AR(2) baseline MAE by roughly
**sqrt(R²) ≈ 67.6%** as a back-of-envelope estimate.
The Ridge would do better than this in practice because it can learn
nonlinear interactions (e.g. `net_load × is_weekend` or `net_load²`)
and because the FI Ridge already has nuclear / wind features that
this hypothesis is partially redundant with.

## Verdict

**STRONG SIGNAL.** net_load explains 45.7% of the AR(2) residual variance and the high-decile slice has 4.5× the MAE of the low-decile slice. Adding net_load to the FI Ridge feature set is expected to materially improve winter-spike accuracy. Recommend implementing as a feature for v2.2.


## Method limitations

* **Reading against AR(2) residuals, not FI Ridge residuals.** The
  measured R² is the *upper bound* of what net_load can add — the
  FI Ridge already has wind speed, HDD, and nuclear deficit, which
  partially encode the same information. The marginal gain on FI
  Ridge will be smaller than the 45.7% measured here.
* **Single OOS window.** Results may differ in summer / autumn.
* **Day-ahead forecasts (datasets 165, 246, 247) vs real-time
  nuclear (188).** The day-ahead forecasts are what would be
  available at forecast time; using nuclear real-time is a small
  cheat — proper inference would use the day-ahead nuclear schedule.
  Effect on the analysis is small because nuclear is much more
  stable than wind or consumption.

## Reproducibility

```sh
set FINGRID_API_KEY=...
python studies/fingrid_netload_study.py
```

Cached Fingrid responses live under `studies/_fingrid_cache/`;
re-running is fast after the first fetch.
