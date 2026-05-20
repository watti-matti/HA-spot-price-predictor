# Bootstrap learning curve — when does the EMA become useful?

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_bootstrap_learning_curve.py`](../exp_bootstrap_learning_curve.py).

Tests how quickly the rank-shift profile and baseline envelope
converge to their long-run values as observation history accrues.
The post-PV window from the reference household is used as the
truth-set (958 valid days); each row truncates
to the first N days and reports deviation from the full-window estimate.

This answers the **fresh-install bootstrap question**: how many days
of HA recorder observations does the EMA module need before the
learned profile is usefully better than the synthetic fallback?

## Convergence table

| N days | days used | top4 % | ratio | share-L1 vs truth | top4 Δ % | baseline rel-MAE % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|    7 |    7 |   0.0 |  0.00 | 1.000 | 17.67 | 743.1 |
|   14 |   14 |  19.2 |  1.69 | 0.256 |  1.56 | 376.3 |
|   30 |   30 |  17.0 |  1.29 | 0.171 |  0.71 | 233.2 |
|   60 |   58 |  16.6 |  1.28 | 0.131 |  1.10 | 266.0 |
|   90 |   88 |  18.2 |  1.54 | 0.073 |  0.52 | 333.8 |
|  180 |  178 |  19.1 |  2.06 | 0.126 |  1.42 | 464.8 |
|  365 |  363 |  17.6 |  1.72 | 0.038 |  0.07 |  17.0 |
|  720 |  716 |  17.6 |  1.72 | 0.020 |  0.05 |  11.6 |
|  964 |  958 |  17.7 |  1.77 | 0.000 |  0.00 |   0.0 |

Columns:

- **share-L1**: total absolute deviation across the 24 ranks,
  normalised. 0 = identical, ~0.5 = uniform-vs-skewed.
- **top4 Δ**: absolute difference in the top-4-rank concentration
  (the headline number for "is the household rank-shifting").
- **baseline rel-MAE**: relative mean absolute error of the
  baseline envelope vs the full-window envelope, weighted by cell.

## Read-out

The first row to inspect is **N=30** — one month of observations.

If 30 days achieves ≤ 5 percentage-points absolute error on the
top4 share and ≤ 20 % relative baseline MAE, the EMA module
becomes a useful upgrade over the synthetic fallback within the
first month of operation on a fresh install. No bulk-import seed
required.

The 90-day row tells us when the profile is "production-grade" —
when the predictor's PV-aware CVaR can be relied on as the
canonical reference signal.

The 365-day row tells us when monthly_factor has been observed
across all 12 months and seasonal extrapolation can stop.

## Practical generalisation for fresh installs

Three bootstrap mechanisms, in order of increasing user friction:

1. **Cold start (default for any HA install)**: synthetic Finnish
   profile scaled to user's `annual_kwh`. data_provenance =
   "synthetic_cold_start". Predictor publishes CVaR with low-
   confidence flag.

2. **Online learning (default everywhere)**: every HA install
   accumulates observations and the EMA module incrementally
   updates baseline + share_by_rank + monthly_factor cells. After
   N days the profile transitions through data_provenance =
   "ema_blended" → "ema_warm" based on the convergence numbers
   above.

3. **Bulk-import accelerator (Finland-only)**: users with Fingrid
   Datahub access can run a one-time
   `extract_household_profile_from_fingrid.py` import to seed the
   monthly_factor and shape from years of metering history. This
   skips the 90-day warm-up directly to "ema_warm". The
   accelerator is **optional convenience**, not a requirement.

The third mechanism is Finland-specific. The first two work for
any HA installation worldwide.

## Open architectural question — monthly_factor bootstrap

The slowest cell to populate is monthly_factor: 12 months of
single-cell observations are needed. The empirical data here
quantifies whether that 12-month wait is actually a problem in
practice (might it be that the shape converges in 90 days, leaving
only monthly_factor as the bottleneck?).

If so, a future addition: regress daily_kwh against
heating-degree-hours per day (HDH = `Σ_h max(0, T_setpoint −
T_outdoor)`). With a few weeks of observations + Open-Meteo
climatology for the user's location, monthly_factor can be
projected onto each month's typical HDH instead of waiting for
each month to be observed individually.

The HDH regression is climate-zone-aware (Open-Meteo provides
historical climatology for any latitude/longitude) and household-
specific (the slope `beta` is learned from local observations).
This converts the 12-month bottleneck into a few-weeks problem
that works anywhere, not just Finland.
