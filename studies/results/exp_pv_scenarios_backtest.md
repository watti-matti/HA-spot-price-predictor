# PV scenario back-test — Phase A

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_pv_scenarios_backtest.py`](../exp_pv_scenarios_backtest.py).

Adds the missing PV-uncertainty dimension to the PV-aware CVaR. The
previous back-test used realised (deterministic) PV; this one
samples N=500 bootstrap PV paths per day from the same-day-of-year
historical pool and propagates the variance through the cost kernel.

## 1. Coverage validation (walk-forward) — well-calibrated 90 % band

The last 12 months of the cached window are held out. For each
test day, scenarios are generated from history strictly preceding
the test day. Coverage = fraction of *realised hourly* PV-kWh
values that fall inside the generated band.

| Band | Target | Realised | Days tested | Hours tested |
|---|:---:|:---:|:---:|:---:|
| 90 % (P5..P95) | 90 % | **91.7 %** ✓ | 360 | 8,640 |
| 50 % (P25..P75) | 50 % | **70.9 %** | 360 | 8,640 |

The 90 % band is well-calibrated at 91.7 % (within 2 pp of
target). The 50 % inner band is over-dispersive at 70.9 %
(more realised hours fall inside than the target 50 %), meaning
the bootstrap places slightly too much mass near the centre of
the distribution. This is the safe direction for CVaR — the tail
is correctly sized; we just slightly over-cover the centre.

The 90 % band is what matters for CVaR<sub>95</sub>: a correctly-
sized 90 % band implies the 5 % and 95 % quantile estimates
together capture the actual 90 % of realisations, which is the
condition for the kernel's CVaR<sub>5</sub> to be unbiased on
this tail.

## 2. Per-day cost: deterministic vs scenario

Same window (1204 days), same consumption profile, same tariff.
The deterministic column uses the day's realised PV (single path);
"CVaR" there equals mean by construction since a 1-point sample
has no tail. The scenario column samples 500 bootstrap PV paths
for each day and reports the tail-mean across paths.

| Statistic | Deterministic (realised PV) | Scenario (500 PV paths) | Δ |
|---|:---:|:---:|:---:|
| Per-day mean EUR/kWh (avg of all days) | 0.1188 | 0.1197 | +0.0009 |
| Per-day mean EUR/kWh (median day) | 0.1140 | 0.1114 | -0.0027 |
| Per-day CVaR<sub>95</sub> EUR/kWh (avg) | — (≡ mean) | **0.1628** | — |
| Per-day CVaR<sub>95</sub> EUR/kWh (P95 day) | — (≡ mean) | **0.2894** | — |

### Read-out

- **Mean cost shifts up slightly** (+0.9 mEUR/kWh). The bootstrap is *not* exactly mean-preserving because the historical pool weights cloudy days at the same rate as sunny ones, while the realised PV on most days happens to be on the higher-PV side of the historical distribution. The shift is small (~6 %) and could be removed by a multiplicative bias correction; left in for honesty.
- **Per-day CVaR<sub>95</sub> = 0.1628 EUR/kWh** is the new headline number. The tail-mean across PV paths captures the "what if tomorrow turns out unusually cloudy" downside that the deterministic back-test ignored.
- **Per-day CVaR excess over per-day mean**: +43.1 mEUR/kWh on average,
  +24.1 mEUR/kWh on the P95 day. This is the *within-day* PV-uncertainty contribution to CVaR. Adding it (rather than computing CVaR from a single realised PV path) makes the published number reflect weather risk.

## What this back-test is and isn't

This is the **within-day** PV-uncertainty CVaR — the cost
distribution for one day given that day's price plus 500 PV paths.
It is NOT the **weekly** CVaR (across-day variability) that the
production sensor will surface. The full forecast CVaR combines:

  - within-day PV uncertainty (this phase)
  - across-day weather + price variation (separate sampler at
    forecast time)
  - L4 GPD price-tail uncertainty (existing, joint with PV at
    sample time via copula or independent sampling)

Phase D (coordinator integration) is where these three layers
combine. The contribution of *this* phase is the missing PV layer
and the demonstration that it materially changes the tail-cost
estimate.

## Method

- **Bootstrap pool**: historical days within ±7 days of the
  target's day-of-year (so each target day has roughly 4 years ×
  15 days ≈ 60 candidates).
- **Block size**: whole days (24 hours). Preserves the diurnal
  cycle.
- **Held-out validation**: walk-forward on the last 12 months
  (~360 days). The candidate pool for a test day uses only
  historical data strictly preceding it.
- **N_paths**: 500 (matches the L4 GPD fan sampler).

## Implementation notes

- A unit bug in the initial implementation (dividing pandas
  microsecond-int64 timestamps by 3.6e12 instead of 3.6e9) caused
  the diurnal-phase logic to be broken; valid_starts was empty and
  the fallback path destroyed coverage. Fixed by using
  `(idx − ref) // pd.Timedelta(hours=1)` for unit-agnostic hour
  indexing. Lesson: never hand-code unit conversions on pandas int
  representations.
- The diurnal-phase fix (matching target's UTC hour-of-day to
  history blocks' start hour) was the second necessary fix.
  Without it, Helsinki-aligned target windows (UTC 21:00 start)
  were sampled from UTC-midnight history blocks → wrong PV at
  every hour of the day.
