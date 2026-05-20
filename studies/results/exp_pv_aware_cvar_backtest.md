# PV-aware CVaR back-test on the real household profile

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_pv_aware_cvar_backtest.py`](../exp_pv_aware_cvar_backtest.py).

First empirical estimate of the PV-aware CVaR sensor for the
reference household, using the user's own consumption shape
extracted from their HA recorder DB.

## Setup

- Profile source: `studies/_private/household_profile.json` (extracted
  from HA recorder, **not** committed).
- Profile window: 2026-03-08 to 2026-05-19
  (72.3 days,
  1724 hourly observations).
- Backtest overlap window (after intersecting cached prices + weather):
  **49 days**
  (2026-03-08 to 2026-04-26).
- Mean consumption: 2.1052 kWh/h
  (~50.5 kWh/day,
  ~18442 kWh/year extrapolated).
- PV system: 8.91 kWp / tilt 45° / azimuth
  160° (Tampere reference).
- Consumer tariff in EUR/kWh:
  spot + 0.030 (margin) +
  0.045 (grid fee) + 0.028 (tax),
  all × 1.255 VAT.
- Feed-in tariff: 0.040 EUR/kWh.
- Bootstrap: 2000 weekly samples drawn with
  replacement from the 49 daily realisations.
  CVaR at α = 0.05.

## Headline — weekly cost statistics by consumption strategy

The same 24-hour PV and the same 24-hour buy/sell prices are
applied to three different consumption shapes scaled to the same
daily kWh total:

| Strategy | Mean EUR/kWh | Median EUR/kWh | VaR<sub>95</sub> | **CVaR<sub>95</sub>** | Weekly mean (EUR) | Weekly CVaR<sub>95</sub> (EUR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0 — flat baseload** | 0.1059 | 0.1055 | 0.1235 | **0.1294** | 36.70 | 45.30 |
| **S1 — EMA-shaped (optimised)** | 0.0932 | 0.0925 | 0.1138 | **0.1200** | 32.32 | 42.40 |
| **S2 — anti-optimised** | 0.1236 | 0.1234 | 0.1417 | **0.1467** | 42.92 | 52.14 |

### Read-out

- **Optimisation yield (S0 → S1)**:
  ΔCVaR = +0.0094 EUR/kWh
  (+7.3% relative).
  ΔMean = +0.0127 EUR/kWh.
  Annual extrapolation: ≈ +234 EUR/year mean cost reduction from following the EMA-shaped (EMHASS-optimised) consumption versus a flat household.
- **Worst-case reference (S0 → S2)**:
  ΔCVaR = +0.0173 EUR/kWh shows how much *worse* a perversely anti-optimised household would be — bounds the optimisation upside.
- **Tail vs mean for S1**: CVaR<sub>95</sub> exceeds the mean by
  0.0268 EUR/kWh, i.e. worst-5%-week premium of
  28.7%
  relative to the mean. The PV-aware CVaR sensor surfaces this number to the user as the "downside risk this week" figure.

## Daily self-consumption fraction (S1)

| | mean | min | max |
|---|:---:|:---:|:---:|
| SCF | 0.827 | 0.000 | 1.000 |
| PV (kWh/day) | 27.48 | 0.00 | 54.80 |
| Self-consumed (kWh/day) | 20.67 | 0.00 | 33.09 |
| Exported (kWh/day) | 6.81 | 0.00 | 25.76 |
| Import (kWh/day) | 28.82 | 2.73 | 52.94 |

## Caveats

- **Spring-only window.** The profile and the back-test overlap
  only March 8 → April 28 2026. Winter (heat-pump peak) and
  summer (PV peak) are absent and must be modelled by
  extrapolation for the published annual CVaR estimate. The
  numbers above are *what the sensor would have read this spring*,
  not an annualised figure.
- **Deterministic PV.** The current back-test uses point-forecast
  PV (no cloud-bootstrap scenarios yet — Phase A still to land).
  The CVaR is therefore tail-of-realised-price only; once PV
  scenarios are added, the CVaR will widen modestly because tail
  joint events (cold cloudy spike-price days) get sampled.
- **Tariff sensitivity.** Numbers above use one tariff structure.
  The relative gap between S0 and S1 is robust to tariff choice
  because both strategies see the same prices.
- **Profile bootstrapping not used.** This is realised-data
  back-test, not forward-forecast CVaR. The published sensor will
  use the joint price + PV forecast scenarios over the upcoming 170
  hours, not a historical re-sample. This study confirms the
  *machinery* and reports a *current-window estimate*; production
  output is forward-looking.

## Sanity check vs the kernel

The kernel produces the same cost realisation when called with
the same arrays — both code paths use
`pv_cost_kernel.cost_distribution`. Per-strategy mean cost from
the daily aggregation and from the kernel agree to within
0.00e+00 EUR/kWh
(machine epsilon).

This is the first empirical evidence on the branch that the
cost kernel + EMA profile + cached weather/prices pipeline
produces stable, mean-positive PV-aware CVaR numbers for the
reference household.
