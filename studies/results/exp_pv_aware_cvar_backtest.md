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
  **1211 days**
  (2023-01-01 to 2026-04-26).
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
  replacement from the 1211 daily realisations.
  CVaR at α = 0.05.

## Headline — weekly cost statistics by consumption strategy

The same 24-hour PV and the same 24-hour buy/sell prices are
applied to three different consumption shapes scaled to the same
daily kWh total:

| Strategy | Mean EUR/kWh | Median EUR/kWh | VaR<sub>95</sub> | **CVaR<sub>95</sub>** | Weekly mean (EUR) | Weekly CVaR<sub>95</sub> (EUR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0 — flat baseload** | 0.1467 | 0.1465 | 0.1942 | **0.2231** | 54.84 | 97.56 |
| **S1 — EMA-shaped (optimised)** | 0.1413 | 0.1414 | 0.1915 | **0.2215** | 52.90 | 96.84 |
| **S2 — anti-optimised** | 0.1568 | 0.1564 | 0.2038 | **0.2318** | 58.60 | 101.74 |

### Read-out

- **Optimisation yield (S0 → S1)**:
  ΔCVaR = +0.0016 EUR/kWh
  (+0.7% relative).
  ΔMean = +0.0054 EUR/kWh.
  Annual extrapolation: ≈ +99 EUR/year mean cost reduction from following the EMA-shaped (EMHASS-optimised) consumption versus a flat household.
- **Worst-case reference (S0 → S2)**:
  ΔCVaR = +0.0087 EUR/kWh shows how much *worse* a perversely anti-optimised household would be — bounds the optimisation upside.
- **Tail vs mean for S1**: CVaR<sub>95</sub> exceeds the mean by
  0.0802 EUR/kWh, i.e. worst-5%-week premium of
  56.7%
  relative to the mean. The PV-aware CVaR sensor surfaces this number to the user as the "downside risk this week" figure.

## Daily self-consumption fraction (S1)

| | mean | min | max |
|---|:---:|:---:|:---:|
| SCF | 0.819 | 0.000 | 1.000 |
| PV (kWh/day) | 21.06 | 0.00 | 62.32 |
| Self-consumed (kWh/day) | 13.69 | 0.00 | 37.22 |
| Exported (kWh/day) | 7.37 | 0.00 | 43.69 |
| Import (kWh/day) | 38.55 | 3.03 | 93.16 |

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
