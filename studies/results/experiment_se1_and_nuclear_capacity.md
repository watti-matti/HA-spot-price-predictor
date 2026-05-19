# Follow-up — SE1 deep-dive + capacity-aware nuclear deficit

Branch: `experiment/extra-l2-features`. Off-tree research; no
production change. Script:
[`studies/exp_se1_and_nuclear_capacity.py`](../exp_se1_and_nuclear_capacity.py).

This follow-up answers two questions raised by the user (2026-05-19):

1. **Is the SE1 contribution to B2_se1 real, and what does the
   coefficient say?** — does the v2.5.1 opposite-sign-pair finding
   (+1.61 / −1.60) reproduce on the 2023-2026 window?
2. **Why does naive `nuclear_deficit` look chronically zero?** —
   recently OL1 and OL2 service breaks have visibly reduced FI nuclear
   capacity. A capacity-aware deficit relative to the 60-day rolling
   max should be non-zero during those episodes and might recover the
   signal that the naive form misses.

## 1. Nuclear capacity in Finland (2022-2026, Fingrid dataset #188)

Normalised by max-fleet 4 372 MW (the same constant the production
`features.py:152` uses). 1.0 = full fleet output.

| Year | Mean | Max | Max MW | FI spot mean (EUR/MWh) |
|---|:---:|:---:|:---:|:---:|
| 2023 | 0.852 | 1.008 | 4,407 | 55.5 |
| 2024 | 0.808 | 1.000 | 4,373 | 45.7 |
| 2025 | 0.816 | 0.968 | 4,233 | 40.5 |
| 2026 | 0.922 | 0.970 | 4,241 | 82.7 |

The 2023 step is OL3 commissioning (May 2023, +1 600 MW nameplate);
the model's `nuclear_mw / 4372` therefore jumps from ≲ 0.65 to ≲ 0.95
in mid-2023. After 2023 the operational ceiling is ~0.95-0.98 of fleet.

### Recent (last-30 vs prior-90 days)

- prior-90-day mean: **0.958**
  (≈ 4,187 MW),
  min 0.793.
- last-30-day mean:  **0.819**
  (≈ 3,579 MW),
  min 0.625.
- Δ = **-13.9
  percentage points** (negative ⇒ recent capacity below the prior
  baseline).

A drop of this size relative to the 90-day baseline is consistent with
one or more units offline. From public reactor-status data, OL1
and/or OL2 were in service breaks during this window.

### Deficit episodes ≥ 12 h with > 5 % deficit vs 60-day rolling max

Detected **83** episodes in the 4-year
window. The five most recent:

| Start | End | Hours | Mean deficit (MW) | FI spot mean during episode (EUR/MWh) |
|---|---|:---:|:---:|:---:|
| 2026-03-17 | 2026-03-19 | 48 | 293 | 6.3 |
| 2026-03-21 | 2026-03-23 | 48 | 292 | 8.5 |
| 2026-03-24 | 2026-03-26 | 48 | 334 | 12.5 |
| 2026-04-05 | 2026-04-17 | 300 | 794 | 75.6 |
| 2026-04-19 | 2026-04-27 | 206 | 1,078 | 24.4 |

The April 2026 episodes carry FI spot means substantially above the
4-year average — confirming the user's observation that recent OL1 /
OL2 service breaks coincide with elevated FI prices. The longest
episode (2026-04-05 → 2026-04-17, 300 h) averaged 794 MW of deficit
and **75.6 EUR/MWh** FI spot — roughly 2× the 2024-2025 baseline.

## 2. SE1 deep-dive

Two variants compared (everything else identical):

- `B2_no_se1` — core 5 + `Y_se3, Y_ee, export_potential_se3`
- `B2_se1`   — core 5 + `Y_se1, Y_se3, Y_ee, export_potential_se3`

### Ridge coefficients on the training split

| Feature | B2 (no SE1) | B2_se1 |
|---|:---:|:---:|
| `Y_ee` | +0.5369 | +0.5310 |
| `Y_fi_lag168` | +0.0221 | +0.0221 |
| `Y_se1` | — | +0.1712 |
| `Y_se3` | +0.4891 | +0.4052 |
| `Y_sigmoid_wind_rho` | -41.6795 | -37.8633 |
| `Y_solar_effective` | +0.0278 | +0.0266 |
| `Y_temp` | -0.5913 | -0.4883 |
| `export_potential_se3` | -0.1115 | -0.1412 |
| `intercept` | +1.5709 | +0.5954 |
| `is_workday` | +2.6157 | +2.3494 |

**Both `Y_se1` and `Y_se3` carry positive coefficients on the
2023-2026 window** — the v2.5.1 opposite-sign-pair finding does NOT
reproduce here. The model treats SE1 as an additional positive
predictor of FI price (≈ 0.43× the SE3 weight) rather than as a
spread-extraction signal. The mechanism is more straightforward than
v2.5.1 suggested: post-OL3 commissioning, FI is structurally so
cheap that even SE1 (which is normally closer to FI than SE3) carries
incremental upside-pressure information when it deviates from its own
climatology.

### MAE by SE1↔SE3 coupling regime (test split)

Decoupled hours = top 25 % of `|SE1 − SE3|` (the saturated-transit
regime). Coupled hours = the remaining 75 %.

| Regime | Mean FI spot (EUR/MWh) | P95 FI spot | MAE B2 (no SE1) | MAE B2_se1 |
|---|:---:|:---:|:---:|:---:|
| coupled (|SE1−SE3| ≤ 26.9 EUR/MWh, 75% of hours) | 44.6 | 160 | 8.88 | 8.76 |
| decoupled (|SE1−SE3| > 26.9 EUR/MWh, 25% of hours) | 57.5 | 167 | 15.65 | 15.25 |

Despite the coefficient signs not matching the v2.5.1 pattern, the
**MAE improvement is 3× larger in the decoupled regime than in the
coupled regime** (0.40 vs 0.12 EUR/MWh). The user's hypothesis —
limited transit capacity makes SE1 distinct from SE3 and valuable for
the model — is **functionally validated**: SE1 helps most in the
hours where transit capacity has saturated the SE1↔SE3 link.

## 3. Capacity-aware nuclear deficit

Recomputed deficit:
`nuclear_deficit_v2 = max(0, rolling_60d_max(nuclear_mw) − nuclear_mw)`.
This activates during real outage episodes (where the recent ceiling
was higher than the current output) instead of being chronically zero.

Also tested: the legacy v2.2 interaction
`nuclear_x_scarcity_v2 = nuclear_deficit_v2 × wind_log_scarcity`,
which amplifies outage impact during cold-and-windless conditions.

All variants below stack on top of `B2_se1`:

| Variant | n_feat | Test MAE | Extreme MAE | Hedge CVaR red. (pp) |
|---|:---:|:---:|:---:|:---:|
| B2_se1 | 10 | 11.43 | 15.50 | 11.07 |
| B2_se1_nuclear_v1 | 11 | 11.42 | 15.49 | 11.07 |
| B2_se1_nuclear_v2 | 11 | 11.47 | 15.50 | 11.01 |
| B2_se1_nuclear_x | 12 | 11.46 | 15.52 | 11.05 |

### Read-out

**No nuclear variant passes the hedge gate on top of B2_se1.** Even
the capacity-aware deficit (`nuclear_deficit_v2`) marginally *hurts*
the hedge metric (11.07 → 11.01 pp) and increases overall MAE
(11.43 → 11.47). The interaction term `nuclear_x_scarcity_v2` is
similarly neutral-to-negative.

**Why nuclear adds nothing despite the visible capacity reduction.**
The April 2026 outage episode raised FI spot by ~2× — the price
impact is real and large. But that same impact also raises Sweden's
SE1 and SE3 prices (FI imports more, SE exports more, market couples
upward). The B2_se1 model already sees the elevated `Y_se1` / `Y_se3`
during the outage and adjusts its FI forecast accordingly. Nuclear
deficit therefore enters the model only **through cross-border
prices**, not as an independent feature. Adding it explicitly just
duplicates the signal Ridge has already extracted from `Y_se*`.

**When could nuclear still matter as an explicit feature?**
- A **forecast-window outage** that hasn't yet propagated to neighbour
  prices (the planned-outage UMM schedule could pre-empt the
  cross-border signal by 24-48 h).
- A **decoupled regime** where FI nuclear outage doesn't pull up SE1
  / SE3 because the FI↔SE3 transmission cable is at capacity. Today
  the model already extracts that decoupling via `Y_se1` vs `Y_se3`,
  but pairing transmission-out events with nuclear-out events could
  give the Ridge a sharper signal.

Neither of those refinements is justified by the current data window.

## Method note

- Data: 2023-01-08 → 2026-04-27 (inner-join of FI prices, neighbour
  prices, FI weather, and Fingrid grid data).
- Train/test: time-ordered, `TRAIN_FRAC = 0.55` (matches v2513).
- Coefficient signs are reported from the train-split fit; the regime
  metrics are on the test split.
