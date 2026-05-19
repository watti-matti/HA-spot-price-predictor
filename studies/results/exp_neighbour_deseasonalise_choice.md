# Should Y_se1 / Y_se3 / Y_ee be deseasonalised?

Branch: `experiment/extra-l2-features`. Off-tree report. Script:
[`studies/exp_neighbour_deseasonalise_choice.py`](../exp_neighbour_deseasonalise_choice.py).

Tests whether the v2.9.0 production choice to deseasonalise the
cross-border neighbour prices (Y_se* = se* − seasonal_se*) is the
right one, or whether raw / hybrid / spread forms encode the
SE-FI coupling better.

## Conceptual framing

- SE1 (Luleå area, hydro-dominated) and SE3 (central, includes
  Stockholm) couple to FI through FennoSkan-1/2 cables (~1100 MW) at
  the **SE3** node. There is no direct FI↔SE1 cable.
- SE3 price has hydro-driven seasonality: winter peak (heating + low
  inflow), spring crash (snowmelt fills reservoirs), summer trough,
  autumn climb.
- Transit capacity is constant, but whether the cable BINDS is
  event-driven: it binds when the SE↔FI spread is large enough
  relative to capacity, which happens disproportionately during
  spike hours (low SE3 hydro + high FI demand or low FI nuclear).
- The deseasonalisation question: does FI's own `seasonal_fi`
  artefact already absorb the SE-correlated seasonal signal (so
  feeding the FI Ridge the *residual* Y_se* is sufficient), or does
  the seasonal SE3 level carry independent information about FI
  cable flow that gets discarded by L1 subtraction?

## Setup

- Data window: same as `exp_extended_retrain.md` (cached parquets,
  2023-01-08 → 2026-04-26).
- 55 / 45 chronological train / test.
- L1 (seasonal_fi) unchanged; L2 Ridge alpha=1.0; L3 AR(1) refit
  per variant. L4 not refit — the comparison is on point error and
  hedge CVaR.

## Headline metrics

| Variant | k (incl. intercept) | MAE | R² | MAE \|spot\|>100 | Hedge ΔCVaR % | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| V_xb | 9 | 11.41 | +0.898 | 15.48 | 11.01 | +0.857 |
| V_xb_raw | 9 | 12.18 | +0.905 | 14.85 | 11.45 | +0.879 |
| V_xb_hybrid | 12 | 11.68 | +0.893 | 15.52 | 11.31 | +0.851 |
| V_xb_spread | 9 | 12.18 | +0.905 | 14.85 | 11.45 | +0.879 |

## SE/EE block coefficients

- **V_xb**: Y_se1=+0.170, Y_se3=+0.403, Y_ee=+0.532
- **V_xb_raw**: raw_se1_mc=+0.234, raw_se3_mc=+0.133, raw_ee_mc=+0.440
- **V_xb_hybrid**: Y_se1=+0.214, Y_se3=+0.419, Y_ee=+0.542, seas_se1_mc=+0.117, seas_se3_mc=+0.165, seas_ee_mc=-0.013
- **V_xb_spread**: raw_se3_mc=+0.808, spread_se1_se3_v2=+0.234, spread_se3_ee_v2=-0.440

## Δ vs V_xb (the production design) — v2.5.6 gate

Threshold: +0.3 pp hedge CVaR reduction per ADDED feature
(or just +0.3 pp if feature count is unchanged but the form
differs).

| Variant | Δ features | Δhedge pp | Threshold pp | ΔMAE | ΔMAE>100 | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|---|
| V_xb_raw | +0 | +0.45 | 0.30 | +0.77 | -0.63 | ✅ beats V_xb |
| V_xb_hybrid | +3 | +0.30 | 0.90 | +0.27 | +0.05 | ❌ does not justify replacing V_xb |
| V_xb_spread | +0 | +0.45 | 0.30 | +0.77 | -0.63 | ✅ beats V_xb |

## Interpretation

The verdict above answers the empirical question directly. The
*conceptual* read-out independent of the numbers:

- **If V_xb_raw wins**: FI's seasonal_fi does NOT fully absorb
  SE-driven seasonality. The seasonal SE3 LEVEL carries information
  the deseasonalised residual loses. Production should switch to
  raw mean-centred neighbour prices.
- **If V_xb_hybrid wins by ≥0.9 pp** (3 extra features × 0.3 pp):
  the seasonal and residual SE components carry independent
  signal. Production should expand to both, or at minimum revisit
  the L1 components for SE3 (the current SE3 climatology may be
  miscalibrated).
- **If V_xb_spread wins**: the operative cross-border signal is
  raw SE3 level + transit-saturation spreads, not three
  deseasonalised neighbour series. This would be the cleanest
  win — fewer features, more physical, no leakage on FI itself.
- **If V_xb is best**: the deseasonalisation choice is justified.
  Seasonal_fi + Y_se* is sufficient because FI seasonality already
  encodes the heating-demand component that SE3 hydro
  seasonality correlates with.

Whichever variant wins, the underlying mechanism documented in
`studies/results/exp_extended_retrain.md` (SE3 carries FI cable
state, SE1 carries upstream hydro inflow, EE carries the
Baltic-side coupling) is unchanged — only the encoding moves.
