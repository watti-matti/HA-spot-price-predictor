# Full retrain with extended L2 features — hedge-gate decision

Branch: `experiment/extra-l2-features`. Off-tree research; **no
production artefact is overwritten**. Candidate JSONs land in
`output/exp_spike_model_<variant>.json` so the production Pipeline
keeps loading the current v2.8.1 artefact.

Script: [`studies/exp_extended_retrain.py`](../exp_extended_retrain.py).
Data: 2023-01-08 → 2026-04-26, 28,824 hourly rows, train = first 15,853 hours, test = last 12,971 hours.

## Multi-year nuclear outage pattern (Fingrid #188, 2022-05 → 2026-05)

Annual deficit profile, deficit MW = `(rolling_60d_max − nuclear_mw)
× 4 372`:

| Year | Mean MW deficit | p95 MW deficit | Max MW deficit | Hours w/ deficit > 100 MW |
|---|:---:|:---:|:---:|:---:|
| 2022 |   566 | 1 179 | 1 736 |  3 773  (65 % of year) |
| 2023 |   611 | 1 642 | 2 511 |  5 485  (63 % of year) |
| 2024 |   584 | 1 596 | 2 496 |  5 532  (63 % of year) |
| 2025 |   606 | 1 667 | 2 689 |  5 656  (65 % of year) |
| 2026 |   316 |   979 | 1 503 |  1 351  (41 % of YTD) |

Every year has 22-27 distinct outage episodes (deficit > 200 MW lasting
≥ 24 h) totalling **~4 700 hours per year** — well over half the year
is spent at less-than-fleet output. The longest single episode each
year tracks the refueling cycle (1 300-1 500 h ≈ 55-60 days):

- **2023**: 2023-01-10 → 03-08 (1 377 h, 1 658 MW peak)
- **2024**: 2024-03-09 → 05-10 (1 508 h, 1 510 MW peak)
- **2025**: 2025-02-28 → 04-30 (1 465 h, 1 760 MW peak)
- **2026 (in progress)**: 2026-04-19 → 05-18 (708 h, 972 MW peak)

Confirms the user's note 2026-05-19: every year sees significant
service breaks across the fleet, and the spring 2026 episode (OL1 /
OL2 maintenance) fits the historical pattern.

The model retrain below tests whether including this signal as an
explicit Ridge feature is worth it.

Three variants refit end-to-end (L2 Ridge + L3 AR(1) + L4 GPD POT;
L1 seasonal components are loaded unchanged from the shipped
artefact):

| Variant | What it adds vs the production v2.8.1 design |
|---|---|
| `V_prod`   | Sanity baseline. Same five Ridge features as the current production `spike_model_default.json`. |
| `V_xb`     | + `Y_se1`, `Y_se3`, `Y_ee` (cross-border, per the experiment_extra_l2_features.md finding). |
| `V_xb_nuc` | `V_xb` + `nuclear_deficit_v2` (capacity-aware: rolling-60-day max minus current `nuclear_mw`). |

## Variant metrics (test split)

| Variant | n_feat | MAE | R² | MAE (|spot|>100) | Hedge CVaR red. (pp) | φ | σ(η) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V_prod | 6 | 10.30 | +0.925 | 20.14 | 1.99 | +0.904 | 22.81 |
| V_xb | 9 | 11.41 | +0.898 | 15.48 | 11.01 | +0.857 | 18.63 |
| V_xb_nuc | 10 | 11.46 | +0.898 | 15.49 | 10.96 | +0.855 | 18.63 |

## Hedge-gate decision vs V_prod

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
| V_xb | +1.11 | -4.66 | +9.02 | 0.90 | **accept (hedge gate + extreme tail)** |
| V_xb_nuc | +1.16 | -4.65 | +8.97 | 1.20 | **accept (hedge gate + extreme tail)** |

The hedge gate is the v2.5.6 acceptance test: **+0.3 pp CVaR-reduction
per added feature**, no severe regression on MAE.

## Heavy-tail (L4) parameter comparison

Reading the η = post-AR residual statistics:

| Variant | σ(η) | skew | excess kurt | GPD u | ξ (shape) | σ (scale) | Hill α̂ right |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V_prod | 22.81 | +4.67 | +251.67 | nan | +0.481 | 20.49 | 2.14 |
| V_xb | 18.63 | +0.37 | +85.69 | nan | +0.357 | 15.75 | 2.92 |
| V_xb_nuc | 18.63 | +0.34 | +85.00 | nan | +0.346 | 15.98 | 2.90 |

A lower σ(η) (variance of the post-AR residual) means the L2+L3 stack
already explains more of the per-hour variation; the L4 spike layer
then has less work to do. ξ > 0 indicates a heavy right tail — the
fan-chart's P95 band depends on this parameter and the GPD scale.

## Method note

- Train/test split chronological (`TRAIN_FRAC = 0.55`).
- Ridge α = 1.0; intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then
  applied one-step-ahead in the test forecast.
- GPD POT fit on the post-AR residual of the train split (right tail
  only; the production v2.8.1 left-tail params are not refit here as
  the experiment focuses on price-spike accuracy).
- Hedge gate: `npk_cvar_hedge.optimize_hedge` at α = 0.05, the model
  prediction as the futures instrument vs realised spot.
- L4 fan-chart sampling and the production calibrators
  (HourlyBiasCorrector, HourlyFanChartCalibrator) are **not** exercised
  here. The hedge metric isolates the L2 contribution.

## Operational follow-up if a variant is accepted

To promote `V_xb` (or `V_xb_nuc`) to production:

1. Copy `output/exp_spike_model_<variant>.json` →
   `custom_components/spot_price_predictor/data/spike_model_default.json`
   in a follow-up commit.
2. Extend `RIDGE_FEATURES` in `pipeline.py:62-69` to match the variant's
   feature order.
3. Wire `fetch_neighbor_prices()` results into
   `coordinator.py:_apply_pipeline_pre_dk` so the pipeline receives a
   `recent_neighbour_prices` dict (deseasonalised SE1/SE3/EE).
4. For `V_xb_nuc`: also pipe the Fingrid `nuclear_mw` history (the
   pipeline needs a rolling-60-day buffer to compute `nuclear_deficit_v2`
   at runtime). Could be a separate calibrator state file.
5. Refit L1 seasonal at the same time (the shipped components are from
   an earlier window).
6. Update the test suite for the new feature count.

None of the above is done by this commit — the experiment is a
side-by-side evaluation only.
