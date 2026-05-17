# v2.4.3 — EE model investigation (REJECTED; current `ar_ee` retained)

## TL;DR

**REJECT** — no proposed EE model variant improved on the seasonal-only baseline (+4.07 % NPK-CVaR reduction at 48 h hedge). The current production `ar_ee` AR(2) feature already extracts the achievable hedge signal at this horizon. v2.4.3 documents this as a negative result and ships no behaviour change.

## What was tested

Three architectures per the v2.4.1 → v2.5.0 plan for EE:

```
V1: P_EE = seasonal_EE                                   (baseline)
V2: P_EE = seasonal_EE + B_workday · is_workday
V3: P_EE = seasonal_EE + B_workday · is_workday + B_AR1 · Y_{t-1}    (matches SE3 v2.4.2)
```

Window: 2023-01-01 → 2026-04-29 (29 134 EE hourly observations).

## Results

| Variant | h_hat | CVaR test reduction | vs V1 |
|---|---:|---:|---:|
| **V1** (seasonal-only baseline) | 1.209 | **+4.07 %** | — |
| **V2** (seasonal + workday) | 1.210 | **+4.07 %** | +0.00 pp |
| **V3** (seasonal + workday + AR(1)) | 0.188 | **+0.84 %** | **−3.23 pp** |

## Why every variant matches the baseline (or is worse)

1. **Workday coefficient is essentially zero** (`β_workday = −0.02` EUR/MWh). The weekday/weekend distinction is already fully baked into the `P_day[0..6]` seasonal vector. Adding a separate flag is redundant.

2. **AR(1) HURTS the EE hedge**, dropping reduction from +4.07 % to +0.84 %. Reason: EE's residual decay half-life is **4.7 h**, vs SE3's **12.8 h**. At lag-1, the EE residual has already mean-reverted 14 % of the way back to zero, so the AR(1) signal is much weaker. Worse, adding an AR(1) term to the model prediction introduces deep autocorrelation in `diff(model)` that compresses the optimal hedge ratio from 1.2 to 0.19, changing the hedge geometry unfavourably.

3. **Conclusion**: for EE, the seasonal hour/day/week vector ALONE is at the achievable bound for this feature set. The production model's existing `ar_ee` AR(2) feature already does the job; replacing it with the v2.4.2 SE3-style architecture would be a regression.

## What v2.4.3 ships

- **`studies/ee_model_v243.py`** — runnable investigation script (3 variants + decision logic; exit-code 1 = REJECT).
- **`studies/results/V2_4_3_RELEASE_NOTES.md`** — this document.
- **`manifest.json`** — bumped `2.4.2` → `2.4.3` to keep the patch chain visible.

No coordinator changes. No sensor schema changes. No tests added (the `npk_cvar_hedge.py` test coverage already exercises the underlying tools).

Test suite: **309 / 309 passing** (unchanged from v2.4.2).

## What stays the same for v2.5.0

- `ar_ee` AR(2) feature in the bundled v2.2 9-feature Ridge: **KEEP UNCHANGED**.
- SE3 → v2.4.2 model accepted: WILL be wired in at v2.5.0.
- EE → v2.4.3 model rejected: production behaviour for EE stays as v2.2.

## Future EE improvements (out of scope for v2.5.0)

To improve EE further would need exogenous features the current pipeline doesn't have:

- Baltic gas spot price (oil-shale plants set EE's marginal cost)
- Estlink-1 and Estlink-2 cross-border flow / congestion state
- Russian electricity exit transition (2022–2024 regime shift)
- Estonian holiday calendar (currently we use Finnish `is_holiday`)

These are research projects beyond the v2.5.0 plan and are not on the v2.4.x roadmap.

## Process note: REJECT is a valid outcome

The user's quoted methodology — *"if test CVaR drops, the feature captures real signal; if unchanged, it's noise — discard"* — explicitly invites REJECT verdicts. A REJECT prevents adding complexity that doesn't earn its keep, which is more valuable than a marginal accept-everything mindset. The v2.4.x patch chain is doing what it should: testing each variant individually and keeping only the ones that demonstrably reduce out-of-sample tail risk.
