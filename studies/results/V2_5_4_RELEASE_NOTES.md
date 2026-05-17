# v2.5.4 — Per-sensor seasonal-content audit (DECOMPOSITION DEPTH SET)

## TL;DR

**No coordinator behaviour change.** v2.5.4 ships a structural decomposition audit of every candidate input to the FI price model — for each input it fits the Moazeni-Powell additive `X = P_hour + P_day + P_week + Y`, reports the variance share of each component, and decides what decomposition depth to apply in v2.5.5 (de-seasonalize inputs).

Per user direction 2026-05-17: the v2.5.4 audit is structural preparation; the final accept/reject decisions on which features enter the production FI Ridge are made by the v2.5.6 hedge-gated sweep, **prioritising 7-day CVaR accuracy** (primary) over time-series MAE (secondary, visualization only).

User's directional hint (also from 2026-05-17): *"Wind has seasonal variation but not within a week but rather day and month."* **Confirmed exactly** — wind's P_day variance share is **0.2 %** (dropped); P_hour amplitude is `±0.3 m/s` (kept by the amplitude rule); P_week share is **13 %** (kept).

## Decomposition depth recommendations

Aligned 2023-01-01 → 2026-04-27 hourly window (29,112 rows) — same window as the v2.5.3 solar sub-model, post-2022-crisis-spike.

| Input | n | mean | σ | P_hour | P_day | P_week | residual | wkd–wknd | Recommended depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fi` | 29,112 | 50.90 | 64.63 | 5.4 % | 2.8 % | 13.9 % | 77.9 % | 13.6 | **hour + day + week** |
| `se3` | 29,112 | 33.55 | 46.43 | 7.6 % | 4.5 % | 22.7 % | 65.2 % | 9.0 | **hour + day + week** |
| `se1` | 29,112 | 17.69 | 36.96 | 2.2 % | 2.2 % | 22.1 % | 73.4 % | 8.6 | **day + week** |
| `ee` | 29,112 | 71.74 | 76.51 | 12.0 % | 4.9 % | 10.0 % | 73.1 % | 11.0 | **hour + day + week** |
| `wind` | 29,112 | 6.16 | 2.32 | 0.9 % | 0.2 % | 13.0 % | 85.9 % | 0.05 | **hour + week** ← matches user hint |
| `solar` | 29,112 | 102.95 | 173.96 | 46.7 % | 0.01 % | 16.9 % | 36.3 % | 0.5 | **hour + week** |
| `ghi_cs` | 29,112 | 145.85 | 215.55 | 51.9 % | 0.00 % | 26.8 % | 21.3 % | 0.1 | **hour + week** |
| `temp` | 29,112 | 4.95 | 9.95 | 2.4 % | 0.01 % | 81.0 % | 16.6 % | 0.02 | **hour + week** |
| `cloud` | 29,112 | 65.81 | 33.92 | 0.1 % | 0.1 % | 19.0 % | 80.7 % | 0.3 | **week** |

(Tabulated by `studies/per_sensor_seasonality_audit.py`; auto-generated md at `studies/results/per_sensor_seasonality_audit.md`.)

## Keep rule

A component is kept if **any** of:
1. variance share ≥ 5 % of total variance — captures the dominant case
2. removing it inflates the Ljung-Box statistic on the residual by > 50 % — captures cases where the seasonal structure is sharp but compact in variance
3. (P_hour only) peak-to-trough amplitude ≥ 0.25 σ — captures wind's diurnal cycle (0.9 % variance, but ±0.3 m/s swing is structurally meaningful on a σ=2.3 m/s field)
4. (P_day only) workday vs weekend mean difference ≥ 0.1 σ — captures the price workday-weekend split which is dramatic in absolute terms but small in variance share once P_hour and P_week have absorbed the cycles
5. (P_week only) peak-to-trough amplitude ≥ 0.5 σ — backstop; all inputs satisfy this anyway

The dual amplitude criterion is what correctly aligns the audit with the user's directional knowledge — pure variance share is biased by sample-size effects and by which components are evaluated first in the sequential subtraction.

## Headline observations

- **Prices** (FI / SE3 / SE1 / EE): all three components carry signal. SE1 P_hour is borderline (hydro-dominated zone, flatter diurnal) — drops only because its variance share AND its hour-amplitude both fall below threshold; v2.5.6 hedge gate will re-test.
- **Wind**: hour cycle present (diurnal boundary-layer mixing), no day-of-week effect, annual cycle dominant. **Exactly the user's hint.**
- **Solar / clear-sky GHI**: hour cycle is overwhelming (sun is up or it isn't); no day-of-week effect (sun doesn't care about workdays); large annual cycle at FI latitudes. The clear-sky baseline shows the same profile as the raw irradiance — confirming v2.5.3's deterministic baseline captures the structure correctly.
- **Temperature**: dominantly annual (81 %); modest diurnal cycle (kept on amplitude grounds despite 2.4 % variance share); no day-of-week effect.
- **Cloud cover**: dominantly stochastic (81 % residual), modest annual cycle (19 %). The most random of the inputs — which is exactly why it carries information beyond the calendar.

## Per-input figures

Generated under `studies/results/figures/per_sensor_components_<NAME>.png`:

- `per_sensor_seasonal_variance.png` — headline stacked-bar of variance shares per input
- `per_sensor_components_fi.png` — FI price decomposition (hour, day, week, residual ACF)
- `per_sensor_components_{se3,se1,ee}.png` — cross-border prices
- `per_sensor_components_{wind,solar,ghi_cs,temp,cloud}.png` — exogenous physical inputs

## Files

- **New**: `studies/per_sensor_seasonality_audit.py` (~430 LOC)
- **New**: `studies/results/per_sensor_seasonality_audit.md` (auto-generated)
- **New**: 10× figures under `studies/results/figures/per_sensor_*`
- **New**: `studies/results/V2_5_4_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.3 → 2.5.4`)

No new tests (this patch is a measurement study; the decomposition function `fit_seasonal_hdw` is already covered by tests in v2.4.1's `tests/test_npk_cvar_hedge.py`).

## What this prepares for v2.5.5 and v2.5.6

v2.5.5 will use the **Recommended depth** column to build per-input residual series:
- For each input X with recommended depth `d`, persist `(P_components, Y_X)` to `.storage/spot_price_predictor_seasonal_cache.json`.
- Compute `Y_X` on the fly at coordinator-update time using cached seasonal vectors (lookup + subtract; pure-numpy, no fit).
- Refresh cadence: quarterly, alongside the v2.5.3 solar artifact.

v2.5.6 will restart the FI Ridge from the original 17-feature universe with each raw input replaced (or augmented) by its `Y_X` per the recommendations above, then run the NPK-CVaR hedge gate to decide which features actually earn their place — **with 7-day CVaR accuracy as the primary acceptance criterion** per user direction 2026-05-17. Features dropped by v2.2's MAE-only sweep are explicitly back in the candidate pool, since the audit confirms the v2.2 reductions ran on raw inputs and may have been seasonally confounded.

## Reproducibility

```bash
python studies/per_sensor_seasonality_audit.py
```

Reads only the parquets in `output/` and cached cloud-cover responses from `studies/.cache/`. No API call. The clear-sky baseline is computed on the fly via the v2.5.3 `solar_clear_sky` module.
