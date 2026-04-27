# Findings: AR(2) vs SARIMAX for Neighbor Price Forecasting

**Validation date:** 2026-04-27
**Training period:** 2022-04-15 → 2025-12-31 (~3.7 years of hourly observations)
**Holdout period:** 2026-01-01 → 2026-04-15 (3.5 months)
**Anchors:** 4 walk-forward points (Jan 1, Feb 2, Mar 7, Apr 9), 168h forecast each
**Decision criteria:** SARIMAX must beat AR(2) on hourly MAE (≥2/3 zones), `dk_cheap[3]` cheap-4h (all zones), and weekend MAE (≥2/3 zones).

## Outcome

**❌ DECISION: KEEP AR(2)**

| Criterion | SARIMAX wins | Need | Pass? |
|-----------|--------------|------|-------|
| Hourly MAE | 1 of 3 (SE1 only) | ≥2/3 | ❌ |
| `dk_cheap[3]` (cheap 4h) | 1 of 3 (SE1 only) | all | ❌ |
| Weekend MAE | 0 of 3 | ≥2/3 | ❌ |

## Per-Zone Picture

### SE1 — SARIMAX is competitive
SARIMAX wins on MAE (-2.5), RMSE (-4.4), and most `dk_cheap` and `dk_peak` levels.
AR(2) marginally wins weekend MAE (+0.7). SE1 has a near-random-walk AR(2)
(`max_root ≈ 0.97`) that is essentially "carry forward the last deviation"; SARIMAX's
calendar exog adds value over this weak baseline.

### SE3 — Mixed; AR(2) better on cheap end
SARIMAX wins MAE marginally (-1.1) and most `dk_peak` levels (priciest hours), but
AR(2) wins `dk_cheap` for k=1..6 by 0.3-2.7 pts and weekend by 9.4 pts. SARIMAX's
weekend Fourier interaction approximation underfits the deep weekend dip that
AR(2)'s explicit `profile_we[24]` captures cleanly.

### EE — AR(2) dominates
AR(2) wins on weekend MAE by 33 points (37.6 vs 70.8) and on `dk_cheap[1..12]` by
6-20 points across the board. SARIMAX wins `dk_peak[7..12]` (priciest 7-12 hours)
modestly. EE has the strongest workday vs weekend price differential, and SARIMAX's
diurnal × weekend interaction captures that less well than AR(2)'s separate
weekend hourly profile.

## Structural Pattern

Across all 3 zones a consistent pattern emerges:

- **SARIMAX is better at predicting peak hours** (`dk_peak` mostly wins from k=2-3 onward)
- **AR(2) is better at predicting cheap hours** (`dk_cheap` mostly wins, especially in EE)
- **AR(2) is much better on weekends** in SE3 and EE; tied in SE1
- **SARIMAX has lower Spearman ρ** in 2/3 zones — produces less well-ordered forecasts
  (which is the relevant rank quality for D(k) at any threshold)

## Why AR(2) Wins on Weekends

AR(2) carries 48 calendar parameters per zone (24-hour `profile_wd` + 24-hour
`profile_we`). The weekend profile is just an arithmetic mean of weekend observations
at each hour — nothing fancy, but flexible enough to capture the (often very different)
weekend price shape.

SARIMAX's `regression-with-ARMA-errors` formulation captures the weekend pattern via:
- 6 day-of-week dummies (raises/lowers daily level)
- 4 diurnal Fourier (intra-day shape, K=2)
- 4 diurnal × weekend interaction (separate weekend shape, K=2)

Effectively only 2 distinct intra-day shapes (workday, weekend). With 4 years of mixed
price regimes (2022 spike + 2023 normal + 2024-25 lower), the OLS-style fit averages
across regimes and can't track the recent weekend dynamics as well as AR(2)'s
profile-based mean.

## Why SARIMAX Wins on Peak Hours (when it wins)

In SE1 and SE3, SARIMAX produces lower-magnitude forecasts at peak hours, which
happens to better match the recent (post-2022) price regime where extreme peaks
have moderated. AR(2)'s profile-based forecast carries more of the 2022-era peaks
into the present, biasing it high on `dk_peak`.

## Why the Single-Anchor Result Was Misleading

A 1-anchor (Feb 19) quick run showed SARIMAX losing on all metrics across all 3
zones. The 4-anchor walk-forward shows mixed results, especially on SE1 where
SARIMAX actually wins. The Feb 19 week happened to be unfavorable for SARIMAX.
**Lesson: validation gates must use multiple anchors** to avoid single-week artifacts.

## Architectural Notes

The originally proposed `SARIMAX(2,0,1)(1,1,0)[168]` (true seasonal SARIMAX with
weekly period) was tested and abandoned — even 30 days of hourly data took >5
minutes to fit due to O(s²) Kalman filter cost. Switched to "regression with
ARMA errors": `SARIMAX(2,0,1)(0,0,0,0)` with rich calendar exog. This is the
industry-standard approach for hourly electricity prices with weekly seasonality
(see Weron 2014 review). 4-year fit takes ~2 minutes, forecast is instantaneous.

This means the SARIMAX we tested **does NOT have explicit per-week recurrence in
the state** (e.g. "this Monday morning is similar to last Monday morning"). It
captures weekly patterns purely through the regression on day-of-week dummies and
diurnal × weekend Fourier interactions. AR(2)'s `profile_wd[24] + profile_we[24]`
gives it 48 calendar parameters, vs SARIMAX's ~14 — AR(2) has more capacity to
fit per-hour-of-week patterns.

## Recommendations

1. **Keep AR(2) as the production neighbor model.** It is robust, fast, and has
   competitive accuracy across the relevant metrics for thermal optimization
   (especially `dk_cheap` for scheduling and weekend behavior).

2. **Do not pursue SARIMAX integration further** unless we can address the weekend
   underfit. Possible avenues (not pursued here):
   - Replace diurnal × weekend Fourier interaction with full hour-of-week dummies
     (167 features) — gives SARIMAX equivalent capacity to AR(2).
   - Use shorter rolling training window (1 year) to reduce regime-mismatch bias.
   - Time-decay weighted SARIMAX fit (would require custom likelihood).

3. **Phase 6 of the original plan (segment reduction 4→2) remains valid
   independently.** The 4-segment duration model can still be simplified to
   day/night without depending on the SARIMAX upgrade. Segments only mattered
   for tariff alignment; AR(2) already captures the weekly + diurnal structure
   adequately.

4. **The D(k) cheap/peak refactor (Phase 1) is independent of this decision and
   should proceed.** It improves the semantic interpretation of the D(k) array
   regardless of which neighbor model produces the underlying price forecasts.

## Artifacts

- Validation script: `studies/validate_neighbor_models.py`
- Full report (4-anchor): `studies/results/validation_20260427_0655.md`
- Raw JSON: `studies/results/validation_20260427_0655.json`
- Quick run (1-anchor, archived): `studies/results/validation_20260427_0637.md`
- D(k) utility (Phase 1): `src/dk_utils.py`
- SARIMAX trainer (kept for future iteration): `src/sarimax_neighbor.py`
