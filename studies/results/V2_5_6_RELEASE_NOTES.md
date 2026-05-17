# v2.5.6 — Hedge-gated input selection sweep (FINDINGS LANDED)

## TL;DR

**No coordinator behaviour change.** v2.5.6 runs the NPK-CVaR hedge gate over a 17-feature candidate universe and selects per-horizon the minimal set whose hedge CVaR-reduction beats the +0.3 pp acceptance threshold per added feature. The selection follows the user direction 2026-05-17: **7-day CVaR accuracy as the primary metric**; secondary horizons (48 h, 24 h) reported alongside.

### Headline numbers (per horizon, accepted features only)

| Horizon | Baseline CVaR % | Accepted features | Final CVaR % | Test MAE |
|---|---:|---|---:|---:|
| **24 h (day-ahead)** | +5.97 % | `Y_fi_lag24`, `is_workday`, `solar_pred_mw` | **+41.22 %** | 35.3 EUR/MWh |
| **48 h (v2.4.x ref)** | +5.68 % | `Y_ghi_cs`, `Y_fi_lag168` | **+7.02 %** | 39.1 EUR/MWh |
| **168 h (7-day, PRIMARY)** | +5.87 % | `Y_fi_lag168`, `is_workday` | **+16.00 %** | 39.3 EUR/MWh |

## Key findings

1. **The hedge value is overwhelmingly horizon-dependent.** At day-ahead the recoverable CVaR-reduction is ~41 %; at 7 days it drops to 16 %. Decisions that depend on 7-day-ahead price clarity are inherently noisier than day-ahead decisions, and the price model cannot manufacture signal that physics doesn't supply at that lead time.

2. **AR lags on the deseasonalized FI residual dominate at every horizon.** At 24 h the winner is `Y_fi_lag24` (+34.4 pp by itself); at 168 h the winner is `Y_fi_lag168` (+9.6 pp). The dominant additional hedge value beyond the seasonal forecast is the "same hour of the same day last week / yesterday" pattern that survives in `Y_fi`.

3. **Cross-border price residuals, weather residuals, and the v2.5.3 solar sub-model do NOT add forward-marginal value at any of the three horizons** under greedy forward-add. The largest non-AR contribution is `Y_ghi_cs` at 48 h (+1.03 pp). The `solar_pred_mw` sub-model only marginally helps at 24 h (+0.36 pp), barely above the +0.3 pp threshold.

4. **`is_workday` adds a small but consistent +0.4–0.5 pp at the longer horizons.** It captures the workday/weekend split that survives both the seasonal decomposition and the AR-lag features.

5. **The v2.2 production model's 9-feature complexity is not justified at the 7-day horizon** under the hedge-gated criterion. The 7-day-optimal feature set is `(Y_fi_lag168, is_workday)` — just two features beyond the seasonal forecast that v2.5.5 already encodes in the target residualisation.

### Caveats — be aware before locking the v2.6.0 design

- **Forward-add is myopic** — it cannot find feature pairs that only help jointly. v2.5.1 found that `Y_se3` + `Y_se1` together (with near-opposite signs `+1.61 / −1.60`) deliver +0.55 pp in joint-hedge mode. The v2.5.6 sweep tries them one-at-a-time and rejects both, because neither helps alone. A pair-aware extension is a candidate follow-up patch (e.g. compute the leave-one-pair-out CVaR for the top-k correlated pairs).
- **The 7-day horizon is a hard target.** The hedge baseline (no features at all) recovers only +5.9 % CVaR — most of the 7-day price-risk is structurally unhedgeable from same-time-now features. The user may want to add a short horizon (e.g. 24 h) as a co-primary metric for sensors / dashboards that actually consume day-ahead forecasts.
- **The MAE / R² story is comparable but slightly different.** At 7 d, MAE drops only marginally with the accepted features (~0.5 EUR/MWh out of 40 baseline). The test-MAE secondary axis matches the user's framing ("for visualization purposes since it is easier to interpret"); the actual production decisions come from the CVaR ranking.

## Detail: full sweep history per horizon

(See `studies/results/v256_input_sweep.md` for auto-generated tables.)

### 24 h — day-ahead

| Step | Feature added | Total | CVaR % | Δ pp | Test MAE |
|---:|---|---:|---:|---:|---:|
| 0 | (baseline) | 0 | +5.97 % | — | 39.87 |
| 1 | `Y_fi_lag24` | 1 | **+40.39 %** | **+34.42** | 35.32 |
| 2 | `is_workday` | 2 | +40.86 % | +0.48 | 35.32 |
| 3 | `solar_pred_mw` | 3 | +41.22 % | +0.36 | 35.32 |
| 4 | `Y_solar` (stop) | 4 | +41.19 % | −0.03 | 35.32 |

### 48 h — v2.4.x reference

| Step | Feature added | Total | CVaR % | Δ pp | Test MAE |
|---:|---|---:|---:|---:|---:|
| 0 | (baseline) | 0 | +5.68 % | — | 39.87 |
| 1 | `Y_ghi_cs` | 1 | +6.71 % | +1.03 | 39.57 |
| 2 | `Y_fi_lag168` | 2 | +7.02 % | +0.31 | 39.09 |
| 3 | `is_workday` (stop) | 3 | +7.28 % | +0.26 | 39.10 |

### 168 h — PRIMARY (user direction)

| Step | Feature added | Total | CVaR % | Δ pp | Test MAE |
|---:|---|---:|---:|---:|---:|
| 0 | (baseline) | 0 | +5.87 % | — | 39.87 |
| 1 | `Y_fi_lag168` | 1 | **+15.51 %** | **+9.64** | 39.30 |
| 2 | `is_workday` | 2 | +16.00 % | +0.49 | 39.32 |
| 3 | `solar_pred_mw` (stop) | 3 | +16.19 % | +0.19 | 39.32 |

## Implications for v2.6.0

Based on the user direction (primary 7-day CVaR) and the sweep results, the candidate production architecture for v2.6.0 is:

```
prediction(t) = seasonal_fi(hour, weekday, week)               # from v2.5.5 artifact
              + ridge_coef_dot([Y_fi_lag168, is_workday])      # learned residual model
```

This is **dramatically simpler** than the v2.2 9-feature Ridge — two features beyond the seasonal layer. Operational benefits:

- Removes the runtime dependency on cross-border price fetch (Elering, elprisetjustnu.se) IF the production model is purely 7-day-optimal. Keep them in the wider feature universe but the candidate selection says they don't help at 7-day.
- Removes runtime dependency on weather features (Open-Meteo wind / solar / temperature) IF same.
- Solar sub-model (v2.5.3) sits at the borderline — it improves day-ahead by +0.36 pp but doesn't beat the threshold at 7 d. Worth retaining for the day-ahead dashboard use case.

But **before locking that simpler design**, two open questions for the user:

1. Should the v2.6.0 model optimise for 7-day CVaR (16 % achievable) or day-ahead CVaR (41 % achievable)? They imply different feature sets, even though the same training infrastructure builds both. A two-model ensemble (separate day-ahead vs 7-day predictors, each picking its own optimal feature subset) is feasible.
2. Is the +0.55 pp v2.5.1 SE3+SE1 joint hedge worth retaining despite forward-add saying neither helps individually? The pair-aware sweep is a one-day patch to add.

## Files

- **New**: `studies/v256_hedge_input_sweep.py` (~470 LOC) — three-horizon forward-add sweep
- **New**: `studies/results/v256_input_sweep.md` (auto-generated; full tables)
- **New**: `studies/results/figures/v256_sweep_path.png` (two-panel: CVaR + MAE across all three horizons)
- **New**: `studies/results/V2_5_6_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.5 → 2.5.6`), `README.md` release-notes index

No tests for the sweep script itself (it's a one-shot study; the underlying `npk_cvar_hedge` and `seasonal_decomposition` infrastructure is tested separately).

## Tests

**363 / 363 passing** (no new tests; full suite verified post-change).

## Reproducibility

```bash
python studies/v256_hedge_input_sweep.py
```

No network call; reads only `output/*.parquet`, the v2.5.3 + v2.5.5 artifacts in `data/`, and the cloud-cover cache in `studies/.cache/`. Sweep at three horizons; ~3 minutes wall clock on a modest laptop.

## Next step — v2.6.0 (consolidation)

With the v2.5.3 → v2.5.6 chain landed, v2.6.0 will:

1. Lock the production feature set per the v2.5.6 verdict (likely `(Y_fi_lag168, is_workday)` for 7-day, with optional `solar_pred_mw` for the day-ahead path).
2. Refit the production Ridge model using the cleaner architecture (target = `Y_fi`, features per the verdict).
3. Replace `model_coefs_default.json` with the new fit.
4. Coordinator wiring: load the v2.5.5 seasonal artifact + the new Ridge coefficients; produce `prediction = seasonal_fi + ridge(Y_features)`.
5. Update README / TECHNICAL_GUIDE / TEKNINEN_TOTEUTUS to reflect the simpler architecture.
6. **First runtime behaviour change in this chain** — every patch from v2.5.3 through v2.5.6 has been research-only.
