# v2.5.5 — De-seasonalized input infrastructure (PRE-HEDGE STRUCTURAL PREP)

## TL;DR

**No coordinator behaviour change yet.** v2.5.5 ships the production-ready infrastructure that turns the v2.5.4 audit into deployable residual features `Y_X(t) = X(t) − seasonal_components(t)` for every candidate input. The v2.5.6 hedge-gated sweep then uses these residuals to rebuild the FI Ridge from the original 17-feature universe — **with 7-day CVaR accuracy as the primary acceptance criterion** per user direction.

The mechanics mirror the v2.5.3 solar sub-model pattern: training is offline (one quarterly run), the shipped JSON artifact carries the frozen vectors, and runtime inference is pure-numpy with no API calls.

## What lands

### New module `custom_components/spot_price_predictor/seasonal_decomposition.py`

Public API:

| Function | Purpose |
|---|---|
| `fit_components(x, ts, depth)` | Offline: sequential Moazeni-Powell fit `P_hour → P_day → P_week` with skippable components per the v2.5.4 depth recommendations. |
| `compute_seasonal_part(ts, components)` | Reconstruct the seasonal sum at given timestamps. Components absent from the dict are treated as zero — this is how the per-input depth choice is honoured at runtime without any separate spec. |
| `compute_residual(x, ts, components)` | Runtime: `Y(t) = x(t) − compute_seasonal_part(ts, components)`. |
| `build_artifact(inputs, ...)` | JSON-serialisable artifact assembly, records depths + training window + stats. |
| `load_components(path)` | Load the persisted artifact; returns None if missing (caller can fall back to raw inputs). |
| `DEFAULT_DEPTHS` | Module-level constant pinning the v2.5.4 verdict — `wind: hour + week`, etc. The `test_default_depths_match_v2_5_4_audit` test enforces it. |

### Builder script `studies/build_seasonal_components.py`

Reads `output/*.parquet` + the v2.5.3 cached cloud-cover responses, fits per-input components per `DEFAULT_DEPTHS`, and writes the artifact to `custom_components/spot_price_predictor/data/seasonal_components_default.json`. Auto-generates `studies/results/seasonal_components_build.md` with the fit statistics. No external API call.

### Shipped artifact `data/seasonal_components_default.json` (22 KB)

Window 2023-01-01 → 2026-04-27 (29,112 hourly rows). Per-input fit statistics:

| Input | Depth | σ raw | σ Y | Var reduction | E[Y] |
|---|---|---:|---:|---:|---:|
| `fi` | hour + day + week | 64.63 | 57.06 | **22.1 %** | −2.5e-16 |
| `se3` | hour + day + week | 44.84 | 36.22 | **34.8 %** | +2.5e-16 |
| `se1` | day + week | 34.91 | 30.37 | **24.3 %** | 0.0 |
| `ee` | hour + day + week | 72.82 | 62.27 | **26.9 %** | +2.5e-16 |
| `wind` | hour + week | 2.32 | 2.16 | **13.8 %** | +4.7e-17 |
| `solar` | hour + week | 201.12 | 121.23 | **63.7 %** | −1.8e-15 |
| `ghi_cs` | hour + week | 219.78 | 101.43 | **78.7 %** | −2.6e-15 |
| `temp` | hour + week | 9.80 | 4.00 | **83.4 %** | −9.4e-17 |
| `cloud` | week only | 27.44 | 24.69 | **19.0 %** | −1.5e-15 |

`E[Y] ≈ 0` to machine precision across all inputs — confirms the sequential subtraction is mechanically correct. Var-reduction numbers match the v2.5.4 audit shares exactly.

### Tests `tests/test_seasonal_decomposition.py` (14 tests)

- Hour/weekday/week-of-year index helpers (3 tests; including the diurnal-cycle coverage and the Jan-1-2024-is-Monday calibration).
- Synthetic-pattern recovery: `X = a·sin(2π·h/24)` should give exact `P_hour` reconstruction; pure-diurnal input should leave `P_day ≈ 0` after the sequential pass.
- Depth honouring: components absent from the artifact are not subtracted; passing wrong-length vectors raises `ValueError`.
- Artifact lifecycle: default-depths recording, custom-depth override, missing-file `load` returns `None`, JSON round-trip preserves residual exactly.
- **`test_default_depths_match_v2_5_4_audit`** — enforces the v2.5.4 verdict as a regression test, so an inadvertent change to `DEFAULT_DEPTHS` breaks the test rather than silently shipping a different model.

## Why no coordinator change

The roadmap originally bundled "de-seasonalize inputs" with "drop `month_cos` / AR-daytype machinery" inside v2.5.5. In practice these split naturally:

- **v2.5.5 (this patch)**: adds the infrastructure that produces `Y_X` for any input. Pure additive feature — nothing in the coordinator reads it yet.
- **v2.5.6 (next)**: builds candidate FI Ridge models on `(raw + residual)` feature combinations and runs the NPK-CVaR hedge gate restarting from the original 17-feature universe. Whichever variant wins the hedge then displaces v2.2 in v2.6.0 — at which point `month_cos` and AR-daytype come out only if the data agreed they should.

This staging keeps each numbered patch easily reversible and the hedge-gate decisions data-driven.

## Files

- **New**: `custom_components/spot_price_predictor/seasonal_decomposition.py` (~240 LOC)
- **New**: `custom_components/spot_price_predictor/data/seasonal_components_default.json` (22 KB; fit stats embedded)
- **New**: `studies/build_seasonal_components.py` (~190 LOC)
- **New**: `tests/test_seasonal_decomposition.py` (14 tests)
- **New**: `studies/results/seasonal_components_build.md` (auto-generated)
- **New**: `studies/results/V2_5_5_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.4 → 2.5.5`), `README.md` release-notes index

## Tests

**363 / 363 passing** (349 prior + 14 new seasonal-decomposition tests).

## Reproducibility

```bash
python studies/build_seasonal_components.py        # refresh the artifact
git add custom_components/spot_price_predictor/data/seasonal_components_default.json
git commit -m "seasonal components quarterly refit"
```

Reads only locally cached data. Refresh quarterly alongside the v2.5.3 solar artifact.

## Next step — v2.5.6

The hedge-gated input-selection sweep, primary metric = 7-day CVaR accuracy:

1. Build candidate features `(raw X, Y_X)` for every input in the original 17-feature universe + the new candidates (solar sub-model output, hydro reservoir, spread features).
2. For each feature, fit a Ridge model with that feature added/dropped and run `npk_cvar_hedge.run_baseline_hedge_analysis()` on 7-day-horizon forecasts.
3. ACCEPT features whose presence improves test CVaR by ≥ 0.3 pp; REJECT otherwise.
4. Output the winning feature set + the per-feature CVaR contribution scorecard.
