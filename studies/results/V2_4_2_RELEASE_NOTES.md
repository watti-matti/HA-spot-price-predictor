# v2.4.2 — SE3 model validation (offline study, ACCEPT)

## TL;DR

**No coordinator behaviour change.** v2.4.2 builds and validates the de-seasonalized SE3 price model proposed by the user, gated on the NPK-CVaR hedge methodology from v2.4.1. The model **PASSES** the acceptance gate with a +3.07 pp improvement over the windowed seasonal-only baseline (−6.06 % → −2.99 %). It will be wired into the coordinator at v2.5.0 once EE (v2.4.3) and FI (v2.4.4) variants are also validated.

## What v2.4.2 ships

### 1. SE3 model build & validation (`studies/se3_model_v242.py`)

The model architecture per the user's directive:

```
P_SE3 = P_hour_SE3 + P_day_SE3 + P_week_SE3
      + β_hydro · hydro_offset_t
      + β_workday · is_workday_t
      + β_AR1 · Y_{t-1}
      + ε_t
```

Where:
- `hydro_offset_t` = Statnett Norwegian total reservoir % − week-of-year baseline (`mean(total_pct) for that week across observed years`)
- `is_workday_t` = 1 if Mon–Fri, 0 if Sat/Sun
- `Y_{t-1}` = previous-hour deseasonalized residual (AR(1) lag)

### 2. Fitted coefficients on real data (104-week Statnett window)

| Coefficient | Value | Interpretation |
|---|---:|---|
| `β_hydro` | **−0.0894** EUR/MWh per % offset | More water → lower price ✓ |
| `β_workday` | **−0.248** EUR/MWh | Slight workday-vs-weekend differential |
| `β_AR(1)` | **+0.942** | Residual half-life ≈ 12 h, matches v2.4.1 baseline |
| R² on Y_SE3 | **0.891** | AR(1) absorbs 89 % of de-seasonalized variance |

### 3. NPK-CVaR hedge result (gate check)

| Variant | h_hat | CVaR test hedged | Reduction |
|---|---:|---:|---:|
| **Baseline** (seasonal-only, same window) | 0.796 | 21.65 | **−6.06 %** |
| **v2.4.2 model** (seasonal + hydro + workday + AR(1)) | 0.175 | 21.02 | **−2.99 %** |
| **Δ improvement** | | | **+3.07 pp ✓ ACCEPT** |

The baseline is windowed to match the Statnett-available period (May 2024 → April 2026) so the comparison is apples-to-apples. The v2.4.1 baseline measurement of −7.78 % was on the full 2023+ window; same metric is −6.06 % on the v2.4.2 window because 2023 (the Russia/Ukraine crisis tail) is excluded.

### 4. Tests

6 new tests, all passing alongside the 303 carried forward from v2.4.1:

- `tests/test_se3_model_v242.py` — `build_hydro_offset` invariants, model coefficient recovery on synthetic data (hydro sign correct, AR(1) coefficient ≈ 0.9), hedge reduction interface

Total: **309 / 309 passing.**

### 5. Auto-written results report (`studies/results/se3_model_v242_results.md`)

The script writes a fresh markdown summary after each run so the documented numbers stay synchronised with the live Statnett window.

## What's NOT changed in v2.4.2

- **Coordinator runtime is unchanged.** The v2.2 9-feature Ridge model continues to drive predictions. The validated SE3 variant lives in `studies/` only.
- **No sensor schema change.**
- **No new config options.**
- HACS users will see version `2.4.2` but observe no runtime difference.

## Next patches in the v2.4.x → v2.5.0 sequence

| Patch | Scope | Gate |
|---|---|---|
| v2.4.3 | EE model: seasonal_EE + is_workday + AR(1) (no hydro) | Beat +4.07 % EE baseline |
| v2.4.4 | FI revised model | Beat +6.06 % FI baseline |
| v2.4.5 | Alternative solar model (clear-sky × cloudiness) | Beat +6.06 % FI baseline |
| **v2.5.0** | Consolidate all accepted variants into coordinator | Combined model beats v2.4 on full backtest |

## Reproducibility

```bash
cd HA-spot-price-predictor
python studies/se3_model_v242.py
```

Output goes to stdout AND writes a fresh `studies/results/se3_model_v242_results.md`. The Statnett window updates each run.

## Files

- **New**: `studies/se3_model_v242.py` (260 LOC, runnable end-to-end)
- **New**: `tests/test_se3_model_v242.py` (6 tests, pure-Python no network)
- **New**: `studies/results/se3_model_v242_results.md` (auto-generated)
- **New**: `studies/_statnett_reservoir_history.json` (Statnett snapshot at build time, for offline reproducibility)
- **Modified**: `manifest.json` (`2.4.1` → `2.4.2`)

Test suite: **309 / 309 passing.**
