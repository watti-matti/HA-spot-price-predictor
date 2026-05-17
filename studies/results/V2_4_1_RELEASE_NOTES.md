# v2.4.1 — Validation framework foundation

## TL;DR

**No model behaviour change.** v2.4.1 builds the validation infrastructure (NPK-CVaR hedge tool + Statnett hydro reservoir client + baseline measurements) that every subsequent v2.4.x patch will be gated against. First step in the v2.4.x → v2.5.0 model upgrade series proposed in the project plan.

## What v2.4.1 ships

### 1. NPK-CVaR hedge analysis tool (`studies/npk_cvar_hedge.py`)

Pure-NumPy/SciPy port of `studies/Matlab_study_on_CVAR/npk_cvar_hedge_demo.m`. Validated against the MATLAB benchmarks documented by the original study:

| Benchmark | MATLAB target | Python port |
|---|---:|---:|
| OU half-life (FI, 2023+) | 10.3 h | **10.2 h** ✅ |
| Raw 48 h hedge `h_hat` | 0.99 | **0.922** ✅ (within ±5 % tolerance) |
| Raw 48 h CVaR reduction | 7.9 % | **6.06 %** ✅ |
| Deseasonalized 2 h `h_hat` | −0.12 | **−0.119** ✅ |
| Deseasonalized 2 h CVaR reduction | 1.4 % | **1.43 %** ✅ |

Full validation report: [`studies/results/npk_cvar_python_port_validation.md`](../../studies/results/npk_cvar_python_port_validation.md).

API:

- `fit_seasonal_hdw(x, ts)` — sequential-subtraction decomposition `x = P_hour + P_day + P_week + Y` (the Moazeni-Powell methodology)
- `fit_ou_ar1(Y)` — discrete-time OU / AR(1) fit with `λ`, `μ`, `σ`, half-life
- `npk_cvar_objective(h, v, rS, rF, alpha)` — Rockafellar form with Gaussian KDE smoothing (Silverman bandwidth)
- `optimize_hedge(rS, rF, alpha)` — Nelder-Mead with bounds penalty
- `historical_cvar(L, alpha)` — empirical CVaR sanity check
- `acf(y, lags)` — sample autocorrelation
- `run_baseline_hedge_analysis(ts, P, mode, futures_lag_hours)` — end-to-end pipeline

### 2. Statnett hydro reservoir client (`custom_components/spot_price_predictor/statnett_client.py`)

Async client for `driftsdata.statnett.no/restapi/Reservoir/` — Norway TSO's public weekly Nordic reservoir data. **No authentication, no API key, no rate limits.** Weekly resolution, 1–2 week publication lag.

Returns Norwegian total reservoir fill % plus per-zone breakdown (NO1-NO5). Norwegian hydro storage drives Nordic prices via cross-border flows (Norway holds ~50 % of European hydro capacity), so this serves as the exogenous hydro signal for both Swedish and Finnish price models.

Features:
- 24 h cache freshness (hydro doesn't change hourly per user directive)
- Persistent `.storage/spot_price_predictor_hydro_cache.json` for restart resilience
- Silent fallback to cache when Statnett unreachable
- Returns normalised JSON: `{fetched_at, weeks: [{year, week, total_pct, zones: {no1, ...}}], latest: {...}}`

**v2.4.1 wires the client up but does NOT yet feed any model input** — that happens in v2.4.2 (SE3 with hydro offset). This isolates the validation framework from any model risk.

### 3. Baseline CVaR measurements (`studies/results/current_model_cvar_baseline.md`)

Per-zone baselines that every v2.4.x patch must beat:

| Zone | n samples | h_hat | CVaR test unhedged | CVaR test hedged | Reduction |
|---|---:|---:|---:|---:|---:|
| **FI** (Finland) | 29 112 | 0.922 | 44.69 | 41.99 | **+6.06 %** |
| **SE3** (southern Sweden) | 44 254 | 0.978 | 21.78 | 23.47 | **−7.78 %** |
| **SE1** (northern Sweden) | 44 254 | 0.413 | 17.13 | 17.18 | **−0.30 %** |
| **EE** (Estonia) | 29 134 | 1.209 | 86.06 | 82.56 | **+4.07 %** |

Notable: **SE3 baseline is NEGATIVE** — the seasonal-only hedge actually hurts out-of-sample CVaR. This is precisely the failure case v2.4.2 (SE3 with hydro reservoir offset + workday/holiday + AR(1)) is designed to fix.

### 4. Tests

23 new tests, all passing alongside the 280 carried forward from v2.4.0:

- `tests/test_npk_cvar_hedge.py` (14) — seasonal decomposition invariants, OU parameter recovery on synthetic data, Rockafellar objective, hedge optimization, ACF
- `tests/test_statnett_client.py` (9) — response normalisation, sorting, zone preservation, malformed-entry skipping, cache fallback on fetch failure, cache freshness

Total: **303/303 passing.**

## Pass/fail gates for next patches

Per the validation methodology in the plan:

```
v2.4.2  (SE3 model w/ hydro + holidays + AR(1)):  reduction must beat -7.78%
v2.4.3  (EE  model w/ holidays + AR(1)):          reduction must beat +4.07%
v2.4.4  (FI  model revision):                     reduction must beat +6.06%
v2.4.5  (solar model alt):                        reduction on FI hedge must beat +6.06%
```

Each patch that doesn't improve its zone's baseline gets rejected. v2.5.0 only ships once all accepted improvements compose into a model that beats v2.4.0 on the full FI backtest.

## What's NOT changed in v2.4.1

- No coordinator behaviour change
- No sensor schema change
- No new config options
- No change to the bundled 9-feature Ridge model
- HACS users will see version `2.4.1` but observe no runtime difference

This is pure infrastructure. The behavioural changes start with v2.4.2.

## Files

- **New**: `studies/npk_cvar_hedge.py` (270 LOC)
- **New**: `custom_components/spot_price_predictor/statnett_client.py` (220 LOC)
- **New**: `tests/test_npk_cvar_hedge.py` (14 tests)
- **New**: `tests/test_statnett_client.py` (9 tests)
- **New**: `studies/results/npk_cvar_python_port_validation.md`
- **New**: `studies/results/current_model_cvar_baseline.md`
- **New**: `studies/README.md`
- **Modified**: `manifest.json` (`2.4.0` → `2.4.1`)

Test suite: **303 / 303 passing**.
