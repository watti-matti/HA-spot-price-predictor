# `studies/` — research scripts, validation tools, and historical experiments

This directory contains research code, calibration scripts, and validation tools that are not part of the HA integration runtime. They support model development and decision-making but are not loaded by the coordinator.

## Privacy invariant (read this before adding new scripts)

**Never commit the following kinds of data to this repository:**

- API keys, tokens, or any other credential (Fingrid `x-api-key`, ENTSO-E security token, HA long-lived tokens, etc.)
- Personal or address-level locations — street addresses, household coordinates beyond regional precision (~0.1°), HA installation site coordinates
- Anything that uniquely identifies an individual user or installation

**Local storage pattern**: put secrets in a `.env` file at the repo root. `.env` is in `.gitignore` and is auto-loaded by the study scripts that need it (see the `_load_dotenv()` helper in `solar_clear_sky_submodel.py` and `fingrid_netload_study.py`). Example:

```
FINGRID_API_KEY=your_free_key_here
ENTSOE_TOKEN=your_token_here
```

Before any commit, run `git log -S "<suspect-value>"` to surface accidental inclusions. Cache directories under `studies/.cache/` and `studies/_fingrid_cache/` are also gitignored — raw API responses stay local even though they don't contain credentials.

Treat any contribution that ships a `.env`, hard-codes a key in source, or includes user-specific coordinates as a security regression.

## Validation tools

### `npk_cvar_hedge.py` (NEW in v2.4.1)

NPK-CVaR hedge analysis — the **canonical model-selection criterion** for the v2.4.x → v2.5.0 release sequence. Ported from `Matlab_study_on_CVAR/npk_cvar_hedge_demo.m` and validated against the MATLAB benchmarks (see `results/npk_cvar_python_port_validation.md`).

Every new model variant proposed for v2.4.2 onwards must reduce out-of-sample CVaR vs the current baseline (documented in `results/current_model_cvar_baseline.md`) to be accepted.

Quick usage:

```python
import sys; sys.path.insert(0, 'studies')
import pandas as pd
from npk_cvar_hedge import run_baseline_hedge_analysis

df = pd.read_parquet('output/fi_prices.parquet')
df = df[df.index >= '2023-01-01']
ts = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=3)
result = run_baseline_hedge_analysis(
    ts, df['price_eur_mwh'].values,
    mode='raw', futures_lag_hours=48, alpha=0.05,
)
print(f"h_hat = {result['hedge']['h_hat']:.3f}")
print(f"CVaR reduction = {100 * (result['hedge']['cvar_test_hist_unhedged'] - result['hedge']['cvar_test_hist_hedged']) / result['hedge']['cvar_test_hist_unhedged']:.2f}%")
```

API:
- `fit_seasonal_hdw(x, ts)` — sequential-subtraction decomposition `x = P_hour + P_day + P_week + Y`
- `fit_ou_ar1(Y)` — discrete-time OU / AR(1) fit; returns `λ`, `μ`, `σ`, half-life, `b`
- `npk_cvar_objective(h, v, rS, rF, alpha)` — Rockafellar form with Gaussian KDE smoothing
- `optimize_hedge(rS, rF, alpha)` — Nelder-Mead optimisation with hedge-ratio bounds
- `historical_cvar(L, alpha)` — empirical CVaR for sanity checks
- `acf(y, lags)` — sample autocorrelation at specified lags
- `run_baseline_hedge_analysis(ts, P, mode, futures_lag_hours)` — end-to-end pipeline

### Results

- `results/npk_cvar_python_port_validation.md` — Python vs MATLAB benchmark comparison (all 4 checks PASS)
- `results/current_model_cvar_baseline.md` — per-zone (FI, SE3, SE1, EE) CVaR baselines that v2.4.2+ must beat
- `results/V2_3_RELEASE_NOTES.md`, `results/V2_3_1_RELEASE_NOTES.md`, `results/V2_4_0_RELEASE_NOTES.md`, `results/V2_4_1_RELEASE_NOTES.md` — historical release notes

## Subdirectories

- `Matlab_study_on_CVAR/` — original MATLAB scripts that defined the Phase 3 methodology (Moazeni-Powell-derived seasonal + OU model + NPK-CVaR hedge). The `nordpool_analysis_findings.md` file in there is the spec for the Python port.
- `results/` — generated reports, release notes, validation outputs.
- `_fingrid_cache/` — cached Fingrid API responses for fast iteration on training scripts.

## Existing scripts (pre-v2.4.1)

| Script | Purpose |
|---|---|
| `validate_forecaster_performance.py` | Walk-forward validation (rolling 540-day refit, 180-day test) |
| `run_validation_sweep.py` | Multi-zone neighbour model comparison (SARIMAX vs AR(2)) |
| `fingrid_netload_study.py` | Net-load feature study (consumption/wind/solar/nuclear) |
| `duration_study.py` | D(k) curve analysis |
| `evaluate.py` | Model evaluation dashboards |
| `extended_dashboard.py` | Extended diagnostic dashboards |
| `time_alignment_study.py`, `coupling_study.py`, `histogram_study.py`, etc. | Various exploratory analyses from v1.x → v2.x development |
