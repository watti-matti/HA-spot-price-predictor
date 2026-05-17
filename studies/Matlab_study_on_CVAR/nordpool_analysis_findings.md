# Nord Pool Electricity Price Analysis — Findings & Implementation Spec

**Date:** May 2026
**Data:** Finnish (FI) hourly spot prices from Sähkötin API, Jan 2023 – Apr 2026 (29,183 observations)
**Target platform:** Python on Intel N100 (low-power, no GPU)
**Reference:** Moazeni, Powell & Hajimiragha (2015), IEEE Trans. Power Systems 30(3), 1222–1232

---

## 1. Summary of Key Findings

### 1.1 Seasonal Decomposition Model

The price decomposes additively:

```
P_t = P_hour(h) + P_day(d) + P_week(w) + Y_t
```

| Component | Description | Size |
|---|---|---|
| `P_hour(h)` | Hour-of-day averages (24 bins) | Dominant intra-day |
| `P_day(d)` | Day-of-week averages (7 bins, Monday=1) | Weekend dip |
| `P_week(w)` | Week-of-year averages (53 bins) | Replaces monthly; captures winter/summer |
| `Y_t` | Deseasonalized residual | Ornstein-Uhlenbeck process |

#### Design decision: weekly > monthly cycle
- 53-bin week-of-year captures seasonal transitions (spring thaw, autumn heating start) more precisely than 12 monthly bins
- Fill unobserved weeks with nearest-neighbor interpolation

### 1.2 Ornstein-Uhlenbeck (OU) Residual Parameters

Fitted on Finnish 2023–2026 data:

| Parameter | Value | Unit |
|---|---|---|
| Half-life | **10.3** | hours |
| λ (mean-reversion rate) | 0.0673 | per hour |
| μ (long-run mean) | ~0 | EUR/MWh (by construction) |
| σ (volatility) | ~varies | EUR/MWh |

**Autocorrelation decay:**

| Lag | ρ(lag) | Practical signal |
|---|---|---|
| 1h | 0.935 | Noisy (microstructure) |
| 2h | 0.874 | Best exploitable lag |
| 10h | 0.500 | Half-life |
| 24h | 0.202 | Weak |
| 48h | 0.041 | ≈ 0 — no signal |

### 1.3 Variance Decomposition

Output from `analyze_sahkotin_seasonal`:
- Seasonal component explains a significant share of total price variance
- Residual Y_t captures the remainder

---

## 2. CVaR Hedge Analysis Results

### 2.1 Method

NPK-CVaR (nonparametric kernel) hedge analysis:
- Objective: minimize CVaR via Rockafellar's reformulation
- Kernel smoothing for optimization surface (Gaussian kernel, Silverman bandwidth)
- Historical CVaR for reporting (assumption-free)
- Train/test split: 55% / 45%
- α = 0.05 (5% tail → 95% confidence)

### 2.2 Results Summary

| Mode | Lag | h (hedge ratio) | Test CVaR unhedged | Test CVaR hedged | Reduction |
|---|---|---|---|---|---|
| Deseasonalized Y_t | 1h | -0.021 | 409.3 | 410.4 | **-0.3%** (noise) |
| Deseasonalized Y_t | 2h | -0.115 | 409.3 | 403.7 | **1.4%** |
| Deseasonalized Y_t | 48h | 0.009 | 409.5 | 409.3 | **0.06%** (≈0) |
| **Raw prices, seasonal futures** | **48h** | **0.992** | **447.2** | **411.7** | **7.9%** |

### 2.3 Interpretation

1. **h ≈ 1.0 for raw prices** — the optimizer independently validated the seasonal model (subtract full seasonal forecast = optimal hedge)
2. **7.9% CVaR reduction at 48h** — the seasonal component is the only hedgeable risk beyond ~24h
3. **10.3h OU half-life** — residual is only exploitable intra-day (≤ ~2h lag)
4. **SARIMA adds no value beyond AR(1)** — fitting higher-order ARMA to the residual is modelling noise at horizons > 20h
5. **Kernel vs Historical CVaR** — < 1% difference with 16,000+ training samples; kernel useful for smooth optimization, historical for reporting

---

## 3. Conclusions for Model Design

### What works
- Simple seasonal averages (hour + day-of-week + week-of-year) capture all hedgeable risk at day-ahead / 48h horizons
- AR(1) / OU on the residual captures intra-day mean-reversion (useful for 0–10h forecasting)

### What doesn't help
- SARIMA(p,d,q)(P,D,Q)_24 without exogenous inputs — adds parameters but no out-of-sample improvement over AR(1)
- Any autoregressive model on Y_t beyond ~24h — OU autocorrelation has decayed to zero

### Where to invest modeling effort
- **Exogenous features** (not autoregressive complexity):
  - Wind forecast (dominant for DK, growing for FI/SE)
  - Temperature forecast (heating demand)
  - Holidays / special days
  - Hydro reservoir levels (critical for Sweden SE1–SE3)
  - Fenno-Skan congestion state / cross-border flows

### Validation approach
- After adding each feature, re-run the CVaR hedge analysis
- Compare test CVaR hedged: if it drops, the feature captures real signal
- If test CVaR unchanged, the feature is noise — discard

---

## 4. Nordic Market Notes (Sweden Extension)

### SE1–SE3 collinearity
- SE1 ≈ SE3 prices ~85–90% of hours (internal transmission rarely congested)
- **Use SE3 as single Swedish reference** — simpler, avoids multicollinearity
- Model the SE1–SE3 spread as a **congestion indicator** (captures tail events when zones decouple)

### Finnish interconnection
- Fenno-Skan 1+2: ~1,400 MW — frequently congested
- When congested, FI prices decouple from SE1/SE3
- Model: SE3 as reference + congestion dummy + flow direction

### Swedish model (recommended)
```
P_t = P_hour(h) + P_day(d) + P_week(w) + Y_t
Y_t = φ · Y_{t-1} + β₁ · reservoir_SE + β₂ · holiday_t + ε_t
```

Reservoir level is the dominant Swedish-specific variable (hydro 40–50% of supply).
Expected CVaR improvement from reservoir: +5–15% beyond seasonal baseline.

### Zone-specific priorities

| Zone | Key exogenous variable |
|---|---|
| SE1–SE2 | Hydro reservoir level (dominant) |
| SE3 | Reservoir + demand (Stockholm) |
| SE4 | Wind (Denmark coupling) + continental prices |
| FI | Wind + Fenno-Skan flow + SE3 reference |

---

## 5. Python Implementation Spec (N100 target)

### Architecture

```
nordpool_forecast/
├── data/
│   ├── loader_sahkotin.py       # Sähkötin API client (FI prices)
│   ├── loader_nordpool.py       # Nord Pool CSV/API (SE prices)
│   └── loader_entsoe.py         # ENTSO-E transparency (flows, NTC)
├── model/
│   ├── seasonal.py              # P_hour + P_day + P_week (numpy only)
│   ├── ou_residual.py           # AR(1) / OU fit on Y_t
│   ├── exogenous.py             # Wind, temperature, reservoir, holidays
│   └── forecast.py              # Seasonal + OU + exogenous → price scenarios
├── risk/
│   ├── cvar.py                  # Historical CVaR (np.percentile based)
│   ├── npk_cvar_hedge.py        # Kernel CVaR hedge analysis (scipy.optimize)
│   └── storage_optimizer.py     # Moazeni et al. storage scheduling
├── config.py                    # Zone, horizon, alpha, paths
└── main.py                      # CLI entry point
```

### N100 Performance Considerations

| Concern | Recommendation |
|---|---|
| CPU only, 4 cores, 6W TDP | No GPU-dependent models; numpy/scipy only |
| RAM ~8–16 GB | 29k hourly obs × 10 features ≈ 2 MB — no issue |
| Kernel CVaR optimization | `scipy.optimize.minimize(method='Nelder-Mead')` — 1–5 sec |
| Monte Carlo scenarios | 2,000 × 168h = 336k points — < 1 sec with numpy vectorization |
| Seasonal fit | `np.bincount` / `pd.groupby` — instant |
| OU fit | OLS via `np.linalg.lstsq` — instant |

### Key Dependencies (minimal)

```
numpy          # Core numerics
scipy          # optimize.minimize, erfc for kernel CVaR
pandas         # Data loading and time alignment
requests       # API calls (Sähkötin, ENTSO-E)
matplotlib     # Optional: diagnostic plots
```

No need for: statsmodels (SARIMA), sklearn, tensorflow, pytorch.

### Porting Notes from MATLAB

| MATLAB | Python equivalent |
|---|---|
| `accumarray(idx, P, [N 1], @mean)` | `pd.Series(P).groupby(idx).mean().reindex(range(1,N+1))` |
| `week(t, 'weekofyear')` | `t.isocalendar().week` (pandas) |
| `weekday(t, 'monday')` | `t.weekday()` (0=Mon in Python, shift to 1-based) |
| `fminsearch` | `scipy.optimize.minimize(method='Nelder-Mead')` |
| `normcdf(z)` / `normpdf(z)` | `0.5 * erfc(-z / sqrt(2))` / `exp(-0.5*z²) / sqrt(2π)` |
| `quantile(L, 1-α)` | `np.percentile(L, 100*(1-α))` |

### Numerical Notes

- Use `erfc` instead of `scipy.stats.norm.cdf` — avoids scipy.stats import overhead and is faster
- Kernel bandwidth: `b = 1.06 * std(L) * T**(-0.2)` (Silverman rule of thumb)
- Clip hedge ratio: `h = np.clip(h, -5, 5)` for numerical stability
- OU lambda from AR(1): `λ = -log(b_clamped) / dt` where `b = OLS slope`, `dt` in hours

---

## 6. Validated Numerical Benchmarks

Use these to verify Python implementation matches MATLAB:

### Finnish data (2023–2026, Sähkötin)

| Quantity | Expected value |
|---|---|
| OU half-life | ~10.3 hours |
| Seasonal CVaR hedge (raw, 48h), h | ~0.99 |
| Seasonal CVaR hedge (raw, 48h), test reduction | ~7.9% |
| Deseasonalized CVaR hedge (2h), h | ~-0.12 |
| Deseasonalized CVaR hedge (2h), test reduction | ~1.4% |
| Deseasonalized CVaR hedge (48h), h | ~0.01 (≈ 0) |

### Sanity checks after porting
1. Seasonal averages: `P_hour` should peak at hours 8–10 and 17–19 (Finnish morning/evening)
2. `P_day`: Saturday/Sunday should be lowest
3. `P_week`: winter weeks (1–8, 45–52) highest; summer (25–35) lowest
4. OU half-life should be 8–15 hours
5. Re-run hedge analysis → h ≈ 1.0 for raw mode confirms correct seasonal model
