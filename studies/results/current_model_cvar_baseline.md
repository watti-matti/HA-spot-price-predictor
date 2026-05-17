# Current-model CVaR baseline (v2.4.1)

**Date:** May 2026
**Tool:** `studies/npk_cvar_hedge.py` (Python port of `npk_cvar_hedge_demo.m`, validated against MATLAB benchmarks — see `npk_cvar_python_port_validation.md`)
**Data window:** 2023-01-01 → 2026-04-28 (matches both the MATLAB study and the locked v2.5.0 calibration window)
**Hedge configuration:** raw mode, 48 h futures lag, α = 0.05 (95 % confidence), 55 %/45 % chronological train/test split

These numbers define the "current model" baseline. Every v2.4.2+ variant must improve the per-zone `Reduction` to be accepted into the next patch release.

## Per-zone baseline

| Zone | n samples | h_hat (optimal hedge ratio) | CVaR test unhedged | CVaR test hedged | Reduction | OU half-life | Seasonal var share |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FI** (Finland)           | 29 112 | 0.922 | 44.69 | 41.99 |   **+6.06 %** | 10.2 h | 22.1 % |
| **SE3** (southern Sweden)  | 44 254 | 0.978 | 21.78 | 23.47 |   **−7.78 %** | 12.8 h | 38.9 % |
| **SE1** (northern Sweden)  | 44 254 | 0.413 | 17.13 | 17.18 |   **−0.30 %** | 26.1 h | 37.5 % |
| **EE** (Estonia)           | 29 134 | 1.209 | 86.06 | 82.56 |   **+4.07 %** |  4.7 h | 26.7 % |

## Interpretation

### FI baseline (the canonical benchmark)
- Matches MATLAB study (their result: ~7.9 % reduction at h ≈ 0.99). Our extended window through April 2026 gives slightly more conservative numbers (6.06 % at h = 0.92) but is well within ±5 % tolerance.
- **v2.4.4 (FI model revision) must beat +6.06 % test CVaR reduction.**

### SE3 baseline
- **Striking negative result:** the seasonal-only hedge on raw SE3 prices actually *hurts* the out-of-sample CVaR (−7.78 %).
- 38.9 % seasonal variance share is the highest in the four zones, so the seasonal pattern IS strong in-sample — but it appears to be unstable across the train/test split. SE3 prices have undergone regime shifts (hydrological cycle, transmission upgrades, demand growth) that the joint train-period averages don't generalise to the test period.
- **This is precisely the failure case the v2.4.2 model is designed to fix**: by adding hydro reservoir level + workday/holiday + AR(1) residual model, we should beat both the seasonal-only baseline AND get into positive territory.
- **v2.4.2 (SE3 model revision) target: convert −7.78 % into a meaningful positive reduction.** Even +0 % would be a success vs the seasonal-alone baseline.

### SE1 baseline
- Essentially no hedge signal at all (−0.30 % is statistical noise) and `h_hat = 0.41` ≪ 1.0 suggests the seasonal forecast over-states the actual residual pattern.
- OU half-life of 26 h reflects very persistent SE1 residuals — likely driven by Norwegian hydro / Swedish nuclear schedule effects, not by the hour/day/week seasonality.
- SE1 is currently retained in our cross-border pipeline mainly for context; the bundled v2.2 9-feature model already dropped `ar_se1` as collinear with `ar_se3`. We do NOT propose a separate v2.4.x for SE1 — its de-seasonalized model will be tested as a side experiment in v2.4.2 but only adopted if it improves materially.

### EE baseline
- Positive 4.07 % reduction is meaningful but small.
- `h_hat = 1.21 > 1.0` indicates the seasonal forecast under-predicts test-period swings (Estonia has had unusual price patterns post-2022 with oil-shale capacity changes).
- OU half-life 4.7 h — very fast mean-reversion suggests dominant high-frequency noise, less exploitable.
- **v2.4.3 (EE model) target: beat +4.07 % test CVaR reduction.** EE doesn't get hydro feature, so the gain must come from de-seasonalization + workday/holiday + AR(1) alone. Lower expected improvement than SE3.

## Pass/fail gate for each v2.4.x patch

```
v2.4.2  (SE3 model w/ hydro + holidays + AR(1)):  CVaR reduction must beat -7.78 %  (very low bar)
v2.4.3  (EE  model w/ holidays + AR(1)):          CVaR reduction must beat +4.07 %
v2.4.4  (FI  model revision):                     CVaR reduction must beat +6.06 %
v2.4.5  (solar model alt):                        CVaR reduction on FI hedge with new solar feature must beat +6.06 %
```

If any patch fails to improve its zone's baseline, it is rejected and the alternative pursued (per the validation methodology documented in plan section 3.15).

## How to reproduce

```bash
cd HA-spot-price-predictor
python -c "
import sys; sys.path.insert(0, 'studies')
import pandas as pd
from npk_cvar_hedge import run_baseline_hedge_analysis
df = pd.read_parquet('output/fi_prices.parquet')
df = df[df.index >= '2023-01-01']
ts = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=3)
r = run_baseline_hedge_analysis(ts, df['price_eur_mwh'].values,
                                 mode='raw', futures_lag_hours=48)
print(r['hedge'])
"
```

To benchmark a new model variant, replace the raw-prices array with the variant's predicted prices and re-run. Compare `cvar_test_hist_hedged` to the FI baseline of 41.99.
