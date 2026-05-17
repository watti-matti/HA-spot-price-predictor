# v2.5.1 — Seasonal visualizations + SE1/SE3 congestion finding (ACCEPT for SE1+SE3 joint hedge)

## TL;DR

**No coordinator behaviour change.** v2.5.1 ships two research artifacts:

1. **MATLAB-style 9-panel seasonal diagnostics** + heatmaps for all four Nordic zones (FI, SE3, SE1, EE) — the visual evidence requested by the user matching the original MATLAB study's diagnostic layout.
2. **SE1+SE3 congestion-aware hedge analysis** revealing that **adding SE1 alongside SE3 improves the FI hedge by +0.55 pp** (4.55 % → 5.10 % CVaR test reduction). The user's transmission-capacity hypothesis is empirically confirmed: the joint hedge essentially uses the **SE3−SE1 spread** as a key signal (`h_SE3 = +1.61, h_SE1 = −1.60`), capturing congestion-state information that SE3 alone misses.

Two assets that re-open one part of the v2.4.4 REJECT decision: the v2.2 9-feature Ridge's choice to drop `ar_se1` as redundant is **wrong under the NPK-CVaR methodology** when evaluated on 2023+ data. A future model-rebuild patch should retain both.

## What v2.5.1 ships

### 1. Seasonal diagnostic visualizations (`studies/seasonal_visualization.py`)

Generates the 9-panel layout from `studies/Matlab_study_on_CVAR/analyze_sahkotin_seasonal.m` for each of FI, SE3, SE1, EE:

- Spot price + seasonal overlay
- Deseasonalized residual Y_t time series
- `P_hour(h)` diurnal pattern bar chart
- `P_day(d)` weekly pattern bar chart
- `P_week(w)` annual pattern bar chart
- Residual histogram + Normal fit overlay
- Residual ACF stem plot (up to 72 hours)
- Residual Q-Q plot vs Normal
- Variance decomposition pie chart (seasonal vs residual share)

Plus two heatmaps per zone:
- Hour × Day-of-week average price
- Hour × Week-of-year average price

Outputs in `studies/results/figures/`:

| Zone | Diagnostic | Heatmap | Seasonal var share | OU half-life |
|---|---|---|---:|---:|
| FI  | `seasonal_diag_fi.png`  | `seasonal_heatmap_fi.png`  | 22.1 % | 10.2 h |
| SE3 | `seasonal_diag_se3.png` | `seasonal_heatmap_se3.png` | 38.9 % | 12.8 h |
| SE1 | `seasonal_diag_se1.png` | `seasonal_heatmap_se1.png` | 37.5 % | 26.1 h |
| EE  | `seasonal_diag_ee.png`  | `seasonal_heatmap_ee.png`  | 26.7 % |  4.7 h |

The diagnostic plots make the key empirical facts visually unmistakable: fat-tailed residual distributions (clear S-curves in Q-Q plots), slow ACF decay characteristic of mean-reversion, dominant winter-week seasonality, weekend lows.

### 2. SE1+SE3 congestion analysis (`studies/se1_se3_congestion_analysis.py`)

User's question: *"hedge analysis should be applied in understanding if both SE1 and SE3 could be included or it sufficient to include one. the transmit capacity is significant factor for Finland prices when full transfer capacity is reach and price is more directly coupled to Finland when need in Finland does not exceeded transmit capacity"*

#### Spread distribution finding

`SE3 − SE1` spread (positive = SE3 expensive, congestion gradient toward FI):

| Statistic | Value |
|---|---:|
| Mean | **+17.73 EUR/MWh** |
| Median | +0.01 (zone is uncongested ~50 % of the time) |
| Std | 31.95 |
| Hours \|spread\| > 5 EUR/MWh | 12,392 (**42.6 %**) |
| Hours \|spread\| > 20 EUR/MWh | 8,935 (**30.7 %**) |
| Hours \|spread\| > 50 EUR/MWh | 3,402 (**11.7 %**) |

Congestion is **frequent and significant** — almost a third of hours have substantial SE3-SE1 decoupling.

#### FI correlation regime-split

Splitting by spread magnitude (since the spread distribution is bi-modal at 0 + long positive tail, magnitude bins are more informative than quartiles):

| Regime (spread magnitude) | n | spread mean | corr(FI, SE3) | corr(FI, SE1) | Δ corr |
|---|---:|---:|---:|---:|---:|
| uncongested (≤ 0.5 EUR/MWh) | 15,334 | −0.28 | 0.727 | 0.732 | −0.005 |
| mild (0.5 – 5) | 1,613 | +2.37 | 0.758 | 0.757 | +0.001 |
| moderate (5 – 30) | 5,607 | +17.70 | 0.684 | 0.688 | −0.004 |
| **severe (> 30)** | **6,558** | **+63.65** | **0.595** | **0.529** | **+0.066** |

In the **severe-congestion regime** (22.5 % of all hours), FI's correlation with SE3 (0.595) **diverges materially** from its correlation with SE1 (0.529) by +0.066. SE1 carries genuinely different signal there — consistent with the user's hypothesis that when Fenno-Skan / SE-internal transmission is saturated, FI decouples from one or both Swedish zones.

#### NPK-CVaR hedge comparison

48 h horizon, α = 0.05, 55 %/45 % chronological train/test split, 29,134 aligned hours (2023+):

| Hedge instrument(s) | Coefficients | CVaR test hedged | Reduction |
|---|---|---:|---:|
| SE3 seasonal forecast alone | h_SE3 = +0.897 | 41.65 | **+4.55 %** |
| SE1 seasonal forecast alone | h_SE1 = +1.335 | 42.61 | +2.36 % |
| **SE3 + SE1 jointly** | **h_SE3 = +1.609, h_SE1 = −1.600** | **41.42** | **+5.10 %** |

**Δ improvement: +0.55 pp → ACCEPT**

The most striking detail: the joint hedge picks `h_SE3 ≈ +1.61` and `h_SE1 ≈ −1.60` — **nearly opposite signs**. This is the optimizer telling us that the **spread** `(SE3 − SE1)` carries information that neither price level alone does. The user's transmission-capacity intuition is empirically borne out: the dual-feature hedge is essentially a level + spread decomposition.

### 3. Implication for the production model

The v2.2 9-feature Ridge dropped `ar_se1` as collinear with `ar_se3` under a leave-one-out redundancy sweep run on the FULL 2022+ training window. That sweep was correct in average terms but **missed the congestion-regime contribution**: SE1 is collinear with SE3 most of the time, but during the ~22 % of hours with severe spread, it carries independent signal worth +0.55 pp of test CVaR.

This is a candidate for re-opening at a future model-rebuild patch (out of scope for v2.5.1, which is documentation-only). The validated change would be:

```
Instead of: ar_se3            (current v2.2)
Use:        ar_se3, spread_se3_se1   (level + spread)
```

The spread feature could be `Y_SE3 − Y_SE1` (deseasonalized spread) so that it's orthogonal to the per-zone seasonal forecasts. Validation: run the FI Ridge retrain with the spread feature added; gate on NPK-CVaR test improvement vs current v2.2 baseline.

### 4. Tests

6 new tests in `tests/test_se1_se3_congestion.py`:

- `hedge_dual_features` smoke test (returns expected shape, finite values, bounded coefficients)
- Collinear features handled gracefully (no crash with f1 ≡ f2)
- `regime_split_analysis` categorises all four bins, skips underpopulated ones
- `red_pct` arithmetic helper

Total: **315 / 315 passing** (6 new + 309 from v2.5.0).

## Files

- **New**: `studies/seasonal_visualization.py` (~250 LOC)
- **New**: `studies/se1_se3_congestion_analysis.py` (~360 LOC)
- **New**: `tests/test_se1_se3_congestion.py` (6 tests)
- **New**: `studies/results/figures/seasonal_diag_{fi,se3,se1,ee}.png` (4 files, ~200 KB each)
- **New**: `studies/results/figures/seasonal_heatmap_{fi,se3,se1,ee}.png` (4 files, ~50 KB each)
- **New**: `studies/results/figures/se1_se3_spread_analysis.png`
- **New**: `studies/results/se1_se3_congestion_results.md` (auto-generated)
- **New**: `studies/results/V2_5_1_RELEASE_NOTES.md` — this document
- **Modified**: `manifest.json` (`2.5.0 → 2.5.1`)

No coordinator changes. No sensor schema changes. HACS users see version `2.5.1` but observe no runtime difference.

## Reproducibility

```bash
cd HA-spot-price-predictor
python studies/seasonal_visualization.py
python studies/se1_se3_congestion_analysis.py
```

Both scripts write fresh outputs each run; figures regenerate; markdown summary auto-updates.
