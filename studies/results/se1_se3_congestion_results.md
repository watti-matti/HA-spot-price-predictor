# SE1 vs SE3 congestion-aware analysis for FI hedging

**Data window:** 2023-01-01 → 2026-04-28 (29,112 aligned hours)
**Methodology:** NPK-CVaR hedge at α = 0.05, 48 h horizon, 55/45 train/test split.

## SE3 − SE1 spread (Fenno-Skan / SE-internal congestion proxy)

The spread `SE3 − SE1` is positive when southern Sweden is more expensive than
northern Sweden — typically when SE-internal transmission bottlenecks decouple
the zones. Since Fenno-Skan terminates in SE3, a large positive spread also
indicates the FI ↔ SE corridor is operating in a stressed regime.

| Statistic | Value |
|---|---:|
| mean | +17.73 EUR/MWh |
| median | +0.01 |
| std | 31.95 |
| p01 / p99 | -4.01 / +137.81 |
| hours \|spread\| > 5 EUR/MWh | 12,392 (42.57 %) |
| hours \|spread\| > 20 EUR/MWh | 8,935 (30.69 %) |
| hours \|spread\| > 50 EUR/MWh | 3,402 (11.69 %) |

## FI–SE3 and FI–SE1 correlations by spread quartile

If `corr(FI, SE1) ≈ corr(FI, SE3)` in all four regimes, SE1 carries no
independent FI signal beyond SE3 (collinear). If they diverge — especially in
extreme regimes (Q1 or Q4) — SE1 captures congestion-state information that
SE3 alone misses.

| Regime | n | spread mean | corr(FI, SE3) | corr(FI, SE1) | Δ corr |
|---|---:|---:|---:|---:|---:|
| uncongested (|spread| ≤ 0.5) | 15,334 | -0.28 | 0.727 | 0.732 | -0.005 |
| mild (0.5 – 5) | 1,613 | +2.37 | 0.758 | 0.757 | +0.000 |
| moderate (5 – 30) | 5,607 | +17.70 | 0.684 | 0.688 | -0.004 |
| severe (> 30) | 6,558 | +63.65 | 0.595 | 0.529 | +0.067 |


## NPK-CVaR hedge: SE3 alone vs SE3 + SE1

| Hedge instrument(s) | h coefficients | CVaR test hedged | Reduction |
|---|---|---:|---:|
| SE3 seasonal forecast alone | h_SE3 = +0.897 | 42.66 | **+4.55 %** |
| SE1 seasonal forecast alone | h_SE1 = +1.335 | 43.64 | **+2.36 %** |
| **SE3 + SE1 jointly** | h_SE3 = +1.609, h_SE1 = -1.600 | 42.42 | **+5.10 %** |

**Δ improvement (dual vs SE3 alone): +0.55 pp**

**Verdict: ACCEPT — keep both SE1 and SE3 as features**

## Interpretation

The current v2.2 9-feature Ridge dropped `ar_se1` as collinear with `ar_se3`
under a leave-one-out redundancy sweep run on the full 2022+ training window.
This analysis confirms or refutes that decision under the v2.4.1 NPK-CVaR
methodology on the 2023+ window:

- Pearson correlation between FI and SE3 vs FI and SE1 was checked in four
  separate spread regimes. SE1 carries materially different signal than SE3 in extreme regimes — adding it as a separate feature pays off in the hedge.
- The joint dual-feature hedge improves the out-of-sample
  CVaR vs the SE3-only hedge by 0.55 pp.

## User's hypothesis on transmission capacity

The user noted: *"the transmit capacity is significant factor for Finland
prices when full transfer capacity is reached and price is more directly
coupled to Finland when need in Finland does not exceed transmit capacity"*.

This is captured by the SE3−SE1 spread regime split:
- **Low spread regimes** (Q1, Q2): SE-internal transmission is not stressed,
  and the FI ↔ SE3 corridor likely has headroom. FI couples to SE3.
- **High spread regimes** (Q4): SE3 is significantly more expensive than SE1,
  often indicating Fenno-Skan-direction congestion. FI may decouple from SE3
  and reflect FI-specific supply-demand (nuclear, FI wind, FI demand).

The regime-split correlation table above quantifies this for our data.

## Files

- `studies/results/figures/se1_se3_spread_analysis.png` — three-panel spread time series,
  scatter coloured by quartile, distribution histogram
- `studies/results/se1_se3_congestion_results.md` — this file (auto-written)

## Reproducibility

```bash
python studies/se1_se3_congestion_analysis.py
```
