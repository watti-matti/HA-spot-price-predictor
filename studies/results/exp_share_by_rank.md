# Empirical share_by_rank[24] on post-PV consumption

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_share_by_rank.py`](../exp_share_by_rank.py).

Validation of the rank-shift consumption model proposed in
`pv_adjusted_cvar_plan.md`. The model says household deferrable load
is placed disproportionately into the cheapest-effective-price hours
of each day. This script measures whether that's actually the case
on the user's data.

## Method

- **Total demand** reconstructed from Fingrid grid_import + cached
  irradiance × pv_estimate − Fingrid grid_export.
- **Baseline envelope** `baseline[7×24]` = q10 of demand per
  (weekday, hour) cell.
- **Deferrable per hour** `defer(h) = max(0, L(h) − baseline(h))`.
- **Effective price per hour** `(1−α)·buy + α·sell` with
  `α = min(1, PV/L)`. Buy/sell are consumer-tariff EUR/kWh.
- **Rank** 0..23 within each day, sorted by effective price.
- **Share** accumulator over 958 valid days in
  2023-09-06 to 2026-04-27: `share_by_rank[r] = Σ_day defer(h_at_rank_r) /
  Σ_day defer(all hours)`.

## Headline — overall share_by_rank

| Rank | share | bar (each ★ ≈ 0.1 %) |
|:---:|:---:|---|
|  0 |  3.84 % | ************************************ |
|  1 |  4.41 % | ****************************************** |
|  2 |  4.60 % | ******************************************** |
|  3 |  4.83 % | ********************************************** |
|  4 |  4.56 % | ******************************************* |
|  5 |  4.80 % | ********************************************** |
|  6 |  4.93 % | *********************************************** |
|  7 |  5.30 % | ************************************************** |
|  8 |  5.70 % | ****************************************************** |
|  9 |  5.26 % | ************************************************** |
| 10 |  5.77 % | ******************************************************* |
| 11 |  5.26 % | ************************************************** |
| 12 |  5.05 % | ************************************************ |
| 13 |  4.53 % | ******************************************* |
| 14 |  4.30 % | ***************************************** |
| 15 |  3.60 % | ********************************** |
| 16 |  3.52 % | ********************************* |
| 17 |  3.59 % | ********************************** |
| 18 |  3.13 % | ****************************** |
| 19 |  3.00 % | **************************** |
| 20 |  2.63 % | ************************* |
| 21 |  2.58 % | ************************ |
| 22 |  2.41 % | *********************** |
| 23 |  2.40 % | *********************** |

**Concentration metrics:**
- Top 4 cheapest ranks (0–3) receive **17.7 %** of deferrable mass.
- Top 8 cheapest ranks (0–7) receive **37.3 %**.
- Bottom 4 most expensive ranks (20–23) receive **10.0 %**.
- Cheap/expensive concentration ratio (top4 / bottom4): **1.8×**.

A uniform distribution would put 16.7 % in any 4-rank bucket; the
observed top4 is 1.1× uniform and the bottom4
is 0.60× uniform.

**Verdict: ⚠ rank-shift effect present but moderate**

## By season

| Season | days | top4 | top8 | bot8 | bot4 | top4/bot4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| winter  |  271 |  19.7 % |  43.5 % |  21.1 % |   9.6 % |   2.0× |
| spring  |  238 |  15.6 % |  31.9 % |  22.9 % |   8.5 % |   1.8× |
| summer  |  184 |  12.7 % |  20.2 % |  37.5 % |  17.0 % |   0.7× |
| autumn  |  265 |  18.2 % |  38.4 % |  22.3 % |   9.8 % |   1.9× |

## Interpretation

The model `L(h, d) = baseline(h) + L_deferrable(d) · share(rank_h(d))`
treats `share_by_rank` as the empirical signature of the household's
optimisation policy. The numbers above either validate it
(⚠ rank-shift effect present but moderate) or argue for a different parameterisation.

If validated, the EMA module's published profile carries
`share_by_rank[24]` alongside `baseline[7×24]` and the predictor
reconstructs per-path consumption at forecast time as:

```
for each path:
    eff[h] = (1−α(h))·buy[h] + α(h)·sell[h]
    rank_h = argsort(argsort(eff))
    L_path[h] = baseline[wd, h] + deferrable_daily × share[rank_h]
```

This correctly preserves the joint distribution of (L, PV, price)
at forecast time and resolves the marginal-product bias documented
in the previous architectural turn.

## Caveats

- **Reconstruction error**: `pv_total` is estimated from cached
  irradiance × `pv_estimate`; small errors propagate into the
  `α = PV/L` coverage fraction. The effect on rank ordering is
  minor because effective-price ordering is dominated by spot.
- **q10 envelope sensitivity**: a different percentile changes the
  absolute shares but not the rank-relative concentration ratio.
- **Per-season counts**: some seasons may be under-sampled. Cross-
  check via the `n_days` column.
