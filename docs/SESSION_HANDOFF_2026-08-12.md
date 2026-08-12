# Session handoff — 2026-08-12

Forecast-accuracy investigation triggered by a field report: summer
**weekday** prices forecast far too high, weekends roughly right.

Read this with `docs/BACKLOG.md`, which carries the measurements. This
file carries the *state* and the *traps*.

---

## 1. Where things stand

| | |
|---|---|
| `main` | `d1c3ebd` + docs commits; v2.18.0 released and live on HACS |
| open PR | **#27** — asinh VST recommendation (measurement only, no code) |
| tests | 672 passing |
| data store | refreshed to **2026-08-12**, snapshot `85fe2b59dbb7f630` |
| harness | production-faithful, pinned by `tests/test_harness_production_parity.py` |

### The symptom is live, not historical

Production-equivalent replay on complete data, weekday **peak** hours:

```
2026-07  actual 20.03  model 48.52   +28.50  (+142%)
2026-08  actual 17.80  model 32.19   +14.39   (+81%)
```

v2.18.0 treats this symptomatically via the bias corrector. It was
cold-started 2026-08-10, so give it until ~2026-08-24 before judging
live behaviour.

---

## 2. What shipped — v2.18.0

**Bias-corrector retune.** Half-life 14 → 3 days, warm-up gate → a
2-observation guard, and `adaptive_init` (CMA → EMA hand-over,
`α_n = max(1/n, λ)`) replacing the zero start. The old setting disabled
correction for exactly one half-life then switched on at 50 % strength —
a zero-initialised EMA reaches `1−(1−λ)ⁿ`, which at n = halflife is 50 %
by construction.

```
config                    MAE  weekday  wd peak  |mth bias|  2026-07 wd peak
no correction           25.76    28.22    35.55        9.54          +29.36
v2.17.3                 24.91    26.83    33.91        7.21          +27.17
v2.18.0                 24.13    25.76    32.36        3.30          +13.66
```

Post-install (3 weeks after a state wipe): bias −5.20 → −0.39.

**Three train/inference mismatches**: UTC-vs-local workday flag (2.93 %
of hours), 15-minute neighbour prices keeping the `:45` quarter, Kolari
weather site 62 km from the trained coordinates.

---

## 3. Measured and REJECTED — do not re-run without new evidence

| hypothesis | result | producer |
|---|---|---|
| Fingrid day-ahead channel | +0.9 %, costs +2.8 EUR/MWh bias | `exp_fingrid_dayahead_channel.py` |
| Nuclear via UMM planned availability | usable signal explains **0.1 %** | `exp_umm_nuclear_leverage.py` |
| Consumption in the price mean | every specification −0.4 … −1.3 % | scratch, recorded in BACKLOG |
| A focused consumption model | spread sensitivity saturates at 3.6 % MAPE | " |
| `P_week` smoothing | honest LOYO 17.10 → 16.44, 4 % | " |
| Lagged-price level anchor | model tracks weekly level at corr 0.848 vs persistence 0.478 | " |
| Norwegian hydro reservoir | corr with FI weekly level −0.015 | " |
| Log-space target | +11.5 % worse | `exp_asinh_vst.py` (`LOG20` arm) |

**Two of my own claims were wrong and are corrected in BACKLOG and the
release notes**: the "+8.5 % Fingrid channel" (measured without the
publication boundary) and "the nuclear result is unsafe" (tested,
refuted).

---

## 4. Measured and RECOMMENDED — asinh VST (PR #27, not yet built)

Against v2.18.0 as it ships, non-overlapping hourly replay:

```
config                         MAE     bias  |mth bias|  2026-07 wd
BASE + corrector             24.03    +0.01        4.10       +9.53
ASINH + corrector            22.63    -0.14        3.67       +6.47
```

**MAE −5.8 %, monthly bias −10 %, July weekday over-prediction −32 %.**
The only change measured that improves MAE *and* bias together.

Non-obvious points, all load-bearing:

* The VST predicts a conditional **median** (that is what makes it
  MAE-optimal), so raw bias worsens −2.72 → −7.59. The corrector absorbs
  it (−0.14) **only because v2.18.0 cut the half-life to 3 days**. Order
  of operations matters; testing the VST first would have rejected it.
* `ASINH_L2` (linear L1, transformed L2 only) is **worse than base**
  (−2.8 %). There is no runtime-only version — shipping means retraining
  the seasonal artifact.
* `log(p+20)` beat asinh marginally on raw MAE (24.79 vs 24.93). asinh
  is chosen on **robustness**: the offset is arbitrary and log breaks
  below −20 EUR/MWh.
* It does **not** replace the conditional-spread model. Amplitude law
  `amp = a + b·level`: actual b = 1.016, base 0.107, asinh 0.314.

**Before shipping:** the retrain changes the model fingerprint and
cold-starts calibrators, and the wind/PV sign constraints and 168 h
neighbour-lag guard must be **re-verified** — both assert on coefficient
values whose scale changes under the transform, so those tests could
pass or fail for the wrong reason.

---

## 5. NEXT: benchmark against published EPF (the open thread)

We are **~2× behind published Nord Pool day-ahead benchmarks** in the
literature's own metric.

| source | market | horizon | rMAE | sMAPE |
|---|---|---|--:|--:|
| Lago, Marcjasz, De Schutter & Weron (2021), *Applied Energy* 293:116983 — DNN ensemble | Nord Pool | day-ahead | **0.403** | 4.85 % |
| same — LEAR ensemble | Nord Pool | day-ahead | **0.420** | 5.01 % |
| Norwegian zones 2025 (arXiv:2604.26634), LightGBM | NO1–NO5 | day-ahead | ≈**0.28–0.34** | — |

Ours, 2025-07 → 2026-08, mean 52.4, sd 59.4:

```
forecast                         MAE   sMAPE %    rMAE
similar-day naive              32.51      84.1    1.000
our model (170 h, production)  24.00      79.0    0.738
```

`rMAE` = MAE(model) / MAE(naive), naive = same hour previous day
(Tue–Fri) or previous week (Sat–Mon). epftoolbox convention.

**Tested and insufficient:** published models are autoregressive on
recent prices while ours zeroes `Y_fi_lag168`. Adding lag24/48/168 on a
D+1-style task gives rMAE 0.814 → **0.717** (MAE 26.52 → 23.35, −12 %).
Real, but nowhere near 0.40. **The gap is not primarily the price lags.**

Remaining explanations, untested:

1. **Regime.** Our sMAPE 77–84 % vs the benchmark's ~5 %. Our window has
   sd > mean (Feb 155, Jul 17 EUR/MWh). The *naive* also scores 84 %
   sMAPE, so the market itself was less predictable — not only our model.
2. **Capacity.** LEAR is ~250 LASSO-selected features; the DNN results
   are ensembles. Ours is a **9-feature ridge**, deliberately, so it runs
   on a Raspberry Pi.
3. **Zone.** FI is import-dependent with no hydro buffer. NO3/NO4, where
   the Norwegian study gets MAE 1.6–3.1, are hydro-rich and calm.

### The concrete next step

**Install `epftoolbox` and run LEAR on FI over our exact window.**
Open-source (AGPL-3.0), `pip install epftoolbox`, repo
`github.com/jeslago/epftoolbox`. Same period, same zone, same metric
turns "roughly 2× behind" into a measured gap against a named benchmark,
and tells us whether the headroom is regime or model.

Feed it from `data_store/fi_prices.parquet` plus the exogenous series;
its native format is a dataframe with `Price` and two exogenous columns.

---

## 6. Traps — things that cost time this session

* **`data_store/*.parquet` is gitignored.** It existed only in the
  worktree, not the main checkout. Seed a fresh clone by copying, then
  `python -m src.data_store update` (needs `PYTHONPATH=<repo>` — plain
  `python src/data_store.py` fails on `No module named 'src'`).
* **The harness had two independent defects** and had not run since
  v2.17.0. Both fixed; `tests/test_harness_production_parity.py` now
  pins it to `Pipeline.compute_forecast` at <1e-6 EUR/MWh. If that test
  fails, trust nothing measured offline.
* **`ridge_coef` has the intercept FIRST**, `ridge_features` omits it.
  Zip them naively and the wind coefficient reads as solar.
* **Nord Pool UMM API returns HTTP 403 to a bare urllib agent** — needs a
  browser User-Agent. `fuelTypes` filters server-side; `areas` does not.
* **`scripts/retrain_model.bat` is a tracked file the user keeps locally
  modified with their Fingrid API key.** Never commit it. The key is in
  `.env` (gitignored) — read it from there, never hardcode.
* **Do not retrain casually.** Any refit jitter (verified: 1e-12 in a
  single `P_hour` bin) changes the model fingerprint, which wipes every
  calibrator and reopens the post-install window.

---

## 7. Reproducible producers

```
studies/backtest_harness.py                 PRODUCTION + FRESH configs, monthly weekday bias
studies/bias_corrector_warmup_study.py      what the v2.18.0 retune is worth
studies/exp_asinh_vst.py                    WP2a, the recommended change
studies/exp_fingrid_dayahead_channel.py     WP2.5, rejected
studies/exp_umm_nuclear_leverage.py         nuclear via UMM, rejected
studies/honest_horizon_study.py             leak-free origin-based evaluation
```

All read the canonical `data_store/`. `exp_extra_features.py` used to
read a stale `output/` snapshot — fixed in #24, but check
`OUTPUT_DIR` if numbers look stale.
