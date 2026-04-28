# Spot Price Predictor — version-over-version comparison

**Comparing**
* **Old**: commit `472295d` ("Run 4: definitive KEEP AR(2)") — the baseline before any of the Phase A / Phase B work landed.
* **New**: commit `cb5d7b8` (head) — Phase A + Phase B v2 + UI integration committed.

**Date**: 2026-04-28
**Test methodology**: bundled-model metrics from `model_coefs_default.json` (4 years of training data, 2022-04 through 2026-04, 35 040 hourly observations); plus walk-forward validation on real Sähkötin / elprisetjustnu.se / Elering hourly data (700–914 days per zone, 2023-Q4 → 2026-Q2).

---

## TL;DR

| Layer | What changed | Measured impact |
|---|---|---|
| **Hourly forecast model** | Retrained with Fingrid nuclear features (15 → 17), bias correction, dual cheap/peak target | **MAE 24.70 → 23.94 EUR/MWh (−3.1 %)**, **R² 0.386 → 0.515 (+33 %)** |
| **Duration model** | Replaced single 24-element cumulative D(k) with dual cheap[12] + peak[12] | **2× the decision-relevant range** (legacy k=13..24 carried no signal); enables risk-aware LP |
| **Online calibration (optional)** | New per-D(i) DtACI layer; off by default | When on: **MAE −4 to −13 %** per zone, **+1.3–3.7 pp coverage** on under-covering zones, **5–15 % sharper intervals** at matched coverage |
| **Operations** | Single bundled model file plus optional `enable_dtaci_dk` toggle | Backward-compatible (legacy `dk_consumer_eur_kwh[24]` still emitted for one transition release) |

The hourly model is materially better; the duration model is architecturally better; and the optional DtACI layer adds calibrated prediction intervals the LP can plan against.

---

## 1. Hourly forecast model — direct, verifiable comparison

`model_coefs_default.json` ships pre-trained metrics. They're computed on the same train/test split (29 784 train + 5 256 test hours) under both versions, so the comparison is apples-to-apples.

| Metric | Old (`472295d`) | New (`cb5d7b8`) | Δ |
|---|---:|---:|---:|
| **MAE (EUR/MWh)** | 24.70 | **23.94** | **−0.76 (−3.1 %)** |
| **R²** | 0.386 | **0.515** | **+0.129 (+33.4 %)** |
| **Feature count** | 15 | **17** | +2 (nuclear deficit + nuclear×scarcity) |
| **Max predicted price** | 1758 EUR/MWh | 1455 EUR/MWh | tighter tail (−17 %) |
| **Data sources** | weather, cross-border | weather, cross-border, **nuclear** | +Fingrid |

The R² jump (0.386 → 0.515) is the headline. Adding `nuclear_deficit` (+0.410 coefficient on standardized scale) makes it the **single most important feature**, dwarfing the cross-border AR neighbour prices (+0.255 each). The Fingrid signal explains a layer of FI price variation that the wind/weather/cross-border features cannot capture — primarily winter price spikes when reactor outages coincide with low-wind cold spells.

**To reproduce**:
```sh
git show 472295d:custom_components/spot_price_predictor/data/model_coefs_default.json | jq '.metrics'
jq '.metrics' custom_components/spot_price_predictor/data/model_coefs_default.json
```

---

## 2. Duration model — architecture redesign

### Old: single cumulative D(k)[24]

```
dk_consumer_eur_kwh[k-1] = mean of cheapest k hours,  k = 1..24
```

* **k = 1..12** — informative range; D(4), D(8), D(12) are scheduling indicators.
* **k = 13..24** — values smoothly approach the daily average. **No decision-relevant signal**: D(20) and D(22) differ by less than 1 % from the daily mean on most days. Half the array is wasted storage and chart space.

The old Lovelace card consequently showed 24 lines, 12 of which were nearly indistinguishable from each other.

### New: dual cheap + peak

```
dk_cheap_eur_kwh[k-1] = mean of cheapest k hours,    k = 1..12  (non-decreasing)
dk_peak_eur_kwh [k-1] = mean of priciest k hours,    k = 1..12  (non-increasing)
```

Both arrays carry decision-relevant signal end-to-end:

| Use case | Reads… | Why it matters |
|---|---|---|
| Schedule a deferrable load for k hours | `dk_cheap[k-1]` | Direct cost estimate when running into the cheapest k slots |
| Storage depletion / worst-case planning | `dk_peak[k-1]` | What's the risk if we have to run during peak k hours |
| Daily mean (LP cost reference) | `(dk_cheap[11] + dk_peak[11]) / 2` | Free from the sum identity |
| Risk-aware LP cost vector | `(1−r)·dk_cheap[k-1] + r·dk_peak[k-1]` | Convex blend, `r` = risk aversion ∈ [0, 1] |

The legacy `dk_consumer_eur_kwh[24]` is **still emitted** for one transition release and can be exactly reconstructed from cheap+peak via the formula in `docs/dk_cheap_peak_migration.md`.

**Architectural property** that wasn't true before: training (Phase A.3) now fits **two** Ridge models per `(segment, k)` — one for cheap-end mean, one for peak-end mean — with PAVA isotonic correction in **opposite** directions per side. Inference (Phase A.3 `DurationModel.predict_day`) emits both arrays directly, and the coordinator prefers them over the sort-based reconstruction when present.

### Spearman ρ on D(4) (last 365 days)

| Version | ρ |
|---|---:|
| Old | 0.913 |
| New | 0.913 |

D(4) Spearman is unchanged (the cheap end was already the production target, and the addition of nuclear features doesn't materially help rank-correlation that's already at 0.91). The structural win is on the **peak end**, which the old model didn't expose at all.

---

## 3. Online calibration (DtACI) — optional, off by default

Enabling `enable_dtaci_dk` in the integration options activates a per-(direction, k) Dynamically-tuned Adaptive Conformal Inference layer (Gibbs & Candès, JMLR 2024). One DtACI instance per D(i) order statistic, 24 instances per zone, 4 zones.

### Walk-forward validation, 700–914 days per zone, day-ahead AR(2) forecaster

Headlines from `studies/results/DTACI_ANALYSIS_V2.md` (re-runnable via `python studies/validate_dtaci_dk.py`):

| Zone | Method | Coverage | Width (EUR/MWh) | MAE (EUR/MWh) | Δ MAE vs raw |
|---|---|---:|---:|---:|---:|
| FI | static (no DtACI) | 0.875 | 123.18 | 31.27 | — |
| FI | DtACI | **0.890** | **114.17** | **30.00** | **−4.1 %** |
| SE1 | static | 0.905 | 51.83 | 11.36 | — |
| SE1 | DtACI | 0.901 | **44.09** | **10.28** | **−9.5 %** |
| SE3 | static | 0.890 | 104.12 | 22.75 | — |
| SE3 | DtACI | **0.903** | **87.52** | **20.35** | **−10.5 %** |
| EE | static | 0.851 | 132.61 | 36.13 | — |
| EE | DtACI | **0.889** | 137.51 | **31.47** | **−12.9 %** |

Three things worth highlighting:

1. **Coverage convergence**: target is 0.90. The static empirical baseline under-covers in 3 of 4 zones (worst: EE at 0.851); DtACI brings every zone within ±0.05 of target. This is exactly the "calibration under arbitrary distribution shift" guarantee the algorithm was designed for.

2. **Sharpness**: SE3 and SE1 see 15–16 % narrower intervals at matched-or-better coverage. This translates directly into less padding in the LP's risk-aware budget envelope.

3. **MAE win is real on every zone**: −4 % on FI (smallest, since FI is already mostly debiased after the retrain), −13 % on EE (largest, since EE has heavy-tailed residuals the AR(2) day-ahead can't fully capture).

### Cross-zone coupling — the 4-zone deployment justification

A separate experiment (`studies/neighbor_bias_propagation.py`) measured the Pearson correlation of daily forecast residuals across all four zones:

```
cor(r_FI, r_SE1) = +0.59
cor(r_FI, r_SE3) = +0.57
cor(r_FI, r_EE)  = +0.75

3-feature OLS  r_FI ~ a·r_SE1 + b·r_SE3 + c·r_EE   →   R² = 0.667
```

Two-thirds of FI day-ahead residual variance is explained by neighbour residuals alone. Nord Pool market shocks (gas, hydro, transmission) propagate across zones simultaneously, and the AR(2) forecaster misses in correlated ways. The 4-zone DtACI deployment captures this signal — neighbour bias correction can later be propagated into the FI Ridge feature row to improve FI accuracy under market disturbances.

---

## 4. What this enables downstream (thermal optimization perspective)

Things that the old version could **not** do, that the new one can:

| Capability | Old | New |
|---|---|---|
| Get the mean cost of the cheapest 4 hours | ✅ `dk[3]` | ✅ `dk_cheap[3]` |
| Get the mean cost of the priciest 4 hours | ❌ (not exposed; would have to invert the legacy array) | ✅ `dk_peak[3]` |
| Risk-aware LP cost vector | ❌ | ✅ `(1−r)·dk_cheap[k-1] + r·dk_peak[k-1]` |
| Calibrated 90 % prediction interval per D(i) | ❌ | ✅ when `enable_dtaci_dk = true`: `dk_cheap_lower/upper_eur_kwh`, `dk_peak_lower/upper_eur_kwh` |
| Per-(direction, k) bias diagnostics | ❌ | ✅ `dtaci_diagnostics.zones.fi.per_k.cheap[k].bias_ema` etc. |
| Detect Nord Pool regime shift | ❌ (no online adaptation) | ✅ DtACI dominant γ jumps + weight entropy spikes |
| Disturbance-robustness against neighbour-zone shocks | ❌ | ✅ planned (neighbour-zone bundles allocated; AR-feature bias correction is the follow-up commit) |

The thermal optimizer's existing `compute_dk_prices_for_lp` already consumes `dk_cheap[k_load]`. The new architecture lets it cheaply add a `risk_aversion` parameter without retraining, and the DtACI layer means the same LP can be made aware of what the model itself is uncertain about (where to widen the budget envelope when the bias EMA is large or the alpha drift is high).

---

## 5. What did *not* change

* **Underlying neighbour AR(2) models** — confirmed the production choice in `studies/results/FINDINGS_v2.md` (4 SARIMAX-vs-AR(2) walk-forward runs). Still AR(2) for SE1 / SE3 / EE.
* **Workday / off-day treatment** — `is_off_day = weekend OR holiday`, mirrored into all training paths.
* **Sensor entity IDs** — `sensor.spot_price_predictor_price_forecast` and `sensor.spot_price_predictor_duration_forecast` keep the same names and primary state semantics. Sensor `state` for the duration sensor is now `dk_cheap_eur_kwh[3]` (cheapest 4h average) instead of the legacy equivalent — same numerical value for the cheap end.
* **Update interval** — still 6 hours.
* **HACS install path** — unchanged.

Existing automations and Lovelace cards keep working without changes for one transition release; the legacy `dk_consumer_eur_kwh` attribute is still emitted in parallel.

---

## 6. Test surface

| Suite | Old | New |
|---|---:|---:|
| Total tests | 189 | **234** |
| New since old | — | **+45** |

New tests added across the four commits:

* `test_dk_utils.py` (10) — cheap/peak utility correctness
* `test_dk_consumers.py` (15) — schema migration round-trip + sum identity
* `test_duration_model_dual.py` (10) — dual cheap/peak inference
* `test_dtaci.py` (16) — DtACI algorithm + persistence round-trip
* `test_dk_dtaci.py` (11) — DkDtACIBundle (per-(direction, k))
* `test_dtaci_ui_wiring.py` (7) — coordinator/sensor wiring contracts

Run: `python -m pytest -q` → 234 passed.

---

## 7. Commits in this version (cumulative diff `472295d` → `cb5d7b8`)

```
d07ec3e  Phase A: D(k) cheap/peak schema migration end-to-end
4fefe2c  Phase B: DtACI online calibration layer + comprehensive validation
2500a10  Phase B v2: per-D(i) DtACI on order statistics, 4-zone deployment
cb5d7b8  Phase B v2 UI: wire DtACI bundle into coordinator + sensor + Lovelace
```

Lines changed: **+5 110 / −615** across 33 files. Bundled `model_coefs_default.json` regenerated; old `model_coefs_user.json` if present is unaffected.

---

## 8. Relationship between the new (cheap, peak) schema and the legacy D[1..24]

The new schema does **not add a new degree of freedom** to the duration forecast — it re-views the same sorted-price information from both tails simultaneously. Given one day's 24 hourly prices `p[1] ≤ p[2] ≤ ... ≤ p[24]` (sorted ascending):

```
dk_cheap[k-1] = (1/k) · Σ_{i=1..k} p[i]            for k = 1..12  (mean of cheapest k)
dk_peak [k-1] = (1/k) · Σ_{i=24-k+1..24} p[i]      for k = 1..12  (mean of priciest k)
D(k)          = (1/k) · Σ_{i=1..k} p[i]            for k = 1..24  (legacy cumulative-cheap)
```

For k = 1..12 the cheap end and the legacy array are numerically identical: `dk_cheap[k-1] ≡ D(k)`. The peak end is a **different** statistic and is *not* a re-indexing of legacy D(13..24).

### Sum identity (cross-check)

The cheapest 12 hours and the priciest 12 hours are a disjoint partition of the 24, so:

```
12·dk_cheap[11] + 12·dk_peak[11] = 24·D(24)
=> (dk_cheap[11] + dk_peak[11]) / 2 = D(24)
```

This identity holds exactly to numerical noise on every day and gives any consumer a free way to verify the two arrays are jointly consistent without seeing the underlying hourly prices.

### Reconstructing legacy D[13..24] from (dk_cheap, dk_peak)

The conservation law `k·D(k) + (24−k)·dk_peak[24−k−1] = 24·D(24)` rearranges to:

```
                12·(dk_cheap[11] + dk_peak[11])  −  (24−k)·dk_peak[24−k−1]
  D(k)  =      ───────────────────────────────────────────────────────────       for k = 13..23
                                          k

  D(24) = (dk_cheap[11] + dk_peak[11]) / 2
```

#### Worked example (k = 16)

Suppose for one day:
- `dk_cheap[11]` = 30 EUR/MWh (mean of cheapest 12 hours)
- `dk_peak[11]`  = 70 EUR/MWh (mean of priciest 12 hours)
- `dk_peak[7]`   = 95 EUR/MWh (mean of priciest 8 hours, i.e. k = 24−16 = 8)

Then:
```
D(16) = (12·(30 + 70)  −  8·95) / 16
      = (1200 − 760) / 16
      = 27.5 EUR/MWh
```

This matches what the legacy `dk_consumer_eur_kwh[15]` would have shown — the cumulative-mean of the 16 cheapest hours that day. Note D(16) is *much closer* to D(24) = (30+70)/2 = 50 than to dk_cheap[11] = 30; this is exactly why legacy D(13..24) carries diminishing decision-relevant signal as k → 24.

### Implication: the two schemas carry equivalent information

Given the full pair (`dk_cheap[12]`, `dk_peak[12]`) you can recover legacy D(k) for all k = 1..24, and vice versa given the full legacy 24-array you can recover the peak end via `dk_peak[24−k−1] = (24·D(24) − k·D(k)) / (24−k)`. The conversion is exact arithmetically (numerically lossy near k = 24 because of the small-difference-of-large-numbers).

What changes between schemas is **which statistics are exposed as first-class API surface**, not the underlying information content. The new schema makes the cheap-end and peak-end cumulative means *both* directly accessible to downstream consumers; the legacy schema exposed only the cheap-end (as D(1..12)) plus a 12-entry tail (D(13..24)) that asymptotes to the daily mean and answers no scheduling question that the cheap end couldn't already answer.

### What this means for DtACI calibration

The `DkDtACIBundle` runs **24 DtACI instances per zone** — one for each `dk_cheap[k]` and `dk_peak[k]` for k = 1..12. Each instance independently tracks:

- a conformity-score window over residuals of *its own* statistic
- an adaptive miscoverage rate α_t with discounted-loss expert weights
- an `OnlineBiasCorrector` EMA over signed residuals of *its own* statistic

There is **no separate DtACI instance for legacy D(k) with k = 13..24**. The implications, in order of operational importance:

1. **Point forecasts for D(13..24) inherit bias correction by derivation.** When you reconstruct D(k) for k = 13..24 from the bias-corrected `dk_cheap` and `dk_peak` forecasts using the formula above, the result is bias-aware: each input to the linear combination has had its own EMA correction applied. This is exact only if the residual structure of D(k) is a clean linear combination of the residual structures of its inputs (`dk_cheap[11]`, `dk_peak[11]`, `dk_peak[24−k−1]`); in practice the approximation is close because the underlying hourly residuals drive all four statistics jointly.

2. **Prediction intervals for D(13..24) are *not* directly calibrated.** The DtACI marginal-coverage guarantee at level 0.9 holds for each of the 24 cheap/peak statistics individually — for those, P(true value ∈ [lower, upper]) → 0.9 over time, distribution-free. A band on D(15) reconstructed by propagating cheap-side and peak-side bands inherits *approximate* coverage but loses the formal guarantee, because the residual distribution of D(15) is its own linear combination of hourly noise terms with its own quantile structure.

3. **Daily mean (D(24)) is sharply calibrated** because both inputs to the identity `D(24) = (dk_cheap[11] + dk_peak[11]) / 2` are first-class DtACI statistics. The propagation here is exact and the propagated band is well-defined.

4. **Operational impact: none.** No production downstream consumer (the thermal LP, Lovelace cards, automations) is reading legacy D(13..23) as a scheduling input. Legacy D(13..24) values approach the daily mean and answer no question the calibrated cheap/peak end couldn't already answer better. If a future consumer ever needs a directly-calibrated band for some specific D(k) in the upper range, the bundle is trivially extensible — add a DtACI instance keyed on that statistic and feed it the daily (forecast, actual) pair, no other change required.

5. **Backward compatibility holds.** `dk_consumer_eur_kwh[24]` is still emitted by the coordinator for one transition release. Automations reading `dk_consumer_eur_kwh[3]` (legacy D(4)) continue to get the identical numerical value as `dk_cheap_eur_kwh[3]`. Automations reading `dk_consumer_eur_kwh[15..23]` continue to get the legacy cumulative-cheap means — they're now derived from cheap+peak via the conservation identity rather than from a parallel sort of forecast hourly prices, but the derivation is exact to numerical noise and verified by `tests/test_dk_consumers.py:test_split_to_legacy_24_array_round_trip_exact`.
