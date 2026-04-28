# DtACI Online Calibration Layer (v2)

> **v2 architecture:** DtACI is applied **per-(direction, k) order
> statistic** — 24 instances per zone (cheap[k] for k=1..12, peak[k] for
> k=1..12), not per hourly forecast. The thermal LP consumes D(k)
> directly, so calibrating the exact statistic the consumer uses gives
> tight, properly-calibrated bands. The reference card in
> `dtaci_info_cards.html` describes the diagnostic surface this design
> exposes.
>
> **Algorithm:** Gibbs & Candès JMLR 2024, with the discounted-loss
> weight update `L[m] ← ρ · L[m] + err_t`, `w[m] = softmax(−η · L[m])`.
> 15-expert γ ladder log-spaced over [0.0005, 0.2].
>
> **Production scope:** four zones (FI, SE1, SE3, EE). FI bundle drives
> consumer-price duration bands; SE1/SE3/EE bundles bias-correct the
> AR(2) features fed into the FI Ridge model. The neighbour-FI residual
> correlation (R² = 0.667 on a 3-feature OLS, see
> `studies/neighbor_bias_propagation.py`) justifies the 4-zone scope.

# Legacy v1 documentation (hourly DtACI) follows for reference

This document describes the Phase B online-calibration layer that wraps
the production point forecaster with calibrated prediction intervals
and online bias correction. The implementation is based on Gibbs &
Candès, *"Conformal Inference for Online Prediction with Arbitrary
Distribution Shifts"*, JMLR 25 (2024), paper 22-1218.

For an empirical analysis of the layer's value on real Nordic spot
prices see [`studies/results/DTACI_ANALYSIS.md`](../studies/results/DTACI_ANALYSIS.md).

---

## Components

```
┌──────────────────────────────────────────────────────────────────┐
│  AR(2) / Ridge / SARIMAX / any point forecaster                  │
└──────────────────────────────────────────────────────────────────┘
                            │ point forecast
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  OnlineBiasCorrector  (bias_corrector.py)                        │
│    EMA half-life 20 days, warm-up 168 steps, winsor 5x           │
└──────────────────────────────────────────────────────────────────┘
                            │ debiased forecast (point)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  DtACI                (dtaci.py)                                 │
│    target=0.9, gammas {0.001, 0.005, 0.01, 0.05, 0.1},           │
│    window=720, min_warmup=24                                     │
└──────────────────────────────────────────────────────────────────┘
                            │ (lower, point, upper)
                            ▼
                 calibrated prediction interval
```

Both modules are pure Python (stdlib only) so they fit the HA custom
component runtime constraint of no extra dependencies.

---

## Configuration

| Parameter         | Default | Where             | Effect                                          |
| ----------------- | ------- | ----------------- | ----------------------------------------------- |
| `target_coverage` | 0.9     | `DtACI`           | Marginal target. 0.9 → 90% prediction intervals.|
| `gammas`          | (0.001, 0.005, 0.01, 0.05, 0.1) | `DtACI` | Per-expert step sizes.                |
| `window`          | 720     | `DtACI`           | Rolling buffer (≈30 days hourly).               |
| `min_warmup`      | 24      | `DtACI`           | Steps before non-trivial intervals are returned.|
| `eta`             | 0.1     | `DtACI`           | Expert-weight learning rate.                    |
| `halflife_days`   | 20      | `OnlineBiasCorrector` | EMA half-life of signed residuals.          |
| `warmup_steps`    | 168     | `OnlineBiasCorrector` | Steps before bias correction is applied.    |
| `winsor_limit`    | 5.0     | `OnlineBiasCorrector` | Per-step residual cap (× running abs mean). |
| `cadence_per_day` | 24      | `OnlineBiasCorrector` | Steps per day (hourly cadence).             |

The defaults are suitable for hourly EUR/MWh-scale data and were used
for all reported validation runs.

---

## Algorithm details

### DtACI — single iteration

Given a new `(forecast, actual)` pair:

1. Apply bias correction: `point = forecast + bias_estimate` once warm.
   Update the bias EMA with the signed residual `actual - forecast`.

2. Compute the conformity score `s = |actual - point|`.

3. For each expert `k` ∈ {0…K-1}:

   a. Compute its quantile threshold `q_k = Q_{1-α_k}(score_window)`.

   b. Did expert `k` cover? `err_k = 1 if s > q_k else 0`.

   c. Update `α_k`:
      ```
      α_k ← α_k + γ_k * (α_target - err_k)        (clamped to (0, 1))
      ```
      A miss decreases `α_k` (widens the interval); a cover increases
      it (narrows the interval).

   d. Compute its pinball loss:
      ```
      L_k = α_target * max(s - q_k, 0)  +  (1 - α_target) * max(q_k - s, 0)
      ```

4. Multiplicative-weights update:
   ```
   w_k ← w_k * exp(-η * (L_k - min_k L_k))         (renormalised)
   ```

5. Append `s` to the score window. Increment `n_updates`.

### DtACI — interval prediction

Given a new `forecast`:

1. Apply bias correction: `point = forecast + bias_estimate` once warm.

2. If still cold (`n_updates < min_warmup` or empty window) return
   `(point, point, point)`.

3. Compute combined `α_eff = Σ_k w_k * α_k`, then half-width
   `q_eff = Q_{1-α_eff}(score_window)`.

4. Return `(point - q_eff, point, point + q_eff)`.

---

## State persistence

Each (zone, forecaster) keeps its state in a JSON file under the HA
config's data directory:

```
<data_dir>/dtaci_state_fi_hourly.json
<data_dir>/dtaci_state_se1.json     # not active in Phase B v1
<data_dir>/dtaci_state_se3.json
<data_dir>/dtaci_state_ee.json
```

Schema (see `DtACI.to_dict`):

```json
{
  "version": 1,
  "target_coverage": 0.9,
  "gammas": [0.001, 0.005, 0.01, 0.05, 0.1],
  "eta": 0.1,
  "window": 720,
  "min_warmup": 24,
  "alphas":  [...],         // length K
  "weights": [...],         // length K
  "score_window": [...],    // up to `window` floats
  "n_updates": 4521,
  "bias_corrector": {
    "version": 1,
    "halflife_days": 20.0,
    "warmup_steps": 168,
    "winsor_limit": 5.0,
    "cadence_per_day": 24,
    "bias_estimate": -2.34,
    "abs_bias_estimate": 8.71,
    "n_updates": 4521
  }
}
```

State files are written atomically (temp file + `os.replace`) so a
power-cut mid-write cannot corrupt the file.

---

## Sensor attributes (when integration is enabled)

The price-forecast sensor gains four interval band fields per hourly
entry (added by `dtaci_integration.attach_intervals`):

| Field | Unit | Meaning |
| --- | --- | --- |
| `forecast_lower_eur_mwh` | EUR/MWh | Lower bound of the 90% interval, spot |
| `forecast_upper_eur_mwh` | EUR/MWh | Upper bound, spot |
| `forecast_lower_eur_kwh` | EUR/kWh | Lower bound, consumer (with day/night tariff) |
| `forecast_upper_eur_kwh` | EUR/kWh | Upper bound, consumer |

A diagnostic block is exposed on the duration-forecast sensor:

| Attribute              | Description                              |
| ---------------------- | ---------------------------------------- |
| `dtaci_n_updates`      | Total number of updates ingested         |
| `dtaci_effective_coverage` | Current combined coverage target     |
| `dtaci_half_width`     | Current empirical half-width (EUR/MWh)   |
| `dtaci_dominant_gamma` | Step size of the highest-weighted expert |
| `dtaci_bias_estimate`  | Current EMA bias (EUR/MWh)               |
| `dtaci_bias_warm`      | True once `OnlineBiasCorrector` is warm  |

---

## Activation

DtACI is **opt-in**. To enable it on an existing install, set the
following key in your config entry options (Settings → Integrations →
Spot Price Predictor → Configure):

```
enable_dtaci: true
```

Default is `false` until production-side validation against thermal-cost
outcomes confirms net benefit (see DTACI_ANALYSIS.md "Recommendations
for production deployment"). The walk-forward backtest validation is
already strongly positive on FI and SE3; the gate is in place to
de-risk the production rollout, not to question the layer's value.

When enabled the coordinator:

1. Loads `dtaci_state_fi_hourly.json` on each cycle (cold-start
   instantiates a fresh DtACI).
2. Feeds any `(forecast, actual)` pairs that have arrived since the
   previous cycle (Sähkötin publishes day-ahead prices the previous
   afternoon, so each cycle has 1–24 fresh actuals).
3. Calls `attach_intervals` to write band fields into the forecast
   array.
4. Saves state atomically.

---

## Validation summary

From `studies/results/DTACI_ANALYSIS.md`:

* Realised marginal coverage on FI: **0.8949** (target 0.9), on SE3:
  **0.8960**.
* Local-calibration improvement (stable-window fraction, ±5% of
  target) over the static empirical baseline: **+15 pp on FI** (0.807
  → 0.957), **+12.6 pp on SE3** (0.874 → 1.000).
* MAE improvement from bias correction: 7.5% on SE3 (where the AR(2)
  forecaster has the largest known bias per FINDINGS_v2.md), 0.5% on
  FI (already nearly unbiased).
* Mean interval width 5–7% sharper than vanilla single-γ ACI at the
  same coverage.
* Width tracks the volatility cycle across calendar years; the static
  baseline does not.

---

## Limitations and future work

1. **Symmetric scoring** — current implementation uses `s = |y - ŷ|`,
   producing symmetric intervals around the (debiased) point. Heavy
   upper-tail electricity-price distributions could be served better
   by an asymmetric (CQR-style) score. The public API is unchanged;
   only `DtACI._update_internal_score()` would need swapping.

2. **One-step horizon only** — validation forecaster sees the previous
   actual at every step. Extending DtACI to multi-step (k-hours-ahead)
   intervals requires per-horizon score windows. The conformity score
   and update rule are horizon-agnostic, so the addition is mechanical.

3. **Window length is uniform** — `window=720` is one hyperparameter
   shared by all zones. Per-zone tuning is feasible future work.

4. **Asymmetric bias winsorisation** — the bias corrector winsorises
   symmetrically. For zones with high upward-spike skewness, an
   asymmetric clip would prevent spike weeks from corrupting the
   slow-drift estimate.

---

## Tests

| File                                | Coverage                                   |
| ----------------------------------- | ------------------------------------------ |
| `tests/test_dtaci.py`               | 16 tests — quantile/pinball, ACI direction, coverage convergence, regime shift, bias-corrector composition, persistence round-trip, cold-start, configuration validation, dominant-expert tracking |

Run with `pytest tests/test_dtaci.py -q`.

---

## Files

| Path | Role |
| --- | --- |
| `custom_components/spot_price_predictor/dtaci.py` | Pure-Python DtACI |
| `custom_components/spot_price_predictor/bias_corrector.py` | EMA bias tracker |
| `custom_components/spot_price_predictor/dtaci_integration.py` | Coordinator glue (load/save/attach) |
| `studies/validate_dtaci.py` | Walk-forward backtest harness |
| `studies/results/dtaci_validation_*.{md,json}` | Per-zone validation reports |
| `studies/results/DTACI_ANALYSIS.md` | Comprehensive analysis |
| `tests/test_dtaci.py` | Unit tests |
| `docs/dtaci_layer.md` | This document |
