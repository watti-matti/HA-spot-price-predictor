# v2.7.0 — Cutover: v26 L1+L2+L3+L4+floor REPLACES v2.2 9-feature Ridge

## TL;DR

**HA users on v2.7.0 get materially better predictions on their existing dashboards.** No reconfiguration required.

Per the v2.6.1 head-to-head benchmark on real FI test data (decisive win for v2.6.0 across every metric), the v26 V_sigmoid_full pipeline now produces the primary `spot_eur_mwh` and `consumer_eur_kwh` values that flow into `sensor.spot_price_predictor_price_forecast` and `sensor.spot_price_predictor_duration_forecast`.

## What changes on update from v2.6.x → v2.7.0

| Attribute | Before (v2.2 9-feature Ridge) | After (v26 L1+L2+L3+L4+floor) |
|---|---|---|
| `forecast[i].spot_eur_mwh` | v2.2 prediction | **v26 prediction** |
| `forecast[i].consumer_eur_kwh` | derived from v2.2 spot | **derived from v26 spot** |
| `dk_cheap_eur_kwh[12]` | computed from v2.2 hourly | **computed from v26 hourly** |
| `dk_peak_eur_kwh[12]` | computed from v2.2 hourly | **computed from v26 hourly** |
| `dk_cheap_v26_eur_mwh[24]` | (added in v2.6.0, additive) | unchanged — 24-entry extended |
| `dk_peak_v26_eur_mwh[24]` | (added in v2.6.0, additive) | unchanged |
| `v26_P5..P95_eur_mwh` | (added in v2.6.0, additive) | unchanged — fan-chart bands |

**Entity IDs unchanged.** Every existing dashboard / automation continues to work and starts seeing better numbers immediately.

## Measured improvement on the user's data

Same v2.6.1 benchmark numbers, restated as "what changes for the user":

| Dimension | v2.2 (was) | v2.7.0 (is now) | Improvement |
|---|---:|---:|---:|
| Hourly point-forecast MAE | 35.20 EUR/MWh | **10.00 EUR/MWh** | **−71.6 %** |
| Hourly R² | +0.49 | **+0.93** | +0.44 |
| Hourly bias | −34.0 EUR/MWh | **−0.2 EUR/MWh** | bias eliminated |
| `cheap_4` MAE | 14.4 EUR/MWh | **4.4 EUR/MWh** | −69 % |
| `peak_4` MAE | 55.6 EUR/MWh | **6.9 EUR/MWh** | **−88 %** |
| Price-spike capture (≥ 100 EUR/MWh) | 29.1 % | **98.4 %** | **3.4× more spikes caught** |
| Per-month MAE | v2.2 wins 0 of 18 | **v26 wins 18 of 18** | dominant across regimes |

## What changed in code

### `coordinator.py` — surgical edit

The `_apply_v26_pipeline` helper is now `_apply_v26_pipeline_pre_dk` and runs BEFORE `_compute_duration_forecast` instead of after. Three things happen in order:

1. v2.2 Ridge runs as before — produces the initial `forecast[i].spot_eur_mwh` values
2. **NEW**: v26 pipeline runs and OVERWRITES `spot_eur_mwh` + `consumer_eur_kwh` per row with v26 values
3. `_compute_duration_forecast` runs on the v26-overwritten forecast → D(k) reflects v26
4. v26 pipeline also computes the extended 24-entry D(k) and injects as `dk_cheap_v26_eur_mwh` / `dk_peak_v26_eur_mwh` keys on each duration_forecast day
5. DkDtACI wraps the (now v26-derived) D(k) as before

This is a ~30 LOC reorder of the existing v2.6.0 wiring. The v2.2 model code (`model.py`, `features.py`, `data/model_coefs_default.json`) still exists in the repo and is still loaded at startup — but its predictions are now shadowed by v26 before reaching the sensor attributes.

### Why we kept the v2.2 code in place

Safety. Removing 400+ LOC of v2.2-specific code in the same patch that introduces the cutover doubles the risk of subtle bugs. v2.7.0 is a *behaviour change without a code removal* — easy to roll back if any field issue surfaces.

A future v2.8.0 (or whichever release follows positive field reports) can perform the cleanup:
- Delete `model.py`, `features.py` AR-with-daytype code, `model_coefs_default.json`
- Net cleanup: ~400 LOC removed
- No additional behaviour change at that point

## Sensor schema after v2.7.0

```yaml
sensor.spot_price_predictor_price_forecast:
  forecast:
    - timestamp: "2026-05-20T10:00:00+00:00"
      spot_eur_mwh: 72.1               # v26 NOW (was v2.2)
      consumer_eur_kwh: 0.1198         # v26-derived NOW (was v2.2-derived)
      wind: 5.8                        # unchanged
      solar: 412.0                     # unchanged
      temp: 12.3                       # unchanged
      # ── still present from v2.6.0 ──
      v26_mean_eur_mwh: 72.1           # = spot_eur_mwh (duplicate intentional)
      v26_P5_eur_mwh:   18.0
      v26_P25_eur_mwh:  48.0
      v26_P50_eur_mwh:  72.0
      v26_P75_eur_mwh:  98.0
      v26_P95_eur_mwh:  155.0

sensor.spot_price_predictor_duration_forecast:
  daily_forecast:
    - date: "2026-05-21"
      dk_cheap_eur_kwh: [...]          # v26-derived NOW (was v2.2-derived), 12 entries
      dk_peak_eur_kwh:  [...]          # v26-derived NOW, 12 entries
      dk_cheap_v26_eur_mwh: [c00, c01, ..., c22, c23]   # 24-entry extended
      dk_peak_v26_eur_mwh:  [p00, p01, ..., p22, p23]   # 24-entry extended

# Both spot price sensors unchanged (orthogonal to model upgrade)
sensor.spot_price_predictor_spot_electricity_price:        unchanged
sensor.spot_price_predictor_spot_electricity_selling_price: unchanged
```

## Operational impact

| Metric | Impact |
|---|---|
| **Sensor entity IDs** | Unchanged. Zero dashboard / automation migration. |
| **Attribute names** | Unchanged. All existing attribute keys preserved. |
| **Attribute values** | Materially better predictions for `spot_eur_mwh`, `consumer_eur_kwh`, `dk_cheap_eur_kwh`, `dk_peak_eur_kwh`. |
| **Runtime cost** | +50 ms per coordinator cycle (v26 pipeline still runs in addition to v2.2). |
| **Memory** | +20 KB per zone for v26 calibrator state. |
| **External APIs** | No new calls. |
| **Persistent storage** | New: `.storage/spot_price_predictor_v26/{hourly_bias,hourly_fan_chart,refit_monitor}.json` (~5 KB total). |

## How thermal / EMHASS users benefit

EMHASS configured against `consumer_eur_kwh` automatically inherits the v26 accuracy improvement — no config change. Households see better day-ahead scheduling decisions. The v26 fan-chart attributes (P5..P95) remain available for users who want to opt into CVaR-aware optimisation.

## Files

- **Modified**: `custom_components/spot_price_predictor/coordinator.py` (~40 LOC: reorder + helper rename + signature change + return tuple)
- **Modified**: `manifest.json` `2.6.1 → 2.7.0`
- **Modified**: `README.md` release-notes index
- **New**: `studies/results/V2_7_0_RELEASE_NOTES.md` — this document

## Tests

**403 / 403 passing** (no new tests; cutover is a runtime data-flow change validated by the existing v26 test suite + v2.6.1 benchmark).

## Field validation recommendation

Watch your `sensor.spot_price_predictor_price_forecast` for the first few days after update. Compare predicted vs actual via the existing chart attributes — predictions should track real prices visibly tighter than before.

If anything looks wrong, the v2.2 model code is still loaded (just shadowed). A future patch can re-enable the v2.2 output path via a config flag if any user reports a regression.

## What's next

- **v2.8.0** (after 2-4 weeks of positive field reports): delete `model.py`, `features.py` AR-with-daytype machinery, `model_coefs_default.json`. Pure cleanup, no behaviour change. ~400 LOC removed.
- **Quarterly artifact refresh**: re-run `python studies/build_seasonal_components.py` + `python studies/solar_clear_sky_submodel.py` to keep L1/L4 in sync with the evolving FI generation mix. The integration picks the refreshed JSONs up automatically on next coordinator restart.

## Reproducibility

The change is mechanical — pull the v2.7.0 tag, restart HA, the coordinator's next update cycle will produce v26-derived primary values.

For offline verification:

```bash
python studies/v261_v22_vs_v26_benchmark.py     # re-confirm v26 wins
python studies/v2516_performance_review.py      # end-to-end performance
```
