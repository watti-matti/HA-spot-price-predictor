# v2.11.3 — PV-aware price consistency fixes + release consistency test gate

Patch release on top of v2.11.2. Two correctness fixes to the PV-aware
pricing path plus a comprehensive data-consistency test suite. No config
or schema changes.

## What's fixed

### 1. Stale PV-aware prices after the L1–L4 pipeline overwrite

`effective_eur_kwh`, `net_household_cost_eur`, and `sell_eur_kwh` were
computed in a first pass from the **raw model** spot price, then the
prediction pipeline (`_apply_pipeline_pre_dk`) overwrote `spot_eur_mwh`
and `consumer_eur_kwh` with its corrected values — but never recomputed
the PV-aware fields. The published row therefore mixed a pipeline-corrected
`consumer_eur_kwh` with PV-aware fields frozen against a different (lower)
price.

Symptom: at night (and any hour with `pv_production_kwh <= baseload_kwh`),
`effective_eur_kwh` should equal `consumer_eur_kwh` — instead it was ~half.
`net_household_cost_eur` was likewise `baseload × stale_buy_price`.

Fix: the pipeline-overwrite loop now recomputes `sell_eur_kwh`,
`effective_eur_kwh`, `net_household_cost_eur`, and `is_export_hour` from
the corrected price, immediately after `consumer_eur_kwh` is recomputed.
This also corrects the downstream `current_effective_eur_kwh` and
`week_*_effective_eur_kwh` aggregates.

### 2. PV-aware D(k) horizon starting one day late

The duration sensor's grid D(k) back-fills "today" from realized spot
prices, but the PV-aware D(k) (`dk_cheap_pv_eur_kwh` /
`dk_peak_pv_eur_kwh`) was only emitted for forecast days with a full 24
hours of data. Because the forecast window starts at `now`, "today" is
partial and was dropped — so PV-aware lines began a day later than the
grid lines on dashboards.

Fix: new `_pv_dk_by_local_date()` reconstructs each local day's full
24-hour `effective_eur_kwh` series from the rolling `_forecast_history`
unioned with the current forecast, and the merge step injects today's
PV-aware D(k) onto the relevant `duration_forecast` entry. PV-aware
duration curves now start the same day as the grid curves.

Known limitation: after a mid-day cold start (empty history), today
cannot reach 24 hours and its PV-aware D(k) appears the next day. Normal
continuous operation is unaffected.

## What's added

- `tests/test_release_data_consistency.py` — a release-gate suite that
  re-derives every published variable (`consumer`, `effective`, `net`,
  `sell`, `is_export_hour`, P5–P95 fan chart, grid/PV D(k), week/current
  aggregates) from its inputs and asserts full internal consistency over a
  realistic 188-hour horizon spanning day/night and export/non-export. It
  includes the actual pre-fix sample as a negative fixture and source-text
  guards that keep the mirrored tariff/sell formulas aligned with the
  coordinator.
- New regression tests in `tests/test_coordinator_pv.py` covering the
  night-time `effective == consumer` invariant and the PV-aware D(k)
  horizon reconstruction.

## What's unchanged

- All sensor entities, attribute names, and schemas.
- Spot/consumer prices, percentiles, and the prediction pipeline itself.
- Integration icon artwork was refreshed (no behavioral effect).

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/coordinator.py` | Recompute PV-aware fields after pipeline overwrite; add `_pv_dk_by_local_date()` and inject today's PV-aware D(k) |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version` 2.11.2 → 2.11.3 |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.2 → 2.11.3 |
| `custom_components/spot_price_predictor/icon.png`, `icon@2x.png`, `brand/icon.png`, `brand/icon@2x.png`, `brand/logo.png` | Refreshed icon/logo artwork; `@2x` now correctly 512×512 |
| `tests/test_coordinator_pv.py` | Regression tests for both fixes |
| `tests/test_release_data_consistency.py` | New release-gate consistency suite |

## Migration

None required. HACS auto-update; existing configs continue to work. The
fixes only correct already-published derived values — no user action.

## Test status

`python -m pytest tests/` → 485 passed, 5 skipped.
