# v2.11.8 — de-scope DtACI to FI (remove redundant cross-border bundles)

Patch release on top of v2.11.7 (shared WM hexagon brand icon). Removes
the SE1/SE3/EE per-D(i) DtACI bundles, which were dead scaffolding. No
sensor attribute/schema changes.

## What was wrong

`DTACI_ZONES` declared four zones (FI, SE1, SE3, EE) and the coordinator
created, persisted, and reported a DtACI bundle for each. But the data
path is FI-only:

* No neighbour D(k) is ever produced (`duration_forecast` carries only FI
  consumer/spot D(k)).
* `_dtaci_record_forecasts` / `_dtaci_reconcile_actuals` /
  `_dtaci_attach_bands` are hardcoded to `"fi"`.

So the SE1/SE3/EE bundles never received an `update()` — they stayed
permanently cold (`n_warm_instances = 0`) while still writing three empty
state files every cycle and showing up "idle" in the diagnostics card.

An analytical estimate (using the measured FI ridge coefficients
0.287/0.252/0.146, the per-zone DtACI MAE reductions, and the 0.57–0.75
neighbour↔FI residual correlations) put the FI-accuracy benefit of
neighbour DtACI bias-correction at **~1% MAE** — and that is largely
capped by the pipeline's existing final-stage FI bias EMA, which already
absorbs the constant component of any propagated neighbour bias. Not worth
the complexity.

## What's changed

* `DTACI_ZONES = ("fi",)` — the FI bundle is the only one created,
  persisted, and reported. The FI bundle drives the sensor's calibrated
  D(k) bands exactly as before.
* `_dtaci_init_bundles` now **removes stale `dtaci_dk_se1/se3/ee.json`**
  state files once, on first start after upgrade.
* Diagnostics now report a single zone (FI); the warmup-status badge
  counts 2 × 24 = 48 instances instead of 96.

## What's explicitly NOT changed

Cross-border **price coupling is untouched** — SE1/SE3/EE still feed the FI
model as features:

* Pipeline: deseasonalised `Y_se1` / `Y_se3` / `Y_ee` ridge terms.
* Feature model: `ar_se1` / `ar_se3` / `ar_ee` AR(2) forecasts +
  `export_potential_se3` spread.

Only the redundant per-D(i) DtACI *calibration bundles* for those zones
were removed. `fetch_neighbor_prices`, `_NEIGHBOUR_ZONES`, and all
neighbour features remain.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/const.py` | `DTACI_ZONES = ("fi",)` + rationale |
| `custom_components/spot_price_predictor/coordinator.py` | FI-only init docstring; stale neighbour-zone state-file cleanup; "four-zone" → "FI" comment |
| `custom_components/spot_price_predictor/sensor.py` | Warmup-status comment 96 → 48 instances; `sw_version` 2.11.7 → 2.11.8 |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.7 → 2.11.8 |
| `tests/test_dk_dtaci.py` | Guard test: DtACI scoped to FI + stale-file cleanup wired |

## Migration

Automatic. On first start after upgrade, the three neighbour-zone state
files are deleted; the FI bundle is unaffected and keeps its calibration.

## Test status

`python -m pytest tests/` → 495 passed, 5 skipped.
