# v2.11.6 — fix spurious 12/13 seam in DtACI per-k bias / coverage

Patch release on top of v2.11.5. Removes a calibration-state migration
artifact that made D_peak / D_cheap per-k bias and coverage look
discontinuous between k=12 and k=13. No schema changes to sensor
attributes.

## What was wrong

`D_peak(k)` (mean of the k priciest hours) is a smooth, monotone function
of k — on real data the step `D(k)−D(k−1)` across the 12→13 boundary is
indistinguishable from every other step. So its forecast bias has no
mathematical reason to jump at 12/13.

The jump came from persisted DtACI state, not the statistic:

* Phase B v2 introduced the per-(direction, k) bundle with **12** levels
  per direction (`CHEAP_PEAK_K_RANGE = range(1, 13)`).
* v2.8.1 expanded it to **24** levels (`range(1, 25)`) but left the
  persisted schema at `version: 1` with **no migration**.
* `DkDtACIBundle.from_dict` restores only the instance keys present in the
  saved JSON and cold-starts the rest.

So on any install that ran the 12-level era, after upgrading:

* `peak_1..12` / `cheap_1..12` reloaded with **warm** bias EMAs, while
* `peak_13..24` / `cheap_13..24` **cold-started** (zero bias, not warm).

That produced "substantially different bias data for k=1..12 vs k=13..24"
with the seam exactly at 12/13 — a state artifact, not market behaviour.

## What's fixed

* Bundle persisted schema bumped to `SCHEMA_VERSION = 2`.
* `from_dict` now **refuses pre-v2 (12-level) state** with a clear message,
  so `load_or_create_bundle` cold-starts a fresh bundle. All 24 levels then
  warm **uniformly** with identical history — no 12/13 seam.
* This is a one-time re-warm of the calibration bands (~1–3 weeks), which
  was already happening to k=13..24 on affected installs.

## What's unchanged

* The D(k) point forecasts, sensor attributes, and all other calibration
  behaviour. Only the persisted DtACI band state is reset once on upgrade.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/dk_dtaci.py` | `SCHEMA_VERSION = 2`; `to_dict` writes v2; `from_dict` rejects v1 (12-level) state |
| `custom_components/spot_price_predictor/dtaci_integration.py` | Loader log wording: "unreadable or incompatible; cold-starting" |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.5 → 2.11.6 |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version` 2.11.5 → 2.11.6 |
| `tests/test_dk_dtaci.py` | v2 schema tests + legacy-v1 cold-start (no-seam) end-to-end test |

## Migration

Automatic. On first start after upgrade, the old `dtaci_dk_<zone>.json`
state is discarded and rebuilt; calibration bands re-warm over ~1–3 weeks.
No user action. (Deleting the state files manually has the same effect.)

## Test status

`python -m pytest tests/` → 494 passed, 5 skipped.
