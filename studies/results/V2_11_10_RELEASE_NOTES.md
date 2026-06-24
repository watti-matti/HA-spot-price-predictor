# v2.11.10 — fix external-PV time misalignment (phantom night PV)

Stable release. Promotes the 2.11.9-beta.1 public-launch prep to a full
release and fixes a **time-alignment bug in the external PV forecast path**
that placed solar production in the middle of the night, collapsing the
PV-aware effective price to 0 during hours when the sun was down.

## What was wrong

With `pv_source: external`, a field report on 2.11.9-beta.1 showed
`pv_production_kwh` of 2–5 kWh at 01:00–03:00 **local** time, while the
model's own (correctly aligned) Open-Meteo irradiance was 0 for the same
hours. Because self-consumed PV is free, `effective_eur_kwh` collapsed to
`0` across those night hours — a flat green band from ~01:00 to "now".

Root cause: the external reader's **`irradiance` branch consumed the value
list positionally**, ignoring the companion time axis the source publishes
(`iso_time` / `time`). For a source whose list starts at local midnight
(e.g. `sensor.meteo_7day_forecast_total`, which exposes `irradiance` in W
plus a naive-**local** `iso_time`), positional indexing against the
`now`-anchored forecast clock dropped the local-13:00 production peak into
the small hours.

The internal Open-Meteo estimator was never affected — `estimate_pv_kwh_per_hour`
is a pure function of irradiance, and the weather series is re-based to
`now` (the v2.11.5 fix). This was strictly the external-entity path.

## What's changed

1. **Timestamp alignment for the `irradiance` branch**
   (`_read_external_pv_forecast`). When the source publishes a parallel
   `iso_time` / `time` axis, the PV values are now aligned by timestamp via
   `_parse_ts` (naive timestamps interpreted in the configured local zone,
   then floored to the UTC hour) instead of positionally. The local-13:00
   peak now maps to 10:00 UTC and renders back at 13:00 local — correct for
   both day **and** night. Sources without a usable time axis fall back to
   positional indexing (still gated — see below).

2. **Physical irradiance gate** (`_compute_pv_forecast`). Defense-in-depth:
   external PV is forced to 0 at any forecast hour where the model's own
   aligned Open-Meteo irradiance says the sun is down (< 5 W/m²). No future
   external-source quirk (wrong tz, positional, spurious night entries) can
   put PV production at night again. The gate is a no-op when weather data
   is unavailable, so it never zeroes all PV.

## Effect

Night `pv_production_kwh` returns to 0, so `effective_eur_kwh` at
01:00–06:00 returns to the real consumer buy price (no PV discount after
sunset). Daytime PV from a timestamped source now peaks at the true ~13:00
local solar noon instead of being smeared by the positional offset.

## Also included (rolled up from 2.11.9-beta.1 + cockpit dashboards)

* Public-launch prep: docs accuracy pass + dashboard rewrite (Phase 0+1).
* Dedicated **effective wind speed** sensor (model's capacity-weighted
  120 m wind).
* PV-aware CVaR fields: `pv_aware_mean/p5/p95_eur_kwh` + no-PV
  `grid_cost_eur_kwh` baseline, and the parallel "expected vs worst-5%" /
  "with vs without PV" cards.
* New **Duration & Risk cockpit** dashboard example
  (`docs/yaml_examples/cockpit.yaml` + section 2c of the v2.11 dashboard):
  nested cheap→peak spread bands, a day-selectable duration curve, and a
  plain-language PVaR risk/PV ledger driven by one `input_select` day
  picker.

## Files changed (this release's fix)

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/coordinator.py` | `irradiance` branch aligns by `iso_time`/`time` via `_parse_ts`; `_compute_pv_forecast` gates external PV by aligned irradiance (sun-down → 0) |
| `custom_components/spot_price_predictor/manifest.json` | `version` → 2.11.10 |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version` → 2.11.10 |
| `tests/test_openmeteo_time_alignment.py` | New: Open-Meteo UTC request guard, real `fetch_weather` exercise, no-PV-after-local-sunset, DST-awareness, external-PV irradiance gate, and `irradiance` + naive-local `iso_time` alignment (built from the reporting source's shape) |
| `docs/yaml_examples/cockpit.yaml`, `forecast_v2_11_dashboard.yaml` | Duration & Risk cockpit example |

## Migration

Automatic. No configuration or attribute-schema changes. Users with an
external PV forecast entity that publishes a time axis get correct
day/night placement on the next coordinator refresh after upgrade.

## Test status

`python -m pytest tests/` → 542 passed, 5 skipped.
