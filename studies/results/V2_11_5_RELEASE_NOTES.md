# v2.11.5 — fix weather / PV time alignment (solar shifted by now.hour)

Patch release on top of v2.11.4. Fixes a time-alignment bug that shifted
all weather and PV data later by the UTC hour-of-update. No schema changes.

## What was wrong

`fetch_weather` calls Open-Meteo `/forecast` with `forecast_days=8` and
`timezone=UTC` but **no start hour**, so the hourly grid begins at **00:00
UTC of the current day**. The coordinator consumed it **positionally**
against a clock anchored at `now`:

* `build_forecast_features`: `weather_data[i]` at hour `start_utc + i`
* forecast rows + internal PV: `weather[i]` at `now + i`

So `weather[i]` (really hour `00:00 + i`) was labelled hour `now + i` —
shifting the entire solar/wind/temp series **later by `now.hour` hours**.
At a 12:00Z update, real solar noon (~10:00Z) appeared at 22:00Z (visible
in the dumped `solar_weighted` peaking at 22:00Z = 01:00 local). The
PV-cheap window therefore landed "right after local midnight."

Two consumers were affected:

1. **Model inputs** — wind/solar/temp fed the spot-price model shifted, so
   the bug also degraded the *price* forecast.
2. **External PV** — `_read_external_pv_forecast` discarded the per-entry
   timestamps / dict keys and returned a positional list consumed as if it
   started at `now`; a source publishing from 00:00 was shifted the same way.

## What's fixed

* `fetch_weather` now carries Open-Meteo's `hourly.time` as a per-row
  `timestamp`.
* New `_align_weather_to_now()` re-bases the weather list so index 0 is the
  current hour (drops already-elapsed rows), fixing model features,
  internal PV, and the displayed solar in one place. Falls back to a
  positional `now.hour` slice if timestamps are absent.
* `_read_external_pv_forecast` now **honours timestamps**: `forecast`
  list-of-dicts (via `period_start` / `datetime` / `timestamp` / `time` /
  `start` keys), `wh_hours`, and `watts` return a `{utc_hour -> kWh}` dict.
  Naive timestamps (e.g. Forecast.Solar local time) are interpreted in the
  configured local timezone, else UTC. `_compute_pv_forecast` aligns that
  dict to each forecast hour (`now + i`). Sources without timestamps
  (`irradiance` list) keep positional behaviour.

## Effect

Solar/PV now line up with wall-clock time: the PV-cheap / `effective ≈ 0`
hours fall around local midday, not after midnight. Because the model now
sees correctly-aligned weather, spot-price forecasts also change (improve).

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/api_client.py` | `fetch_weather` carries `hourly.time` as per-row `timestamp` |
| `custom_components/spot_price_predictor/coordinator.py` | `_align_weather_to_now()`; weather re-based to `now`; `_read_external_pv_forecast` timestamp-aware (dict); `_compute_pv_forecast(..., start_utc)` aligns by time |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.4 → 2.11.5 |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version` 2.11.4 → 2.11.5 |
| `tests/test_coordinator_pv.py` | Weather/external-PV alignment tests + source guards |

## Migration

None required. HACS auto-update. The fix changes published forecast values
(weather, PV, and price) to their correctly-timed values.

## Test status

`python -m pytest tests/` → 491 passed, 5 skipped.
