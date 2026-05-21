# v2.11.1 — Nordpool 15-minute schema + dashboard polish (patch)

Patch release on top of v2.11.0. No new features. Bug fix and dashboard
hardening — entirely additive to v2.11.0; safe upgrade for all v2.11.0
users.

The version bump exists to force HACS to actually re-download the
integration files; the in-place tag hot-fixes that landed on v2.11.0
were ignored by HACS because the version string in `manifest.json`
hadn't changed.

## What's fixed

### sensor.spot_price_predictor_spot_electricity_price was stuck at "unknown"

Root cause: the Nordpool HACS integration migrated to **15-minute
resolution** in 2025+ (EU 15-minute-settlement rollout). Its
`raw_today` / `raw_tomorrow` entries now contain Python `datetime`
objects in the `start` key (instead of ISO strings), and there are 96
entries per day rather than 24.

The integration's `_process_nordpool_data` used `str(ts)` on the
`start` value, producing strings with a space separator
(`"2026-05-21 00:00:00+03:00"`). Downstream, `datetime.fromisoformat()`
silently failed on those (it strictly required a `T` separator on
Python ≤ 3.10), and the resulting timeline was empty.

Fix:

- New `_normalize_ts_to_iso()` helper coerces any reasonable timestamp
  format (datetime object, int/float UNIX seconds, ISO string with
  either `T` or space separator, trailing `Z`) to canonical ISO 8601
  with `T` separator.
- `_process_nordpool_data` now calls it on every timestamp before
  using it as a dict key. Downstream consumers can rely on
  `datetime.fromisoformat()` working on any Python version.
- The legacy `data`-attribute branch also recognises `start`/`value`
  keys (the new format) in addition to the legacy `Timestamp` /
  `TotalPrice` keys.

Validated end-to-end with the exact format reported by an affected
user (96 quarter-hour entries with `datetime(..., tzinfo=ZoneInfo)`
values) — all 96 entries returned with parseable ISO timestamps.

### Dashboard hardening

`docs/yaml_examples/forecast_v2_11_dashboard.yaml`:

- Removed `graph_span: 24h` from the "Today's cheapest-k-hours" chart;
  the integer k values 1..24 were being interpreted as UNIX timestamps
  (1970-01-01 + 1..24 seconds) under apexcharts-card's time-series
  mode. Switched to `xaxis.type: category` with object-form data
  points.
- Replaced markdown tables with embedded HTML tables across both
  dashboard YAMLs (`forecast_v2_11_dashboard.yaml`,
  `dtaci_diagnostics_card.yaml`). HA's GUI dashboard editor was
  reformatting `content: |` to `content: >` on save, which folded
  markdown table rows into a single line. HTML tables survive any
  YAML scalar style because browsers ignore whitespace between
  tags. Explicit column widths via `<colgroup>`.
- DtACI per-k coverage panels: switched from horizontal bars + numeric
  xaxis (which left a "strange y-axis scale" and no visible data) to
  vertical bars with `xaxis.type: category` and a percentage y-axis;
  added a dashed target=90 % annotation line.
- Both dashboard YAMLs restructured to complete-dashboard format
  (top-level `title:` + `views:`) so they paste into HA's Raw
  Configuration Editor without the "views: expected an array value"
  error.

## What's unchanged

- All integration code outside `sensor.py::_process_nordpool_data` and
  the new `_normalize_ts_to_iso` helper.
- All sensor entities, attribute names, and schemas.
- All v2.11.0 architectural plans and validation results.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/sensor.py` | `_normalize_ts_to_iso()` added; `_process_nordpool_data` rewritten; `sw_version` 2.11.0 → 2.11.1 |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.0 → 2.11.1 |
| `docs/yaml_examples/forecast_v2_11_dashboard.yaml` | k-axis fix; markdown → HTML tables; complete-dashboard wrapper |
| `docs/yaml_examples/dtaci_diagnostics_card.yaml` | Per-k coverage panels fixed; markdown → HTML tables; complete-dashboard wrapper |

## Migration

None required. HACS auto-update; after the next coordinator cycle the
Nordpool-derived sensors will populate their timelines from the new
15-minute Nordpool schema. The forecast / duration / spot-forecast-fi
sensors were unaffected by this bug and continue to work as in v2.11.0.

## Test status

`python -m pytest tests/` → 471 passed, 0 failed.
