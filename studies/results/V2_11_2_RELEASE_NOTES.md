# v2.11.2 — auto-prefix Nordpool entity ID (defensive robustness)

Patch release on top of v2.11.1. Pure defensive fix; no new features.

## What's fixed

`sensor.spot_price_predictor_spot_electricity_price` and
`sensor.spot_price_predictor_spot_electricity_selling_price` would
stay at `state: unknown` with empty `timeline: []` when the user's
`nordpool_entity` config field stored the **bare object ID** (e.g.
`nordpool_kwh_fi_eur_3_10_0`) instead of the full HA entity ID
(`sensor.nordpool_kwh_fi_eur_3_10_0`).

Root cause: `hass.states.get("nordpool_kwh_fi_eur_3_10_0")` returns
`None` because HA requires `domain.object_id` form for entity IDs.
The bare object ID slipped through config validation because the
field is a free-form string.

Fix: new `_normalize_entity_id()` helper auto-prepends `sensor.`
when the stored value contains no `.`. Applied at all four sites
in `sensor.py` that read `CONF_NORDPOOL_ENTITY`:

- `SpotElectricityPriceSensor.native_value`
- `SpotElectricityPriceSensor.extra_state_attributes`
- `SpotElectricitySellingPriceSensor.native_value`
- `SpotElectricitySellingPriceSensor.extra_state_attributes`

User experience: the integration now forgives the missing `sensor.`
prefix transparently. The user's previous workaround (manually
editing the config field to add `sensor.`) is still valid but no
longer required.

## What's unchanged

- All other integration code.
- The v2.11.1 Nordpool 15-minute schema fix (`_normalize_ts_to_iso`)
  still works the same way; this v2.11.2 patch is independent.
- All sensor entities, attribute names, and schemas.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/sensor.py` | `_normalize_entity_id()` added; applied at 4 call sites; `sw_version` 2.11.1 → 2.11.2 |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.1 → 2.11.2 |

## Migration

None required. HACS auto-update; new integration installs and
existing configs both work — the helper is purely additive.

## Test status

`python -m pytest tests/` → 471 passed, 0 failed.
