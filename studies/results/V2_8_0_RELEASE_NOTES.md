# v2.8.0 — Consolidated retraining service + v2.2 legacy cleanup

## TL;DR

**HA users can now retrain the entire v26 model stack from the Home Assistant UI** — a single service call refreshes the L1 seasonal artifact, the L2+L3+L4 spike model, and (optionally) the solar sub-model, then atomically reloads the V26Pipeline on every coordinator. Pair it with the v2.5.15 `RefitMonitor` flag and you have an end-to-end "drift detected → user retrains → fresh model live" workflow without ever leaving Home Assistant.

Per user direction 2026-05-17 ("there should also be a consolidated retraining script that can be triggered from HA side that retrains all necessary parts of the model. We do not need to maintain v2.2 legacy but replace the same functionality with new model"), v2.8.0 delivers:

1. New `spot_price_predictor.retrain_models` HA service.
2. New `retrain.py` orchestrator that wraps the existing study scripts.
3. Atomic artifact replacement (no risk of half-written JSON).
4. Automatic V26Pipeline reload across all coordinators after success.
5. New HA event `spot_price_predictor_models_retrained` for automation chaining.

## How to use it

### From Home Assistant UI

Developer Tools → Services → `Spot Price Predictor: Retrain v26 model artifacts`.

```yaml
service: spot_price_predictor.retrain_models
data:
  layers: ["seasonal", "spike"]      # optional, default: all three
  fingrid_api_key: "your_key_here"   # only needed for the solar layer
```

If you omit `layers`, all three (`seasonal`, `spike`, `solar`) are refit. If you omit the Fingrid key, it falls back to the `FINGRID_API_KEY` env var; if that's also missing the solar layer is skipped cleanly and the other two proceed.

### From an automation (drift-triggered)

```yaml
automation:
  - alias: "Auto-retrain when model drifts"
    trigger:
      - platform: template
        value_template: >
          {{ state_attr('sensor.spot_price_predictor_duration_forecast',
             'pipeline_diagnostics').get('refit_recommended', false) }}
    action:
      - service: spot_price_predictor.retrain_models
        data:
          layers: ["seasonal", "spike"]
      - service: persistent_notification.create
        data:
          title: "Spot Price Predictor"
          message: "Auto-retrained after coverage drift was detected."
```

When `RefitMonitor` raises the `refit_recommended` flag (after 14 consecutive days of >5 pp coverage drift), the automation triggers a retrain, and the V26Pipeline picks up the fresh artifact on the next coordinator cycle.

### From the command line (debugging)

```bash
python -m custom_components.spot_price_predictor.retrain --layers spike seasonal
python -m custom_components.spot_price_predictor.retrain --fingrid-key XXX --layers solar
```

Returns a JSON dict with per-layer metadata. Useful for sanity-checking after an artifact change.

## Service behaviour in detail

1. **Validation**: schema-checks the `layers` and `fingrid_api_key` inputs.
2. **Execution**: runs `retrain.retrain_all()` in `hass.async_add_executor_job` so HA's event loop isn't blocked.
3. **Per-layer flow**: each layer's refitter cd's into the repo root, imports the relevant `studies/*.py` module, and runs its `main()` which writes the artifact via `_atomic_write_json` (temp file + `os.replace`).
4. **Reload**: after the orchestrator returns, every active `SpotPriceCoordinator` gets a fresh `V26Pipeline(data_dir, storage_dir)` and is told to `async_request_refresh()`.
5. **Notification**: a `persistent_notification` with the per-layer outcomes is created.
6. **Event**: `spot_price_predictor_models_retrained` fires with `{result: {...}, reloaded_coordinators: N}` for automation chaining.

## What the orchestrator does per layer

| Layer | Script invoked | Output artifact | Typical runtime |
|---|---|---|---|
| **seasonal** | `studies/build_seasonal_components.py` | `data/seasonal_components_default.json` | ~30 s (uses cached weather parquets) |
| **spike** | `studies/v2513_layer4_spike_model.py` | `data/spike_model_default.json` | ~30 s (Ridge + AR + GPD POT fit) |
| **solar** | `studies/solar_clear_sky_submodel.py` | `data/solar_submodel_default.json` | 2–5 min (Fingrid + Open-Meteo fetch if cache stale) |

All atomic: the `_atomic_write_json` helper writes to a `.tmp` sibling file then `os.replace`s into the target. If anything fails mid-flight, the live artifact is untouched.

## What's still v2.2 in code (not removed in this patch)

The user direction was to "replace the same functionality" — done at the runtime path level in v2.7.0. v2.8.0 does NOT delete:

- `custom_components/spot_price_predictor/model.py` — `SpotPriceModel` class still imported by `__init__.py` for the legacy `upload_coefficients` and `reset_coefficients` services. Those services are themselves legacy from the v2.2 era but stay for now (they're harmless when nobody uses them).
- `custom_components/spot_price_predictor/features.py` — still imported by `coordinator.py` for `build_forecast_features()` which produces the v2.2 features. The v2.2 prediction is computed but shadowed by v26 in `_apply_v26_pipeline_pre_dk`.
- `data/model_coefs_default.json` — still loaded at startup.

Why we deferred the full deletion:

1. The v2.2 service handlers (`upload_coefficients`, `reset_coefficients`, `model_info`) need a non-trivial rewrite to operate against v26 artifacts. Cleaner as a separate v2.9.0 patch.
2. Removing the v2.2 prediction loop in the coordinator carries a non-trivial risk of breaking the forecast row population code that v26 currently overwrites. Easier to keep the shadow pattern.
3. v2.8.0 is already substantial; one focused user-facing feature (the retraining service) is easier to validate than feature + cleanup combined.

**v2.9.0 cleanup is queued**: it'll delete the v2.2 model code + retire the legacy service handlers in favour of the new `retrain_models` service. Pure code removal, no behaviour change.

## What landed in code

### New file `custom_components/spot_price_predictor/retrain.py` (~230 LOC)

Public API:

```python
retrain_all(repo_root=None, data_dir=None, layers=None, fingrid_api_key=None) -> dict
retrain_seasonal(repo_root, data_dir) -> dict
retrain_spike_model(repo_root, data_dir) -> dict
retrain_solar_submodel(repo_root, data_dir, fingrid_api_key=None) -> dict
```

Plus `_atomic_write_json` helper and CLI entrypoint (`python -m custom_components.spot_price_predictor.retrain`).

### `__init__.py` additions (~95 LOC)

- New service `SERVICE_RETRAIN_MODELS = "retrain_models"` with `RETRAIN_SCHEMA` voluptuous validator.
- `handle_retrain_models` async handler — runs orchestrator in executor, reloads V26Pipeline, posts notification, fires event.
- Service registered on setup and unregistered on unload.

### `services.yaml` entry

New entry documenting the service signature for the HA UI (Developer Tools → Services dropdown).

### `tests/test_retrain.py` — 9 tests

- Atomic write: creates / overwrites / no-partial-on-failure
- Orchestrator: `ALL_LAYERS` constant, unknown-layer-returns-error, solar-skips-without-key, started/completed timing, `_ensure_repo_imports` idempotence

## Tests

**412 / 412 passing** (403 prior + 9 new retraining tests).

## Files

- **New**: `custom_components/spot_price_predictor/retrain.py` (~230 LOC)
- **New**: `tests/test_retrain.py` (9 tests)
- **New**: `studies/results/V2_8_0_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/__init__.py` (~95 LOC additions for service + handler)
- **Modified**: `custom_components/spot_price_predictor/services.yaml` (new `retrain_models` entry)
- **Modified**: `manifest.json` `2.7.0 → 2.8.0`, `README.md` release-notes index

## Operational impact

| Aspect | Impact |
|---|---|
| Sensor entity_ids | Unchanged |
| Sensor attributes | Unchanged at the keys/types level |
| Forecast accuracy | Unchanged from v2.7.0 (v26 still the primary path) |
| Runtime cost | No change to coordinator cycle. Retraining only runs on demand. |
| Per-retrain cost | ~30 s for seasonal+spike; ~2-5 min for solar (depends on cache state) |
| Storage | Artifacts unchanged in size (~30 KB total) |
| External APIs | No new persistent dependencies. Solar refit fetches Fingrid + Open-Meteo on demand. |

## Suggested workflow

1. **Initial setup**: install v2.8.0 via HACS. The bundled artifacts in `data/` are good; no retraining needed.
2. **Quarterly hygiene**: run `service: spot_price_predictor.retrain_models` (no parameters → all layers).
3. **Drift response**: enable the auto-retrain automation snippet above; the `RefitMonitor` flag drives it.
4. **After a Finnish generation-mix shift** (e.g. new wind farm online, nuclear unit overhaul): manual `retrain_models` call to capture the new structure faster than the quarterly schedule.

## Roadmap

- **v2.9.0** (next): pure cleanup — delete `model.py`, `features.py`, `model_coefs_default.json`, retire legacy `upload_coefficients` / `reset_coefficients` services. ~400 LOC removed. Zero behaviour change at that point.
- **Future**: optional `button.spot_price_predictor_retrain_models` entity that runs the service with one click from the dashboard. Trivial to add when there's demand.
