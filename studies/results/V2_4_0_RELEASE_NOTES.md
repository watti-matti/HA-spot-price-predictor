# v2.4.0 — Friendlier baseload UX with `annual_consumption_kwh`

## Why v2.4

The v2.3.x baseload schema (`baseload_kwh_per_hour` + `baseload_day_factor` + `baseload_night_factor`) asked the user to do conversions and pick arbitrary day/night splits. Almost no Finnish HA user can tell you their baseload in kW; everyone knows their **annual kWh** — it's on the bill, on every contract offer, in every monitoring app.

v2.4 replaces those three fields with two friendlier ones, and adds optional HA-sensor-driven smoothing for users who want accuracy without manual tuning.

## What's new

### Configuration schema

Two fields replace the three v2.3 fields:

- **`annual_consumption_kwh`** (default `12 000`) — the user's typical TOTAL annual household demand from the electricity bill, including PV self-consumption AND optimizer-controlled loads (heat pump, EV, sauna, water heater). Bill-derived single number. The integration multiplies by a built-in Finnish residential monthly seasonal profile to get per-hour baseload. No day/night split — that's the optimizer's domain.

- **`consumption_entity`** (optional, e.g. `sensor.energy_yesterday`) — any HA consumption sensor. The integration auto-detects the sensor type and applies internal long-window smoothing (14-day rolling average + 5 % hysteresis), so EMHASS's daily scheduling decisions don't propagate back into the forecast. No need to build a `filter:` template yourself.

### Auto-detection of sensor types

When `consumption_entity` is set, the integration handles smoothing internally based on sensor type:

| Detected type | Detection (HA attrs) | Strategy |
|---|---|---|
| Cumulative-kWh counter | `unit=kWh`, `state_class=total_increasing` | 14-day delta ÷ 14 → daily kWh |
| Daily / monthly `utility_meter` | `state_class=total` with cycle | Recorder history, 14 daily totals averaged |
| Instantaneous power | `unit=W` or `kW`, `device_class=power` | `statistics_during_period(28d, mean) × 24` |
| Unknown | (fallback) | Silent fallback to `annual_consumption_kwh` config; warn |

Smoothed value cached in `.storage/spot_price_predictor_consumption_cache.json`, recomputed at most once per day. 5 % hysteresis dead-band on the cached value prevents minor sensor noise from re-triggering coordinator updates.

### Finnish residential monthly seasonal profile

Hardcoded in `const.py` as `FINLAND_RESIDENTIAL_MONTHLY_FACTORS`:

```
Jan: 1.18   Feb: 1.12   Mar: 1.08   Apr: 1.00
May: 0.92   Jun: 0.85   Jul: 0.80   Aug: 0.85
Sep: 0.93   Oct: 1.02   Nov: 1.10   Dec: 1.15
```

Sum = 12.00 exactly (normalization invariant). Variation ≈ ±19 % around the mean — characteristic of Finnish 60°N latitude where lighting load drives the strong winter peak and vacation/long-day combination drives the July trough.

Source: literature-derived from Finnish residential load profile research (VTT Publications 289 "Load research and load estimation in electricity distribution", Adato Energia DSO standard load profiles "tyyppikäyrät", Statistics Finland "Energy consumption in households" survey 2024). Calibrated to the published BE03 (residential, no electric heating) shape.

**TODO** in a future v2.4.x patch: replace with verbatim values from Fingrid Open Data dataset #360 (BE03 typing curve) — the official Finnish standard used by every DSO for unmetered residential billing. Requires the same Fingrid API key the integration already uses for nuclear data.

### Stability under the v2.4 schema

- **Default mode** (`consumption_entity = ""`): `baseload(h)` is a deterministic function of `(annual_consumption_kwh, h)` only — no HA entity reads, fully open-loop. Identical safety property to v2.3.

- **HA-sensor mode**: 14-day rolling average dampens a single-day perturbation to `1/14 ≈ 7 %`. Combined with the 5 % hysteresis dead-band, EMHASS rescheduling a 5 kWh load between days produces `5/14 ≈ 0.36 kWh` rolling change, only ~3 % of a 12 kWh/day baseline — within the dead-band, so the cached baseload value doesn't move and EMHASS sees a stable forecast.

The new `tests/test_baseload_v24.py` covers the monthly factor invariants (sum = 12.00 exactly, winter peak / summer trough), the marginal-cost formula at typical Finnish values, the v2.3 → v2.4 migration, and the stability proof: a single-day perturbation cannot escape the hysteresis dead-band when smoothed over 14 days.

## Migration from v2.3.x

Automatic. When a config entry only carries the legacy `baseload_kwh_per_hour` field, the coordinator's `__init__` infers the equivalent annual value:

```
inferred_annual_kwh = baseload_kwh_per_hour
                    × ((day_factor × 15 + night_factor × 9) / 24)
                    × 8760
```

and logs an INFO line. The legacy fields stay in `entry.data` untouched until the user opens the Options dialog and re-saves, at which point they are dropped cleanly.

A user who configured the v2.3.0 default (`baseload_kwh_per_hour = 0.8`, day_factor = 1.2, night_factor = 0.7) gets `inferred_annual_kwh ≈ 7660 kWh/yr` — clearly low for a typical Finnish heat-pump house, prompting them to re-tune to their actual bill.

## Test suite

**280 / 280 tests passing** (13 new in `test_baseload_v24.py` for v2.4: monthly factor invariants, baseload formula, migration logic, hysteresis behaviour, smoothing window stability proof).

## Files changed

- `custom_components/spot_price_predictor/const.py` — add `CONF_ANNUAL_CONSUMPTION_KWH`, `CONF_CONSUMPTION_ENTITY`, defaults, `CONSUMPTION_SMOOTHING_DAYS`, `CONSUMPTION_HYSTERESIS_PCT`, `FINLAND_RESIDENTIAL_MONTHLY_FACTORS` with normalization assertion. Legacy v2.3 baseload constants kept for backwards compatibility.
- `custom_components/spot_price_predictor/coordinator.py` — rewrite `_resolve_baseload(ts)` to use annual + monthly factor; add `_smooth_consumption_entity()`, `_fetch_consumption_daily_kwh()`, `_smooth_kwh_counter()`, `_smooth_power_sensor()`, `_get_local_dt()`. Add v2.3 → v2.4 migration in `__init__`.
- `custom_components/spot_price_predictor/config_flow.py` — replace 3 baseload form fields with 2 in both new-install step and Options flow. Drop legacy fields when user re-saves Options. Add `_infer_legacy_annual_kwh()` helper for default values during migration.
- `custom_components/spot_price_predictor/data/finland.yaml` — add v2.4 fields, document the migration path, keep legacy fields for backwards compatibility.
- `manifest.json` — version `2.3.1 → 2.4.0`.
- `tests/test_baseload_v24.py` — new, 13 tests.
- `README.md`, `TECHNICAL_GUIDE.md`, `TEKNINEN_TOTEUTUS.md` — document the new schema, smoothing behaviour, migration path, and the worked stability calculation.

## What's still on the roadmap

- Replace literature-derived `FINLAND_RESIDENTIAL_MONTHLY_FACTORS` with verbatim values from Fingrid Open Data dataset #360 BE03 (v2.4.x patch).
- Battery storage (adds a temporal state variable; needs DP / MILP).
- Capacitated water-filling D(k) for very large flexible loads.
- Per-tilt second Open-Meteo fetch for non-default panel orientations.
