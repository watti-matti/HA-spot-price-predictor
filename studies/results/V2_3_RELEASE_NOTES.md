# v2.3.0 — PV-aware duration forecasts

## Why v2.3

Until v2.2 the duration curves `dk_cheap` / `dk_peak` were grid-only — they answered _"what does it cost to run k hours of load at retail prices?"_. For the growing share of households with rooftop PV that question is wrong: a sunny midday hour costs the **opportunity cost of foregone export**, not the retail buy price. v2.3 closes that gap by layering an optional PV-aware effective price (and parallel D(k) curves) on top of the unchanged v2.2 spot-price model — without retraining anything.

## What's new

### Marginal effective price per hour

When `pv_capacity_kwp > 0`, each forecast hour gains:

```
pv_avail_h = max(0, p_h − baseload_h)         # PV surplus on top of baseload
from_pv    = min(1, pv_avail_h)               # share of new 1 kWh from PV
m_h        = from_pv · s_h + (1 − from_pv) · b_h
```

`m_h` is the cost of running 1 additional kWh of flexible load at hour `h`. It is bounded analytically by `[s_h, b_h]` (sell ≤ effective ≤ buy) — no baseload-divisor pathology, no extreme tails. `s_h` is **not** clipped at zero, so deep oversupply (negative spot + high PV → user pays to export) propagates correctly.

### PV-aware D(k) cheap/peak

Each daily forecast entry now also exposes `dk_cheap_pv_eur_kwh[12]` and `dk_peak_pv_eur_kwh[12]`, computed by sorting the day's 24 `effective_eur_kwh` values and applying the same `compute_dk_cheap_peak` reduction as the grid-only curves. The duration sensor surfaces convenience scalars `today_cheap_pv_{1,4,8,12}h_eur_kwh` and the peak counterparts.

**Validated on 4 years of real Finnish data** (1,460 complete days, 5 kWp / 1 kWh-h baseload reference): zero PAVA-monotonicity violations, D(1) mean +6.90 c/kWh with std 6.0 c/kWh — bounded, realistic, optimization-ready. Annual savings vs grid-only D(4) ≈ 600 EUR/yr for a typical 5 kWp system.

### Optional PV input — basic params in the HA config UI

A new "PV system" step asks for the same four numbers Forecast.Solar uses:

| Field | Default | Notes |
|---|---|---|
| `pv_capacity_kwp` | 0 (off) | Set > 0 to enable PV-aware outputs |
| `pv_tilt_deg` | 45 | Matches Open-Meteo's existing fetch tilt |
| `pv_azimuth_deg` | 180 | 0 = N, 90 = E, 180 = S, 270 = W |
| `pv_system_efficiency` | 0.85 | DC/AC + soiling + losses, lumped |
| `pv_export_grid_fee` | 0 | Optional extra €/kWh on export, above seller commission |
| `baseload_kwh_per_hour` | 0.8 | Non-flexible household consumption — see invariant below |
| `baseload_day_factor` | 1.2 | 07–22 multiplier |
| `baseload_night_factor` | 0.7 | 22–07 multiplier |

Internal estimator uses Open-Meteo's `global_tilted_irradiance_instant` × kWp × tilt/azimuth correction × efficiency, capped at the physical ceiling `kWp · efficiency`. Free, 7-day horizon, no rate limit.

### Source-agnostic external PV entity

Set `pv_external_entity` to any HA sensor that publishes one of four common attribute conventions; the reader auto-detects:

| Convention | Attribute | Shape | Unit | Conversion |
|---|---|---|---|---|
| Generic forecast list | `forecast` | `list[dict]` | kWh | direct (keys: `pv_kwh`, `kwh`, `energy`, `value`) |
| Forecast.Solar Wh dict | `wh_hours` | `dict {ISO ts → number}` | Wh | `/ 1000` |
| Forecast.Solar W dict | `watts` | `dict {ISO ts → number}` | W | `/ 1000` |
| EMHASS template list | `irradiance` | `list[number]` | W or kWh | magnitude > 50 → assume W and `/ 1000`; else kWh |

Use this if you have multi-array setups, want shading-corrected values from Forecast.Solar's `horizon` parameter, or already maintain a custom Open-Meteo template. Values are clamped per-element to `[0, kWp · efficiency]` to defend against template errors. Silent fallback to the internal estimator if no convention matches.

## Stability invariant — open-loop wrt the optimizer

A coupled forecaster ↔ optimizer system can oscillate: optimizer schedules into cheap hours → those hours' baseload rises → next forecast shifts the cheap window → schedule chases. To prevent this, **the price forecaster must remain a deterministic function of `(spot, weather, PV config, baseload config)`**, with no entity reads that reflect the optimizer's own decisions.

v2.3 enforces this by construction:

- `_resolve_baseload(ts)` MUST NOT call `hass.states.get` or any HA entity-read API — verified by a grep test on coordinator source (`tests/test_coordinator_pv.py`).
- The configured `baseload_kwh_per_hour` represents **non-flexible** consumption only. Heat pump, EV, sauna, and other optimizer-controlled loads are excluded by contract.
- `_read_external_pv_forecast()` MAY read an HA entity because PV is weather-driven and independent of optimizer decisions — no feedback path is created.

A future Phase 2 may allow an opt-in HA energy-sensor baseload with explicit warnings; v2.3 deliberately does not.

## Backwards compatibility

`pv_capacity_kwp = 0` (default) and empty `pv_external_entity` → all PV-aware attributes are silently omitted and the integration produces **byte-identical** outputs to v2.2.0. Existing dashboards, automations, and downstream consumers continue to work without changes.

The legacy `dk_consumer_eur_kwh[24]` and `dk_spot_eur_mwh[24]` arrays are still emitted for one transition release alongside the v2.2 cheap/peak split. Migration guide: [`docs/dk_cheap_peak_migration.md`](../../docs/dk_cheap_peak_migration.md).

## Sensor API additions

### `sensor.price_forecast`

When PV is configured, every `forecast[i]` entry gains:

| Attribute | Type | Description |
|---|---|---|
| `pv_production_kwh` | float | Estimated hourly PV output |
| `baseload_kwh` | float | Configured non-flexible consumption for the hour |
| `effective_eur_kwh` | float | Marginal cost `m_h`, bounded in `[s_h, b_h]` |
| `net_household_cost_eur` | float | Informational raw EUR/h flow (export revenue ↔ grid import) |
| `is_export_hour` | bool | True when PV exceeds baseload |
| `sell_eur_kwh` | float | Sell price (can be negative during deep oversupply) |

Plus top-level: `current_effective_eur_kwh`, `pv_capacity_kwp`, `pv_source` (`internal` / `external` / `disabled`), `baseload_kwh_per_hour`, `week_min/avg/max_effective_eur_kwh`.

### `sensor.duration_forecast`

When PV is configured, every `daily_forecast[i]` entry gains:

| Attribute | Shape | Unit | Description |
|---|---|---|---|
| `dk_cheap_pv_eur_kwh` | `float[12]` | EUR/kWh | PV-aware D(k) cheap, k = 1..12 |
| `dk_peak_pv_eur_kwh` | `float[12]` | EUR/kWh | PV-aware D(k) peak, k = 1..12 |

Plus convenience top-level scalars `today_cheap_pv_{1,4,8,12}h_eur_kwh` and `today_peak_pv_{1,4,8,12}h_eur_kwh`.

## Tests

- `tests/test_pv_estimate.py` — 20 tests covering PV physics (zero / ceiling / tilt / azimuth / efficiency), marginal-cost boundary cases (no PV, partial cover, full self-consumption, net export, negative-sell liability), and the analytical bound `m_h ∈ [s_h, b_h]` on randomized inputs.
- `tests/test_coordinator_pv.py` — 13 tests including the stability-invariant grep on coordinator source, all four external-entity format auto-detections, ceiling clamping, D(k) monotonicity on synthetic data, and an end-to-end synthetic Finnish summer day with bound checks.
- Full suite: **267 / 267 passing**, 33 net-new tests on top of the v2.2 baseline.

## Out of scope (Phase 1)

Deferred to a future release:

- Battery storage (adds a temporal state variable; needs DP / MILP).
- Capacitated water-filling D(k) for very large flexible loads.
- HA energy-entity-driven baseload (opt-in only, with stability warnings).
- Per-tilt second Open-Meteo fetch.
- Spot-model retraining with PV (model unchanged; PV is a post-prediction transform).

## Migration

No action required for existing users — defaults preserve v2.2 behaviour exactly. To enable PV awareness, edit the integration's options in **Settings → Devices & Services → Spot Price** and fill in the PV system fields; the new sensor attributes start appearing on the next coordinator update.
