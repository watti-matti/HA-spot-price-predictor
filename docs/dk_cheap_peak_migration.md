# D(k) Cheap/Peak Schema Migration Guide

**Status:** Phase A migration of the Duration Forecast sensor schema. The
sensor still emits the legacy `dk_consumer_eur_kwh[24]` / `dk_spot_eur_mwh[24]`
arrays for **one transition release** while downstream consumers move to
the new cheap/peak split.

## Why split

The legacy single-array D(k) was `dk[k-1] = mean of the cheapest k hours`,
k = 1..24. That definition produces a sensible cumulative curve for
k = 1..12, but for k = 13..24 the values smoothly approach the daily
average and carry no decision-relevant signal — every entry is just
"daily mean shifted by a few percent." Half of the array is wasted.

The cheap/peak split keeps both halves meaningful:

```
dk_cheap[k-1]  = mean price of the cheapest k hours,    k = 1..12
                 (monotone non-decreasing in k)
dk_peak [k-1]  = mean price of the priciest k hours,    k = 1..12
                 (monotone non-increasing in k)
```

Both arrays are CVaR-equivalent — `dk_cheap[k-1]` is CVaR at α = k/24 of
the lower tail of the daily price distribution, `dk_peak[k-1]` is CVaR
at α = k/24 of the upper tail. Together they cover the entire
decision-relevant span:

- Thermal optimization / deferrable-load scheduling reads `dk_cheap` to
  estimate "expected cost if I shift k hours into the cheapest slots."
- Risk-aware planning / storage depletion reads `dk_peak` to estimate
  "worst case if I'm forced to run during k peak hours."

## Sum identity (cross-check)

```
dk_cheap[11] + dk_peak[11] = 2 × daily_average
```

Because the cheapest 12 hours and the priciest 12 hours partition the
24-hour day, their averages sum to twice the daily mean. This identity
is exact (to numerical noise) and lets any consumer cross-check that
the two arrays are jointly consistent without ever seeing the underlying
hourly prices.

## Sensor attributes (new)

The `sensor.spot_price_predictor_duration_forecast` entity now emits
both schemas side-by-side. Each entry in `daily_forecast[]` carries:

| Attribute (per day)                     | Type      | Status     |
| --------------------------------------- | --------- | ---------- |
| `date`, `weekday`, `source`             | string    | unchanged  |
| `dk_cheap_eur_kwh`                      | float[12] | **new**    |
| `dk_peak_eur_kwh`                       | float[12] | **new**    |
| `dk_cheap_spot_eur_mwh`                 | float[12] | **new**    |
| `dk_peak_spot_eur_mwh`                  | float[12] | **new**    |
| `dk_consumer_eur_kwh`                   | float[24] | deprecated |
| `dk_spot_eur_mwh`                       | float[24] | deprecated |

Top-level convenience scalars (for templates and Lovelace badges):

| Attribute                  | Description                                  |
| -------------------------- | -------------------------------------------- |
| `today_cheap_1h_eur_kwh`   | Today's `dk_cheap_eur_kwh[0]`                |
| `today_cheap_4h_eur_kwh`   | Today's `dk_cheap_eur_kwh[3]` (sensor state) |
| `today_cheap_8h_eur_kwh`   | Today's `dk_cheap_eur_kwh[7]`                |
| `today_cheap_12h_eur_kwh`  | Today's `dk_cheap_eur_kwh[11]`               |
| `today_peak_1h_eur_kwh`    | Today's `dk_peak_eur_kwh[0]`  (most expensive 1h) |
| `today_peak_4h_eur_kwh`    | Today's `dk_peak_eur_kwh[3]`  (mean of priciest 4h) |
| `today_peak_8h_eur_kwh`    | Today's `dk_peak_eur_kwh[7]`                 |
| `today_peak_12h_eur_kwh`   | Today's `dk_peak_eur_kwh[11]`                |

Sensor `state` is now `dk_cheap_eur_kwh[3]` (cheapest 4h average,
EUR/kWh) — the single most-used D(k) value for thermal scheduling. The
fallback to legacy `dk_consumer_eur_kwh[3]` happens automatically if the
new array is missing (e.g., immediately after upgrade before the next
coordinator refresh).

## Migration paths for downstream consumers

### Pattern 1: Read either schema, prefer new

```jinja2
{# Home Assistant template — graceful fallback #}
{% set day = state_attr('sensor.spot_price_predictor_duration_forecast',
                        'daily_forecast')[0] %}
{% set cheap = day.get('dk_cheap_eur_kwh', day.get('dk_consumer_eur_kwh', [])) %}
{{ cheap[3] if cheap | length >= 4 else 0 }}
```

```javascript
// Lovelace data_generator — same pattern
const arr = d.dk_cheap_eur_kwh || d.dk_consumer_eur_kwh;
return [new Date(d.date + 'T12:00:00').getTime(),
        (arr && arr[3] != null) ? arr[3] * 100 : null];
```

### Pattern 2: Reconstruct legacy from new

If a consumer hard-codes the legacy 24-array shape (e.g., a Python
script that builds a `[24][n_days]` matrix), it can rebuild that matrix
from cheap+peak using:

```python
def synthesize_legacy_24(cheap: list[float], peak: list[float]) -> list[float]:
    """cheap[12], peak[12] → legacy cumulative-ascending D(k)[24].

    legacy[k-1] = mean of cheapest k hours of the day, k=1..24.
    """
    assert len(cheap) == 12 and len(peak) == 12
    total_sum = 12.0 * (cheap[11] + peak[11])  # 24 × daily mean
    out = list(cheap)  # k=1..12 → indices 0..11 (identical to legacy)
    for k in range(13, 25):
        j = 24 - k  # priciest j hours, j = 11 .. 0
        if j == 0:
            out.append(total_sum / 24.0)         # daily mean
        else:
            out.append((total_sum - j * peak[j - 1]) / k)
    return out
```

This recovers the legacy array **exactly** to numerical noise (verified
in `tests/test_dk_consumers.py:test_split_to_legacy_24_array_round_trip_exact`).

### Pattern 3: Use the dedicated utility

For Python consumers (training pipeline, dashboards, validation):

```python
from src.dk_utils import (
    compute_dk_cheap_peak,        # 24 hourly → (cheap[12], peak[12])
    reconstruct_sorted_prices,     # (cheap[12], peak[12]) → sorted hourly
    is_monotone_cheap,             # validation
    is_monotone_peak,              # validation
)
```

The HA component ships a self-contained mirror at
`custom_components/spot_price_predictor/dk_utils.py` (it cannot import
from `src/`). Both modules are tested for byte-equivalence in
`tests/test_dk_consumers.py:test_ha_mirror_matches_src_compute_dk_cheap_peak`.

## Thermal optimizer (`DkForecast`) changes

The `multi_load_ha_integration.DkForecast` dataclass gained four new
fields with sensible defaults:

```python
@dataclass
class DkForecast:
    # Existing legacy fields (unchanged)
    dk_eur_kwh: list[list[float]]   # [24][n_days]
    dk_ckwh:    list[list[float]]   # [24][n_days]
    dk_spot:    list[list[float]]   # [24][n_days]
    dates:      list[str]
    weekdays:   list[str]
    n_days:     int

    # Phase A additions (default = empty list when sensor has not been upgraded)
    dk_cheap_eur_kwh: list[list[float]] = field(default_factory=list)  # [12][n_days]
    dk_peak_eur_kwh:  list[list[float]] = field(default_factory=list)  # [12][n_days]
    dk_cheap_spot:    list[list[float]] = field(default_factory=list)  # [12][n_days]
    dk_peak_spot:     list[list[float]] = field(default_factory=list)  # [12][n_days]
```

New methods:

| Method                          | Returns                              |
| ------------------------------- | ------------------------------------ |
| `cheap(k, day, unit)`           | D_cheap(k) for that day, with legacy fallback |
| `peak(k, day, unit)`            | D_peak(k) for that day, NaN if absent |
| `has_peak()`                    | True if peak array is populated      |
| `compute_load_peak_cost(...)`   | Worst-case cost for a deferrable load |

The existing `d(k, day, unit)` (legacy cumulative D(k)) continues to
work unchanged. `fetch_dk_forecast()` now accepts either schema:

- If only legacy is present → reads it, leaves cheap/peak fields empty;
  consumers calling `cheap(k, day)` get the legacy fallback automatically.
- If only new is present → reads it, **synthesizes** the legacy 24-array
  via the formula above so old consumers continue to work.
- If both are present → both are populated directly.

## Lovelace card changes

`lovelace/electricity_forecast_card.yaml` (in the
thermal-energy-optimization repo) replaced the 24 monotone-cumulative
D(k) series with two compact groups:

- **Cheap end** (cool colors, solid lines): D_cheap(1), D_cheap(4),
  D_cheap(8), D_cheap(12).
- **Peak end** (warm colors, dashed lines): D_peak(12), D_peak(8),
  D_peak(4), D_peak(1).

Each generator falls back to legacy `dk_consumer_eur_kwh[k-1]` for the
cheap end if the new attribute is missing; peak series render empty
during the first refresh after upgrade.

## Test coverage

| Test file                       | Coverage                                  |
| ------------------------------- | ----------------------------------------- |
| `tests/test_dk_utils.py`        | 10 tests — utility correctness, monotonicity, identities, round-trip |
| `tests/test_dk_consumers.py`    | 15 tests — HA mirror parity, sum identity, legacy synthesis exactness, sensor state fallback, load cost contracts |

Run with `pytest tests/test_dk_utils.py tests/test_dk_consumers.py -q`.

## Removal timeline

The legacy `dk_consumer_eur_kwh[24]` and `dk_spot_eur_mwh[24]` attributes
will be removed in a future release **after** all known downstream
consumers (thermal optimizer, dashboards, third-party Lovelace cards,
user automations) have been updated. The minimum support window is one
release cycle.

If you maintain a downstream consumer, the migration path is:

1. Now: keep reading legacy; new attributes are present but unused.
2. Next release: switch to `dk_cheap_eur_kwh` for scheduling, add
   `dk_peak_eur_kwh` for risk-aware features. Leave legacy as fallback.
3. Future release: drop legacy fallback. The legacy attributes will
   stop being emitted around this point.
