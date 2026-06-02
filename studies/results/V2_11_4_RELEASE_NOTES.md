# v2.11.4 — self-consumed PV valued as free in effective price / PV D(k)

Patch release on top of v2.11.3. Changes the PV-value convention used by
`effective_eur_kwh` (and therefore the PV-aware duration curves
`dk_cheap_pv_eur_kwh` / `dk_peak_pv_eur_kwh`). No schema changes.

## What changed

Previously the marginal effective price valued the PV-served share of an
additional kWh at the **export opportunity cost** (`sell` price): consuming
your own solar was treated as "giving up" the export revenue, so in a
high-spot hour `effective_eur_kwh` stayed at roughly the spot level even
when PV fully covered the load.

That conflicts with household economics: **on-site PV carries no spot
price, no transmission tariff, and no energy tax.** A kWh you self-consume
is free.

New convention (`marginal_effective_eur_kwh` in `pv_estimate.py`):

```
pv_avail     = max(0, pv − baseload)
from_pv      = min(1, pv_avail)
from_grid    = 1 − from_pv
pv_unit_cost = min(0, sell)          # self-consumed PV is free
effective    = from_pv · pv_unit_cost + from_grid · buy
```

* The PV-served share costs **0** instead of the (positive) sell price.
* It only goes **negative** when the export price itself is negative
  (deep oversupply, you would pay to export): self-consuming then avoids
  the export penalty, a genuine saving worth `sell` (< 0).
* The grid-served share still costs the full consumer buy price.

Result: `effective_eur_kwh ∈ [min(0, sell), buy]`. When surplus PV can
serve the extra load, the effective price is **≤ 0**, as expected.

Unchanged invariant: when `pv ≤ baseload` (night / low sun), there is no
surplus, so `effective_eur_kwh == consumer_eur_kwh` — exactly as before.

## What is NOT changed

* `net_household_cost_eur` — already treats self-consumed PV as free and
  is the realized hourly euro cost. Unchanged.
* `consumer_eur_kwh`, `sell_eur_kwh`, `spot_eur_mwh`, percentiles, the
  prediction pipeline, and all entity/attribute schemas.
* The night-time `effective == consumer` identity.

## Downstream effect

The PV-aware duration curves (`dk_cheap_pv_eur_kwh` /
`dk_peak_pv_eur_kwh`) are order statistics over `effective_eur_kwh`, so
they now reflect free self-consumed PV automatically — high-PV hours
become the genuinely cheapest hours for flexible load, which is the
intended scheduling signal.

## Files changed

| File | Change |
|---|---|
| `custom_components/spot_price_predictor/pv_estimate.py` | `marginal_effective_eur_kwh`: PV-served share valued at `min(0, sell)` (free), not `sell` |
| `custom_components/spot_price_predictor/manifest.json` | `version` 2.11.3 → 2.11.4 |
| `custom_components/spot_price_predictor/sensor.py` | `sw_version` 2.11.3 → 2.11.4 |
| `tests/test_pv_estimate.py` | Updated full-cover / partial / bound tests; added free-PV regardless-of-export-price tests |
| `tests/test_coordinator_pv.py` | End-to-end bound updated to `[min(0, sell), buy]` |
| `tests/test_release_data_consistency.py` | Free-PV effective bound in the validator |

## Migration

None required. HACS auto-update; existing configs work unchanged. Only the
PV-aware effective price and PV duration curves change in value; the grid
(non-PV) curves are identical.

## Test status

`python -m pytest tests/` → 486 passed, 5 skipped.
