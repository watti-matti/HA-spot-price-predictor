# v2.3.1 — Doc-only fix: baseload guidance

## TL;DR

**Behaviour did not change.** Only help text and documentation changed.

The v2.3.0 release shipped with help text instructing users to set `baseload_kwh_per_hour` to **non-flexible** consumption only (lighting + fridge + electronics, excluding heat pump / EV / sauna / water heater). **That guidance was wrong** and produces a systematic optimism bias on heat-pump days. v2.3.1 corrects the guidance: baseload should represent the user's **typical TOTAL hourly consumption** (the bill-derived total demand including all loads).

If you configured baseload following the old "non-flex only" advice, please raise it to typical TOTAL hourly consumption (≈ `annual_bill_kWh / 8760`). For a Finnish single-family heat-pump house with 12 000 kWh/yr that's ≈ 1.4 kWh/h instead of the previously-suggested ≈ 0.5 kWh/h.

## Why the original guidance was wrong

The v2.3 stability-invariant section conflated two separate concerns:

1. **What the predictor reads from HA** — must not be an optimizer-influenced entity. Otherwise EMHASS's daily decisions feed back into the next forecast cycle and the loop oscillates. **This is the actual stability requirement.**
2. **What the static configured value represents** — was incorrectly extended to "must exclude flexible loads". Static configuration cannot create feedback regardless of what it represents, because it doesn't update based on observed consumption.

The marginal-cost arithmetic also favours "typical total" by construction. Worked example, sunny noon, 4 kWh PV, heat-pump house with 16 000 kWh/yr typical demand:

| Scenario | baseload | pv_avail | m_h | Behaviour |
|---|---|---|---|---|
| **A: non-flex only (~0.5 kWh/h, the v2.3 advice)** | 0.5 | 3.5 kWh | ≈ 4 c/kWh | Over-optimistic. Forecast claims all PV is free for extra load. EMHASS schedules heat pump there + further loads on top → second load actually pulls 16 c/kWh from grid. **Systematic optimism bias.** |
| **B: typical total (~1.83 kWh/h, the corrected advice)** | ~1.83 | 2.17 kWh | ≈ 4 c/kWh | Self-consistent. Forecast assumes typical demand (heat pump etc.) is happening; EMHASS plans around that; reality matches assumption; equilibrium. |

With PV at only 2 kWh, Case B correctly returns m_h ≈ 14 c/kWh (PV mostly absorbed by typical demand, only 0.17 kWh headroom). Case A returns ~10 c/kWh — still optimistic. The PV/grid ratio that drives the marginal cost is genuinely a function of total demand, not of non-flex demand.

## What changed in v2.3.1

Doc-only patch — no schema change, no behaviour change:

- `config_flow.py` — corrected the comment block above the PV system schema; help text in the options flow continues to describe the field accurately.
- `data/finland.yaml` — replaced the misleading "STABILITY INVARIANT (do not relax in Phase 1)" comment block with the corrected statement plus an explicit v2.3 → v2.3.1 doc-fix note. The default value `baseload_kwh_per_hour: 0.8` is preserved for backwards compatibility but commented as the legacy default and re-tune-per-bill recommendation.
- `README.md` — sensor section's stability-invariant paragraph rewritten with the corrected statement and a doc-fix note.
- `TECHNICAL_GUIDE.md` — the "Stability invariant — open-loop wrt the optimizer" section now includes the worked Case A vs Case B example, the corrected statement, and the v2.3 → v2.3.1 doc-fix note. The configuration table notes the legacy default and gives concrete typical-total values (≈ 1.4 for mid-range Finnish heat-pump house, ≈ 0.5 for apartment without electric heating). The "Out of scope" entry for HA-energy-entity baseload now points forward to v2.4.0 with internal smoothing.
- `TEKNINEN_TOTEUTUS.md` — same updates in Finnish.
- `manifest.json` — version 2.3.0 → 2.3.1.

Test suite: 267 / 267 passing (unchanged from v2.3.0).

## What's coming in v2.4.0

The forthcoming v2.4.0 schema overhaul replaces the three baseload fields (`baseload_kwh_per_hour`, `baseload_day_factor`, `baseload_night_factor`) with two friendlier ones:

- `annual_consumption_kwh` (default 12 000) — total typical annual demand including PV self-consumption and optimizer-controlled loads.
- `consumption_entity` (optional) — any HA consumption sensor (smart-meter counter, daily/monthly `utility_meter`, instantaneous power); the integration auto-detects the sensor type and applies long-window internal smoothing (14- to 28-day) so EMHASS's daily decisions don't propagate back. Hint placeholder `sensor.energy_yesterday`.

Plus a hardcoded Finnish residential monthly seasonal profile (Fingrid Datahub Type 1) so the per-hour baseload reflects winter/summer variation without requiring user-side day/night factor tuning.

v2.3.1 ships now to correct the guidance immediately; the schema overhaul ships when v2.4.0 is ready.
