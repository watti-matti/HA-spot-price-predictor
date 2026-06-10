# v2.11.9-beta.1 — public-launch prep: docs + dashboards (Phase 0+1)

**Pre-release for friendly testers.** Documentation accuracy pass and a
rewrite of the example dashboards so copy-paste cards actually render. No
forecasting/behaviour changes — docs, example YAML, and a new lint test only.

## Phase 0 — documentation accuracy

- README version badge `2.11.0 → 2.11.8`; added a "Recent changes
  (v2.11.3 → v2.11.8)" summary.
- Documented that **self-consumed PV is free** (`effective_eur_kwh` floors
  at 0 since v2.11.4).
- README + TECHNICAL_GUIDE: DtACI is **FI-only** (the SE1/SE3/EE bundles
  were removed in v2.11.8); warm-up wording standardised to ≈ 5–7 days.
- `docs/dtaci_layer.md`: removed the stale "4-zone production scope" claim;
  flagged the legacy-v1 section as reference-only.
- TECHNICAL_GUIDE header `v2.11.0 → v2.11.8`.

## Phase 1 — example dashboards (the launch blocker)

The shipped example cards used apexcharts-card anti-patterns that render
nothing. Fixed across `docs/yaml_examples/forecast_v2_11_dashboard.yaml`
and `docs/yaml_examples/dtaci_diagnostics_card.yaml`:

- `{x, y}` object data points → `[x, y]` arrays.
- `xaxis.type: category` k-axes (don't render in this time-series-only
  card) → k mapped onto today's 24 hours with the axis/tooltip relabelled
  to "k".
- D(k) cards now select **today** by date (were using `days[0]`, the
  historical day).
- CVaR strip → datetime axis with `[timestamp, value]` arrays.
- DtACI diagnostics: removed the loop over the deleted `se1/se3/ee` zones
  (now iterates whatever zones exist — FI only).
- INSTALLATION Step 9 now points at the current
  `forecast_v2_11_dashboard.yaml` (full, PV-aware) and labels
  `ha_dashboard.yaml` as the lightweight starter.

## New guard — `tests/test_dashboard_examples.py`

Lints every shipped dashboard for the four recurring apexcharts-card
gotchas (bare `entity.<attr>`, `{x,y}` object returns, `type: category`
axes, removed DtACI zones) and validates the YAML — so these regressions
can't ship again.

## Not yet done (later phases, tracked for the public launch)

- Phase 2: install-doc completeness (`consumption_profile_entity` setup,
  external-PV payload examples, cold-start accuracy note, expanded
  troubleshooting, privacy + support sections).
- Phase 3: genericise `docs/household_profile_schema.md` example (currently
  uses the author's real 8.91 kWp / 160° system).
- Phase 4: CHANGELOG, issue template, SUPPORT.md, screenshots refresh.

## Releasing / testing

This is a **GitHub pre-release**; `main` stays on stable **2.11.8**. Testers
with "show beta versions" enabled in HACS get `2.11.9-beta.1`. Promote to a
stable `2.11.9` (fast-forward main) once validated.

## Test status

`python -m pytest tests/` → 515 passed, 5 skipped.
