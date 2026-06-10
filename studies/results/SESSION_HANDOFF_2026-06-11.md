# Session handoff — 2026-06-11

Working context for resuming the public-launch prep. Records what shipped,
what's staged but unreleased, and the open decisions.

## TL;DR state

- **`main` = `15ff79a` = stable v2.11.8.** All v2.11.3–v2.11.8 work is
  released and on main.
- **Beta branch `claude/dreamy-kapitsa-a7693a` = `ab66dc8`** (ahead of
  `v2.11.9-beta.1` tag `cff1d7a` by 6 dashboard commits — these are the
  **uncut `2.11.9-beta.2` candidate**).
- **`v2.11.9-beta.1`** is a published **GitHub pre-release** (manifest
  `2.11.9-beta.1`); main intentionally stays on stable 2.11.8 until the
  beta is validated, then promote to stable `2.11.9` (fast-forward main).

## Released today (all on main unless noted)

| Tag | What |
|---|---|
| v2.11.3 | Fix stale PV-aware prices after pipeline overwrite (`effective_eur_kwh`/`net_household_cost_eur`/`sell_eur_kwh`); PV D(k) now covers *today* (reconstructed from history); added `tests/test_release_data_consistency.py`. |
| v2.11.4 | Self-consumed PV valued as **free**; `effective_eur_kwh` floors at 0 (`min(0, sell)` on the PV-served share). |
| v2.11.5 | Fix weather/PV **time alignment** — Open-Meteo grid starts 00:00 UTC but was indexed as `now`, shifting solar/PV by `now.hour`. Added `_align_weather_to_now` + timestamp-aware external-PV reader. |
| v2.11.6 | Fix spurious **12/13 seam** in DtACI per-k bias/coverage (12-level v1 state reloaded next to cold 13-24); bumped bundle `SCHEMA_VERSION = 2`, reject v1 → cold-start. |
| v2.11.7 | (separate PR #7) shared WM hexagon brand icon. Took the 2.11.7 number, forcing the DtACI de-scope to 2.11.8. |
| v2.11.8 | **De-scope DtACI to FI** (`DTACI_ZONES = ("fi",)`); remove dead SE1/SE3/EE bundles + stale state files. Neighbour PRICES still feed the FI model as features (`Y_se*`, `ar_se*`). |
| v2.11.9-beta.1 | **Pre-release.** Launch-prep Phase 0+1: doc accuracy + rewrote broken example dashboards + `tests/test_dashboard_examples.py` lint guard. |

## Staged on branch but NOT yet released (→ beta.2)

6 commits on top of beta.1 (all dashboards, all lint+YAML green):

- `77b62f2` — add the user's **7-day duration overview** card (cheap/peak + PV-compensated) to `forecast_v2_11_dashboard.yaml`; dropped duplicate Average(24).
- `4b15c19` — 10 c/kWh annotation anchor lines (correct nesting under `apex_config.annotations.yaxis`).
- `4679be2` — DtACI card: **plain-language status panel, defensively FI-only** (uses `dtaci_fi_*` scalars so neighbour zones can't appear even pre-2.11.8).
- `91f62ba`/`db7a3b5`/`ab66dc8` — daily x-axis label centering attempts → **reverted to responsive default** (offsetX/tickAmount/function-min-max all rejected as non-responsive or unsupported).

## OPEN ITEMS (next session)

1. **Plotly daily card** — provided a `plotly-graph-card` version of the
   7-day duration chart **in chat (NOT committed)** for the user to test on
   PC / folded phone (small+large). It uses a category x-axis (responsive
   centered labels) + `yaxis.dtick: 10` (native 10 c gridlines). Awaiting
   the user's render test. If good → add as
   `docs/yaml_examples/forecast_duration_plotly.yaml` + note the
   plotly-graph-card dependency in INSTALLATION; keep apexcharts as the
   no-dependency default. Verify `$ex` scope (`entity.attributes...` vs
   `hass.states[...]`).
2. **CVaR card title/units** — user likes "7-day CVaR95 price outlook";
   decide whether to rename the repo card and whether to keep **c/kWh**
   (recommended, readable) vs €/kWh.
3. **Ship `2.11.9-beta.2`** once 1–2 settle (bump manifest/sensor to
   `2.11.9-beta.2`, tag pre-release off branch, write notes).
4. **Phases 2–4 of the launch plan (not started):**
   - Phase 2: INSTALLATION completeness — `consumption_profile_entity`
     setup + provenance table, external-PV payload examples + silent-
     fallback note, cold-start accuracy note, expanded troubleshooting,
     Finland-only + privacy + support sections.
   - Phase 3: genericise `docs/household_profile_schema.md` (currently the
     author's real **8.91 kWp / 160° / Tampere** values) + disclaimer;
     optionally scrub `studies/results/pv_adjusted_price_plan.md`.
   - Phase 4: `CHANGELOG.md`, GitHub issue template, `SUPPORT.md`,
     refreshed screenshots; promote beta → stable `2.11.9` on main.

## Key learnings / guardrails (so we don't relitigate)

- **apexcharts-card is time-series only.** Data generators must return
  `[x, y]` arrays (not `{x,y}` objects); `xaxis.type: category` and
  numeric/k axes don't render → map index→timestamps. `xaxis.min/max`
  **cannot be functions** (only yaxis); `tickAmount` is ignored on datetime
  axes; `labels.offsetX` is pixel-hardcoded (not responsive). These are
  enforced by `tests/test_dashboard_examples.py`.
- Always read attrs via `entity.attributes.<name>`.
- PV-aware D(k) / CVaR are absent on historical/"today-partial" days →
  filter null before indexing.
- DtACI is FI-only; the per-D(i) bundle does NOT feed the model. Neighbour
  calibration's estimated FI MAE benefit is ~1 % (analytical;
  `studies/_dtaci_dk_*_prices_cache.json` needed for a real ablation, not
  in repo).

## Repo facts

- Releases go to `main` via fast-forward; betas tagged off the branch as
  GitHub pre-releases (don't move main).
- Full suite at branch HEAD: **515 passed, 5 skipped** (incl. 20 dashboard
  lint checks). Version strings live in `manifest.json` + `sensor.py`
  (`sw_version`).
