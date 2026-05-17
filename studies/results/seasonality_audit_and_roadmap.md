# Seasonality audit + v2.5.3 → v2.6.0 roadmap

**Trigger**: user directive (2026-05-17) — *"In our earlier model we used to have different fitted model that try to compensate mechanisms (cos()-fitting etc.) that would be a proper seasonal. What should be used as direct input. Update PV production prediction model based on clear sky model that has represents deterministic characteristics for clear sky production and implement open-meteo clear data for appropriate signal for cloudiness to modulate the clear sky data. Solar production model should be evaluated against separate data … as a submodel that could be trained separately to ensure that errors are not propagated to solar model via FI model fitting. After this we can see what input sensors need seasonal analysis and at what extent."*

## Part 1 — Current-model seasonality audit (findings)

The current production v2.2 9-feature FI Ridge classified by how each feature handles seasonality:

| # | Feature | Class | What it really does | Refactor verdict |
|---|---|---|---|---|
| 1 | `wind_speed_weighted` | Pure exogenous physics | Capacity-weighted wind across 7 FI locations. Has day-of-year + diurnal seasonality but it's CAUSAL not compensating. | Keep — but extract its seasonal share via `P_hour + P_day + P_week` so the model fits its *residual*, not its trend. |
| 2 | `month_cos` | **Explicit compensator** | `cos(2π · month/12)` — a one-harmonic Fourier proxy for whatever annual cycle the residual still carries. | **Remove.** Replace with explicit `P_week(w)` decomposition on the target (FI price residual). |
| 3 | `is_holiday` | Structural | Discrete state. | Keep — but combine with workday/weekend to form a single `daytype` regressor that goes INTO the seasonal decomposition's `P_day(d)`. |
| 4 | `hdd_sq` | Pure exogenous physics | Heating-degree-day squared, derived from temperature. Heavily seasonal but causal. | Keep — extract `T̃ = T − (T_hour + T_day + T_week)` and recompute HDD on the residual, or keep raw and let downstream de-seasonalization absorb the trend. Pick whichever shows lower test-CVaR. |
| 5 | `wind_log_scarcity` | Pure exogenous physics | `log1p(max(0, 8 − wind))` — non-linear supply-shortage indicator. | Keep — derives from wind; same treatment as #1. |
| 6 | `ar_se3` | **Implicit compensator** | AR(2) on SE3 prices computed against a separate weekday vs weekend hour-of-day profile. The daytype split IS the seasonality being smuggled in. | **Refactor.** Replace with `Y_SE3 = SE3 − (SE3_hour + SE3_day + SE3_week)` and a simple AR(1) on `Y_SE3`. The v2.4.2 study already proved this works for SE3. |
| 7 | `ar_ee` | **Implicit compensator** | Same construction as `ar_se3` but for EE. | **Refactor or drop.** v2.4.3 said the de-seasonalized version doesn't beat baseline; v2.5.1 + v2.5.2 add information here as `spread_se3_se1`. Re-test under the new architecture. |
| 8 | `export_potential_se3` | Structural | `max(0, −spread_FI_SE3)` — discrete arbitrage state. | Keep — interaction feature, no seasonality content. |
| 9 | `nuclear_x_scarcity` | Structural | Nuclear deficit × wind-scarcity interaction. | Keep — both factors handled elsewhere. |

**Solar/PV inside the FI Ridge today**: none. `solar_irradiance_weighted` is computed in `BASE_FEATURES` (src/features.py) but the production v2.0.0 coefficients in `model_coefs_default.json` do NOT select it. The `pv_estimate.py` Phase-1 module from v2.4.0 is used ONLY in the downstream consumer-effective-price layer (D(k) reshaping); it is **not** an input to price prediction.

This is actually a deliberate design choice: the duration-curve segments use `solar_mean` from a separate model in `data/finland.yaml`. So adding a proper clear-sky × cloudiness solar sub-model is greenfield for the price-forecast side.

**Hardcoded seasonal coefficients** outside the Ridge:
- `FINLAND_RESIDENTIAL_MONTHLY_FACTORS` (const.py) — used ONLY for baseload sizing, NOT for price.
- `data/finland.yaml` location weights for the seven wind sites — capacity weights, not seasonal.
- No monthly day/night factors in the price path.

**Net audit conclusion**: there are exactly two explicit compensators in the price model (`month_cos`, the AR-with-daytype-profile machinery for SE3/EE). Both are candidates for removal once the explicit `P_hour + P_day + P_week` decomposition is in place per-input. Everything else is either pure exogenous physics that needs de-seasonalization in its own right, or a structural feature that has no seasonal content.

## Part 2 — Public solar ground-truth dataset (revised per user direction)

User direction (2026-05-17): *"Again ENTSO access is problematic. We need to use public source that are currently available such as Fingrid should have this information as well as historic data."*

**Decision: Fingrid dataset 248 ("Solar power generation forecast") as the de-facto historical Finnish solar series.**

Fingrid does retain all historically-published forecast values back to **2017-02-24** (~9 years), queryable at any past date. This is the dataset Fingrid itself uses as the operational national figure for solar generation. The "forecast" label is misleading — every Finnish dashboard, statistical report and grid operator cites this series as Finland's nationwide solar production.

Supporting datasets:
- **267** "Total solar PV capacity used in forecast" — for normalising 248 to capacity factor; history from 2019-04-23.
- **362** "Small-scale electricity surplus production by production type" — measured prosumer net-export of PV (type AV06); history from 2023-08-01. **Caveat**: this is net surplus AFTER own consumption, so it under-reports total PV by ~50 %. Useful as a sanity-cross-check post-2023 but not as ground truth.
- **247** "Solar power generation forecast (daily update)" — 15-min res since 2023, hourly before; alternative to 248 with slightly different update cadence.

URL pattern (uses existing `studies/fingrid_netload_study.py:fetch_dataset` plumbing — already in the repo, already paginates, already disk-caches):
```
GET https://data.fingrid.fi/api/datasets/248/data
   ?startTime=...&endTime=...&format=json&pageSize=20000
Header: x-api-key: $FINGRID_API_KEY
```

The API key is **instant and free** at https://developer-data.fingrid.fi/ (no manual approval delay, no email confirmation), throttle 10,000 req/day, 1 req/2 s. Same friction profile as the project's existing Fingrid usage for dataset 188 (nuclear deficit) and 246/247 (wind/solar forecasts in the runtime coordinator). No new auth burden.

Why this is acceptable as "ground truth" even though it's a forecast:
- Fingrid's own forecast uses "production measurements from large-scale solar parks" as an input (per their dataset 248 description), so it tracks actual large-scale PV closely.
- Distributed PV (~50 % of Finnish capacity) is intrinsically not metered at TSO level — ANY nationwide series is necessarily a model, including ENTSO-E B16 which uses the same back-fill approach.
- Quote the user implicitly: if the operational system trusts this as the national figure, the sub-model can be validated against it.

ENTSO-E B16 is **rejected** per user direction (problematic access). Open Power System Data + JRC EMHIRES remain unsuitable (stale).

## Part 3 — Proposed v2.5.3 → v2.6.0 patch chain

The work splits into four sequential research patches plus one consolidation milestone, following the established v2.4.x discipline (NPK-CVaR hedge gate per change; ACCEPT/REJECT/DEFER first-class; no version bumps for incremental progress).

### v2.5.3 — Clear-sky solar sub-model (isolated training and validation)

**Goal**: produce a stand-alone solar production model that takes (clear-sky GHI, cloudiness) → predicted FI nationwide PV (MWh/h), trained and validated against ENTSO-E B16 with NO connection to the FI price fit.

Steps:

1. **`studies/entso_e_solar_client.py`** — pull FI hourly B16 for 2023-01-01 → today. Persist to `output/fi_solar_actual.parquet`. Reuse the existing ENTSO-E token plumbing.
2. **`custom_components/.../clear_sky.py`** — implement a deterministic clear-sky GHI for FI:
   - Use **Haurwitz** (single-formula, no atmospheric inputs) as the fast baseline: `GHI_cs(zenith) = 1098 · cos(z) · exp(−0.057 / cos(z))`. Free of any external parameters; depends only on solar zenith (lat, lon, datetime).
   - Optionally **Ineichen-Perez** (needs Linke turbidity climatology — embed a 12-month FI climatology constant) as the production candidate. The study script compares both against ENTSO-E and picks the lower-MAE.
   - Aggregate over the same 7 location weights already used for wind/solar irradiance, scaled by Finnish installed-PV-capacity time series (Energiavirasto/Motiva annual numbers interpolated to daily — capacity drift is slow, daily is fine).
3. **`studies/solar_clear_sky_submodel.py`** — fit the cloudiness modulator:
   - Inputs: `GHI_cs(t)` from clear-sky model, `cloud_cover(t)` from Open-Meteo `cloud_cover` variable (already fetched in `weather.py`).
   - Model: `production(t) = GHI_cs(t) · capacity(t) · f(cloud_cover(t))` where `f(·)` is a monotone-decreasing modulator. Candidate forms (all cheap to fit):
     - **Linear**: `f(c) = 1 − a·c/100` (one parameter)
     - **Affine + floor**: `f(c) = max(0, 1 − a·c/100) + b` (two parameters, captures diffuse-radiation floor)
     - **Kasten-Czeplak empirical**: `f(c) = 1 − 0.75·(c/100)^3.4`
   - Train on first 70 % chronological, validate on remaining 30 %.
   - Report MAE, RMSE, R², bias by hour-of-day, bias by month.
4. **Acceptance gate (isolated)**: solar sub-model achieves R² ≥ 0.85 on test set OR matches Open-Meteo's published `global_tilted_irradiance` direct-irradiance accuracy on FI sites (whichever is the harder bar). If both fail, document why and defer.
5. **No FI Ridge change in v2.5.3.** This patch ships the sub-model + validation. The FI model continues to use the existing `solar_irradiance_weighted` (which it ignores anyway in the v2.0.0 coefficients).

### v2.5.4 — Per-sensor seasonal-content analysis

**Goal**: for every candidate input, quantify what fraction of its variance is deterministic-seasonal (`P_hour + P_day + P_week`) versus residual `Y_t`, so we know where de-seasonalization is worth the implementation cost.

The user's hint: *"Wind has seasonal variation but not within a week but rather day and month."* — translate this into the framework: wind needs `P_hour + P_week` only, not `P_day` (no weekday/weekend effect on physical wind). Similar audit needed per input.

Steps:

1. Extend `studies/seasonal_visualization.py` (v2.5.1) to also report variance shares for every candidate input: temperature, wind, cloudiness, FI / SE3 / SE1 / EE prices, hydro reservoir, the new clear-sky solar baseline.
2. For each input compute:
   - `var(P_hour) / var(X)`
   - `var(P_day) / var(X)`
   - `var(P_week) / var(X)`
   - `var(Y) / var(X)` (residual share)
3. **Decision rule per input** for whether to de-seasonalize each component:
   - Component included if its variance share > 5 % of total AND its inclusion lowers the Ljung-Box statistic on `Y` versus omitting it.
   - Otherwise the component is dropped (its variance gets absorbed back into `Y` — fine if the model handles it).
4. Output: `studies/results/per_sensor_seasonality_audit.md` — one row per input with the recommended decomposition depth (e.g. wind: hour + week; temperature: hour + week; prices: hour + day + week).
5. **No model change**. This is a measurement patch.

### v2.5.5 — De-seasonalize inputs + remove explicit compensators

**Goal**: build a v2.6.0-candidate FI Ridge that fits residual-on-residual, removing `month_cos` and the AR-with-daytype machinery.

Steps:

1. New module `custom_components/.../seasonal_cache.py` — refit per-input seasonal vectors quarterly, persist in `.storage/spot_price_predictor_seasonal_cache.json`. Honors the per-input depth recommendations from v2.5.4.
2. Rebuild feature matrix using `Ỹ_X` instead of raw `X` for each input where v2.5.4 recommended decomposition.
3. Re-run the v2.2 leave-one-out redundancy sweep on the new feature set. Expect:
   - `month_cos` drops out (its information is now in `P_week(FI_price)` on the target side, predicted directly from `Y_target`).
   - `ar_se3` becomes a simple AR(1) on `Y_SE3`.
   - `ar_ee` may drop entirely if v2.5.1's `spread_se3_se1` covers it.
   - Solar enters as the new `solar_submodel_prediction` (output of v2.5.3) — first time solar contributes to FI price.
4. **NPK-CVaR hedge gate**: rebuilt model must beat current v2.2 9-feature baseline on test CVaR at α ∈ {0.05, 0.01}. ACCEPT/REJECT/DEFER per established pattern.

### v2.5.6 — Hedge-gated input selection sweep, restarting from the 17-feature universe

User direction (2026-05-17): *"we used earlier effort analyzing which input are valid for FI model. However, when we reduced model from 17→9 the reason may be due to poor seasonal characterization. So instead of reducing 9 input even further I propose that we start from larger set of 17 and reduce the model from there using hedge analysis."*

**Goal**: under the cleaner architecture (de-seasonalized inputs from v2.5.5 + clear-sky solar from v2.5.3), restart selection from the **original 17-feature universe** and let the NPK-CVaR hedge gate decide what stays. Features dropped during the v2.2 17→9 reduction may earn back their place once their previously-confounded seasonal share is properly separated.

**Original 17-feature universe** (recovered from src/features.py and commit 55f6be7):

Kept in v2.2:
1. `wind_speed_weighted`, 6. `month_cos`, 7. `is_holiday`, 8. `hdd_sq`, 9. `wind_log_scarcity`, 13. `ar_se3`, 14. `ar_ee`, 15. `export_potential_se3`, 17. `nuclear_x_scarcity`

Dropped in v2.2 (LOO sweep, ~30 % MAE improvement at the time, but on RAW data with no seasonal decomposition):
2. `solar_irradiance_weighted` — dropped as "collinear with wind"; **rehabilitation candidate** now that we have a clear-sky-based solar sub-model output (v2.5.3) that should NOT be collinear with wind in the same way.
3. `hour_sin`, 4. `hour_cos` — explicit Fourier hour. **Drop permanently** — the new architecture handles hour-of-day via the `P_hour(h)` decomposition on the target, making these doubly-redundant.
5. `month_sin` — Fourier month pair with #6. **Drop permanently** (same reason).
10. `wind_calm_x_peak_am`, 11. `wind_calm_x_peak_pm` — wind-calm × morning/evening peak. **Rehabilitation candidate** if the peak indicator is reconstructed from `P_hour(load)` rather than raw hour.
12. `ar_se1` — dropped as "collinear with ar_se3". v2.5.1 already proved this was WRONG under NPK-CVaR (`spread_se3_se1` adds +0.55 pp). **Strong rehabilitation candidate** — reintroduce as either the de-seasonalized `Y_SE1` or directly as `spread_se3_se1` per v2.5.1.
16. `nuclear_deficit` — dropped as "captured by nuclear_x_scarcity". **Rehabilitation candidate** — the interaction term may have hidden a marginal main-effect contribution.

**Plus new candidates** added since the 17-feature era:
- `solar_submodel_prediction` (output of v2.5.3 — clear-sky × cloudiness)
- `hydro_reservoir_se` (v2.4.1 — Statnett, never made it into the FI model)
- `spread_se3_se1` (v2.5.1 — proven valuable as joint hedge)

**Effective starting universe for v2.5.6**: ≈ 12–14 features (original 17 minus the four hour/month Fourier compensators that are doubly-redundant under the new architecture, plus three new candidates).

Steps:

1. Build the v2.5.6 candidate matrix from the v2.5.5 de-seasonalized base, adding back every "rehabilitation candidate" listed above.
2. **Reduction rule**: feature stays if its individual drop-out increases test CVaR by ≥ 0.3 pp (i.e. the feature's contribution to risk-adjusted accuracy is non-trivial). Drop otherwise.
3. Multi-feature collinearity: VIF check on the surviving set, but only DROP a collinear pair if removing either side improves test CVaR — pure-correlation collinearity is no longer a sufficient reason after v2.5.1's lesson.
4. Output: `studies/results/v256_input_selection.md` with the full scorecard (each feature: MAE delta, CVaR-test delta, VIF, kept/dropped, why).

### v2.6.0 — Consolidated model

Ship the validated cleaner-architecture FI model + cross-border seasonal layer + clear-sky solar sub-model. Update README, TECHNICAL_GUIDE, TEKNINEN_TOTEUTUS. No new sensor schema (Option A still — internal model upgrade only).

## Locked decisions (from the user's directive)

1. **Solar is a separately-trained sub-model.** Errors do NOT propagate via FI fit — sub-model is validated on its own ground truth (ENTSO-E B16) before its output is allowed into the FI feature set.
2. **Clear-sky × cloudiness architecture for solar.** Deterministic baseline (clear-sky) modulated by Open-Meteo cloudiness, not the lumped tilted-irradiance currently exposed by Open-Meteo.
3. **Wind has hour + month seasonality, not day-of-week.** Translate to the framework as `P_hour + P_week` only.
4. **Hedge analysis gates every change.** Consistent with v2.4.x methodology.
5. **No premature version bumps.** v2.5.3, .4, .5, .6 are all incremental research patches; v2.6.0 is the consolidation milestone.

## Open questions to resolve in v2.5.3 planning

1. Which clear-sky model — Haurwitz (one-line, no atmospheric inputs) or Ineichen-Perez (needs Linke turbidity climatology)? **Decide empirically** by running both against ENTSO-E B16 and picking lower-MAE.
2. Where to source Finnish installed-PV-capacity time series for the capacity-scaling factor? Energiavirasto publishes annual numbers; Motiva has slightly higher resolution. Need to pick + interpolate. **Sub-task for v2.5.3.**
3. ENTSO-E B16 is itself a TSO estimate (back-filled distributed PV) — what's the appropriate error floor? The sub-model can't beat the ground truth's own uncertainty. **Document but don't gate on.**

## Reproducibility plan

Each patch ships a self-contained study script + auto-generated markdown summary + tests (when there's code that lives inside `custom_components/`). Follows the existing `studies/results/` convention.
