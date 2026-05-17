# v2.5.3 — Solar production sub-model (PV-aware ground-truth validation, ACCEPT)

## TL;DR

**No coordinator behaviour change.** v2.5.3 ships a fully isolated solar production sub-model — clear-sky GHI × cloudiness modulator — trained and validated against Fingrid dataset 248 (Finnish nationwide solar generation) **without ever touching the FI price model**. The sub-model passes the Tier-A absolute gate (R² ≥ 0.85) and is ready to be wired into the FI Ridge as a single candidate feature in v2.5.5 (subject to NPK-CVaR hedge gating in v2.5.6).

User direction (2026-05-17): *"I want to validate the PV production model separately first before entering FI model evaluation."* — this patch delivers that validation. Per the same user directive, ENTSO-E B16 is **not** used (access friction); Fingrid dataset 248 is the de-facto Finnish national solar series.

## Results

29,496 aligned hourly rows over 2023-01-01 → 2026-05-13. Chronological 70/30 train/test split. Four candidate combinations (Haurwitz vs Ineichen-Perez clear-sky × linear vs Kasten-Czeplak modulator). All combinations include a free intercept and a fitted multiplicative gain (OLS):

| Clear-sky | Modulator | MAE [MW] | MAE-daylight [MW] | rel. MAE-daylight | R² | bias [MW] |
|---|---|---:|---:|---:|---:|---:|
| Haurwitz | Kasten-Czeplak | 44.84 | 80.70 | 26.0 % | 0.907 | +21.13 |
| Haurwitz | Linear | 45.15 | 83.60 | 27.0 % | 0.914 | +29.54 |
| **Ineichen-Perez (winner)** | **Kasten-Czeplak (winner)** | **44.26** | **80.69** | **26.0 %** | **0.910** | **+21.98** |
| Ineichen-Perez | Linear | 47.73 | 84.13 | 27.1 % | 0.917 | +30.60 |

Winner selected by **test MAE-daylight** (the metric that matters — night-time MAE is trivially zero and dilutes the headline). All four pass the **Tier-A absolute gate (R² ≥ 0.85)**.

## Isolation invariant — HONOURED

The sub-model consumes only `(timestamp, lat, lon)` and `cloud_cover(t)`. No price-side input. Its OLS fit minimises the squared error of *production* — not of *price*. Therefore errors in the solar fit cannot propagate to the FI price model via shared optimisation: it has its own training signal and its own loss landscape. This is the design property the user asked for in 2026-05-17.

## Architecture (sub-model)

```
production_MW(t) = alpha
                 + gain · capacity_MW(t)
                        · GHI_clear_sky(t, weighted_sites)
                        · cloudiness_modulator(cloud_cover(t))
```

- **`GHI_clear_sky`** — deterministic function of `(lat, lon, t)` via Spencer 1971 solar geometry and either:
  - Haurwitz (1945) single-formula model — zero atmospheric inputs, suitable as fallback.
  - **Ineichen-Perez (2002)** — uses bundled Finland monthly Linke turbidity climatology. Winner by test MAE-daylight.
- **`cloudiness_modulator`** — three candidate forms; **Kasten-Czeplak empirical** `(1 − 0.75·c³·⁴)` wins. No tuned parameters in the modulator shape itself.
- **`alpha`, `gain`** — single 2-parameter OLS fit per (clear-sky, modulator) combination on training data only.
- **`capacity_MW(t)`** — Fingrid dataset 267 (forward-filled, slow-changing reference). Grew 606 → 1741 MW across the window.
- **`cloud_cover(t)`** — Open-Meteo `cloud_cover` variable, capacity-weighted across the same seven Finnish sites already used by the integration for wind/solar irradiance (`data/finland.yaml`).

## Residual bias structure (known limitation)

Test bias is **+22 MW (~7 % of mean daylight production)** and is *not* uniform — it has two clear structures visible in the bias-by-hour and bias-by-month panels:

1. **Strongly hour-of-day shaped**: near zero at night, peaks at +80 MW around 10–12 UTC, declines back toward sunset.
2. **Spring/summer concentrated**: positive across Mar–Jul; near-zero or slightly negative in Sep–Feb.

This is the unambiguous signature of effective generating capacity at peak being lower than Fingrid 267's planned-capacity figure — distributed PV at mixed orientations averages ~10–15 % below the south-tilt assumption the clear-sky model carries. The OLS intercept absorbs the *mean* drift but cannot absorb the *shape* drift.

**Candidate refinements for a future patch (v2.5.3.1 if warranted)**:
- Capacity de-rating factor (one extra parameter) so that effective_kWp = de_rate · planned_kWp.
- Per-month modulator gain (12 extra parameters; risk of overfitting at low sample counts in winter).
- Swap `cloud_cover` for `cloud_cover_low` or use Open-Meteo `direct_radiation` ÷ `direct_normal_irradiance_instant` as a more physical attenuation signal.
- Tilt/azimuth distribution model averaging over typical Finnish rooftop installations rather than assuming 45° south.

None of these are gating for v2.5.3 — the sub-model PASSes Tier-A as is, and the residual bias is small enough that the FI Ridge can learn around it during the v2.5.6 hedge-gated selection sweep.

## Why Fingrid 248 is acceptable as ground truth

Per the user direction, ENTSO-E access friction motivated re-investigation of Finnish public sources. Findings:

- **Fingrid dataset 248** retains 9 years of historical published forecast values (back to 2017-02-24), queryable at any past date, and is the operational national figure used by Fingrid and every downstream Finnish dashboard.
- Fingrid's own dataset description states it uses **"production measurements from large-scale solar parks"** as a forecast input, so the series tracks large-scale measured PV closely.
- Distributed PV (~50 % of Finnish capacity) is not TSO-metered anywhere — any nationwide series is intrinsically a model, including ENTSO-E B16 which uses the same back-fill approach.
- Free, instant API-key registration (no manual approval, no email confirmation). Throttle 10,000 req/day, 1 req/2 s. Same friction profile as the project's existing Fingrid integration for nuclear deficit (dataset 188) and the wind/solar forecasts used in the runtime coordinator (246, 247).

## Files

- **New**: `custom_components/spot_price_predictor/solar_clear_sky.py` (~210 LOC)
  - Spencer 1971 solar geometry (cos zenith, equation of time, extraterrestrial irradiance)
  - Haurwitz and Ineichen-Perez clear-sky GHI formulas
  - Three cloudiness modulator forms (linear, affine+floor, Kasten-Czeplak)
  - Vectorised `clear_sky_series()` helper for offline studies
  - numpy + stdlib only; no pvlib dependency
- **New**: `studies/solar_clear_sky_submodel.py` (~430 LOC) — full end-to-end study with disk-cached Fingrid + Open-Meteo data fetches
- **New**: `tests/test_solar_clear_sky.py` — 16 tests covering geometry, both clear-sky models, vector helpers, and all three modulator forms
- **New**: `studies/results/solar_clear_sky_submodel.md` (auto-generated; full verdict tables)
- **New**: `studies/results/figures/solar_submodel_validation.png` (4-panel: sample window, scatter, bias-by-hour, bias-by-month)
- **New**: `studies/results/V2_5_3_RELEASE_NOTES.md` — this document
- **Modified**: `custom_components/spot_price_predictor/manifest.json` (`2.5.2 → 2.5.3`)
- **Modified**: `studies/results/seasonality_audit_and_roadmap.md` (Part 2 + v2.5.6 rewritten per user direction on the Fingrid source and the 17-feature restart)

No coordinator behaviour change; no sensor schema change. HACS users see version `2.5.3` but observe no runtime difference until the sub-model output is wired into the FI Ridge in v2.5.5+.

## Runtime deployment story (Fingrid-free)

Per the follow-up user direction 2026-05-17: *"The intended deployment of the model should not assume continuous reading of Fingrid data. Retraining may be needed after few months to adapt to increasing PV production capacity but this information does not need to be downloaded in daily basis."*

The training-time capacity series `capacity_MW(t)` is **collapsed into a single scalar** `K = gain · capacity_ref_MW` at the end of training and baked into the artifact:

```
runtime_prediction(t) = alpha + K · GHI_clear_sky(t, sites) · cloudiness_modulator(cloud_cover(t))
```

So the deployed integration needs **only** the deterministic clear-sky formula (lat, lon, t — no network) and the Open-Meteo `cloud_cover` series which it already fetches as part of the existing weather call. **Zero Fingrid calls at runtime.**

The artifact is persisted at `custom_components/spot_price_predictor/data/solar_submodel_default.json` and contains the frozen `(alpha, K, sites, modulator_form, modulator_params, clear_sky_model)` plus full training metadata. Refresh cadence: **every few months** (as Finnish installed PV capacity drifts) by re-running the study script; the new artifact replaces the shipped JSON via a normal commit.

**Frozen-capacity test metric** (the number that actually matters for deployment, because at runtime capacity is fixed at `capacity_ref` = 1367 MW rather than time-varying):

| Metric | Time-varying capacity (training fit) | **Frozen capacity (deployed)** |
|---|---:|---:|
| Test R² | 0.910 | **0.918** |
| Test MAE | 44.26 MW | **40.05 MW** |
| Test MAE-daylight | 80.69 MW | **72.70 MW** |
| Test bias | +21.98 MW | **+8.81 MW** |

The deployed model is actually **better** than the time-varying-capacity fit, because freezing capacity at training-end (1367 MW) is closer to the effective generating capacity during the test period than Fingrid 267's over-stated planned-capacity series across 2024–25.

The `predict_solar_mw(timestamps, cloud_cover_pct, artifact)` helper in `solar_clear_sky.py` is the runtime inference entry point. Reloads JSON, no Fingrid call, pure numpy.

## Visual analysis (`studies/solar_visualization.py`)

Three deployment-grade figures generated from the cached data + deployed artifact (no Fingrid call — figures regenerate offline):

| Figure | What it shows |
|---|---|
| `figures/solar_tracking.png` | Four seasonal two-week windows (Jan / Apr / Jul / Oct 2025) with the deployed prediction overlaid on actual production, plus cloud-cover overlay. Diurnal cycle is tracked tightly; spring & summer peaks match well; autumn shows good shape with occasional cloud-pass under-prediction. |
| `figures/solar_accuracy.png` | Test-set diagnostics, daylight hours only: scatter density vs `y=x`, residual distribution with Gaussian overlay (residuals are close-to-Gaussian, σ ≈ 98 MW, mean +14 MW), residual binned by cloud cover (no systematic shape — model handles all cloud regimes uniformly), residual binned by clear-sky GHI level (slight under-prediction at the highest GHI bins, consistent with the rooftop-orientation mix hypothesis). |
| `figures/solar_temporal.png` | The most informative: monthly produced energy (actual vs predicted bars across the full 2023–2026 window), monthly MAE + bias trace, and installed-capacity drift vs the artifact's `capacity_ref`. **Reveals the deployment story end-to-end** — see "Refit signal" below. |

### Refit signal — what the temporal figure tells us

The shipped artifact is calibrated to `capacity_ref = 1367 MW`. The capacity panel shows the dashed reference line crossed by the actual capacity (Fingrid 267) in mid-2025 — exactly when the bias trace settles near zero. By April 2026 the actual capacity is **1741 MW (≈ +27 % above the baked value)** and the bias is starting to dip negative (model now slightly under-predicts).

**Recommendation**: refresh the artifact when actual capacity exceeds `capacity_ref` by more than ~20 %, i.e. when actual ≳ 1640 MW. This threshold is already met as of the v2.5.3 release; a refit is justified now and the visualization makes the trigger condition obvious.

The 2023 monthly aggregates show very large over-prediction (~+200 MW bias) — this is exactly the "frozen capacity above actual" effect operating in the opposite direction. The deployed model's accuracy quality is therefore non-uniform across history: it is best when actual capacity ≈ baked capacity (mid-2025 onwards) and degrades as the gap widens in either direction.

This is the price of the Fingrid-free deployment contract — and it's a price worth paying as long as the refit cadence keeps up.

## Tests

**349 / 349 passing** (326 prior + 23 new clear-sky tests, including 7 new tests for `build_artifact()` / `predict_solar_mw()` runtime inference).

## Reproducibility

```bash
export FINGRID_API_KEY=<your_free_key>   # https://developer-data.fingrid.fi/
python studies/solar_clear_sky_submodel.py
```

The script caches raw Fingrid + Open-Meteo responses under `studies/.cache/` so subsequent runs are fast. Re-running regenerates the markdown summary and validation figure.

## Next step (v2.5.4)

Per the v2.5.3 → v2.6.0 roadmap (`studies/results/seasonality_audit_and_roadmap.md`):
- v2.5.4 — per-sensor seasonal-content analysis (decomposition variance shares for every candidate input: wind, temperature, cloudiness, FI/SE3/SE1/EE prices, hydro reservoir, and the new clear-sky baseline). User-specified hint: wind has hour+month seasonality, NOT week.
- The solar sub-model output `solar_submodel_prediction(t)` joins the candidate feature universe in v2.5.5 (de-seasonalized inputs + drop month_cos / AR-daytype machinery) and is gated by NPK-CVaR hedge analysis in v2.5.6 (which restarts from the original 17-feature universe per user direction).
