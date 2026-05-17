# v2.4.4 — FI model revision investigation (REJECTED; production v2.2 Ridge retained for FI)

## TL;DR

**REJECT** — applying the validated SE3-style architecture to FI does not improve over the windowed seasonal-only baseline (+7.01 % CVaR reduction). Best augmented variant (V4, lean lossless: hydro + workday + Y_SE3 + Y_WIND, no AR(1)) achieves only +4.14 %. The current production v2.2 9-feature Ridge model **stays** as the FI predictor in v2.5.0.

## What was tested

Five variants on the Statnett-covered window (May 2024 → April 2026, 17,163 hourly observations):

| Variant | Architecture | h_hat | CVaR reduction |
|---|---|---:|---:|
| **V0 baseline** | seasonal-only | 1.135 | **+7.01 %** |
| V1 SE3-style | seasonal + hydro + workday + AR(1) | 0.191 | +2.87 % |
| V2 + Y_SE3 | V1 + de-seasonalized SE3 coupling | 0.209 | +3.11 % |
| V3 + Y_WIND | V2 + de-seasonalized wind | 0.218 | +3.17 % |
| V4 lean | hydro + workday + Y_SE3 + Y_WIND (no AR(1)) | 0.559 | +4.14 % |

Best is V4 at +4.14 %, still below baseline +7.01 %. **Δ = −2.87 pp → REJECT.**

## Why the SE3 architecture does not transfer to FI

1. **AR(1) compresses the hedge geometry**: same pathology as v2.4.3 EE — adding `β_AR1 · Y_{t-1}` introduces deep autocorrelation in `diff(model)` that drops the optimal hedge ratio from ~1.1 to ~0.2, destroying more CVaR-reduction than the lagged signal contributes.

2. **FI has more diverse price drivers than SE3**:
   - Nuclear outage events (OL3 + Loviisa 1-2) — captured by `nuclear_x_scarcity` in v2.2 Ridge, missing here
   - Fenno-Skan congestion (FI ↔ SE3 transmission bottleneck) — implicit in `ar_se3` AR(2) feature, only partially captured by `Y_SE3` here
   - Strong wind nonlinearity — `wind_log_scarcity` log-nonlinear feature in v2.2 captures it; the linear `Y_WIND` here doesn't

3. **The Moazeni-Powell additive linear architecture saturates** faster on FI than on SE3 because of these structural FI-specific features. The v2.2 production model's combination of log-linear formulation + 9 sign-validated features + power-stretch optimization extracts signal that simple seasonal + 2-3 exogenous features cannot.

## Decision for v2.5.0

| Zone | v2.5.0 model | Source |
|---|---|---|
| **FI**  | **KEEP v2.2 9-feature Ridge (unchanged)** | v2.4.4 REJECT |
| **SE3** | Wire in new model: seasonal + hydro + workday + AR(1) | v2.4.2 ACCEPT |
| **EE**  | **KEEP v2.2 `ar_ee` AR(2) feature (unchanged)** | v2.4.3 REJECT |
| **SE1** | KEEP (not in production model since v2.2 pruning) | — |

The v2.5.0 model rewire becomes much narrower than initially planned: only SE3's `ar_se3` feature gets replaced with the validated v2.4.2 model. The rest of the v2.2 9-feature Ridge stays untouched.

## What v2.4.4 ships

- **`studies/fi_model_v244.py`** — runnable investigation script (5 variants + decision logic)
- **`studies/results/V2_4_4_RELEASE_NOTES.md`** — this document
- **`manifest.json`** — bumped `2.4.3` → `2.4.4`

No coordinator changes, no sensor schema changes, no new tests.

Test suite: **309 / 309 passing** (unchanged from v2.4.2).

## Lesson from the v2.4.x patch chain

Out of 4 model-investigation patches (v2.4.2, v2.4.3, v2.4.4 + planned v2.4.5), so far **1 accept (SE3), 2 reject (EE, FI)**. This is exactly the value the NPK-CVaR hedge methodology was set up to deliver: prevent adding complexity to the production model that doesn't earn its keep on out-of-sample data.

The user's directive — *"if test CVaR drops, the feature captures real signal; if unchanged, it's noise — discard"* — operationalized as objective gates protects the production model from speculative rewrites motivated by aesthetic preference for "cleaner" architectures. v2.2's 9-feature Ridge remains the production FI model because no candidate has demonstrated empirical superiority.
