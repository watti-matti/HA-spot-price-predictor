# v2.5.0 — Phase 3 methodology milestone

## TL;DR

**v2.5.0 is a methodology release, not a model upgrade.** It marks the completion of the v2.4.x patch chain (v2.4.0 → v2.4.5) that systematically investigated whether the Moazeni-Powell-derived seasonal + OU + spike framework should replace the current v2.2 9-feature Ridge production model. The empirical answer is **mostly no** — only the SE3 cross-border model was accepted, and SE3 wire-in requires re-training the FI Ridge (which v2.4.4 explicitly rejected). The production model therefore stays as v2.2 9-feature Ridge with the v2.3.x PV-aware extensions and v2.4.0 baseload UX.

**No coordinator behaviour change in v2.5.0.** HACS users will see version 2.5.0 but observe no runtime difference. The validated model investigations live in `studies/` for future re-use.

## What the v2.4.x patch chain delivered

| Patch | Verdict | Outcome |
|---|---|---|
| **v2.4.1** | infrastructure | NPK-CVaR hedge tool + Statnett hydro reservoir client + per-zone CVaR baselines (FI +6.06 %, SE3 −7.78 %, SE1 −0.30 %, EE +4.07 %) |
| **v2.4.2** | **ACCEPT** | SE3 model (seasonal + hydro + workday + AR(1)) beats SE3 baseline by +3.07 pp |
| **v2.4.3** | REJECT | EE model variants tie or hurt vs baseline — current `ar_ee` AR(2) retained |
| **v2.4.4** | REJECT | FI model variants do not beat windowed baseline — v2.2 Ridge retained |
| **v2.4.5** | DEFER | Alternative solar model needs Open-Meteo `cloud_cover` data not currently fetched |

The methodology — *"if test CVaR drops, the feature captures real signal; if unchanged, it's noise — discard"* — prevented adding complexity that wouldn't earn its keep. **1 accept / 2 reject / 1 defer is exactly the value the gate-driven approach delivers.**

## What changes in v2.5.0

### Nothing in the coordinator runtime

- FI prediction continues to use the v2.2 9-feature Ridge (per v2.4.4 REJECT)
- EE feature continues to use v2.2 `ar_ee` AR(2) (per v2.4.3 REJECT)
- Solar feature continues to use Open-Meteo `global_tilted_irradiance_instant` (per v2.4.5 DEFER)
- All sensor attributes unchanged
- All config options unchanged

### What's NEW in the codebase (carried from v2.4.x)

| Asset | Provenance | Purpose |
|---|---|---|
| `studies/npk_cvar_hedge.py` | v2.4.1 | Validation tool for any future model variant; Python port of MATLAB hedge analysis, validated against MATLAB benchmarks within ±5 % |
| `custom_components/spot_price_predictor/statnett_client.py` | v2.4.1 | Async client for Statnett Norwegian reservoir data (free, no auth, weekly); wired up but not yet a model input |
| `studies/se3_model_v242.py` | v2.4.2 | Runnable build-and-validate script for the accepted SE3 model; ready for production wire-in when FI Ridge retrain is in scope |
| `studies/ee_model_v243.py` | v2.4.3 | EE rejection regression record |
| `studies/fi_model_v244.py` | v2.4.4 | FI rejection regression record (5 variants, all worse than baseline) |
| `studies/results/current_model_cvar_baseline.md` | v2.4.1 | Per-zone CVaR baselines — the gates against which future variants must compare |
| `studies/results/npk_cvar_python_port_validation.md` | v2.4.1 | Python vs MATLAB benchmark comparison (all 4 PASS) |
| `studies/results/V2_4_*_RELEASE_NOTES.md` | v2.4.x | Full investigation history |

### When the SE3 model will actually be wired in

The validated v2.4.2 SE3 architecture replaces the current AR(2) `ar_se3` feature path. Wire-in requires:

1. Storing `seasonal_SE3` + `β_hydro` + `β_workday` + `β_AR1` coefficients in `model_coefs.json`
2. New `compute_se3_v242_forecast()` function in the coordinator
3. **Re-training the FI Ridge** with the new SE3 feature substituted for `ar_se3` (because Ridge coefficients are fit on a specific input scale)
4. Validation that the re-trained FI Ridge still beats v2.4.4's REJECT bar (otherwise we've replaced one feature path with another without overall gain)

That's a substantial undertaking (~v2.6.0 or v3.0.0 scope). Triggering it requires a decision to re-open the FI Ridge — currently blocked by v2.4.4 REJECT. The SE3 v2.4.2 model is preserved in `studies/` so the decision can be revisited at any time.

## Files in v2.5.0

- **Modified**: `README.md` — version badge `2.4.0 → 2.5.0`; release-notes link list extended through v2.4.5
- **Modified**: `manifest.json` — `2.4.5 → 2.5.0`
- **New**: `studies/results/V2_5_0_RELEASE_NOTES.md` — this document

That's it. No source code changes, no test additions, no schema migrations.

Test suite: **309 / 309 passing** (unchanged from v2.4.2 onwards).

## Process highlights

The v2.4.x → v2.5.0 chain operated as a **research-quality release cadence**:

- Each patch shipped a concrete artifact (validation tool, dataset, model variant, decision record)
- Every model claim was gated on an objective out-of-sample metric (NPK-CVaR test reduction at α = 0.05, 48 h hedge horizon)
- Negative results (REJECT, DEFER) were documented as carefully as positive ones (ACCEPT)
- Production behaviour was protected — no coordinator changes shipped without empirical justification
- The MATLAB-validated user methodology drove every accept/reject decision

This pattern is reusable for future model upgrade investigations: build the validation tool first, measure baselines, gate each variant, ship the verdicts.

## What comes next (out of v2.5.0 scope)

Potential v2.5.x / v2.6.0 candidates:

- **Open-Meteo `cloud_cover` integration** (enables v2.4.5 follow-up — alternative solar model gate)
- **FI Ridge retrain** with the v2.4.2 SE3 feature substituted for `ar_se3` (requires re-opening the v2.4.4 decision)
- **EE exogenous features** (Baltic gas spot, Estlink congestion) for a future EE model investigation
- **Battery storage modelling** (Phase 4 placeholder from earlier plan iterations)

None of these are committed; they're enumerated for future planning sessions.
