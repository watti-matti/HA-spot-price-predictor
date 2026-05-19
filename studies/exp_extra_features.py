"""Experiment — does adding nuclear-deficit and/or cross-border features to
the v2.8.1 L2 Ridge improve the spot forecast?

This is research code that runs OFF the production path:
  * does NOT modify any artifact under data/
  * does NOT touch RIDGE_FEATURES, the pipeline, or sensors
  * writes findings to studies/results/experiment_extra_l2_features.{md,json}

Variants
--------
  B0          baseline (v2.8.1 six features — sanity replication)
  B1          + nuclear_deficit
  B2          + Y_se3, Y_ee, export_potential_se3 (v2.2 lineage; no SE1)
  B2_se1      + Y_se1, Y_se3, Y_ee, export_potential_se3 (adds SE1 — user
              note 2026-05-19: limited transit capacity makes SE1 distinct
              from SE3; the v2.2 collinearity-rejection was wrong)
  B2_transit  + Y_se1, Y_se3, Y_ee + signed spreads (transit decoupling)
  B3          B1 + B2_se1 (all candidate signals together)

For each variant we:
  1. Fit a Ridge on the train split (matches v2513 TRAIN_FRAC).
  2. Fit AR(1) on the Ridge residual.
  3. Compose L1 + L2 + L3 prediction on the test split.
  4. Report MAE, R², and the same metrics restricted to the extreme-
     price subset (|spot| > 100 EUR/MWh).
  5. Apply the NPK-CVaR hedge gate (npk_cvar_hedge.optimize_hedge) using
     the model's prediction as a hedge instrument vs realised spot.
     Per v2.5.6, this is the primary validity criterion (threshold:
     +0.3 pp CVaR-reduction per added feature).

Data window: cached parquets cover 2018-2026 weather but FI prices only
start meaningfully in 2023. The inner join gives 2023-01-08 → 2026-04-27
≈ 28.8 k hours, all within the user's specified "2023-2026" scope.

Prerequisites: cached parquets under output/ (fi_prices, fi_weather,
fi_neighbor_prices, fi_grid_data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))
sys.path.insert(0, str(REPO / "studies"))

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402

from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)
from v2510_layer3_ar_wind import fit_ridge, fit_ar1, TRAIN_FRAC  # noqa: E402
from v2512_sigmoid_turbine_curve import sigmoid_turbine_rho  # noqa: E402
from v2511_physics_features import solar_effective  # noqa: E402
from npk_cvar_hedge import optimize_hedge, historical_cvar  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


REPO_DATA = REPO / "custom_components" / "spot_price_predictor" / "data"
RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = REPO / "output"

# Shipped seasonal artifact — same L1 components the production Pipeline uses.
SEASONAL_ARTIFACT = json.loads(
    (REPO_DATA / "seasonal_components_default.json").read_text()
)
SOLAR_ARTIFACT = json.loads(
    (REPO_DATA / "solar_submodel_default.json").read_text()
)


# ── Data assembly ────────────────────────────────────────────────────


def load_grid() -> dict[str, pd.Series]:
    """Fingrid streams (nuclear_mw is already normalised to [0, 1])."""
    df = pd.read_parquet(OUTPUT_DIR / "fi_grid_data.parquet")
    return {col: df[col] for col in df.columns}


def build_dataframe() -> pd.DataFrame:
    """Build the same dataframe `v2513_layer4_spike_model._build_dataframe`
    produces, plus the candidate extra features (`nuclear_deficit`,
    `ar_se3`, `ar_ee`, `export_potential_se3`)."""
    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    inputs.update(load_grid())

    import yaml
    region = yaml.safe_load((REPO_DATA / "finland.yaml").read_text())
    sites = region["weather_source"]["locations"]
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)

    # GHI clear-sky proxy (matches v2513) — used for solar_effective.
    if wea:
        ws_idx = None
        for s in wea.values():
            ws_idx = s.index if ws_idx is None else ws_idx.intersection(s.index)
        ts_np = ws_idx.values
        ghi = np.zeros(len(ws_idx), dtype=float)
        w_total = 0.0
        for site in SOLAR_ARTIFACT["sites"]:
            sw = float(site.get("solar_weight", 0.0))
            if sw <= 0:
                continue
            ghi += sw * scs.clear_sky_series(
                ts_np, lat_deg=float(site["lat"]),
                lon_deg=float(site["lon"]),
                model=SOLAR_ARTIFACT["clear_sky_model"])
            w_total += sw
        if w_total > 0:
            ghi /= w_total
        inputs["ghi_cs"] = pd.Series(ghi, index=ws_idx, name="ghi_cs")

    # Inner join on shared timestamps.
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common).dropna() for k, s in inputs.items()}
    common = None
    for s in inputs.values():
        common = s.index if common is None else common.intersection(s.index)
    inputs = {k: s.reindex(common) for k, s in inputs.items()}

    df = pd.concat(inputs.values(), axis=1)
    df.columns = list(inputs.keys())
    ts_np = pd.DatetimeIndex(common, tz="UTC").values

    # L1 deseasonalisation using the shipped components.
    for name in df.columns:
        if name not in SEASONAL_ARTIFACT["components"]:
            continue
        components = SEASONAL_ARTIFACT["components"][name]
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, components)
        df[f"Y_{name}"] = df[name].values - df[f"seasonal_{name}"].values

    # L2 physics features.
    df["Y_fi_lag168"] = df["Y_fi"].shift(168)
    df["is_workday"] = (df.index.weekday < 5).astype(float)
    df["sigmoid_wind_rho"] = sigmoid_turbine_rho(
        df["wind"].values, df["temp"].values)
    df["solar_effective"] = solar_effective(
        df["solar"].values, df["temp"].values)
    for name in ("sigmoid_wind_rho", "solar_effective"):
        comp = sd.fit_components(df[name].values, ts_np,
                                  depth=("P_hour", "P_week"),
                                  smooth={"P_week": 7})
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, comp)
        df[f"Y_{name}"] = df[name].values - df[f"seasonal_{name}"].values

    # Candidate extra features.
    # nuclear_deficit ∈ [0, 1]: bigger ⇒ more outage. Mean-centre so the
    # Ridge intercept stays meaningful when added/removed across variants.
    nuc_def = np.clip(1.0 - df["nuclear_mw"].values, 0.0, 1.0)
    df["nuclear_deficit"] = nuc_def - nuc_def[~np.isnan(nuc_def)].mean()

    # Cross-border: deseasonalised neighbour prices.
    # legacy v2.2 used a proper AR(2) day-type model; here we reuse the
    # L1 components already shipped for se1/se3/ee, which is the same
    # deseasonalising step the FI side uses. Result: Y_se3 ≈ "AR(2) -ish"
    # deviation of SE3 from its hour-of-week climatology.
    for nb in ("se3", "ee", "se1"):
        col = f"Y_{nb}"
        if col not in df.columns:
            df[col] = 0.0

    # export_potential_se3: legacy v2.2 used a 7-day rolling spread
    # `fi_7d − se3_7d` (lagged, leak-free). The same-hour form
    # `max(0, −(fi(t) − se3(t)))` is NOT safe — it contains the target
    # variable on the right-hand side and the Ridge can trivially
    # recover fi(t) from `se3(t) + spread`. We rebuild it from a
    # 7-day shift+rolling-mean of fi vs se3 so the feature stays
    # leak-free.
    fi_7d  = df["fi"].shift(168).rolling(168, min_periods=24).mean()
    se3_7d = df["se3"].shift(168).rolling(168, min_periods=24).mean()
    ep_raw = np.maximum(0.0, -(fi_7d.values - se3_7d.values))
    df["export_potential_se3"] = (
        ep_raw - np.nanmean(ep_raw[np.isfinite(ep_raw)])
    )

    # Cross-border transit-saturation indicators (user note 2026-05-19).
    # When the spread between zones is small the markets are coupled
    # (transit capacity unsaturated, prices level). When the spread is
    # large the transit is saturated — the zones decouple. The *signed*
    # spread carries direction-of-imbalance information; the *absolute*
    # spread carries saturation level.
    #
    # CRITICAL: only neighbour-vs-neighbour spreads are leak-free.
    # `fi − se1` etc. inject the target into the feature set and the
    # Ridge will trivially overfit; do NOT compute those.
    for left, right in (("se1", "se3"), ("se3", "ee"), ("se1", "ee")):
        signed = df[left].values - df[right].values
        abs_sp = np.abs(signed)
        df[f"spread_{left}_{right}"] = signed - np.nanmean(signed)
        df[f"abs_spread_{left}_{right}"] = abs_sp - np.nanmean(abs_sp)

    return df.dropna()


# ── Variant fitting and evaluation ───────────────────────────────────


_CORE = [
    "Y_fi_lag168", "is_workday",
    "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
]

VARIANTS: dict[str, list[str]] = {
    "B0_baseline":      list(_CORE),
    "B1_nuclear":       _CORE + ["nuclear_deficit"],
    "B2_cross_border":  _CORE + ["Y_se3", "Y_ee", "export_potential_se3"],
    # User direction 2026-05-19: SE1 was rejected in v2.2 as collinear
    # with SE3, but limited transit capacity makes SE1 distinct and
    # potentially valuable. v2.5.1 already showed that the (Y_se1, Y_se3)
    # pair with opposite signs (+1.61 / −1.60) delivers +0.55 pp CVaR
    # reduction at 7 d. Re-test with hedge gate.
    "B2_se1":           _CORE + ["Y_se1", "Y_se3", "Y_ee",
                                  "export_potential_se3"],
    # Explicit transit-saturation features: SE1↔SE3 and SE3↔EE signed
    # spreads (direction of imbalance) plus the absolute SE1↔SE3
    # spread (saturation level). Tests whether the spread *itself* —
    # rather than individual Y_se1 / Y_se3 — is the operative signal.
    # All neighbour-only — no FI on the RHS to avoid target leakage.
    "B2_transit":       _CORE + ["Y_se1", "Y_se3", "Y_ee",
                                  "spread_se1_se3", "abs_spread_se1_se3",
                                  "spread_se3_ee"],
    "B3_combined":      _CORE + ["nuclear_deficit",
                                  "Y_se1", "Y_se3", "Y_ee",
                                  "export_potential_se3"],
}


def fit_and_evaluate(df: pd.DataFrame, features: list[str],
                     alpha: float = 1.0) -> dict:
    """Fit Ridge + AR(1) on the train split, evaluate on the test split.

    The forecast under test mirrors `Pipeline.compute_forecast` for a
    cold-start single-step horizon: ŷ(t) = L1_seasonal(t) + L2_ridge(t)
    + φ·η(t−1). Bias EMA / fan chart / softplus floor are intentionally
    not applied here — we want to isolate the impact of the L2 feature
    set on the point forecast.
    """
    n = len(df)
    split = int(n * TRAIN_FRAC)

    y = df["Y_fi"].values
    X = np.column_stack(
        [np.ones(n)] + [df[f].values for f in features]
    )

    coef = fit_ridge(X[:split], y[:split], alpha=alpha)
    ridge_pred = X @ coef
    eps = y - ridge_pred
    phi, _ = fit_ar1(eps[:split])

    # AR(1) one-step-ahead on the test split.
    ar_contribution = np.zeros(n, dtype=float)
    ar_contribution[1:] = phi * eps[:-1]

    pred_residual = ridge_pred + ar_contribution
    spot_actual = df["fi"].values
    spot_pred = df["seasonal_fi"].values + pred_residual

    err = spot_actual - spot_pred

    def _metrics(mask: np.ndarray) -> dict:
        e = err[mask]
        y_ = spot_actual[mask]
        if e.size == 0:
            return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                    "r2": float("nan")}
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
        return {
            "n": int(e.size),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }

    test_mask = np.zeros(n, dtype=bool)
    test_mask[split:] = True
    extreme_mask = test_mask & (np.abs(spot_actual) > 100.0)

    # ── NPK-CVaR hedge gate (v2.5.6 acceptance criterion) ──
    # Use the model's prediction as the hedge instrument vs realised
    # spot. Threshold per v2.5.6: +0.3 pp CVaR-reduction per added
    # feature beyond B0 baseline.
    hedge = {}
    try:
        rS = np.diff(spot_actual)
        rF = np.diff(spot_pred)
        hedge_full = optimize_hedge(rS, rF, alpha=0.05, train_frac=TRAIN_FRAC)
        unhedged = hedge_full["cvar_test_hist_unhedged"]
        hedged   = hedge_full["cvar_test_hist_hedged"]
        reduction_pp = (
            100.0 * (unhedged - hedged) / unhedged if unhedged > 0
            else float("nan")
        )
        hedge = {
            "h_hat": hedge_full["h_hat"],
            "cvar_test_unhedged": unhedged,
            "cvar_test_hedged":   hedged,
            "cvar_reduction_pp":  reduction_pp,
        }
    except Exception as exc:
        hedge = {"error": repr(exc)}

    return {
        "features": features,
        "n_features_with_intercept": len(features) + 1,
        "ridge_coef": coef.tolist(),
        "phi": float(phi),
        "test_overall": _metrics(test_mask),
        "test_extreme_gt100": _metrics(extreme_mask),
        "hedge": hedge,
        "train_size": split,
        "test_size": n - split,
    }


# ── Reporting ────────────────────────────────────────────────────────


# Note: the findings .md carries a hand-authored TL;DR + recommendation
# section at the top, written after the leak fix. Re-running this script
# regenerates only the variants table, deltas table, decision rule,
# method, and feature-list sections — it preserves the TL;DR block by
# splicing it back in. If the underlying numbers change, update the
# TL;DR by hand to match.
_TLDR_MARKER_BEGIN = "<!-- TLDR-BEGIN -->"
_TLDR_MARKER_END = "<!-- TLDR-END -->"


def _read_tldr(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if _TLDR_MARKER_BEGIN not in text or _TLDR_MARKER_END not in text:
        # Fallback: heuristic — preserve everything between "# Experiment"
        # title and the first "## Variants" heading, exclusive.
        start = text.find("\n## TL;DR")
        end = text.find("\n## Variants")
        if start > 0 and end > start:
            return text[start:end].rstrip() + "\n\n"
        return None
    s = text.find(_TLDR_MARKER_BEGIN) + len(_TLDR_MARKER_BEGIN)
    e = text.find(_TLDR_MARKER_END)
    return text[s:e].strip("\n") + "\n\n"


def write_findings_md(results: dict, out_path: Path) -> None:
    rows = []
    for name, res in results.items():
        ov = res["test_overall"]
        ex = res["test_extreme_gt100"]
        h = res.get("hedge") or {}
        hedge_cell = (f"{h['cvar_reduction_pp']:.2f}"
                      if "cvar_reduction_pp" in h else "—")
        rows.append(
            f"| {name} | {res['n_features_with_intercept']} | "
            f"{ov['mae']:.2f} | {ov['r2']:+.3f} | "
            f"{ex['mae']:.2f} | {ex['r2']:+.3f} | "
            f"{hedge_cell} | {res['phi']:+.3f} |"
        )
    table = "\n".join(rows)

    baseline = results["B0_baseline"]
    base_hedge = baseline.get("hedge") or {}
    base_cvar_red = base_hedge.get("cvar_reduction_pp", 0.0)
    deltas = []
    for name, res in results.items():
        if name == "B0_baseline":
            continue
        d_mae = res["test_overall"]["mae"] - baseline["test_overall"]["mae"]
        d_r2 = res["test_overall"]["r2"] - baseline["test_overall"]["r2"]
        d_mae_ext = (res["test_extreme_gt100"]["mae"]
                     - baseline["test_extreme_gt100"]["mae"])
        h = res.get("hedge") or {}
        d_hedge = (h.get("cvar_reduction_pp", float("nan")) - base_cvar_red
                   if base_cvar_red == base_cvar_red else float("nan"))
        n_extra = (res["n_features_with_intercept"]
                   - baseline["n_features_with_intercept"])
        hedge_threshold = 0.3 * max(1, n_extra)   # +0.3 pp per added feature

        # Primary criterion per v2.5.6: hedge-CVaR reduction relative to
        # the baseline must beat 0.3 pp per added feature. Secondary:
        # extreme-tail MAE drop ≥ 1 EUR/MWh. Tertiary: overall MAE is
        # reported for interpretability but the v2.5.6 explicit design
        # rationale ("7-day CVaR accuracy as the primary metric") means
        # we tolerate up to +2 EUR/MWh drift on overall MAE provided the
        # hedge gate passes — on calm hours the cross-border features
        # add a little noise; on spike hours they pay off, and the spike
        # hours are what carries the tail-risk cost.
        hedge_accept = (d_hedge == d_hedge) and (d_hedge >= hedge_threshold)
        extreme_accept = d_mae_ext <= -1.0
        overall_severe = d_mae > +2.0   # only block if drift is severe
        extreme_regress = d_mae_ext > +1.0
        if overall_severe or extreme_regress:
            verdict = "**reject (severe regression)**"
        elif hedge_accept and extreme_accept:
            verdict = "**accept (hedge gate + extreme tail)**"
        elif hedge_accept:
            verdict = "**accept (hedge gate)**"
        elif extreme_accept:
            verdict = "**provisional (extreme-tail only; hedge gate missed)**"
        else:
            verdict = "neutral (no material gain)"
        deltas.append(
            f"| {name} | {d_mae:+.2f} | {d_mae_ext:+.2f} | "
            f"{d_hedge:+.2f} | {hedge_threshold:.2f} | {verdict} |"
        )
    deltas_table = "\n".join(deltas)

    tldr = _read_tldr(out_path)
    tldr_block = (tldr if tldr else "")

    md = f"""# Experiment — extra L2 features (nuclear deficit, cross-border + SE1)

Branch: `experiment/extra-l2-features`. Off-tree research only — no
production artefact change. Script:
[`studies/exp_extra_features.py`](../exp_extra_features.py).

{tldr_block}## Variants

| Variant | n_feat | MAE | R² | MAE (|spot|>100) | R² (|spot|>100) | Hedge CVaR red. (pp) | φ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table}

## Delta vs B0 baseline (test split)

| Variant | Δ MAE overall | Δ MAE \|spot\|>100 | Δ hedge CVaR (pp) | hedge threshold (pp) | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|
{deltas_table}

**Decision rule (per v2.5.6).**

- **Primary criterion: NPK-CVaR hedge gate.** A variant adds real
  signal iff its hedged-portfolio CVaR is lower than the baseline's by
  at least **+0.3 pp per added feature**. This is the v2.5.6
  acceptance threshold; passing it means the model captures
  hedge-relevant tail risk that the baseline misses.
- **Secondary criterion: extreme-price-hour MAE.** Test hours with
  |spot| > 100 EUR/MWh — the spike subset where the v2.5.13 work
  showed the v2.8.1 baseline is weakest. A drop of ≥ 1 EUR/MWh on this
  bucket is operationally meaningful.
- **Regression guard.** Reject only on *severe* regression — overall
  MAE drift > 2 EUR/MWh, or extreme-tail MAE worsening > 1 EUR/MWh.
  Small overall-MAE drift (≤ 2 EUR/MWh) is tolerated when the hedge
  gate passes, because v2.5.6 established the hedge gate (and not
  average MAE) as the operational acceptance test: cross-border
  features add a little variance on calm hours but pay off
  disproportionately on the spike hours that carry the tail cost.

The hedge gate is the canonical v2.5.x acceptance test; MAE / extreme
MAE are reported for interpretability but do not substitute for the
hedge gate.

## Method

- Data: cached parquets under `output/` (≈ 4 years, hourly).
- Train/test: time-ordered, `TRAIN_FRAC = {TRAIN_FRAC}`
  (train = first {results['B0_baseline']['train_size']} hours,
   test  = last  {results['B0_baseline']['test_size']} hours).
- Ridge α = 1.0, intercept un-penalised.
- AR(1) φ fitted on the Ridge residual of the train split, then applied
  one-step-ahead on the test split.
- Forecast under test: L1 seasonal + L2 ridge + φ·ε(t−1).
  Hourly-bias EMA, softplus floor, and L4 GPD POT bands are NOT applied
  here — the goal is to isolate the impact of the L2 feature set on
  the point forecast.
- Extreme-price bucket: test hours with |spot| > 100 EUR/MWh, where the
  v2.5.13 work showed the model is weakest.

## Candidate features (legacy v2.2 lineage, leak-free)

- `nuclear_deficit ∈ [0, 1]` — `max(0, 1 − nuclear_mw)` where
  `nuclear_mw` is Fingrid #188 normalised by max-fleet 4 372 MW.
  Mean-centred for stable Ridge weighting.
- `Y_se1`, `Y_se3`, `Y_ee` — neighbour spot prices deseasonalised
  against the shipped per-zone hourly+weekly L1 components. The legacy
  v2.2 `ar_se3` / `ar_ee` used a proper AR(2) daytype-deviation; this
  is a simpler analogue. **SE1 is included** per user direction
  2026-05-19: limited FI↔SE3 / SE3↔SE1 transit capacity makes SE1
  distinct from SE3 (the v2.2 collinearity-rejection assumed perfect
  coupling; in reality the transit decouples SE1 from SE3 whenever
  capacity saturates).
- `spread_se1_se3 = se1 − se3`, mean-centred. Signed neighbour-zone
  spread: when SE1 and SE3 prices diverge the transit capacity between
  them is saturated and the two zones decouple. Leak-free (no FI on
  the RHS).
- `abs_spread_se1_se3 = |se1 − se3|`, mean-centred. Magnitude of the
  same spread — saturation level.
- `spread_se3_ee = se3 − ee`, mean-centred. Signed SE3↔EE spread,
  same logic.
- `export_potential_se3` — built on **lagged** FI and SE3 (7-day
  rolling means shifted by 168 h) so the feature is leak-free. The
  same-hour form `max(0, −(fi(t) − se3(t)))` was rejected after a
  same-hour FI value in the feature set let the Ridge trivially
  recover the target (B2_transit reached MAE 2.4 / hedge CVaR red.
  78 pp — implausibly good — under that buggy form).

## Open follow-ups (deferred to separate experiments)

- **SE3 / SE1 / EE transit-capacity saturation.** A continuous
  saturation indicator (e.g. `min(|spread|, cap) / cap` calibrated on
  historical transit-capacity-out data) was *not* tested here. The
  existing `Y_se3` / `Y_ee` already proxy coupling; a saturation
  indicator only adds information when transit *de*couples (large
  spread regime). Belongs in a follow-up if B2 / B3 are accepted.
- **`nuclear_x_scarcity` interaction.** Legacy v2.2 multiplied
  `nuclear_deficit × wind_log_scarcity` to amplify outage impact under
  cold-and-windless conditions. Not tested here; can be a follow-up if
  B1 is accepted.
"""
    out_path.write_text(md, encoding="utf-8")


def main() -> None:
    print("Building dataframe…", flush=True)
    df = build_dataframe()
    print(f"  rows = {len(df):,}, span = "
          f"{df.index[0].date()} → {df.index[-1].date()}", flush=True)

    results = {}
    for name, features in VARIANTS.items():
        print(f"Fitting {name} ({len(features) + 1} feats incl. intercept)…",
              flush=True)
        results[name] = fit_and_evaluate(df, features)
        ov = results[name]["test_overall"]
        ex = results[name]["test_extreme_gt100"]
        h = results[name].get("hedge") or {}
        hedge_txt = (f"hedge red = {h['cvar_reduction_pp']:.2f} pp"
                     if "cvar_reduction_pp" in h else "hedge n/a")
        print(f"  test MAE = {ov['mae']:.2f}  R² = {ov['r2']:+.3f}  "
              f"|spot|>100 MAE = {ex['mae']:.2f}  {hedge_txt}", flush=True)

    md_path = RESULTS_DIR / "experiment_extra_l2_features.md"
    json_path = RESULTS_DIR / "experiment_extra_l2_features.json"
    write_findings_md(results, md_path)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {md_path.relative_to(REPO)}")
    print(f"Wrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
