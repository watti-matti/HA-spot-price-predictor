"""v2.5.10 — Layer 3 (AR on Ridge residual) + targeted test of wind contribution.

User direction 2026-05-17 (after the v2.5.9 prediction-decomposition
figure exposed that Layer 1+2 alone give only R²=0.27): build Layer 3
on top of the v2.5.6 Ridge winner and see whether including wind
specifically improves the solution.

Architecture under test:

    L1 seasonal:     deterministic per-input vectors (v2.5.8 artifact)
    L2 Ridge:        Y_fi(t) ≈ β · features(t)            (residual model)
    L3 AR(1) / OU:   ε(t)   ≈ φ · ε(t-1) + η(t)           (persistence)

    full prediction at t made for t+h:
        ŷ(t+h | t) = seasonal_fi(t+h)
                   + β · features(t+h)            (assumes feature foresight)
                   + φ^h · ε(t)                   (AR decay over h steps)

Four variants compared in a single run:

  V0 — L1 only (seasonal baseline)
  V1 — L1 + L2(Y_fi_lag168, is_workday)         ← v2.5.6 winner, no L3
  V2 — V1 + L3 AR(1) on Ridge residual          ← adds Layer 3
  V3 — V2 + Y_wind                              ← user's question
  V4 — V3 + Y_solar + Y_temp                    ← full weather + L3

Reports test R², MAE, hedge CVaR at the three horizons (24 h, 48 h, 168 h)
for every variant. Also renders a side-by-side prediction overlay.

Reads only locally cached data + the v2.5.8 artifact + v2.5.3 solar
sub-model. No network call.
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

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402
from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)
from npk_cvar_hedge import optimize_hedge  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO / "custom_components" / "spot_price_predictor" / "data"

ARTIFACT = json.loads((DATA_DIR / "seasonal_components_default.json").read_text())
SOLAR_ART = json.loads((DATA_DIR / "solar_submodel_default.json").read_text())

# Match the v2.5.6 sweep
ALPHA = 0.05
TRAIN_FRAC = 0.55
HORIZONS = (24, 48, 168)
SAMPLE_START = pd.Timestamp("2025-08-04", tz="UTC")
SAMPLE_END   = pd.Timestamp("2025-08-18", tz="UTC")


# ── Data loading (mirror input_output_mapping.py) ───────────────────


def _build_dataframe() -> pd.DataFrame:
    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    import yaml
    region = yaml.safe_load((DATA_DIR / "finland.yaml").read_text())
    sites = region["weather_source"]["locations"]
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)
    # Clear-sky GHI
    if wea:
        ws_idx = None
        for s in wea.values():
            ws_idx = s.index if ws_idx is None else ws_idx.intersection(s.index)
        ts_np = ws_idx.values
        ghi = np.zeros(len(ws_idx), dtype=float)
        w_total = 0.0
        for site in SOLAR_ART["sites"]:
            sw = float(site.get("solar_weight", 0.0))
            if sw <= 0:
                continue
            ghi += sw * scs.clear_sky_series(
                ts_np, lat_deg=float(site["lat"]),
                lon_deg=float(site["lon"]),
                model=SOLAR_ART["clear_sky_model"])
            w_total += sw
        if w_total > 0:
            ghi /= w_total
        inputs["ghi_cs"] = pd.Series(ghi, index=ws_idx, name="ghi_cs")

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
    for name in df.columns:
        if name not in ARTIFACT["components"]:
            continue
        components = ARTIFACT["components"][name]
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, components)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    df["Y_fi_lag168"] = df["Y_fi"].shift(168)
    df["is_workday"]  = (df.index.weekday < 5).astype(float)
    return df.dropna()


# ── Ridge / AR fits ────────────────────────────────────────────────


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray,
              alpha: float = 1.0) -> np.ndarray:
    """L2-regularised normal-equation Ridge; intercept un-penalised."""
    p = X_train.shape[1]
    pen = alpha * np.eye(p)
    pen[0, 0] = 0.0
    return np.linalg.solve(X_train.T @ X_train + pen, X_train.T @ y_train)


def fit_ar1(eps_train: np.ndarray) -> tuple[float, float]:
    """Discrete-time AR(1) on the Ridge residual: ε_t = φ · ε_{t-1} + η.

    Returns (φ, residual σ). Robust to small samples and to constant
    residual (returns φ=0)."""
    x = np.asarray(eps_train, dtype=float)
    x_lag = x[:-1]
    x_now = x[1:]
    var = float(np.dot(x_lag, x_lag))
    if var <= 0:
        return 0.0, float(np.std(x))
    phi = float(np.dot(x_lag, x_now) / var)
    phi = max(-0.999, min(0.999, phi))  # enforce stationarity
    eta = x_now - phi * x_lag
    sigma_eta = float(np.std(eta))
    return phi, sigma_eta


def ar_propagate(eps_known: np.ndarray, phi: float, horizons: int) -> np.ndarray:
    """ε̂(t+h | t) = φ^h · ε(t) for h ≥ 1.

    Returns array shape (len(eps_known), horizons). For each t, the
    row gives the AR prediction at t+1 .. t+horizons (no noise — we
    care about the conditional mean, not stochastic simulation).
    """
    out = np.empty((len(eps_known), horizons), dtype=float)
    for h in range(1, horizons + 1):
        out[:, h - 1] = (phi ** h) * eps_known
    return out


# ── Evaluation helpers ─────────────────────────────────────────────


def hedge_cvar_pct(actual: np.ndarray, model: np.ndarray, lag: int,
                   alpha: float) -> float:
    fwd = np.concatenate([model[lag:], np.repeat(model[-1], lag)])
    try:
        res = optimize_hedge(np.diff(actual), np.diff(fwd), alpha=alpha)
        return 100.0 * (res["cvar_test_hist_unhedged"]
                        - res["cvar_test_hist_hedged"]) \
            / res["cvar_test_hist_unhedged"]
    except Exception:
        return float("nan")


def evaluate_variant(name: str, df: pd.DataFrame, features: list[str],
                     use_layer3: bool) -> dict:
    """Fit Ridge ± AR(1), reconstruct h-step-ahead predictions properly.

    For each horizon h ∈ HORIZONS we build a SEPARATE forecast series:
      prediction_h(t) = seasonal_fi(t)
                      + β · features(t)           (the row at t)
                      + φ^h · ε(t - h)            (AR propagated h steps)

    The AR contribution decays as φ^h so the L3 boost shrinks at long
    horizons — which is correct: AR can only carry information forward
    until it has decayed to ~0.

    For each horizon we compute test MAE, R², and hedge CVaR
    (without additional forward-shifting; the prediction is already
    for the same time as the actual at that index).
    """
    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split]
    test  = df.iloc[split:]
    n_full = len(df)

    # Layer 2 Ridge — fit on train residual Y_fi
    if features:
        X_train = np.column_stack([np.ones(len(train))]
                                  + [train[f].values for f in features])
        X_full  = np.column_stack([np.ones(n_full)]
                                  + [df[f].values    for f in features])
        coef = fit_ridge(X_train, train["Y_fi"].values, alpha=1.0)
        ridge_pred_full = X_full @ coef
    else:
        coef = np.array([0.0])
        ridge_pred_full = np.zeros(n_full)

    # Layer 3 AR(1) on Ridge residual (fit on training residual only)
    eps_train = train["Y_fi"].values - ridge_pred_full[:split]
    if use_layer3:
        phi, _ = fit_ar1(eps_train)
    else:
        phi = 0.0
    # ε(t) computed across full set so we can shift it by h for each
    # horizon. ε(t) is observable at time t under the assumption that
    # Y_fi(t) is known when forecasting t+h (true if h ≥ 1).
    eps_full = df["Y_fi"].values - ridge_pred_full

    # Per-horizon evaluation
    per_horizon: dict[int, dict[str, float]] = {}
    pred_series_h1: np.ndarray | None = None  # 1-step for plotting
    for h in (1,) + HORIZONS:
        # AR contribution at time t = φ^h · ε(t - h)
        ar_full = np.zeros(n_full, dtype=float)
        if use_layer3 and phi != 0:
            ar_full[h:] = (phi ** h) * eps_full[:n_full - h]
        full_pred = df["seasonal_fi"].values + ridge_pred_full + ar_full
        actual = df["fi"].values
        # Test portion only
        test_pred = full_pred[split:]
        test_act  = actual[split:]
        err = test_pred - test_act
        mae = float(np.mean(np.abs(err)))
        var_y = float(np.var(test_act))
        r2 = 1.0 - float(np.var(err)) / var_y if var_y > 0 else float("nan")
        # Hedge CVaR at horizon h: pass the full predicted series — the
        # hedge tool's `lag` parameter applies forward-shift; here we
        # want to test whether the prediction made *for time t* is a
        # good hedge for actual at *the same time t*. Set lag=h so that
        # `model[h:]` aligns with `actual[:-h]`, which is the correct
        # alignment: we predicted for t using ε(t-h), tested against
        # actual at t.
        if h in HORIZONS:
            cvar = hedge_cvar_pct(actual, full_pred, lag=h, alpha=ALPHA)
            per_horizon[h] = {"mae": mae, "r2": r2, "cvar_pct": cvar}
        if h == 1:
            pred_series_h1 = test_pred

    return {
        "name": name,
        "features": features,
        "use_layer3": use_layer3,
        "ridge_coef": coef.tolist(),
        "phi": phi,
        "per_horizon": per_horizon,
        # For plotting use the 1-step-ahead prediction (most informative
        # view of how all three layers contribute together)
        "fi_pred_test":   pred_series_h1,
        "fi_actual_test": df["fi"].values[split:],
        "ridge_pred_test": ridge_pred_full[split:],
        "ar_correction_test": (
            (phi * eps_full[split-1:-1] if (use_layer3 and phi and split >= 1)
             else np.zeros(len(test)))),
        "test_index": test.index,
    }


# ── Plotting ───────────────────────────────────────────────────────


def fig_variant_comparison(variants: list[dict], df: pd.DataFrame,
                           out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(variants), 1, figsize=(15, 3.0 * len(variants)),
                             sharex=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        ts_test = v["test_index"]
        mask = (ts_test >= SAMPLE_START) & (ts_test <= SAMPLE_END)
        idx_sample = ts_test[mask]
        actual = v["fi_actual_test"][mask]
        pred   = v["fi_pred_test"][mask]
        ax.plot(idx_sample, actual, "k-", lw=1.2, label="actual")
        ax.plot(idx_sample, pred,   "C0-", lw=1.2, label="predicted")
        ax.axhline(0, color="grey", lw=0.4)
        ax.set_ylabel("EUR/MWh")
        ph1 = v["per_horizon"][24]
        ph168 = v["per_horizon"][168]
        ax.set_title(
            f"{v['name']}  |  features: "
            f"{', '.join(v['features']) if v['features'] else '(none)'}  "
            f"|  L3 φ={v['phi']:.2f}\n"
            f"24h: MAE {ph1['mae']:.1f}, R² {ph1['r2']:.3f}, CVaR {ph1['cvar_pct']:+.1f}%   |   "
            f"168h: MAE {ph168['mae']:.1f}, R² {ph168['r2']:.3f}, CVaR {ph168['cvar_pct']:+.1f}%"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    axes[-1].tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(f"v2.5.10 — Layer 3 (AR) + wind contribution test  |  "
                 f"sample {SAMPLE_START.date()} → {SAMPLE_END.date()}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_layer_decomposition(v: dict, df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ts_test = v["test_index"]
    mask = (ts_test >= SAMPLE_START) & (ts_test <= SAMPLE_END)
    idx_sample = ts_test[mask]
    # Reconstruct per-layer contributions on the sample window
    seasonal_sample = df.loc[idx_sample, "seasonal_fi"].values
    ridge_sample    = v["ridge_pred_test"][mask]
    ar_sample       = v["ar_correction_test"][mask]
    actual_sample   = v["fi_actual_test"][mask]
    pred_sample     = v["fi_pred_test"][mask]

    ax = axes[0]
    ax.plot(idx_sample, actual_sample, "k-", lw=1.3, label="actual FI price")
    ax.plot(idx_sample, pred_sample,   "C0-", lw=1.3, label="full prediction (L1+L2+L3)")
    ax.plot(idx_sample, seasonal_sample, "C2-", lw=1.0, alpha=0.8,
            label="L1 seasonal")
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{v['name']} — actual vs prediction (with L3 AR)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    ax = axes[1]
    ax.fill_between(idx_sample, 0, ridge_sample, color="C0", alpha=0.3,
                    label="L2 Ridge contribution")
    ax.plot(idx_sample, ridge_sample, "C0-", lw=1.0)
    ax.fill_between(idx_sample, 0, ar_sample, color="C3", alpha=0.4,
                    label="L3 AR contribution")
    ax.plot(idx_sample, ar_sample, "C3-", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("EUR/MWh added on top of L1")
    ax.set_title("Per-layer contribution (what each layer adds to L1 seasonal)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(f"v2.5.10 layer decomposition — {v['name']}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Loading aligned data...", flush=True)
    df = _build_dataframe()
    print(f"  {len(df):,} hourly rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})", flush=True)

    variants_spec = [
        ("V0 L1 only",                            [],                              False),
        ("V1 L1+L2 (v2.5.6 winner, no L3)",       ["Y_fi_lag168", "is_workday"],   False),
        ("V2 V1 + L3 AR(1)",                      ["Y_fi_lag168", "is_workday"],   True),
        ("V3 V2 + Y_wind",                        ["Y_fi_lag168", "is_workday",
                                                    "Y_wind"],                       True),
        ("V4 V3 + Y_solar + Y_temp",              ["Y_fi_lag168", "is_workday",
                                                    "Y_wind", "Y_solar", "Y_temp"],   True),
    ]
    print("\nFitting variants...", flush=True)
    variants: list[dict] = []
    for name, features, use_layer3 in variants_spec:
        v = evaluate_variant(name, df, features, use_layer3)
        variants.append(v)
        coef_str = ", ".join(f"{c:+.3f}" for c in v["ridge_coef"])
        print(f"\n  {name}  (φ={v['phi']:+.3f})", flush=True)
        print(f"     ridge coefs: [{coef_str}]", flush=True)
        print(f"     {'horizon':>8s}  {'MAE':>7s}  {'R²':>7s}  {'CVaR':>8s}",
              flush=True)
        for h in HORIZONS:
            ph = v["per_horizon"][h]
            print(f"     {h:7d}h  {ph['mae']:7.2f}  {ph['r2']:+7.3f}  "
                  f"{ph['cvar_pct']:+7.2f}%",
                  flush=True)

    print("\nRendering variant-comparison figure...", flush=True)
    fig_variant_comparison(variants,
                           df, FIGURES_DIR / "v2510_variants_comparison.png")

    print("Rendering layer decomposition for the 168h winner (best CVaR)...",
          flush=True)
    def _cvar_168(v):
        c = v["per_horizon"][168]["cvar_pct"]
        return c if not (c is None or np.isnan(c)) else -1e9
    winner = max(variants, key=_cvar_168)
    fig_layer_decomposition(winner, df,
                            FIGURES_DIR / "v2510_winner_layer_decomp.png")
    print(f"  168h-CVaR winner: {winner['name']}", flush=True)

    # Markdown
    md = RESULTS_DIR / "v2510_layer3_ar_wind.md"
    lines = [
        "# v2.5.10 — Layer 3 (AR on Ridge residual) + wind contribution",
        "",
        f"**Window:** {df.index[0].date()} → {df.index[-1].date()} "
        f"({len(df):,} hourly rows)",
        f"**Split:** chronological 55 / 45",
        f"**Layer 3:** AR(1) on Layer-2 Ridge residual `ε(t) = φ · ε(t-1) + η(t)`",
        "",
        "User question 2026-05-17: does adding `Y_wind` to the Ridge layer "
        "(with Layer 3 in place) improve the FI prediction?",
        "",
        "## Variant comparison (per-horizon, properly evaluated)",
        "",
        "MAE / R² / CVaR-reduction reported separately at each horizon.",
        "The AR(1) contribution at horizon h decays as φ^h, so its boost",
        "shrinks at long lead times.",
        "",
        "| Variant | L3 | φ | h=24h MAE / R² / CVaR | h=48h MAE / R² / CVaR | h=168h MAE / R² / CVaR |",
        "|---|:-:|---:|---|---|---|",
    ]
    for v in variants:
        p24, p48, p168 = (v["per_horizon"][h] for h in (24, 48, 168))
        lines.append(
            f"| {v['name']} | "
            f"{'✓' if v['use_layer3'] else '·'} | "
            f"{v['phi']:+.2f} | "
            f"{p24['mae']:.2f} / {p24['r2']:+.3f} / {p24['cvar_pct']:+.1f}% | "
            f"{p48['mae']:.2f} / {p48['r2']:+.3f} / {p48['cvar_pct']:+.1f}% | "
            f"{p168['mae']:.2f} / {p168['r2']:+.3f} / {p168['cvar_pct']:+.1f}% |"
        )

    lines += [
        "",
        "## Variant overlay",
        "",
        "![Variants](figures/v2510_variants_comparison.png)",
        "",
        f"## Winning variant ({winner['name']}) — layer decomposition",
        "",
        "![Layers](figures/v2510_winner_layer_decomp.png)",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v2510_layer3_ar_wind.py",
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
