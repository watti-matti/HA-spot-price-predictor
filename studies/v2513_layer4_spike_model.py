"""v2.5.13 — Layer 4: GPD POT spike model on FI Ridge+AR residual.

User direction 2026-05-17: after the sigmoid turbine curve (v2.5.12),
move to Layer 4. The architectural plan from v2.5.10 calls for:

    L1 seasonal       (v2.5.8 artifact)
    L2 Ridge          (v2.5.12 V_sigmoid_full: sigmoid_wind_rho +
                       solar_effective + temp + Y_fi_lag168 + is_workday)
    L3 AR(1)          (φ ≈ 0.90 on Ridge residual)
    L4 GPD POT spike  (this patch — heavy-tail noise structure)

v2.5.2 already proved GPD POT feasibility on cross-border zones
(SE3/SE1/EE). Same methodology applied here to the FI post-AR
residual `η(t)` — what the AR layer couldn't explain. If FI shares
the heavy-tail structure of its neighbours, GPD POT should give
accurate CVaR predictions at α ∈ {0.05, 0.01, 0.001}.

Output:
  studies/results/v2513_layer4_spike.md
  studies/results/figures/v2513_layer4_diagnostics.png
  studies/results/figures/v2513_layer4_cvar_backtest.png
  custom_components/spot_price_predictor/data/spike_model_default.json
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

import matplotlib.pyplot as plt  # noqa: E402

import seasonal_decomposition as sd  # noqa: E402
import solar_clear_sky as scs  # noqa: E402
import importlib.util as _ilu
spec = _ilu.spec_from_file_location(
    "peak_model_test",
    REPO / "studies" / "peak_model_feasibility.py",
)
_pmod = _ilu.module_from_spec(spec); spec.loader.exec_module(_pmod)
fit_gpd_pot = _pmod.fit_gpd_pot
cvar_normal = _pmod.cvar_normal
hill_estimator = _pmod.hill_estimator
mean_excess_curve = _pmod.mean_excess_curve

from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)
from v2510_layer3_ar_wind import (  # noqa: E402
    fit_ridge, fit_ar1, TRAIN_FRAC,
)
from v2512_sigmoid_turbine_curve import (  # noqa: E402
    sigmoid_turbine_rho,
)
from v2511_physics_features import solar_effective  # noqa: E402

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


# ── Data + L1–L3 model (replicate V_sigmoid_full from v2.5.12) ─────


def _build_dataframe() -> pd.DataFrame:
    inputs: dict[str, pd.Series] = {}
    inputs["fi"] = load_fi_prices()
    inputs.update(load_neighbor_prices())
    import yaml
    region = yaml.safe_load((DATA_DIR / "finland.yaml").read_text())
    sites = region["weather_source"]["locations"]
    wea = load_weather_extended(WEATHER_WINDOW_START, WINDOW_END, sites)
    inputs.update(wea)
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
    df["sigmoid_wind_rho"] = sigmoid_turbine_rho(
        df["wind"].values, df["temp"].values)
    df["solar_effective"] = solar_effective(
        df["solar"].values, df["temp"].values)

    for name in ("sigmoid_wind_rho", "solar_effective"):
        comp = sd.fit_components(df[name].values, ts_np,
                                  depth=("P_hour", "P_week"),
                                  smooth={"P_week": 7})
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, comp)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    return df.dropna()


def fit_full_model(df: pd.DataFrame) -> dict:
    """Fit L1 (already in df) + L2 Ridge + L3 AR(1) per v2.5.12 V_sigmoid_full.

    Returns:
        dict with ridge coefs, φ, and `eta`: post-AR residual at every t
        in the FULL dataframe (eta is what Layer 4 will model).
    """
    features = ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
                "Y_solar_effective", "Y_temp"]
    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split]
    n_full = len(df)
    X_train = np.column_stack([np.ones(len(train))]
                              + [train[f].values for f in features])
    X_full  = np.column_stack([np.ones(n_full)]
                              + [df[f].values    for f in features])
    coef = fit_ridge(X_train, train["Y_fi"].values, alpha=1.0)
    ridge_pred = X_full @ coef
    eps_train = train["Y_fi"].values - ridge_pred[:split]
    phi, sigma_eta_fit = fit_ar1(eps_train)
    # AR(1) one-step-ahead residual: η(t) = ε(t) - φ · ε(t-1)
    eps_full = df["Y_fi"].values - ridge_pred
    eta = np.zeros_like(eps_full)
    eta[1:] = eps_full[1:] - phi * eps_full[:-1]
    eta[0] = eps_full[0]   # boundary
    return {
        "features": features,
        "ridge_coef": coef.tolist(),
        "phi": phi,
        "ridge_pred": ridge_pred,
        "eps":        eps_full,
        "eta":        eta,
        "split":      split,
    }


# ── CVaR helpers ───────────────────────────────────────────────────


def empirical_cvar(x: np.ndarray, alpha: float) -> float:
    """Conditional Value-at-Risk: average of worst α-fraction of LOSSES.
    Convention here: loss = -x for the "left-tail" side, but for the
    symmetric tail we report E[|x| : |x| > q_{1-α}]."""
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x >= q]
    return float(tail.mean()) if tail.size > 0 else float("nan")


def gpd_cvar(gpd_fit: dict, alpha: float, n_obs: int) -> float:
    """CVaR_α derived from the GPD POT right-tail parameters.

    Standard formula: above threshold u, exceedances follow
    GP(σ, ξ); the (1-α)-quantile and the corresponding tail mean are
    closed-form."""
    if gpd_fit is None:
        return float("nan")
    n_exc = gpd_fit.get("n", 0)
    if n_exc < 20:
        return float("nan")
    u = gpd_fit["threshold"]
    sigma = gpd_fit["scale"]
    xi    = gpd_fit["shape"]
    p_exc = gpd_fit.get("p_exceed", n_exc / max(1, n_obs))
    if p_exc <= 0 or alpha <= 0:
        return float("nan")
    # GPD POT VaR_alpha at right tail
    if alpha >= p_exc:
        return float("nan")
    if abs(xi) < 1e-6:
        var_alpha = u + sigma * np.log(p_exc / alpha)
    else:
        var_alpha = u + (sigma / xi) * ((p_exc / alpha) ** xi - 1.0)
    # CVaR conditional on exceeding VaR: GPD mean above VaR_alpha
    if xi < 1.0:
        cvar_alpha = var_alpha + (sigma + xi * (var_alpha - u)) / (1.0 - xi)
    else:
        return float("inf")
    return float(cvar_alpha)


# ── Plotting ───────────────────────────────────────────────────────


def fig_diagnostics(eta_train: np.ndarray, gpd_fit: dict,
                    hill_alpha_right: float, hill_alpha_left: float,
                    out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) histogram + Normal overlay
    ax = axes[0, 0]
    mu, sigma = eta_train.mean(), eta_train.std()
    bins = np.linspace(eta_train.min(), eta_train.max(), 80)
    ax.hist(eta_train, bins=bins, density=True, alpha=0.6, color="C0",
            label=f"η(t) residual  σ={sigma:.1f}")
    xs = np.linspace(eta_train.min(), eta_train.max(), 200)
    gauss = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(xs, gauss, "k--", lw=1.0, label=f"N({mu:+.1f}, {sigma:.1f}²)")
    ax.set_xlabel("η [EUR/MWh]")
    ax.set_ylabel("density")
    ax.set_title("Post-AR residual distribution")
    ax.legend(loc="upper right", fontsize=9)

    # (b) Q-Q vs Normal
    ax = axes[0, 1]
    sorted_eta = np.sort(eta_train)
    n = len(sorted_eta)
    p = (np.arange(n) + 0.5) / n
    from scipy.stats import norm
    theoretical = norm.ppf(p, loc=mu, scale=sigma)
    ax.plot(theoretical, sorted_eta, "C0.", ms=2, alpha=0.6)
    lim = [min(theoretical.min(), sorted_eta.min()),
           max(theoretical.max(), sorted_eta.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Normal theoretical quantile")
    ax.set_ylabel("Empirical quantile η(t)")
    ax.set_title("Q-Q plot — divergence at tails = heavy tail")

    # (c) Mean excess function (right tail)
    ax = axes[1, 0]
    us, me = mean_excess_curve(eta_train, n_points=40)
    ax.plot(us, me, "C2-o", ms=3)
    if gpd_fit:
        ax.axvline(gpd_fit["threshold"], color="r", lw=1.0, ls="--",
                    label=f"chosen threshold u={gpd_fit['threshold']:.1f}")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("threshold u")
    ax.set_ylabel("e(u) = E[Y - u | Y > u]")
    ax.set_title("Mean excess function — linear ⇒ GPD fits")

    # (d) Hill plot estimate (text summary, not full plot)
    ax = axes[1, 1]
    ax.axis("off")
    info = [
        f"FI post-AR residual statistics (training)",
        f"",
        f"  n          = {len(eta_train):,}",
        f"  mean       = {mu:+.3f}",
        f"  σ          = {sigma:.3f}",
        f"  skew       = {float(((eta_train - mu)**3).mean() / sigma**3):+.3f}",
        f"  ex.kurt    = {float(((eta_train - mu)**4).mean() / sigma**4 - 3):+.3f}",
        f"  |η| > 3 σ  = {(np.abs(eta_train - mu) > 3*sigma).mean()*100:.2f}%  (Normal: 0.27%)",
        f"  |η| > 5 σ  = {(np.abs(eta_train - mu) > 5*sigma).mean()*100:.3f}%  (Normal: 5.7e-5%)",
        f"",
        f"  Hill α̂ right = {hill_alpha_right:.2f}  (smaller ⇒ heavier)",
        f"  Hill α̂ left  = {hill_alpha_left:.2f}",
        f"",
    ]
    if gpd_fit:
        info += [
            f"  GPD POT right tail (u={gpd_fit['threshold']:.1f}):",
            f"    n_exceed = {gpd_fit['n']}",
            f"    ξ        = {gpd_fit['shape']:+.3f}",
            f"    σ        = {gpd_fit['scale']:.2f}",
            f"    p_exceed = {gpd_fit['p_exceed']:.4f}",
        ]
    ax.text(0.02, 0.98, "\n".join(info), va="top", ha="left",
            family="monospace", fontsize=10)

    fig.suptitle("v2.5.13 — FI post-AR residual diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_cvar_backtest(rows: list[dict], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    alphas = [r["alpha"] for r in rows]
    realised = [r["realised"] for r in rows]
    normal   = [r["normal"] for r in rows]
    gpd      = [r["gpd"] for r in rows]
    emp_train = [r["empirical_train"] for r in rows]

    x = np.arange(len(alphas))
    w = 0.20
    ax.bar(x - 1.5*w, realised, w, color="k",   label="Realised (test)")
    ax.bar(x - 0.5*w, normal,   w, color="C7",  label="Normal model")
    ax.bar(x + 0.5*w, gpd,      w, color="C0",  label="GPD POT (L4)")
    ax.bar(x + 1.5*w, emp_train,w, color="C2",  label="Empirical train")
    ax.set_xticks(x); ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylabel("CVaR of post-AR residual [EUR/MWh]")
    ax.set_title("v2.5.13 — CVaR back-test on FI post-AR residual")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Loading data + fitting L1-L3...", flush=True)
    df = _build_dataframe()
    model = fit_full_model(df)
    eta = model["eta"]
    split = model["split"]
    eta_train, eta_test = eta[:split], eta[split:]
    print(f"  {len(df):,} rows; train {len(eta_train):,}, test {len(eta_test):,}",
          flush=True)
    print(f"  L2 Ridge coefs: "
          f"[{', '.join(f'{c:+.3f}' for c in model['ridge_coef'])}]",
          flush=True)
    print(f"  L3 φ = {model['phi']:+.3f}", flush=True)
    print(f"  L4 input η: mean {eta_train.mean():+.3f}, σ {eta_train.std():.3f}",
          flush=True)

    # GPD POT fit on training η (right tail only — focus on price spikes)
    print("\nFitting GPD POT on η (right tail, 95th percentile threshold)...",
          flush=True)
    gpd = fit_gpd_pot(eta_train, threshold_pct=95)
    threshold = gpd["threshold"]
    right_fit = dict(gpd["right"], threshold=threshold)
    left_fit  = dict(gpd["left"],  threshold=threshold)
    if right_fit.get("n", 0) >= 30 and not np.isnan(right_fit["shape"]):
        print(f"  RIGHT  ξ = {right_fit['shape']:+.3f}  "
              f"σ = {right_fit['scale']:.2f}  u = {threshold:.2f}  "
              f"n_exc = {right_fit['n']}", flush=True)
    else:
        print("  WARNING: GPD POT right-tail underpopulated", flush=True)
    if left_fit.get("n", 0) >= 30 and not np.isnan(left_fit["shape"]):
        print(f"  LEFT   ξ = {left_fit['shape']:+.3f}  "
              f"σ = {left_fit['scale']:.2f}  u = {threshold:.2f}  "
              f"n_exc = {left_fit['n']}", flush=True)

    # Hill estimators
    sorted_pos = np.sort(eta_train - eta_train.mean())[::-1]
    sorted_neg = np.sort(-(eta_train - eta_train.mean()))[::-1]
    k = max(100, int(0.02 * len(eta_train)))
    hill_right = hill_estimator(sorted_pos, k)
    hill_left  = hill_estimator(sorted_neg, k)

    # Render diagnostics
    fig_diagnostics(eta_train, right_fit, hill_right, hill_left,
                    FIGURES_DIR / "v2513_layer4_diagnostics.png")

    # CVaR back-test at multiple α
    print("\nCVaR back-test on test η:", flush=True)
    rows = []
    for alpha in (0.05, 0.01, 0.001):
        realised = empirical_cvar(eta_test, alpha)
        mu, sigma = eta_train.mean(), eta_train.std()
        normal   = cvar_normal(mu, sigma, alpha)
        gpd_val  = gpd_cvar(right_fit, alpha, len(eta_train))
        emp_tr   = empirical_cvar(eta_train, alpha)
        rows.append(dict(alpha=alpha, realised=realised, normal=normal,
                          gpd=gpd_val, empirical_train=emp_tr))
        print(f"  α={alpha:.3f}  realised={realised:7.2f}  "
              f"normal={normal:7.2f}  gpd_pot={gpd_val:7.2f}  "
              f"emp_train={emp_tr:7.2f}", flush=True)

    fig_cvar_backtest(rows, FIGURES_DIR / "v2513_layer4_cvar_backtest.png")

    # Persist as artifact (frozen GPD parameters for runtime sampling)
    artifact_path = DATA_DIR / "spike_model_default.json"
    payload = {
        "version": "2.5.13",
        "layer":   "L4 GPD POT on FI post-AR residual",
        "ridge_features":  model["features"],
        "ridge_coef":      model["ridge_coef"],
        "ar1_phi":         model["phi"],
        "gpd_right": (right_fit if right_fit else None),
        "hill_right_alpha": hill_right,
        "hill_left_alpha":  hill_left,
        "train_window": [str(df.index[0]), str(df.index[split - 1])],
        "stats": {
            "n_train": int(len(eta_train)),
            "eta_train_mean": float(eta_train.mean()),
            "eta_train_sigma": float(eta_train.std()),
            "eta_train_skew": float(
                ((eta_train - eta_train.mean())**3).mean()
                / eta_train.std()**3),
            "eta_train_excess_kurt": float(
                ((eta_train - eta_train.mean())**4).mean()
                / eta_train.std()**4 - 3),
        },
        "cvar_backtest": rows,
        "notes": (
            "L4 GPD POT spike model on the post-AR(1) residual of the "
            "V_sigmoid_full architecture (v2.5.12). Runtime: sample "
            "post-AR shocks via GPD inverse-CDF for the right tail. "
            "Bulk uses Normal(μ, σ); only exceedances above threshold "
            "u draw from GPD."
        ),
    }
    artifact_path.write_text(json.dumps(payload, indent=2),
                             encoding="utf-8")
    print(f"\nArtifact: {artifact_path}")

    # Markdown
    md = RESULTS_DIR / "v2513_layer4_spike.md"
    lines = [
        "# v2.5.13 — Layer 4 GPD POT spike model",
        "",
        "Applies the v2.5.2 GPD POT methodology to the FI post-AR(1) "
        "residual produced by the v2.5.12 V_sigmoid_full architecture.",
        "",
        f"**Window**: {df.index[0].date()} → {df.index[-1].date()} "
        f"({len(df):,} hourly rows)",
        f"**Train / test split**: chronological 55 / 45",
        f"**L1 + L2 + L3 architecture**:",
        f"  - L2 Ridge coefs (intercept, lag168, workday, sigmoid_wind_rho, "
        f"solar_effective, Y_temp):",
        f"    `[{', '.join(f'{c:+.4f}' for c in model['ridge_coef'])}]`",
        f"  - L3 AR(1) φ = **{model['phi']:+.3f}**",
        "",
        "## η(t) post-AR residual statistics (training)",
        "",
        f"- n = {len(eta_train):,}",
        f"- mean = {eta_train.mean():+.3f}, σ = {eta_train.std():.3f}",
        f"- skew = {((eta_train - eta_train.mean())**3).mean() / eta_train.std()**3:+.3f}",
        f"- excess kurtosis = {((eta_train - eta_train.mean())**4).mean() / eta_train.std()**4 - 3:+.2f}",
        f"- |η| > 3 σ frequency: "
        f"{(np.abs(eta_train - eta_train.mean()) > 3*eta_train.std()).mean()*100:.2f} % "
        f"(Gaussian baseline: 0.27 %)",
        f"- Hill α̂ right tail: **{hill_right:.2f}**  "
        f"(α̂ < 4 ⇒ infinite kurtosis ⇒ heavy)",
        f"- Hill α̂ left tail: {hill_left:.2f}",
        "",
        "## GPD POT right-tail fit",
        "",
    ]
    if right_fit and right_fit.get("n", 0) >= 30:
        lines += [
            f"- threshold u = {right_fit['threshold']:.2f}",
            f"- shape ξ = **{right_fit['shape']:+.3f}**  "
            f"(ξ > 0 ⇒ heavy; ξ = 0 ⇒ exponential; ξ < 0 ⇒ bounded)",
            f"- scale σ = {right_fit['scale']:.2f}",
            f"- n exceedances = {right_fit['n']}",
            f"- p_exceed = {right_fit['p_exceed']:.4f}",
            "",
        ]
    lines += [
        "## CVaR back-test on held-out η_test",
        "",
        "| α | Realised | Normal | GPD POT | Empirical train |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['alpha']:.3f} | {r['realised']:.2f} | "
            f"{r['normal']:.2f} | {r['gpd']:.2f} | "
            f"{r['empirical_train']:.2f} |"
        )

    lines += [
        "",
        "Interpretation: a model is **accurate** when its CVaR prediction "
        "matches realised. GPD POT closer to realised than Normal at low α "
        "(rare-spike tail) ⇒ Layer 4 is doing its job.",
        "",
        "## Figures",
        "",
        "![Residual diagnostics](figures/v2513_layer4_diagnostics.png)",
        "",
        "![CVaR back-test](figures/v2513_layer4_cvar_backtest.png)",
        "",
        "## Persisted artifact",
        "",
        "`custom_components/spot_price_predictor/data/spike_model_default.json` "
        "carries the frozen GPD POT parameters along with the parent L1+L2+L3 "
        "configuration that produced them. Runtime use: sample post-AR shocks "
        "via GPD inverse-CDF in the right tail; bulk uses Normal(μ, σ).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v2513_layer4_spike_model.py",
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
