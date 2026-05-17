"""v2.5.12 — Sigmoid turbine power curve as a wind feature.

User direction 2026-05-17 (after v2.5.11): the cubic ρ·v³ proxy
over-extrapolated at high winds. Try a sigmoid (S-curve) instead —
matches the actual turbine power curve shape more closely:

    P(v) = 1 / (1 + exp(-k · (v − v_mid)))

The Ridge then has a feature that smoothly transitions from 0 at low
wind through 0.5 at v_mid to 1 at high wind, capturing both cut-in
and rated-saturation regions in one shot.

We test three sigmoid parameterisations alongside the v2.5.11
baseline, and also the air-density-multiplied version:

    sigmoid_wind         = sigmoid((v − 7.5) / 1.5)
    sigmoid_wind_rho     = sigmoid_wind · ρ(T)
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
from build_seasonal_components import (  # noqa: E402
    load_fi_prices, load_neighbor_prices, load_weather_extended,
    WEATHER_WINDOW_START, WINDOW_END,
)
from v2510_layer3_ar_wind import (  # noqa: E402
    fit_ridge, fit_ar1, hedge_cvar_pct, ALPHA, TRAIN_FRAC, HORIZONS,
)
from v2511_physics_features import air_density, solar_effective  # noqa: E402

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


# ── Sigmoid turbine curve ──────────────────────────────────────────


def sigmoid_turbine(v: np.ndarray, v_mid: float = 7.5,
                    k_steep: float = 1.5) -> np.ndarray:
    """Normalised turbine power curve. v_mid is the 50 %-of-rated wind
    speed, k_steep controls transition width (~3·k_steep spans the
    cut-in to rated band).

    Default (7.5, 1.5) ⇒ 5 % at v=3 m/s, 50 % at 7.5, 95 % at 12 —
    matches typical 120 m hub-height fleet curve."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(v, dtype=float) - v_mid) / k_steep))


def sigmoid_turbine_rho(v: np.ndarray, temp_celsius: np.ndarray,
                        v_mid: float = 7.5, k_steep: float = 1.5,
                        rho_ref: float = 1.225) -> np.ndarray:
    """Sigmoid × air-density correction (normalised by ρ_ref)."""
    rho = air_density(temp_celsius)
    return sigmoid_turbine(v, v_mid, k_steep) * (rho / rho_ref)


# ── Data loading ───────────────────────────────────────────────────


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

    # Physics features
    df["sigmoid_wind"]     = sigmoid_turbine(df["wind"].values)
    df["sigmoid_wind_rho"] = sigmoid_turbine_rho(df["wind"].values,
                                                   df["temp"].values)
    df["solar_effective"]  = solar_effective(df["solar"].values,
                                              df["temp"].values)

    for name in ("sigmoid_wind", "sigmoid_wind_rho", "solar_effective"):
        comp = sd.fit_components(df[name].values, ts_np,
                                  depth=("P_hour", "P_week"),
                                  smooth={"P_week": 7})
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, comp)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    return df.dropna()


# ── Evaluation ─────────────────────────────────────────────────────


def evaluate(name: str, df: pd.DataFrame, features: list[str]) -> dict:
    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split]
    n_full = len(df)
    X_train = np.column_stack([np.ones(len(train))]
                              + [train[f].values for f in features])
    X_full  = np.column_stack([np.ones(n_full)]
                              + [df[f].values    for f in features])
    coef = fit_ridge(X_train, train["Y_fi"].values, alpha=1.0)
    ridge_pred_full = X_full @ coef

    eps_train = train["Y_fi"].values - ridge_pred_full[:split]
    phi, _ = fit_ar1(eps_train)
    eps_full = df["Y_fi"].values - ridge_pred_full

    per_horizon = {}
    for h in HORIZONS:
        ar_full = np.zeros(n_full, dtype=float)
        if phi != 0:
            ar_full[h:] = (phi ** h) * eps_full[:n_full - h]
        full_pred = df["seasonal_fi"].values + ridge_pred_full + ar_full
        actual = df["fi"].values
        test_pred = full_pred[split:]
        test_act  = actual[split:]
        err = test_pred - test_act
        mae = float(np.mean(np.abs(err)))
        var_y = float(np.var(test_act))
        r2 = 1.0 - float(np.var(err)) / var_y if var_y > 0 else float("nan")
        cvar = hedge_cvar_pct(actual, full_pred, lag=h, alpha=ALPHA)
        per_horizon[h] = {"mae": mae, "r2": r2, "cvar_pct": cvar}

    eps_demean = eps_train - eps_train.mean()
    var = float(np.var(eps_demean))
    rho1 = (float(np.dot(eps_demean[:-1], eps_demean[1:]))
            / ((len(eps_demean) - 1) * var) if var > 0 else 0.0)
    return {
        "name": name, "features": features, "phi": phi,
        "ridge_coef": coef.tolist(),
        "resid_acf_lag1": rho1,
        "per_horizon": per_horizon,
    }


# ── Plots ──────────────────────────────────────────────────────────


def fig_curves(out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    v = np.linspace(0, 25, 200)
    ax.plot(v, sigmoid_turbine(v), "C0-", lw=2.5, label="sigmoid (v_mid=7.5, k=1.5)")
    # Plot cubic-truncated reference for comparison
    cube = (v / 12.0) ** 3
    cube[v > 12.0] = 1.0
    cube[v > 25.0] = 0.0
    ax.plot(v, cube, "C2--", lw=1.5, alpha=0.8,
            label="clipped cubic at v_rated=12 m/s (real turbine)")
    # Raw cubic (v2.5.11 form)
    cube_raw = (v / 12.0) ** 3
    ax.plot(v, np.minimum(cube_raw, 5), "C3:", lw=1.2, alpha=0.7,
            label="v³ unsaturated (v2.5.11 cubic, over-extrapolates)")
    # Cut-in / cut-out markers
    for vc, label in [(3.0, "cut-in"), (12.0, "rated"), (25.0, "cut-out")]:
        ax.axvline(vc, color="grey", lw=0.5, ls=":")
        ax.text(vc, 1.02, label, ha="center", fontsize=8, color="grey")
    ax.set_xlabel("hub-height wind speed [m/s]")
    ax.set_ylabel("normalised power output [0..1]")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 1.1)
    ax.set_title("Turbine power curve — sigmoid vs cubic candidates")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_variant_comparison(variants: list[dict], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [v["name"] for v in variants]
    phis = [v["phi"] for v in variants]
    rhos = [v["resid_acf_lag1"] for v in variants]
    mae_24  = [v["per_horizon"][24]["mae"]    for v in variants]
    mae_168 = [v["per_horizon"][168]["mae"]   for v in variants]
    r2_24   = [v["per_horizon"][24]["r2"]     for v in variants]
    r2_168  = [v["per_horizon"][168]["r2"]    for v in variants]
    cvar_24 = [v["per_horizon"][24]["cvar_pct"]  for v in variants]
    cvar_168= [v["per_horizon"][168]["cvar_pct"] for v in variants]
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x - 0.18, phis, width=0.36, color="C3",
           label="AR(1) φ")
    ax.bar(x + 0.18, rhos, width=0.36, color="C7", alpha=0.7,
           label="ρ(1) of Ridge residual")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("φ / ρ(1)")
    ax.set_title("Residual autocorrelation (lower ⇒ Ridge captures more)")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.bar(x - 0.18, mae_24,  width=0.36, color="C0", label="MAE 24 h")
    ax.bar(x + 0.18, mae_168, width=0.36, color="C2", label="MAE 168 h")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Test MAE")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.bar(x - 0.18, r2_24,  width=0.36, color="C0", label="R² 24 h")
    ax.bar(x + 0.18, r2_168, width=0.36, color="C2", label="R² 168 h")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("Test R²")
    ax.set_title("Test R²")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("v2.5.12 — sigmoid turbine curve vs cubic vs raw wind",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Building data + features...", flush=True)
    df = _build_dataframe()
    print(f"  {len(df):,} rows  ({df.index[0].date()} → {df.index[-1].date()})",
          flush=True)
    print(f"  sigmoid_wind range:     "
          f"{df['sigmoid_wind'].min():.3f} – {df['sigmoid_wind'].max():.3f}",
          flush=True)
    print(f"  sigmoid_wind_rho range: "
          f"{df['sigmoid_wind_rho'].min():.3f} – {df['sigmoid_wind_rho'].max():.3f}",
          flush=True)

    variants_spec = [
        ("V_base (v2.5.10 V4)",
            ["Y_fi_lag168", "is_workday", "Y_wind", "Y_solar", "Y_temp"]),
        ("V_sigmoid (replaces Y_wind)",
            ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind",
             "Y_solar", "Y_temp"]),
        ("V_sigmoid_rho (adds air density)",
            ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
             "Y_solar", "Y_temp"]),
        ("V_sigmoid_full (sigmoid + solar derate)",
            ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
             "Y_solar_effective", "Y_temp"]),
        ("V_sigmoid_plus_raw_wind (both)",
            ["Y_fi_lag168", "is_workday", "Y_wind", "Y_sigmoid_wind_rho",
             "Y_solar_effective", "Y_temp"]),
    ]
    print("\nFitting variants...", flush=True)
    variants = []
    for name, features in variants_spec:
        v = evaluate(name, df, features)
        variants.append(v)
        coef_str = ", ".join(f"{c:+.3f}" for c in v["ridge_coef"])
        print(f"\n  {name}  φ={v['phi']:+.3f}  "
              f"ρ_residual(1)={v['resid_acf_lag1']:+.3f}", flush=True)
        print(f"     features: {features}", flush=True)
        print(f"     ridge coefs: [{coef_str}]", flush=True)
        for h in HORIZONS:
            ph = v["per_horizon"][h]
            print(f"     h={h:4d}h  MAE={ph['mae']:6.2f}  "
                  f"R²={ph['r2']:+.3f}  CVaR={ph['cvar_pct']:+6.2f}%",
                  flush=True)

    print("\nRendering figures...", flush=True)
    fig_curves(FIGURES_DIR / "v2512_turbine_curves.png")
    fig_variant_comparison(variants,
                            FIGURES_DIR / "v2512_sigmoid_variants.png")

    md = RESULTS_DIR / "v2512_sigmoid_turbine_curve.md"
    lines = [
        "# v2.5.12 — Sigmoid turbine power curve",
        "",
        "Replaces the v2.5.11 cubic ρ·v³ proxy with a sigmoid "
        "`P(v) = 1/(1 + exp(-(v - 7.5)/1.5))` that matches the actual "
        "S-shape of a turbine power curve (cut-in around 3 m/s, rated "
        "around 12 m/s, cut-out at 25 m/s).",
        "",
        "## Sigmoid vs cubic candidates",
        "",
        "![Turbine curves](figures/v2512_turbine_curves.png)",
        "",
        "## Variant comparison",
        "",
        "| Variant | φ | ρ(1) | h=24 MAE/R²/CVaR | h=168 MAE/R²/CVaR |",
        "|---|---:|---:|---|---|",
    ]
    for v in variants:
        p24, p168 = v["per_horizon"][24], v["per_horizon"][168]
        lines.append(
            f"| {v['name']} | {v['phi']:+.3f} | {v['resid_acf_lag1']:+.3f} | "
            f"{p24['mae']:.2f} / {p24['r2']:+.3f} / {p24['cvar_pct']:+.1f}% | "
            f"{p168['mae']:.2f} / {p168['r2']:+.3f} / {p168['cvar_pct']:+.1f}% |"
        )

    lines += [
        "",
        "![Variants](figures/v2512_sigmoid_variants.png)",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v2512_sigmoid_turbine_curve.py",
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
