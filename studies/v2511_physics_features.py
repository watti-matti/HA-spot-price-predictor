"""v2.5.11 — Physics-based feature engineering for wind and solar.

User direction 2026-05-17: the v2.5.10 sweep relied on AR(1) at
φ=0.93 to soak up large residuals. The hypothesis: that strong AR
masks missing feature engineering — the Ridge sees raw `wind_speed`
and raw `temperature` separately and can't learn the non-linear
`ρ_air · v³` coupling. Add the physics, see whether φ drops and
prediction improves.

Physics added (deterministic functions of wind / solar / temp):

  ρ_air(T) = 101_325 / (287.05 · (T_°C + 273.15))    kg/m³
  wind_power_proxy = wind_speed³ · ρ_air(T)          ∝ kW per swept m²

  cell_temp(T_ambient, ghi)
      ≈ T_ambient + 0.03 · ghi_W_m2                  NOCT approximation
  η_temp = 1 − 0.004 · max(0, cell_temp − 25)        Si PV coefficient
  solar_effective = solar_irradiance · η_temp        attenuated by heat

Both feed the FI Ridge as deseasonalized residuals (`Y_wind_power`,
`Y_solar_effective`) so they cohabit with the v2.5.5 seasonal layer.

Variant comparison (all use L1 + L3 AR(1); only the L2 Ridge feature
set differs):

  V_base       v2.5.10 V4 (Y_fi_lag168, is_workday, Y_wind, Y_solar, Y_temp)
  V_phys_wind  V_base − Y_wind + Y_wind_power
  V_phys_solar V_base − Y_solar + Y_solar_effective
  V_phys_both  V_base − Y_wind − Y_solar + Y_wind_power + Y_solar_effective

If physics features genuinely reduce the unexplained residual, φ
should drop from 0.90 toward 0.7-0.8 and MAE should fall further.
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
from v2510_layer3_ar_wind import (  # noqa: E402
    fit_ridge, fit_ar1, hedge_cvar_pct, ALPHA, TRAIN_FRAC, HORIZONS,
    SAMPLE_START, SAMPLE_END,
)

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


# ── Physics ────────────────────────────────────────────────────────


def air_density(temp_celsius: np.ndarray, pressure_pa: float = 101_325.0,
                R_specific: float = 287.05) -> np.ndarray:
    """Dry-air mass density (kg/m³) from temperature (°C) and pressure.

    Sea-level pressure default; FI altitudes are low so the error from
    assuming 101_325 Pa is < 1 %.
    """
    T_K = np.asarray(temp_celsius, dtype=float) + 273.15
    return pressure_pa / (R_specific * T_K)


def wind_power_proxy(wind_speed: np.ndarray,
                     temp_celsius: np.ndarray) -> np.ndarray:
    """v³·ρ(T) — proportional to mechanical kinetic power flux per swept
    area. Units are arbitrary (constants absorbed by Ridge fit)."""
    rho = air_density(temp_celsius)
    return rho * wind_speed ** 3


def pv_cell_temp(temp_ambient: np.ndarray,
                 ghi_w_m2: np.ndarray) -> np.ndarray:
    """Approximate panel cell temperature using NOCT-style derating:
        T_cell ≈ T_ambient + 0.03 · GHI
    The 0.03 K·m²/W coefficient is the standard rule-of-thumb for free-
    standing crystalline modules with NOCT ≈ 45 °C and 800 W/m²
    reference. Adequate for the residual-level signal we need."""
    return np.asarray(temp_ambient, dtype=float) + 0.03 * np.asarray(
        ghi_w_m2, dtype=float)


def pv_efficiency_derating(cell_temp: np.ndarray,
                           coeff_per_C: float = 0.004) -> np.ndarray:
    """1 − 0.004·max(0, T_cell − 25). Above 25 °C the silicon PV
    efficiency falls; below 25 °C the cell is at reference efficiency
    (efficiency does not RISE further with cold in this simple model)."""
    over = np.maximum(0.0, np.asarray(cell_temp, dtype=float) - 25.0)
    return 1.0 - coeff_per_C * over


def solar_effective(ghi_w_m2: np.ndarray,
                    temp_ambient: np.ndarray) -> np.ndarray:
    """GHI attenuated by PV temperature derating."""
    T_cell = pv_cell_temp(temp_ambient, ghi_w_m2)
    return np.asarray(ghi_w_m2, dtype=float) * pv_efficiency_derating(T_cell)


# ── Build dataframe with physics features ───────────────────────────


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

    # Standard seasonal residuals from shipped artifact
    for name in df.columns:
        if name not in ARTIFACT["components"]:
            continue
        components = ARTIFACT["components"][name]
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, components)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    df["Y_fi_lag168"] = df["Y_fi"].shift(168)
    df["is_workday"]  = (df.index.weekday < 5).astype(float)

    # Physics features computed on the RAW inputs (deterministic from
    # wind, solar, temp). Then deseasonalised on the fly so they cohabit
    # with the seasonal layer.
    df["wind_power"]      = wind_power_proxy(df["wind"].values,
                                              df["temp"].values)
    df["solar_effective"] = solar_effective(df["solar"].values,
                                             df["temp"].values)
    df["cell_temp"]       = pv_cell_temp(df["temp"].values, df["solar"].values)
    df["air_density"]     = air_density(df["temp"].values)

    # Deseasonalise the new physics features locally (per the audit
    # depth choices for the underlying inputs)
    for name, depth in [("wind_power",      ("P_hour", "P_week")),
                        ("solar_effective", ("P_hour", "P_week"))]:
        comp = sd.fit_components(df[name].values, ts_np, depth=depth,
                                  smooth={"P_week": 7})
        df[f"seasonal_{name}"] = sd.compute_seasonal_part(ts_np, comp)
        df[f"Y_{name}"]        = df[name].values - df[f"seasonal_{name}"].values

    return df.dropna()


# ── Variant evaluation (reuse v2.5.10 logic but with new features) ──


def evaluate(name: str, df: pd.DataFrame, features: list[str],
             use_layer3: bool = True) -> dict:
    split = int(len(df) * TRAIN_FRAC)
    train = df.iloc[:split]
    n_full = len(df)

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

    eps_train = train["Y_fi"].values - ridge_pred_full[:split]
    phi = (fit_ar1(eps_train)[0] if use_layer3 else 0.0)
    eps_full = df["Y_fi"].values - ridge_pred_full

    per_horizon = {}
    for h in HORIZONS:
        ar_full = np.zeros(n_full, dtype=float)
        if use_layer3 and phi != 0:
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

    # In-sample residual autocorrelation diagnostic
    eps_train_demean = eps_train - eps_train.mean()
    var = float(np.var(eps_train_demean))
    if var > 0:
        rho1 = float(np.dot(eps_train_demean[:-1], eps_train_demean[1:])) \
            / ((len(eps_train_demean) - 1) * var)
    else:
        rho1 = 0.0

    return {
        "name": name, "features": features, "phi": phi,
        "ridge_coef": coef.tolist(),
        "resid_acf_lag1": rho1,
        "per_horizon": per_horizon,
    }


# ── Plotting ───────────────────────────────────────────────────────


def fig_phi_vs_features(variants: list[dict], out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [v["name"] for v in variants]
    phis = [v["phi"] for v in variants]
    rhos = [v["resid_acf_lag1"] for v in variants]
    mae_24 = [v["per_horizon"][24]["mae"] for v in variants]
    mae_168 = [v["per_horizon"][168]["mae"] for v in variants]
    cvar_24 = [v["per_horizon"][24]["cvar_pct"] for v in variants]
    cvar_168 = [v["per_horizon"][168]["cvar_pct"] for v in variants]
    x = np.arange(len(labels))

    ax = axes[0]
    ax.bar(x - 0.18, phis, width=0.36, color="C3",
           label="AR(1) φ (fitted on train residual)")
    ax.bar(x + 0.18, rhos, width=0.36, color="C7", alpha=0.7,
           label="ρ(1) of Ridge residual (no AR)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("AR coefficient / lag-1 autocorrelation")
    ax.set_title("Does Ridge residual autocorrelation drop with physics?")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.bar(x - 0.18, mae_24,  width=0.36, color="C0", label="MAE 24 h")
    ax.bar(x + 0.18, mae_168, width=0.36, color="C2", label="MAE 168 h")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Test MAE per horizon")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.bar(x - 0.18, cvar_24,  width=0.36, color="C0", label="CVaR 24 h")
    ax.bar(x + 0.18, cvar_168, width=0.36, color="C2", label="CVaR 168 h")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
    ax.set_ylabel("Hedge CVaR-reduction [%]")
    ax.set_title("Hedge CVaR per horizon")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("v2.5.11 — physics features test  |  φ / MAE / CVaR per variant",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def fig_physics_relationship(df: pd.DataFrame, out_path: Path) -> None:
    """Show the physics: wind_power vs (wind_speed, temp);
    solar_effective vs (solar, temp)."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sample = df.iloc[::24]  # daily samples (~3 y of data) for scatter

    ax = axes[0, 0]
    sc = ax.scatter(sample["wind"], sample["wind_power"],
                     c=sample["temp"], cmap="coolwarm", s=10, alpha=0.6)
    ax.set_xlabel("wind_speed [m/s]")
    ax.set_ylabel("wind_power = ρ(T)·v³  [arbitrary units]")
    ax.set_title("Wind power proxy — cubic in wind, density-corrected for T")
    fig.colorbar(sc, ax=ax, label="temperature [°C]")

    ax = axes[0, 1]
    T = np.linspace(-30, 35, 200)
    ax.plot(T, air_density(T), "C0-", lw=1.6)
    ax.set_xlabel("temperature [°C]")
    ax.set_ylabel("air density ρ [kg/m³]")
    ax.set_title("Air density vs temperature (101_325 Pa)")
    ax.axvline(0, color="grey", lw=0.5)
    ax.axhline(1.225, color="grey", lw=0.5, ls="--",
                label="standard ρ at 15 °C = 1.225 kg/m³")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1, 0]
    sc = ax.scatter(sample["solar"], sample["solar_effective"],
                     c=sample["temp"], cmap="coolwarm", s=10, alpha=0.6)
    ax.plot([0, sample["solar"].max()], [0, sample["solar"].max()],
            "k--", lw=0.8, alpha=0.6, label="no derating")
    ax.set_xlabel("GHI [W/m²]")
    ax.set_ylabel("solar_effective = GHI · η_temp [W/m²]")
    ax.set_title("Solar effective — PV temperature derating")
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(sc, ax=ax, label="ambient temperature [°C]")

    ax = axes[1, 1]
    T_amb = np.linspace(-10, 35, 200)
    for ghi in (200, 400, 600, 800, 1000):
        T_cell = pv_cell_temp(T_amb, ghi)
        ax.plot(T_amb, pv_efficiency_derating(T_cell),
                label=f"GHI {ghi} W/m²")
    ax.set_xlabel("ambient temperature [°C]")
    ax.set_ylabel("PV efficiency derating factor")
    ax.set_title("η_temp = 1 − 0.004 · max(0, T_cell − 25)")
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("v2.5.11 — physics features", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    print("Loading aligned data + computing physics features...", flush=True)
    df = _build_dataframe()
    print(f"  {len(df):,} hourly rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})", flush=True)
    print(f"  air_density range: {df['air_density'].min():.3f} – "
          f"{df['air_density'].max():.3f} kg/m³", flush=True)
    print(f"  wind_power range : {df['wind_power'].min():.1f} – "
          f"{df['wind_power'].max():.1f}", flush=True)
    print(f"  cell_temp range  : {df['cell_temp'].min():.1f} – "
          f"{df['cell_temp'].max():.1f} °C", flush=True)
    print(f"  solar_effective vs raw ratio (high-solar hours): "
          f"{(df.loc[df['solar'] > 600, 'solar_effective'].mean() / df.loc[df['solar'] > 600, 'solar'].mean()):.3f}",
          flush=True)

    variants_spec = [
        ("V_base (V4 from v2.5.10)",
            ["Y_fi_lag168", "is_workday", "Y_wind", "Y_solar", "Y_temp"]),
        ("V_phys_wind (replaces Y_wind)",
            ["Y_fi_lag168", "is_workday", "Y_wind_power", "Y_solar", "Y_temp"]),
        ("V_phys_solar (replaces Y_solar)",
            ["Y_fi_lag168", "is_workday", "Y_wind", "Y_solar_effective", "Y_temp"]),
        ("V_phys_both (replaces both)",
            ["Y_fi_lag168", "is_workday", "Y_wind_power", "Y_solar_effective", "Y_temp"]),
        ("V_phys_plus_raw_wind (both physics + raw wind)",
            ["Y_fi_lag168", "is_workday", "Y_wind", "Y_wind_power",
             "Y_solar_effective", "Y_temp"]),
    ]
    print("\nFitting variants...", flush=True)
    variants = []
    for name, features in variants_spec:
        v = evaluate(name, df, features, use_layer3=True)
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
    fig_phi_vs_features(variants,
                         FIGURES_DIR / "v2511_phi_vs_features.png")
    fig_physics_relationship(df,
                              FIGURES_DIR / "v2511_physics_relationship.png")

    md = RESULTS_DIR / "v2511_physics_features.md"
    lines = [
        "# v2.5.11 — Physics-based wind and solar features",
        "",
        f"**Window:** {df.index[0].date()} → {df.index[-1].date()} "
        f"({len(df):,} hourly rows)",
        "",
        "Physics:",
        "- `wind_power = ρ_air(T) · v³` (kinetic flux per swept area, "
        "T in K via Boyle's law)",
        "- `solar_effective = GHI · η_temp(T_cell)` with "
        "`T_cell ≈ T_ambient + 0.03·GHI` (NOCT) and "
        "`η_temp = 1 − 0.004·max(0, T_cell − 25)` (Si PV temperature "
        "coefficient)",
        "",
        "## Variant comparison",
        "",
        "| Variant | φ | ρ(1) | "
        "h=24 MAE/R²/CVaR | h=168 MAE/R²/CVaR |",
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
        "Interpretation:",
        "- **φ** = fitted AR(1) coefficient on the Ridge residual; high φ "
        "  (~0.93) means much of the residual is autocorrelated noise the "
        "  Ridge couldn't explain. Lower φ ⇒ Ridge captures more structure.",
        "- **ρ(1)** = lag-1 autocorrelation of the Ridge residual ITSELF "
        "  (independent of whether AR is used). Same metric phrased "
        "  differently — directly diagnostic of feature quality.",
        "- If physics features genuinely capture missing structure, both",
        "  φ and ρ(1) should drop relative to the V_base baseline.",
        "",
        "## Figures",
        "",
        "![φ / MAE / CVaR by variant](figures/v2511_phi_vs_features.png)",
        "",
        "![Physics relationships](figures/v2511_physics_relationship.png)",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python studies/v2511_physics_features.py",
        "```",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
