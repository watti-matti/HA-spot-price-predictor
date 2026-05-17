"""MATLAB-style seasonal-price diagnostic plots for Nordic zones.

Replicates the 9-panel + 2-heatmap layout from
`studies/Matlab_study_on_CVAR/analyze_sahkotin_seasonal.m` for each of
FI, SE3, SE1, EE on 2023+ hourly data.

Outputs (per zone, written to studies/results/figures/):
    seasonal_diag_{zone}.png   — 9-panel diagnostic (time series, residual,
                                  P_hour, P_day, P_week, residual histogram,
                                  ACF, QQ-plot, variance pie)
    seasonal_heatmap_{zone}.png — hour×day and hour×week heatmaps

Run:
    python studies/seasonal_visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")  # non-interactive (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfcinv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))
from npk_cvar_hedge import fit_seasonal_hdw, fit_ou_ar1, acf  # noqa: E402

FIG_DIR = REPO / "studies" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_zone(zone: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load 2023+ hourly prices for a Nordic zone (returns price + local-time index)."""
    if zone == "fi":
        df = pd.read_parquet(REPO / "output" / "fi_prices.parquet")
        col = "price_eur_mwh"
    else:
        df = pd.read_parquet(REPO / "output" / "fi_neighbor_prices.parquet")
        col = zone
    df = df[df.index >= "2023-01-01"]
    ts_local = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=3)
    P = df[col].values.astype(float)
    mask = np.isfinite(P)
    return P[mask], ts_local[mask]


def normal_qq(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample quantiles vs theoretical normal quantiles (Q-Q plot data)."""
    xs = np.sort(x[np.isfinite(x)])
    n = len(xs)
    plotting_pos = (np.arange(1, n + 1) - 0.5) / n
    # Normal quantile via inverse erfc
    theoretical = np.sqrt(2.0) * (-erfcinv(2.0 * plotting_pos))
    mu, sd = float(np.mean(xs)), float(np.std(xs))
    return mu + sd * theoretical, xs


def plot_diagnostic_9panel(zone: str, P: np.ndarray, ts: pd.DatetimeIndex) -> Path:
    """The MATLAB 9-panel diagnostic for one zone."""
    P_hour, P_day, P_week, seasonal, Y = fit_seasonal_hdw(P, ts)
    ou = fit_ou_ar1(Y)
    var_pct_seasonal = 100 * np.var(seasonal) / np.var(P)
    var_pct_residual = 100 * np.var(Y) / np.var(P)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(
        f"Seasonal price analysis — {zone.upper()} "
        f"(n={len(P):,}, {ts.min().date()} → {ts.max().date()})\n"
        f"Sequential subtraction: P = P_hour + P_day + P_week + Y; "
        f"OU half-life = {ou['half_life_hours']:.1f} h",
        fontsize=12,
    )

    # 1. Raw price + seasonal overlay
    ax = axes[0, 0]
    ax.plot(ts, P, color="#3060c0", lw=0.3, label="Spot price")
    ax.plot(ts, seasonal, color="#c03030", lw=0.8, label="Seasonal forecast")
    ax.set_title("Spot price vs seasonal forecast")
    ax.set_ylabel("EUR/MWh")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Deseasonalized residual
    ax = axes[0, 1]
    ax.plot(ts, Y, color="#2e9050", lw=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Deseasonalized Y_t (residual)")
    ax.set_ylabel("EUR/MWh")
    ax.grid(alpha=0.3)

    # 3. Hourly pattern (P_hour bars)
    ax = axes[0, 2]
    ax.bar(np.arange(24), P_hour, color="#5080c0", edgecolor="#205080")
    ax.set_title("P_hour(h) — diurnal seasonality")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks([0, 6, 12, 18])
    ax.grid(axis="y", alpha=0.3)

    # 4. Day-of-week pattern
    ax = axes[1, 0]
    ax.bar(np.arange(7), P_day, color="#c08040", edgecolor="#804020")
    ax.set_title("P_day(d) — weekly seasonality")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    ax.set_ylabel("EUR/MWh (residual after P_hour)")
    ax.grid(axis="y", alpha=0.3)

    # 5. Week-of-year pattern
    ax = axes[1, 1]
    ax.bar(np.arange(1, 54), P_week, color="#3e9070", edgecolor="#206040")
    ax.set_title("P_week(w) — annual seasonality")
    ax.set_xlabel("Week of year (ISO)")
    ax.set_ylabel("EUR/MWh (after P_hour + P_day)")
    ax.grid(axis="y", alpha=0.3)

    # 6. Residual histogram + normal fit
    ax = axes[1, 2]
    ax.hist(Y, bins=80, density=True, color="#8080d0", edgecolor="#404080",
            alpha=0.75, label="Empirical")
    mu, sd = float(np.mean(Y)), float(np.std(Y))
    xx = np.linspace(np.quantile(Y, 0.001), np.quantile(Y, 0.999), 400)
    nrm = np.exp(-0.5 * ((xx - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(xx, nrm, "r-", lw=2, label=f"N(μ={mu:.1f}, σ={sd:.1f})")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Y_t (EUR/MWh)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 7. ACF of Y
    ax = axes[2, 0]
    lags = list(range(1, 73))
    rho = acf(Y, lags=lags)
    ax.stem([0] + lags, [1.0] + [rho[k] for k in lags], basefmt=" ")
    ax.axhline(0, color="k", lw=0.5)
    # 95 % confidence band ≈ ±1.96/sqrt(n)
    ci = 1.96 / np.sqrt(len(Y))
    ax.axhline(ci, color="gray", lw=0.5, ls="--")
    ax.axhline(-ci, color="gray", lw=0.5, ls="--")
    ax.set_title(f"Residual ACF (OU half-life {ou['half_life_hours']:.1f} h)")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ρ(lag)")
    ax.grid(alpha=0.3)

    # 8. Q-Q plot
    ax = axes[2, 1]
    theo, samp = normal_qq(Y)
    ax.plot(theo, samp, "o", markersize=1.5, color="#3060c0", alpha=0.6)
    qrange = [min(theo.min(), samp.min()), max(theo.max(), samp.max())]
    ax.plot(qrange, qrange, "r-", lw=1.5, label="y=x")
    ax.set_title("Residual Q-Q vs Normal")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 9. Variance decomposition pie
    ax = axes[2, 2]
    sizes = [var_pct_seasonal, var_pct_residual]
    labels = [f"Seasonal\n{var_pct_seasonal:.1f}%",
              f"Residual Y_t\n{var_pct_residual:.1f}%"]
    ax.pie(sizes, labels=labels,
           colors=["#5080c0", "#c08040"],
           autopct="%.1f%%",
           startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title("Variance decomposition")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / f"seasonal_diag_{zone}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_heatmaps(zone: str, P: np.ndarray, ts: pd.DatetimeIndex) -> Path:
    """Two heatmaps: hour × day-of-week, hour × week-of-year."""
    h = ts.hour.to_numpy()
    d = ts.weekday.to_numpy()
    w = (ts.isocalendar().week.to_numpy()) - 1
    w = np.clip(w, 0, 52)

    hd = np.full((24, 7), np.nan)
    for hh in range(24):
        for dd in range(7):
            m = (h == hh) & (d == dd)
            if m.any():
                hd[hh, dd] = np.nanmean(P[m])

    hw = np.full((24, 53), np.nan)
    for hh in range(24):
        for ww in range(53):
            m = (h == hh) & (w == ww)
            if m.any():
                hw[hh, ww] = np.nanmean(P[m])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"Average price heatmaps — {zone.upper()} (EUR/MWh)",
                 fontsize=12)

    ax = axes[0]
    im = ax.imshow(hd, aspect="auto", cmap="turbo", origin="lower",
                   interpolation="nearest")
    ax.set_title("Hour × Day-of-week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Hour of day")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"])
    ax.set_yticks([0, 6, 12, 18, 23])
    fig.colorbar(im, ax=ax, label="EUR/MWh")

    ax = axes[1]
    im = ax.imshow(hw, aspect="auto", cmap="turbo", origin="lower",
                   interpolation="nearest")
    ax.set_title("Hour × Week-of-year")
    ax.set_xlabel("Week of year (ISO)")
    ax.set_ylabel("Hour of day")
    ax.set_xticks([0, 12, 25, 38, 52])
    ax.set_yticks([0, 6, 12, 18, 23])
    fig.colorbar(im, ax=ax, label="EUR/MWh")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / f"seasonal_heatmap_{zone}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> int:
    zones = ["fi", "se3", "se1", "ee"]
    print(f"Generating seasonal diagnostics for {len(zones)} zones → {FIG_DIR}\n")
    results = []
    for z in zones:
        P, ts = load_zone(z)
        f1 = plot_diagnostic_9panel(z, P, ts)
        f2 = plot_heatmaps(z, P, ts)
        ou = fit_ou_ar1(fit_seasonal_hdw(P, ts)[4])
        seasonal_var = 100 * np.var(fit_seasonal_hdw(P, ts)[3]) / np.var(P)
        print(f"  {z.upper():4s}: n={len(P):6d}, "
              f"OU half-life {ou['half_life_hours']:5.1f}h, "
              f"seasonal var {seasonal_var:4.1f}%  → {f1.name}, {f2.name}")
        results.append({"zone": z, "ou_half_life_h": ou['half_life_hours'],
                        "seasonal_var_pct": seasonal_var, "n": len(P),
                        "fig_diag": f1.name, "fig_heatmap": f2.name})
    print(f"\n{len(zones) * 2} figures written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
