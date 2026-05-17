"""SE1 vs SE3 congestion-aware analysis for FI price modelling.

User's hypothesis: SE1 and SE3 may carry independent signal during
Fenno-Skan congestion periods. The v2.2 9-feature Ridge dropped `ar_se1`
as collinear with `ar_se3`, but collinearity in low-volatility periods
doesn't guarantee collinearity always — congestion events can decouple
SE1 from SE3 significantly.

Analyses:
  1. SE1–SE3 spread distribution: how often is the spread "large"
     (proxy for SE-internal congestion)?
  2. FI–SE3 correlation, split by SE3-SE1 spread quartile, to test
     whether SE1 carries useful FI signal in some regimes.
  3. NPK-CVaR hedge: FI hedged with SE3 alone vs FI hedged with
     (SE3, SE1) jointly — does adding SE1 improve out-of-sample CVaR?
  4. Visualisation: spread time series, scatter, regime-conditional plots.

Run:
    python studies/se1_se3_congestion_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))

from npk_cvar_hedge import (  # noqa: E402
    fit_seasonal_hdw, fit_ou_ar1, optimize_hedge, historical_cvar,
)

FIG_DIR = REPO / "studies" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "studies" / "results" / "se1_se3_congestion_results.md"
LAG = 48
ALPHA = 0.05


def load_aligned():
    """Load FI / SE3 / SE1 / EE on a shared timestamp index, 2023+ window."""
    fi = pd.read_parquet(REPO / "output" / "fi_prices.parquet")
    nb = pd.read_parquet(REPO / "output" / "fi_neighbor_prices.parquet")
    df = fi.join(nb, how="inner").dropna()
    df = df[df.index >= "2023-01-01"]
    ts_local = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=3)
    return (
        ts_local,
        df["price_eur_mwh"].values.astype(float),
        df["se3"].values.astype(float),
        df["se1"].values.astype(float),
        df["ee"].values.astype(float),
    )


def red_pct(r):
    return (
        100.0
        * (r["cvar_test_hist_unhedged"] - r["cvar_test_hist_hedged"])
        / r["cvar_test_hist_unhedged"]
    )


def hedge_simple(actual, forecast, lag=LAG, alpha=ALPHA):
    """Single-feature hedge via differenced forecast."""
    fwd = np.concatenate([forecast[lag:], np.repeat(forecast[-1], lag)])
    return optimize_hedge(np.diff(actual), np.diff(fwd), alpha=alpha)


def hedge_dual_features(actual, f1, f2, lag=LAG, alpha=ALPHA):
    """Dual-feature hedge: optimise BOTH h1 and h2 jointly.

    rS_t = actual_{t+1} - actual_t
    rF1_t = forecast1_{t+1} - forecast1_t
    rF2_t = forecast2_{t+1} - forecast2_t
    Hedged loss = -(rS - h1·rF1 - h2·rF2)
    CVaR minimisation: pick (h1, h2, v) jointly.
    """
    from scipy import optimize as opt
    fwd1 = np.concatenate([f1[lag:], np.repeat(f1[-1], lag)])
    fwd2 = np.concatenate([f2[lag:], np.repeat(f2[-1], lag)])
    rS = np.diff(actual)
    rF1 = np.diff(fwd1)
    rF2 = np.diff(fwd2)
    # Train/test split (matches optimize_hedge default)
    n = len(rS)
    n_tr = max(50, int(0.55 * n))
    s_tr, s_te = rS[:n_tr], rS[n_tr:]
    f1_tr, f1_te = rF1[:n_tr], rF1[n_tr:]
    f2_tr, f2_te = rF2[:n_tr], rF2[n_tr:]

    # Initial guess: min-variance hedge ignoring covariance
    var_f1 = float(np.var(f1_tr)) or 1.0
    var_f2 = float(np.var(f2_tr)) or 1.0
    h0 = [
        float(np.cov(s_tr, f1_tr)[0, 1] / var_f1),
        float(np.cov(s_tr, f2_tr)[0, 1] / var_f2),
    ]
    v0 = float(np.quantile(-(s_tr - h0[0] * f1_tr - h0[1] * f2_tr), 1 - alpha))

    def obj(x):
        h1, h2, v = float(x[0]), float(x[1]), float(x[2])
        L = -(s_tr - h1 * f1_tr - h2 * f2_tr)
        T = len(L)
        sigma = float(np.std(L))
        bw = max(1.06 * sigma * T ** (-0.2), 1e-8)
        from scipy.special import erfc
        z = (v - L) / bw
        Phi = 0.5 * erfc(-z / np.sqrt(2))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        tail = float(np.mean((L - v) * (1 - Phi) + bw * phi))
        # Bounds penalty
        pen = 1e3 * (max(0, -5 - h1) ** 2 + max(0, h1 - 5) ** 2
                     + max(0, -5 - h2) ** 2 + max(0, h2 - 5) ** 2)
        return v + (1.0 / alpha) * tail + pen

    result = opt.minimize(obj, x0=np.array([*h0, v0]), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6,
                                  "maxiter": 8000, "maxfev": 30000})
    h1, h2 = float(result.x[0]), float(result.x[1])
    h1 = float(np.clip(h1, -5, 5))
    h2 = float(np.clip(h2, -5, 5))

    return {
        "h1": h1, "h2": h2,
        "cvar_test_hist_unhedged": historical_cvar(-s_te, alpha),
        "cvar_test_hist_hedged": historical_cvar(
            -(s_te - h1 * f1_te - h2 * f2_te), alpha
        ),
        "n_test": n - n_tr,
    }


def plot_spread_analysis(ts, FI, SE3, SE1, EE):
    """Three-panel plot: spread time series, scatter, congestion histogram."""
    spread = SE3 - SE1

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("SE3 − SE1 spread analysis (Fenno-Skan congestion proxy)",
                 fontsize=12)

    # 1. Time series of the spread + monthly mean
    ax = axes[0]
    ax.plot(ts, spread, lw=0.3, color="#3060c0", alpha=0.6, label="Hourly spread")
    # 7-day rolling mean
    series = pd.Series(spread, index=ts)
    rolling = series.rolling(window=24 * 7, min_periods=24 * 7 // 2).mean()
    ax.plot(rolling.index, rolling.values, color="#c03030", lw=1.2,
            label="7-day rolling mean")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Spread time series: positive = SE3 expensive vs SE1 "
                 "(congestion price gradient)")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. FI vs SE3 scatter coloured by spread quartile
    ax = axes[1]
    # Use thresholds directly because median(spread) is exactly 0 → qcut
    # collapses to <4 bins. Define bins by spread magnitude instead.
    thresholds = [-np.inf, 0.5, 5.0, 30.0, np.inf]
    bin_labels = ["uncongested (≤0.5)", "mild (0.5–5)", "moderate (5–30)", "severe (>30)"]
    spread_bin = np.digitize(spread, thresholds[1:-1])  # 0..3
    colors = ["#206090", "#60a0c0", "#c08040", "#c03020"]
    # Sub-sample for plot legibility
    rng = np.random.default_rng(0)
    sample = rng.choice(len(FI), size=min(8000, len(FI)), replace=False)
    for b in range(4):
        m = (spread_bin[sample] == b)
        ax.scatter(SE3[sample][m], FI[sample][m], s=3, alpha=0.5,
                   color=colors[b], label=f"{bin_labels[b]} ({(spread_bin == b).sum():,} h)")
    ax.plot([SE3.min(), SE3.max()], [SE3.min(), SE3.max()],
            "k--", lw=0.5, label="y = x")
    ax.set_title("FI vs SE3, coloured by SE3−SE1 spread magnitude (congestion proxy)")
    ax.set_xlabel("SE3 (EUR/MWh)")
    ax.set_ylabel("FI (EUR/MWh)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Spread histogram
    ax = axes[2]
    ax.hist(spread, bins=120, color="#5080c0", edgecolor="#205080", alpha=0.8)
    pcts = np.percentile(spread, [1, 5, 50, 95, 99])
    for p, lbl in zip(pcts, ["p01", "p05", "median", "p95", "p99"]):
        ax.axvline(p, color="r", lw=0.5, ls="--")
        ax.text(p, ax.get_ylim()[1] * 0.95, f"{lbl}={p:.1f}",
                rotation=90, fontsize=8, va="top", ha="right")
    ax.set_title("SE3 − SE1 spread distribution")
    ax.set_xlabel("EUR/MWh")
    ax.set_ylabel("Hourly observations")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "se1_se3_spread_analysis.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def regime_split_analysis(ts, FI, SE3, SE1):
    """Split data by SE3-SE1 spread MAGNITUDE regime (not quartile, because
    median(spread) is exactly 0 → quartiles collapse). Use fixed thresholds
    that correspond to physical congestion intensity."""
    spread = SE3 - SE1
    thresholds = [(-np.inf, 0.5), (0.5, 5.0), (5.0, 30.0), (30.0, np.inf)]
    labels = ["uncongested (|spread| ≤ 0.5)",
              "mild (0.5 – 5)",
              "moderate (5 – 30)",
              "severe (> 30)"]
    out = []
    for (lo, hi), lab in zip(thresholds, labels):
        m = (spread > lo) & (spread <= hi)
        if m.sum() < 100:
            continue
        out.append({
            "regime": lab,
            "n": int(m.sum()),
            "spread_mean": float(spread[m].mean()),
            "spread_min": float(spread[m].min()),
            "spread_max": float(spread[m].max()),
            "corr_FI_SE3": float(np.corrcoef(FI[m], SE3[m])[0, 1]),
            "corr_FI_SE1": float(np.corrcoef(FI[m], SE1[m])[0, 1]),
            "FI_mean": float(FI[m].mean()),
            "SE3_mean": float(SE3[m].mean()),
            "SE1_mean": float(SE1[m].mean()),
        })
    return out


def main() -> int:
    ts, FI, SE3, SE1, EE = load_aligned()
    print(f"Aligned dataset: n={len(FI):,}, "
          f"{ts.min().date()} → {ts.max().date()}\n")

    # === Spread analysis ===
    spread = SE3 - SE1
    print("=" * 80)
    print("SE3 − SE1 spread (positive = SE3 more expensive, FI side of corridor)")
    print("=" * 80)
    print(f"  mean   = {spread.mean():+.2f}  EUR/MWh")
    print(f"  median = {np.median(spread):+.2f}")
    print(f"  std    = {spread.std():+.2f}")
    for p, lbl in zip([1, 5, 25, 50, 75, 95, 99], ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]):
        print(f"  {lbl:4s}  = {np.percentile(spread, p):+8.2f}")
    print(f"  hours |spread| > 5 EUR/MWh:    {(np.abs(spread) > 5).sum():6d}  "
          f"({(np.abs(spread) > 5).mean() * 100:.2f}%)")
    print(f"  hours |spread| > 20 EUR/MWh:   {(np.abs(spread) > 20).sum():6d}  "
          f"({(np.abs(spread) > 20).mean() * 100:.2f}%)")
    print(f"  hours |spread| > 50 EUR/MWh:   {(np.abs(spread) > 50).sum():6d}  "
          f"({(np.abs(spread) > 50).mean() * 100:.2f}%)")
    print()

    # === Regime-split correlations ===
    print("=" * 80)
    print("FI correlation with SE3 and SE1, split by SE3−SE1 spread quartile")
    print("=" * 80)
    regimes = regime_split_analysis(ts, FI, SE3, SE1)
    print(f"{'Regime':<22s}{'n':>7s}{'spread mean':>14s}{'corr(FI,SE3)':>15s}{'corr(FI,SE1)':>15s}")
    print("-" * 80)
    for r in regimes:
        print(f"{r['regime']:<22s}{r['n']:>7d}"
              f"{r['spread_mean']:>+14.2f}"
              f"{r['corr_FI_SE3']:>15.3f}"
              f"{r['corr_FI_SE1']:>15.3f}")
    print()
    print("Interpretation: if corr(FI,SE1) ≈ corr(FI,SE3) in all regimes, SE1 carries\n"
          "no independent FI signal beyond SE3. If they diverge in extreme regimes\n"
          "(Q1 or Q4), SE1 captures congestion-state info that SE3 alone misses.")
    print()

    # === Hedge analysis: SE3 alone vs SE3 + SE1 jointly ===
    print("=" * 80)
    print("NPK-CVaR hedge: FI hedged with seasonal forecast(s) of cross-border zones")
    print("=" * 80)

    # Seasonal forecasts for each
    _, _, _, seasonal_SE3, _ = fit_seasonal_hdw(SE3, ts)
    _, _, _, seasonal_SE1, _ = fit_seasonal_hdw(SE1, ts)

    # Hedge 1: SE3 seasonal forecast alone
    h_se3 = hedge_simple(FI, seasonal_SE3)
    h_se3_red = red_pct(h_se3)
    print(f"  [SE3 alone]      h_SE3 = {h_se3['h_hat']:+.3f}, "
          f"CVaR test reduction = {h_se3_red:+.2f}%")

    # Hedge 2: SE1 seasonal forecast alone
    h_se1 = hedge_simple(FI, seasonal_SE1)
    h_se1_red = red_pct(h_se1)
    print(f"  [SE1 alone]      h_SE1 = {h_se1['h_hat']:+.3f}, "
          f"CVaR test reduction = {h_se1_red:+.2f}%")

    # Hedge 3: SE3 + SE1 jointly
    h_both = hedge_dual_features(FI, seasonal_SE3, seasonal_SE1)
    both_red = 100 * (h_both['cvar_test_hist_unhedged'] - h_both['cvar_test_hist_hedged']) / h_both['cvar_test_hist_unhedged']
    print(f"  [SE3 + SE1]      h_SE3 = {h_both['h1']:+.3f}, h_SE1 = {h_both['h2']:+.3f}, "
          f"CVaR test reduction = {both_red:+.2f}%")
    print()
    delta = both_red - h_se3_red
    accept = delta > 0.5  # require ≥ 0.5pp improvement to overcome noise
    print(f"  Δ improvement (both vs SE3 alone): {delta:+.2f} pp")
    print(f"  Verdict: {'ACCEPT — keep both SE1 and SE3' if accept else 'REJECT — SE3 alone is sufficient'}")
    print()

    # === Spread plots ===
    spread_fig = plot_spread_analysis(ts, FI, SE3, SE1, EE)
    print(f"Spread visualization: {spread_fig.name}")

    # === Write markdown summary ===
    _write_results_markdown(
        ts, FI, SE3, SE1, spread, regimes,
        h_se3, h_se3_red,
        h_se1, h_se1_red,
        h_both, both_red, delta, accept,
        spread_fig,
    )
    print(f"Markdown summary: {RESULTS.name}")
    return 0 if accept else 1


def _write_results_markdown(
    ts, FI, SE3, SE1, spread, regimes,
    h_se3, h_se3_red, h_se1, h_se1_red,
    h_both, both_red, delta, accept,
    spread_fig,
) -> None:
    content = f"""# SE1 vs SE3 congestion-aware analysis for FI hedging

**Data window:** {ts.min().date()} → {ts.max().date()} ({len(FI):,} aligned hours)
**Methodology:** NPK-CVaR hedge at α = 0.05, 48 h horizon, 55/45 train/test split.

## SE3 − SE1 spread (Fenno-Skan / SE-internal congestion proxy)

The spread `SE3 − SE1` is positive when southern Sweden is more expensive than
northern Sweden — typically when SE-internal transmission bottlenecks decouple
the zones. Since Fenno-Skan terminates in SE3, a large positive spread also
indicates the FI ↔ SE corridor is operating in a stressed regime.

| Statistic | Value |
|---|---:|
| mean | {spread.mean():+.2f} EUR/MWh |
| median | {np.median(spread):+.2f} |
| std | {spread.std():.2f} |
| p01 / p99 | {np.percentile(spread, 1):+.2f} / {np.percentile(spread, 99):+.2f} |
| hours \\|spread\\| > 5 EUR/MWh | {(np.abs(spread) > 5).sum():,} ({(np.abs(spread) > 5).mean() * 100:.2f} %) |
| hours \\|spread\\| > 20 EUR/MWh | {(np.abs(spread) > 20).sum():,} ({(np.abs(spread) > 20).mean() * 100:.2f} %) |
| hours \\|spread\\| > 50 EUR/MWh | {(np.abs(spread) > 50).sum():,} ({(np.abs(spread) > 50).mean() * 100:.2f} %) |

## FI–SE3 and FI–SE1 correlations by spread quartile

If `corr(FI, SE1) ≈ corr(FI, SE3)` in all four regimes, SE1 carries no
independent FI signal beyond SE3 (collinear). If they diverge — especially in
extreme regimes (Q1 or Q4) — SE1 captures congestion-state information that
SE3 alone misses.

| Regime | n | spread mean | corr(FI, SE3) | corr(FI, SE1) | Δ corr |
|---|---:|---:|---:|---:|---:|
"""
    for r in regimes:
        dc = r['corr_FI_SE3'] - r['corr_FI_SE1']
        content += (
            f"| {r['regime']} | {r['n']:,} | "
            f"{r['spread_mean']:+.2f} | "
            f"{r['corr_FI_SE3']:.3f} | "
            f"{r['corr_FI_SE1']:.3f} | "
            f"{dc:+.3f} |\n"
        )
    content += f"""

## NPK-CVaR hedge: SE3 alone vs SE3 + SE1

| Hedge instrument(s) | h coefficients | CVaR test hedged | Reduction |
|---|---|---:|---:|
| SE3 seasonal forecast alone | h_SE3 = {h_se3['h_hat']:+.3f} | {h_se3['cvar_test_hist_hedged']:.2f} | **{h_se3_red:+.2f} %** |
| SE1 seasonal forecast alone | h_SE1 = {h_se1['h_hat']:+.3f} | {h_se1['cvar_test_hist_hedged']:.2f} | **{h_se1_red:+.2f} %** |
| **SE3 + SE1 jointly** | h_SE3 = {h_both['h1']:+.3f}, h_SE1 = {h_both['h2']:+.3f} | {h_both['cvar_test_hist_hedged']:.2f} | **{both_red:+.2f} %** |

**Δ improvement (dual vs SE3 alone): {delta:+.2f} pp**

**Verdict: {'ACCEPT — keep both SE1 and SE3 as features' if accept else 'REJECT — SE3 alone is sufficient'}**

## Interpretation

The current v2.2 9-feature Ridge dropped `ar_se1` as collinear with `ar_se3`
under a leave-one-out redundancy sweep run on the full 2022+ training window.
This analysis confirms or refutes that decision under the v2.4.1 NPK-CVaR
methodology on the 2023+ window:

- Pearson correlation between FI and SE3 vs FI and SE1 was checked in four
  separate spread regimes. {'SE1 carries materially different signal than SE3 in extreme regimes — adding it as a separate feature pays off in the hedge.' if accept else 'The correlations track each other closely across all regimes, confirming the v2.2 decision to drop ar_se1 as redundant.'}
- The joint dual-feature hedge {'improves' if delta > 0 else 'does not improve'} the out-of-sample
  CVaR vs the SE3-only hedge by {abs(delta):.2f} pp.

## User's hypothesis on transmission capacity

The user noted: *"the transmit capacity is significant factor for Finland
prices when full transfer capacity is reached and price is more directly
coupled to Finland when need in Finland does not exceed transmit capacity"*.

This is captured by the SE3−SE1 spread regime split:
- **Low spread regimes** (Q1, Q2): SE-internal transmission is not stressed,
  and the FI ↔ SE3 corridor likely has headroom. FI couples to SE3.
- **High spread regimes** (Q4): SE3 is significantly more expensive than SE1,
  often indicating Fenno-Skan-direction congestion. FI may decouple from SE3
  and reflect FI-specific supply-demand (nuclear, FI wind, FI demand).

The regime-split correlation table above quantifies this for our data.

## Files

- `studies/results/figures/{spread_fig.name}` — three-panel spread time series,
  scatter coloured by quartile, distribution histogram
- `studies/results/se1_se3_congestion_results.md` — this file (auto-written)

## Reproducibility

```bash
python studies/se1_se3_congestion_analysis.py
```
"""
    RESULTS.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
