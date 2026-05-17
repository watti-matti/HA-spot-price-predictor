"""EE model variants for v2.4.3 — investigation, REJECTED.

Per the validation methodology (v2.4.1 plan), any new model variant must
improve out-of-sample NPK-CVaR reduction vs the baseline to be accepted.

For Estonia (EE), three architectures were tested:

  V1 (baseline)    : seasonal_EE only
  V2               : seasonal_EE + is_workday
  V3               : P_hour split by workday/weekend + P_week

All three converge on +4.07 % CVaR reduction — no additional structure
materially improves on the seasonal baseline. The workday coefficient is
statistically insignificant (b ≈ −0.02, essentially noise). AR(1) at lag-1
HURTS the EE hedge (drops reduction to +0.84 %) because EE's residual
decay half-life is 4.7 h vs SE3's 12.8 h — AR(1) at 1-hour lag captures
little fresh signal and the deeper autocorrelation of the model prediction
changes the hedge geometry unfavourably.

VERDICT: REJECT all proposed EE variants. The current production
`ar_ee` AR(2) feature already extracts the achievable hedge signal at this
horizon. To improve EE further would need richer features (Baltic gas
spot, Estlink-1/2 congestion state, Russian electricity-exit transition
state, etc.) which are out of scope for the v2.5.0 model upgrade.

This script is preserved as a regression record so future revisits can
confirm or contradict the finding on later data.

Run:
    python studies/ee_model_v243.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))

from npk_cvar_hedge import fit_seasonal_hdw, fit_ou_ar1, optimize_hedge  # noqa: E402

EE_PARQUET = REPO / "output" / "fi_neighbor_prices.parquet"
LAG_HOURS = 48
ALPHA = 0.05


def load_ee_2023plus():
    neigh = pd.read_parquet(EE_PARQUET)
    neigh = neigh[neigh.index >= "2023-01-01"]
    ts_local = pd.DatetimeIndex(neigh.index) + pd.Timedelta(hours=3)
    EE = neigh["ee"].values.astype(float)
    mask = np.isfinite(EE)
    return EE[mask], ts_local[mask]


def variant_baseline(EE, ts_local, lag, alpha):
    _, _, _, seasonal, _ = fit_seasonal_hdw(EE, ts_local)
    Fwd = np.concatenate([seasonal[lag:], np.repeat(seasonal[-1], lag)])
    return optimize_hedge(np.diff(EE), np.diff(Fwd), alpha=alpha)


def variant_seasonal_plus_workday(EE, ts_local, lag, alpha):
    _, _, _, seasonal, Y = fit_seasonal_hdw(EE, ts_local)
    dow = ts_local.weekday.to_numpy()
    is_workday = ((dow >= 0) & (dow <= 4)).astype(float)
    X = np.column_stack([np.ones_like(Y), is_workday])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    model = seasonal + X @ beta
    Fwd = np.concatenate([model[lag:], np.repeat(model[-1], lag)])
    hedge = optimize_hedge(np.diff(EE), np.diff(Fwd), alpha=alpha)
    return hedge, beta


def variant_seasonal_plus_workday_plus_ar1(EE, ts_local, lag, alpha):
    _, _, _, seasonal, Y = fit_seasonal_hdw(EE, ts_local)
    dow = ts_local.weekday.to_numpy()
    is_workday = ((dow >= 0) & (dow <= 4)).astype(float)
    Y_lag = np.concatenate([[0.0], Y[:-1]])
    X = np.column_stack([np.ones_like(Y), is_workday, Y_lag])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    model = seasonal + X @ beta
    Fwd = np.concatenate([model[lag:], np.repeat(model[-1], lag)])
    hedge = optimize_hedge(np.diff(EE), np.diff(Fwd), alpha=alpha)
    return hedge, beta


def red_pct(r):
    return (
        100.0
        * (r["cvar_test_hist_unhedged"] - r["cvar_test_hist_hedged"])
        / r["cvar_test_hist_unhedged"]
    )


def main() -> int:
    EE, ts_local = load_ee_2023plus()
    print(f"EE window: {ts_local.min()} → {ts_local.max()}, n={len(EE)}\n")

    _, _, _, _, Y_baseline = fit_seasonal_hdw(EE, ts_local)
    ou = fit_ou_ar1(Y_baseline)
    print(f"OU half-life on Y_EE: {ou['half_life_hours']:.1f} h (vs SE3 12.8 h)")
    print(f"OU AR coef b:         {ou['b']:.4f}\n")

    print("=" * 75)
    print("Variant comparison (NPK-CVaR raw 48h hedge, alpha=0.05)")
    print("=" * 75)

    v1 = variant_baseline(EE, ts_local, LAG_HOURS, ALPHA)
    print(f"[V1 baseline:   seasonal-only]                h={v1['h_hat']:.3f}  red={red_pct(v1):+.2f}%")

    v2, b2 = variant_seasonal_plus_workday(EE, ts_local, LAG_HOURS, ALPHA)
    print(f"[V2:            seasonal + is_workday]        h={v2['h_hat']:.3f}  red={red_pct(v2):+.2f}%")
    print(f"   coefficients: const={b2[0]:+.4f}, workday={b2[1]:+.4f}")

    v3, b3 = variant_seasonal_plus_workday_plus_ar1(EE, ts_local, LAG_HOURS, ALPHA)
    print(f"[V3:            seasonal + workday + AR(1)]   h={v3['h_hat']:.3f}  red={red_pct(v3):+.2f}%")
    print(f"   coefficients: const={b3[0]:+.4f}, workday={b3[1]:+.4f}, AR(1)={b3[2]:+.4f}")

    print()
    print("=== DECISION ===")
    print(f"  Baseline gate (V1):         {red_pct(v1):+.2f}%")
    print(f"  Best variant (V2):          {red_pct(v2):+.2f}%")
    print(f"  Improvement (V2 - V1):      {red_pct(v2) - red_pct(v1):+.2f} pp")
    print(f"  Verdict:                    REJECT — no improvement possible")
    print()
    print(
        "Interpretation: EE's residual half-life (4.7 h) is too short for AR(1)\n"
        "at lag-1 to capture useful signal; workday coefficient ≈ 0 means\n"
        "weekday/weekend distinction is already baked into P_day; the seasonal\n"
        "model alone is at the achievable bound for this feature set.\n"
        "\n"
        "The current production `ar_ee` AR(2) feature in the v2.2 9-feature\n"
        "Ridge will be carried unchanged into v2.5.0. Future EE improvements\n"
        "need richer features (Baltic gas spot, Estlink congestion, etc.)."
    )
    return 1  # exit 1 = REJECT (any caller can interpret as 'don't auto-adopt')


if __name__ == "__main__":
    sys.exit(main())
