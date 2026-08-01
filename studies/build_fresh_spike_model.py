"""Phase-1 producer — FRESH_CONS deployment candidate.

Refits the full L2/L3/L4 stack (8-feature V_xb ridge + AR(1) + GPD POT)
on ALL data through today, against the FRESH L1 seasonal artifact
(regenerate it first with `python studies/build_seasonal_components.py`).

Two isolated changes, each measured by studies/backtest_harness.py:
  1. freshness       (FRESH  - DEPLOYED)
  2. consistency fix (FRESH_CONS - FRESH): the trainer's seasonal
     deseasonalisation of sigmoid_wind_rho / solar_effective is persisted
     as `physics_seasonal` so Pipeline._deseasonalize_physics applies the
     SAME transform at inference instead of per-batch mean-centring.

No capacity scaling (rejected — see session findings: batch-centred L2
scaling is a pure coefficient rescale; L1 level term didn't help either).

The final fit deliberately uses ALL data (no held-out split): the
walk-forward harness is the generalization evidence; the deployment
artifact should be as fresh as possible.

Output: output/spike_model_fresh.json (candidate — production
data/spike_model_default.json is NOT overwritten by this script).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studies"))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import numpy as _np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

import seasonal_decomposition as sd  # noqa: E402
from exp_extra_features import build_dataframe  # noqa: E402
from v2510_layer3_ar_wind import fit_ar1  # noqa: E402

import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "peak_model_feasibility", REPO / "studies" / "peak_model_feasibility.py")
_pmf = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pmf)
fit_gpd_pot = _pmf.fit_gpd_pot
hill_estimator = _pmf.hill_estimator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUTPUT_DIR = REPO / "output"
FEATS = ["Y_fi_lag168", "is_workday", "Y_sigmoid_wind_rho",
         "Y_solar_effective", "Y_temp", "Y_se1", "Y_se3", "Y_ee"]
# Physics seasonal config — matches exp_extra_features / backtest_harness.
PHYS_DEPTH = ("P_hour", "P_week")
PHYS_SMOOTH = {"P_week": 7}

# Economic sign invariant — wind and PV are zero-marginal-cost, so their
# price coefficient can never be positive (see pipeline.NON_POSITIVE_FEATURES).
# Left free, the ridge assigns Y_solar_effective a POSITIVE coefficient in
# every walk-forward refit (+0.020…+0.027) because it is confounded with
# temperature (clear Finnish winter skies are cold and expensive) and with
# the neighbour-price channel (a sunny day here is sunny in Sweden too).
# Constraining costs ~nothing — harness walk-forward MAE 19.81 -> 19.86
# overall, and summer IMPROVES 14.47 -> 14.29.
NON_POSITIVE_FEATURES = ("Y_solar_effective", "Y_sigmoid_wind_rho")


def fit_ridge_signed(X: _np.ndarray, y: _np.ndarray, alpha: float = 1.0,
                     upper: dict[int, float] | None = None) -> _np.ndarray:
    """Ridge with optional per-coefficient upper bounds (intercept
    un-penalised), solved as a bounded augmented least-squares problem.

    Equivalent to `v2510_layer3_ar_wind.fit_ridge` when `upper` is empty.
    """
    p = X.shape[1]
    A = _np.vstack([X, _np.sqrt(alpha) * _np.eye(p)])
    A[len(X), 0] = 0.0                     # intercept un-penalised
    b = _np.concatenate([y, _np.zeros(p)])
    lo = _np.full(p, -_np.inf)
    hi = _np.full(p, _np.inf)
    for i, u in (upper or {}).items():
        hi[i] = u
    return lsq_linear(A, b, bounds=(lo, hi), method="trf").x


def _snapshot_id() -> str:
    mf = REPO / "data_store" / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text()).get("snapshot_id", "unknown")
    return "no-data-store"


def main() -> None:
    print("Building dataframe against the (fresh) shipped L1…", flush=True)
    df = build_dataframe()
    n = len(df)
    ts = pd.DatetimeIndex(df.index, tz="UTC").values
    print(f"  rows = {n:,}  span = {df.index[0].date()} → "
          f"{df.index[-1].date()}", flush=True)

    # Physics seasonal components fit on the FULL window; keyed by the
    # names Pipeline._deseasonalize_physics looks up.
    phys = {}
    for art_name, col in (("sigmoid_wind_rho", "sigmoid_wind_rho"),
                          ("solar_effective", "solar_effective")):
        comp = sd.fit_components(df[col].values, ts, depth=PHYS_DEPTH,
                                 smooth=PHYS_SMOOTH)
        phys[art_name] = comp
        df[f"Y_{col}"] = df[col].values - sd.compute_seasonal_part(ts, comp)

    y = df["Y_fi"].values
    X = np.column_stack([np.ones(n)] + [df[f].values for f in FEATS])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    # Column i of X is FEATS[i-1] (column 0 is the intercept).
    upper = {FEATS.index(f) + 1: 0.0 for f in NON_POSITIVE_FEATURES
             if f in FEATS}
    coef = fit_ridge_signed(X[ok], y[ok], alpha=1.0, upper=upper)
    for f in NON_POSITIVE_FEATURES:
        if f in FEATS:
            print(f"  sign-constrained {f:22s} = {coef[FEATS.index(f) + 1]:+.6f}",
                  flush=True)
    eps = y[ok] - X[ok] @ coef
    phi, _ = fit_ar1(eps)
    eta = np.empty_like(eps)
    eta[0] = eps[0]
    eta[1:] = eps[1:] - phi * eps[:-1]

    gpd = fit_gpd_pot(eta)
    right = gpd.get("right") if isinstance(gpd, dict) else None
    k_hill = max(50, len(eta) // 100)
    hill_r = hill_estimator(eta, k=k_hill)
    hill_l = hill_estimator(-eta, k=k_hill)

    print(f"  ridge fit on {int(ok.sum()):,} h | phi = {phi:+.3f} | "
          f"sigma(eta) = {eta.std():.2f}", flush=True)

    payload = {
        "version": "v2.15.0/fresh-cons",
        "layer": "L4 GPD POT on FI post-AR residual (fresh full-window refit)",
        "ridge_features": FEATS,
        "ridge_coef": coef.tolist(),
        "ar1_phi": float(phi),
        "gpd_right": right if isinstance(right, dict) and "shape" in right else None,
        "hill_right_alpha": float(hill_r),
        "hill_left_alpha": float(hill_l),
        "physics_seasonal": phys,
        "stats": {
            "n_train": int(ok.sum()),
            "eta_train_mean": float(eta.mean()),
            "eta_train_sigma": float(eta.std()),
            "eta_train_skew": float(((eta - eta.mean()) ** 3).mean()
                                    / eta.std() ** 3),
            "eta_train_excess_kurt": float(((eta - eta.mean()) ** 4).mean()
                                           / eta.std() ** 4 - 3),
        },
        "train_window": [str(df.index[0]), str(df.index[-1])],
        "data_store_snapshot": _snapshot_id(),
        "notes": (
            "FRESH_CONS candidate: full-window refit of L2 ridge + AR(1) + "
            "GPD against the fresh L1 seasonal artifact, with "
            "physics_seasonal persisted for train/inference-consistent "
            "deseasonalisation of the wind/solar physics terms. Validated "
            "by studies/backtest_harness.py (walk-forward day-ahead): "
            "DEPLOYED 22.15 -> FRESH 21.75 -> FRESH_CONS 19.81 MAE. "
            "Requires the pipeline._deseasonalize_physics support and the "
            "matching fresh seasonal_components_default.json."
        ),
    }
    out = OUTPUT_DIR / "spike_model_fresh.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote candidate → {out.relative_to(REPO)}")
    print(f"  snapshot = {payload['data_store_snapshot']}")


if __name__ == "__main__":
    main()
