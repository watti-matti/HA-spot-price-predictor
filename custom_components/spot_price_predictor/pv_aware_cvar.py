"""PV-aware CVaR computation for a single forecast day.

Production-side counterpart to `studies/exp_pv_scenarios_backtest.py`.
The studies version uses an empirical bootstrap over 4+ years of
cached weather to generate PV scenarios (`sim_pv_scenarios.py`); that
isn't available at HA runtime, so this module ships a lightweight
parametric scenario sampler calibrated to deliver the same first-
moment / second-moment behaviour:

  - mean of sampled paths = point-forecast PV (mean-preserving)
  - relative std = ``REL_STD_PARAM`` (default 0.30, matches the
    empirical bootstrap's marginal width at peak hours in spring
    months — within 2 pp of the 91.7 % coverage measured in the
    Phase A validation)

The trade-off: the parametric sampler under-correlates *across*
hours, which slightly under-widens the daily CVaR vs the empirical
bootstrap. The integration's published CVaR is therefore a modestly
conservative estimate; refinement to a runtime-cached weather
history is a follow-up if it ever proves limiting.

Inputs are scalar numpy arrays for one day (24 hours each); output
is a dict suitable for direct injection onto a
``duration_forecast.daily_forecast[i]`` row.
"""
from __future__ import annotations

import numpy as np

try:  # package import (HA runtime)
    from .pv_cost_kernel import cost_distribution
except ImportError:  # bare import (test or studies path)
    from pv_cost_kernel import cost_distribution  # type: ignore[no-redef]


# Relative std for the lognormal PV perturbation. Anchored to the
# Phase A back-test (studies/exp_pv_scenarios_backtest.md) — see
# CVaR-mean spread of +43 mEUR/kWh on the reference household.
REL_STD_PARAM = 0.30

# Number of Monte-Carlo paths per day. 200 gives a CVaR_5% standard
# error of ≈ σ/√(200·0.05) ≈ σ/3, plenty for a published number that
# the user reads to one or two decimal places.
N_PATHS_PARAM = 200


def _sample_pv_paths(
    pv_point_kwh: np.ndarray,
    *,
    n_paths: int = N_PATHS_PARAM,
    rel_std: float = REL_STD_PARAM,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Lognormal mean-preserving multiplicative perturbation."""
    if rng is None:
        rng = np.random.default_rng()
    n_h = len(pv_point_kwh)
    # Lognormal with mean 1 and the requested relative std.
    sigma_log = np.sqrt(np.log(1.0 + rel_std ** 2))
    mu_log = -0.5 * sigma_log ** 2  # so E[exp(X)] = 1
    z = rng.normal(mu_log, sigma_log, size=(n_paths, n_h))
    multipliers = np.exp(z)
    paths = pv_point_kwh[None, :] * multipliers
    # Zero-PV hours (night) stay zero — don't sample noise there.
    paths = np.where(pv_point_kwh[None, :] > 0, paths, 0.0)
    return paths


def compute_pv_aware_cvar_for_day(
    buy_eur_kwh:     np.ndarray,   # [24]
    sell_eur_kwh:    np.ndarray,   # [24]
    pv_kwh:          np.ndarray,   # [24] point forecast
    consumption_kwh: np.ndarray,   # [24]
    *,
    n_paths: int = N_PATHS_PARAM,
    rel_std: float = REL_STD_PARAM,
    alpha:   float = 0.05,
    rng:     np.random.Generator | None = None,
) -> dict:
    """Per-day PV-aware cost distribution summary.

    Returns a dict with keys:

    - ``mean_eur_kwh``: expected effective cost / kWh
    - ``cvar95_eur_kwh``: tail-mean of the worst 5 % of paths
    - ``p5_eur_kwh`` / ``p50_eur_kwh`` / ``p95_eur_kwh``: fan-chart
      quantiles per kWh
    - ``mean_eur``: expected total daily cost in EUR
    - ``cvar95_eur``: tail-mean of total daily cost
    - ``pv_self_consumed_kwh``: mean self-consumed PV across paths
    - ``pv_exported_kwh``: mean exported surplus across paths
    - ``n_paths``: paths used
    - ``rel_std``: PV perturbation std parameter

    All inputs are length-24 numpy arrays of float. PV must be the
    point forecast (deterministic); this function samples the
    uncertainty band internally.
    """
    if len(buy_eur_kwh) != 24 or len(sell_eur_kwh) != 24 \
            or len(pv_kwh) != 24 or len(consumption_kwh) != 24:
        raise ValueError(
            "all input arrays must have length 24; got "
            f"buy={len(buy_eur_kwh)} sell={len(sell_eur_kwh)} "
            f"pv={len(pv_kwh)} cons={len(consumption_kwh)}"
        )

    pv_paths = _sample_pv_paths(
        np.asarray(pv_kwh, dtype=float),
        n_paths=n_paths,
        rel_std=rel_std,
        rng=rng,
    )
    buy_paths = np.broadcast_to(
        np.asarray(buy_eur_kwh, dtype=float)[None, :],
        (n_paths, 24),
    )
    sell_paths = np.broadcast_to(
        np.asarray(sell_eur_kwh, dtype=float)[None, :],
        (n_paths, 24),
    )

    out = cost_distribution(
        buy_paths, sell_paths, pv_paths,
        np.asarray(consumption_kwh, dtype=float),
        alpha=alpha,
    )

    return {
        "mean_eur_kwh":            float(out.mean_eur_kwh),
        "cvar95_eur_kwh":          float(out.cvar_eur_kwh),
        "p5_eur_kwh":              float(np.quantile(
            out.cost_per_kwh_eur, 0.05
        )),
        "p50_eur_kwh":             float(np.median(out.cost_per_kwh_eur)),
        "p95_eur_kwh":             float(np.quantile(
            out.cost_per_kwh_eur, 0.95
        )),
        "mean_eur":                float(out.mean_eur),
        "cvar95_eur":              float(out.cvar_eur),
        "pv_self_consumed_kwh":    float(out.pv_self_consumed_kwh_mean),
        "pv_exported_kwh":         float(out.pv_exported_kwh_mean),
        "n_paths":                 int(n_paths),
        "rel_std":                 float(rel_std),
    }
