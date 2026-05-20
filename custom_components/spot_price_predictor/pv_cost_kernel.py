"""Shared cost-realisation kernel for PV-aware CVaR statistics.

Stateless library used by both the predictor (with a slow-EMA
reference consumption profile) and the downstream thermal optimiser
(with its actual planned schedule). Same maths, different
``consumption_kwh`` arguments.

Per `studies/results/pv_adjusted_cvar_plan.md`:

* The realised hourly cost for one Monte-Carlo path is::

      C_h = max(L_h - PV_h, 0) * buy_h  -  max(PV_h - L_h, 0) * sell_h

  where ``L_h`` is household load and ``PV_h`` is on-site production
  (both kWh/hour). ``buy_h`` and ``sell_h`` are EUR/kWh.

* CVaR is non-linear in price and PV, so we sample the joint
  distribution and take CVaR of the realised cost across paths.

The function deliberately knows nothing about the source of its
inputs (slow-EMA profile vs planner schedule). That keeps it safe
to reuse across repos without import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostDistribution:
    """Summary of a per-path cost realisation."""

    cost_per_path_eur:  np.ndarray   # [N_paths]
    cost_per_kwh_eur:   np.ndarray   # [N_paths]
    mean_eur_kwh:       float
    var_eur_kwh:        float        # VaR(alpha)
    cvar_eur_kwh:       float        # CVaR(alpha) = E[X | X >= VaR]
    mean_eur:           float
    cvar_eur:           float
    pv_self_consumed_kwh_mean: float
    pv_exported_kwh_mean:      float
    alpha:              float
    n_paths:            int


def cost_distribution(
    buy_eur_kwh:     np.ndarray,
    sell_eur_kwh:    np.ndarray,
    pv_kwh:          np.ndarray,
    consumption_kwh: np.ndarray,
    *,
    alpha: float = 0.05,
) -> CostDistribution:
    """Realise hourly cost across joint price/PV paths, return CVaR.

    Parameters
    ----------
    buy_eur_kwh, sell_eur_kwh : ndarray of shape ``[N_paths, n_hours]``
        Buy and sell prices per path. Both may be the same array if
        prices are deterministic. ``buy >= sell`` is not enforced
        (e.g. negative spot with a fixed feed-in tariff).
    pv_kwh : ndarray of shape ``[N_paths, n_hours]``
        Production in kWh per hour, per path.
    consumption_kwh : ndarray
        Either ``[n_hours]`` (deterministic profile — broadcast over
        all paths) or ``[N_paths, n_hours]`` (one path per scenario).
    alpha : float, default 0.05
        Tail probability for VaR/CVaR. ``alpha=0.05`` reports the
        mean of the worst 5 percent of paths.

    Returns
    -------
    CostDistribution
        Aggregated per-horizon cost statistics across paths.

    Raises
    ------
    ValueError if shapes are inconsistent or ``alpha`` is outside
    ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    buy = np.asarray(buy_eur_kwh, dtype=float)
    sell = np.asarray(sell_eur_kwh, dtype=float)
    pv = np.asarray(pv_kwh, dtype=float)
    cons = np.asarray(consumption_kwh, dtype=float)

    if buy.ndim != 2:
        raise ValueError(
            f"buy_eur_kwh must be 2-D [N_paths, n_hours]; got shape {buy.shape}"
        )
    n_paths, n_hours = buy.shape

    if sell.shape != buy.shape:
        raise ValueError(
            f"sell shape {sell.shape} must match buy shape {buy.shape}"
        )
    if pv.shape != buy.shape:
        raise ValueError(
            f"pv shape {pv.shape} must match buy shape {buy.shape}"
        )
    if cons.ndim == 1:
        if cons.shape[0] != n_hours:
            raise ValueError(
                f"consumption (1-D) length {cons.shape[0]} != n_hours {n_hours}"
            )
        cons = np.broadcast_to(cons, (n_paths, n_hours))
    elif cons.ndim == 2:
        if cons.shape != buy.shape:
            raise ValueError(
                f"consumption (2-D) shape {cons.shape} must match {buy.shape}"
            )
    else:
        raise ValueError(f"consumption must be 1-D or 2-D; got {cons.ndim}-D")

    # Per-hour, per-path cost realisation.
    deficit = np.maximum(cons - pv, 0.0)
    surplus = np.maximum(pv - cons, 0.0)
    cost_per_hour = deficit * buy - surplus * sell

    cost_per_path = cost_per_hour.sum(axis=1)
    consumption_per_path = cons.sum(axis=1)
    # Guard against zero-consumption paths (would produce inf).
    safe_consumption = np.where(
        consumption_per_path > 0.0, consumption_per_path, np.nan
    )
    cost_per_kwh = cost_per_path / safe_consumption

    # CVaR_alpha = E[X | X >= VaR_alpha] over the worst alpha of paths.
    var_threshold = float(np.nanquantile(cost_per_kwh, 1.0 - alpha))
    tail_mask = cost_per_kwh >= var_threshold
    cvar_per_kwh = float(np.nanmean(cost_per_kwh[tail_mask])) if tail_mask.any() else var_threshold

    cvar_per_path = float(np.nanmean(
        cost_per_path[cost_per_path >= float(np.nanquantile(cost_per_path, 1.0 - alpha))]
    ))

    return CostDistribution(
        cost_per_path_eur=cost_per_path,
        cost_per_kwh_eur=cost_per_kwh,
        mean_eur_kwh=float(np.nanmean(cost_per_kwh)),
        var_eur_kwh=var_threshold,
        cvar_eur_kwh=cvar_per_kwh,
        mean_eur=float(np.nanmean(cost_per_path)),
        cvar_eur=cvar_per_path,
        pv_self_consumed_kwh_mean=float(np.minimum(pv, cons).sum(axis=1).mean()),
        pv_exported_kwh_mean=float(surplus.sum(axis=1).mean()),
        alpha=alpha,
        n_paths=n_paths,
    )
