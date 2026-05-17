"""Smooth softplus price floor.

Physical motivation: when the electricity price threatens to go
substantially negative, dispatchable thermal plants (gas, coal,
biomass — anything with operating cost) curtail their output. This
puts a soft floor near 0 EUR/MWh on most negative-price periods.

Empirically on FI 2023+:
  * 5.73 % of hours have negative price
  * 99 % of negatives cluster between 0 and −5 EUR/MWh
  * 0.04 % go below −50 (true extreme curtailment failures)
  * Hard Nord Pool floor: −500 EUR/MWh

The softplus floor at level `f` is

    floored(p) = f + softplus(p − f)
               = f + log(1 + exp(p − f))

which:
  * Returns ≈ p when p ≫ f (identity above the floor)
  * Smoothly asymptotes to f when p ≪ f
  * Is C∞ — no kink, gradient-friendly for any downstream optimizer
  * Adds ~0.7 EUR/MWh to the prediction right AT the floor (log 2)

This is applied to the L1+L2+L3 deterministic mean forecast ONLY.
GPD POT (Layer 4) samples are NOT floored because the real extreme
spikes down to −500 EUR/MWh are part of the distribution we want to
preserve for tail-risk awareness.
"""

from __future__ import annotations

import numpy as np


# Empirical default: 99 % of FI negative-price hours sit above −5.
DEFAULT_FLOOR_EUR_MWH = -5.0


def softplus_safe(x: np.ndarray) -> np.ndarray:
    """`log(1 + exp(x))` with overflow protection.

    For large positive x the function is ≈ x; for large negative x it
    is ≈ 0. The naive `np.log(1 + np.exp(x))` overflows around x ≈ 700.
    """
    x = np.asarray(x, dtype=float)
    # Use the identity log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|))
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def apply_floor(price: np.ndarray,
                floor: float = DEFAULT_FLOOR_EUR_MWH) -> np.ndarray:
    """Apply the smooth softplus floor to a price array.

    Args:
        price: array of point-forecast prices (EUR/MWh).
        floor: floor level (default −5 EUR/MWh empirical FI value).

    Returns:
        Same-shape array with prices floored smoothly. Above `floor` is
        approximately the identity; below `floor` asymptotically
        approaches `floor`.
    """
    return floor + softplus_safe(np.asarray(price, dtype=float) - floor)


def floor_curve(floor: float = DEFAULT_FLOOR_EUR_MWH,
                lo: float = -100.0, hi: float = +100.0,
                n: int = 401) -> tuple[np.ndarray, np.ndarray]:
    """Return `(x, apply_floor(x))` for plotting the floor's shape."""
    x = np.linspace(lo, hi, n)
    return x, apply_floor(x, floor=floor)
