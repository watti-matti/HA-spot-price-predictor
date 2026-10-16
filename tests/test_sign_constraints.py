"""Sign constraints must actually bind, and must refuse contradictions.

A feature whose sign is known a priori should be constrained, because an
unconstrained ridge will fit a physically impossible coefficient when
the feature is collinear with a trend. Measured instance: the
cross-border transfer-capacity term fitted +1.397 EUR/MWh per GW when
more import capacity must LOWER the Finnish price — it was proxying the
time trend, and nothing caught it.

Shipped constraints: wind and PV <= 0 (zero marginal cost cannot raise
the price). Residual load is the natural >= 0 case: scarcity moves you
up a convex supply stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "studies", REPO / "custom_components" / "spot_price_predictor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from honest_horizon_study import sfit      # noqa: E402


def _data(slope: float, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = slope * x + rng.normal(scale=0.1, size=n)
    return np.column_stack([np.ones(n), x]), y


def test_unconstrained_recovers_the_true_sign() -> None:
    X, y = _data(+3.0)
    assert sfit(X, y, 1.0)[1] > 2.5
    X, y = _data(-3.0)
    assert sfit(X, y, 1.0)[1] < -2.5


def test_upper_bound_binds_against_the_data() -> None:
    """The existing wind/PV guard: data says +3, bound says <= 0."""
    X, y = _data(+3.0)
    c = sfit(X, y, 1.0, upper={1: 0.0})
    assert c[1] <= 1e-9, f"upper bound did not bind, got {c[1]}"


def test_lower_bound_binds_against_the_data() -> None:
    """The new residual-load guard: data says -3, bound says >= 0."""
    X, y = _data(-3.0)
    c = sfit(X, y, 1.0, lower={1: 0.0})
    assert c[1] >= -1e-9, f"lower bound did not bind, got {c[1]}"


def test_lower_bound_is_inert_when_the_data_agrees() -> None:
    """A non-binding constraint must not perturb the fit — this is why
    adding it to the residual-load terms left every number unchanged."""
    X, y = _data(+3.0)
    free = sfit(X, y, 1.0)
    bound = sfit(X, y, 1.0, lower={1: 0.0})
    np.testing.assert_allclose(free, bound, rtol=1e-6, atol=1e-8)


def test_both_bounds_together() -> None:
    X, y = _data(+3.0)
    c = sfit(X, y, 1.0, upper={1: 1.0}, lower={1: 0.5})
    assert 0.5 - 1e-9 <= c[1] <= 1.0 + 1e-9


def test_contradictory_bounds_raise() -> None:
    """Silently returning something for lower > upper would hide a
    configuration error in a fit nobody inspects by hand."""
    X, y = _data(+3.0)
    with pytest.raises(ValueError, match="contradictory bounds"):
        sfit(X, y, 1.0, upper={1: 0.0}, lower={1: 1.0})


def test_intercept_stays_unpenalised() -> None:
    """The ridge row for column 0 is zeroed; a large offset must still
    be recovered rather than shrunk toward zero."""
    X, y = _data(+1.0)
    y = y + 50.0
    c = sfit(X, y, 1.0)
    assert abs(c[0] - 50.0) < 0.5, f"intercept shrunk to {c[0]}"
