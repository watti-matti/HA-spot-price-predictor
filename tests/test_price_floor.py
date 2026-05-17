"""Tests for custom_components/spot_price_predictor/price_floor.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import price_floor as pf  # noqa: E402


def test_floor_is_approximately_identity_well_above_floor() -> None:
    """For prices much higher than the floor, output ≈ input."""
    p = np.array([10.0, 50.0, 100.0])
    out = pf.apply_floor(p, floor=-5.0)
    assert np.allclose(out, p, atol=1e-4)


def test_floor_asymptotes_to_floor_well_below_floor() -> None:
    """For prices much lower than the floor, output → floor."""
    p = np.array([-100.0, -500.0, -1000.0])
    out = pf.apply_floor(p, floor=-5.0)
    assert np.allclose(out, -5.0, atol=1e-4)


def test_floor_at_floor_adds_log_2() -> None:
    """At p = floor exactly, softplus adds log(2) ≈ 0.693."""
    out = pf.apply_floor(np.array([-5.0]), floor=-5.0)
    assert out[0] == pytest.approx(-5.0 + np.log(2), abs=1e-9)


def test_floor_is_monotone() -> None:
    """Floored output must be a non-decreasing function of input."""
    x = np.linspace(-200, 200, 401)
    y = pf.apply_floor(x, floor=-5.0)
    assert (np.diff(y) >= 0).all()


def test_floor_never_below_floor_minus_epsilon() -> None:
    """Output should never fall below the floor value (within floating
    precision)."""
    x = np.linspace(-1e6, 1e6, 2000)
    y = pf.apply_floor(x, floor=-5.0)
    assert y.min() >= -5.0 - 1e-9


def test_softplus_safe_no_overflow_for_large_input() -> None:
    """log(1 + exp(1000)) would overflow without the protection."""
    out = pf.softplus_safe(np.array([1000.0]))
    assert np.isfinite(out[0])
    assert out[0] == pytest.approx(1000.0, abs=1e-6)


def test_softplus_safe_matches_naive_in_safe_range() -> None:
    """In the [-100, +100] range the protected and naive forms agree."""
    x = np.linspace(-50, 50, 101)
    out = pf.softplus_safe(x)
    naive = np.log(1.0 + np.exp(x))
    assert np.allclose(out, naive, atol=1e-9)


def test_floor_curve_returns_matching_arrays() -> None:
    x, y = pf.floor_curve()
    assert x.shape == y.shape
    assert (y >= -5.0 - 1e-9).all()


def test_default_floor_matches_empirical_fi_value() -> None:
    """Documented invariant: −5 EUR/MWh = 99-th-percentile of FI negative
    prices. If this constant changes the calibration data should be
    refreshed (see v2.5.14 release notes)."""
    assert pf.DEFAULT_FLOOR_EUR_MWH == -5.0


def test_apply_floor_handles_scalar_via_zero_dim_array() -> None:
    out = pf.apply_floor(np.array(10.0), floor=-5.0)
    assert float(out) == pytest.approx(10.0, abs=1e-4)
