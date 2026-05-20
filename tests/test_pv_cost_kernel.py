"""Tests for `custom_components.spot_price_predictor.pv_cost_kernel`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from pv_cost_kernel import cost_distribution  # noqa: E402


# ── Shape / validation ───────────────────────────────────────────────


def test_alpha_must_be_in_unit_interval() -> None:
    buy = np.ones((4, 5))
    with pytest.raises(ValueError):
        cost_distribution(buy, buy, buy, buy, alpha=0.0)
    with pytest.raises(ValueError):
        cost_distribution(buy, buy, buy, buy, alpha=1.0)


def test_inconsistent_shapes_raise() -> None:
    buy = np.ones((4, 5))
    sell = np.ones((4, 5))
    pv = np.ones((4, 6))  # n_hours mismatch
    cons = np.ones(5)
    with pytest.raises(ValueError):
        cost_distribution(buy, sell, pv, cons)


def test_buy_must_be_2d_raises() -> None:
    buy = np.ones(24)   # 1-D
    sell = np.ones((4, 24))
    pv = np.ones((4, 24))
    cons = np.ones(24)
    with pytest.raises(ValueError, match="2-D"):
        cost_distribution(buy, sell, pv, cons)


def test_sell_shape_mismatch_raises() -> None:
    buy = np.ones((4, 24))
    sell = np.ones((4, 23))  # mismatched n_hours
    pv = np.ones((4, 24))
    cons = np.ones(24)
    with pytest.raises(ValueError, match="sell shape"):
        cost_distribution(buy, sell, pv, cons)


def test_pv_shape_mismatch_raises() -> None:
    buy = np.ones((4, 24))
    sell = np.ones((4, 24))
    pv = np.ones((3, 24))  # mismatched n_paths
    cons = np.ones(24)
    with pytest.raises(ValueError, match="pv shape"):
        cost_distribution(buy, sell, pv, cons)


def test_consumption_2d_with_wrong_shape_raises() -> None:
    buy = np.ones((4, 24))
    sell = np.ones((4, 24))
    pv = np.ones((4, 24))
    cons = np.ones((4, 23))  # 2-D, wrong shape
    with pytest.raises(ValueError, match="consumption \\(2-D\\)"):
        cost_distribution(buy, sell, pv, cons)


def test_consumption_1d_wrong_length_raises() -> None:
    buy = np.ones((4, 24))
    sell = np.ones((4, 24))
    pv = np.ones((4, 24))
    cons = np.ones(20)  # 1-D, wrong length
    with pytest.raises(ValueError, match="consumption \\(1-D\\)"):
        cost_distribution(buy, sell, pv, cons)


def test_consumption_3d_raises() -> None:
    buy = np.ones((4, 24))
    sell = np.ones((4, 24))
    pv = np.ones((4, 24))
    cons = np.ones((4, 24, 1))  # 3-D
    with pytest.raises(ValueError, match="1-D or 2-D"):
        cost_distribution(buy, sell, pv, cons)


def test_consumption_2d_accepted() -> None:
    """The 2-D consumption path (one consumption per scenario) works."""
    rng = np.random.default_rng(99)
    buy = np.full((10, 24), 0.20)
    sell = np.full((10, 24), 0.05)
    pv = np.zeros((10, 24))
    # 2-D consumption: different load per scenario
    cons = rng.uniform(0.5, 2.0, size=(10, 24))
    out = cost_distribution(buy, sell, pv, cons)
    # Each path's cost = sum(buy * cons), distinct since cons varies
    expected = (buy * cons).sum(axis=1)
    assert np.allclose(out.cost_per_path_eur, expected)


def test_consumption_1d_is_broadcast_over_paths() -> None:
    buy = np.full((10, 24), 0.20)
    sell = np.full((10, 24), 0.05)
    pv = np.zeros((10, 24))
    cons = np.ones(24)  # 1 kWh per hour
    out = cost_distribution(buy, sell, pv, cons)
    # Same prices, same consumption, no PV → all paths produce identical cost.
    assert np.allclose(out.cost_per_path_eur, 24 * 0.20)


# ── Limit cases ──────────────────────────────────────────────────────


def test_no_pv_reduces_to_buy_times_consumption() -> None:
    n_paths, n_hours = 50, 24
    rng = np.random.default_rng(0)
    buy = rng.uniform(0.05, 0.30, size=(n_paths, n_hours))
    sell = buy * 0.5
    pv = np.zeros_like(buy)
    cons = np.full(n_hours, 1.5)
    out = cost_distribution(buy, sell, pv, cons)
    expected_per_path = (buy * cons).sum(axis=1)
    assert np.allclose(out.cost_per_path_eur, expected_per_path)
    assert out.pv_self_consumed_kwh_mean == 0.0
    assert out.pv_exported_kwh_mean == 0.0


def test_pv_far_above_consumption_yields_pure_export() -> None:
    n_paths, n_hours = 30, 12
    buy = np.full((n_paths, n_hours), 0.25)
    sell = np.full((n_paths, n_hours), 0.04)
    cons = np.full(n_hours, 0.5)
    pv = np.full((n_paths, n_hours), 10.0)  # way above consumption
    out = cost_distribution(buy, sell, pv, cons)
    # Pure surplus: cost = -sell * (pv - cons) per hour.
    expected = -(sell * (pv - cons)).sum(axis=1)
    assert np.allclose(out.cost_per_path_eur, expected)


def test_pv_equals_consumption_zero_cost() -> None:
    n_paths, n_hours = 5, 8
    buy = np.full((n_paths, n_hours), 0.20)
    sell = np.full((n_paths, n_hours), 0.05)
    cons = np.full(n_hours, 1.0)
    pv = np.full((n_paths, n_hours), 1.0)
    out = cost_distribution(buy, sell, pv, cons)
    assert np.allclose(out.cost_per_path_eur, 0.0)


# ── CVaR identities ──────────────────────────────────────────────────


def test_cvar_of_constant_cost_equals_constant() -> None:
    n_paths, n_hours = 100, 24
    buy = np.full((n_paths, n_hours), 0.20)
    sell = np.full((n_paths, n_hours), 0.05)
    pv = np.zeros_like(buy)
    cons = np.ones(n_hours)
    out = cost_distribution(buy, sell, pv, cons, alpha=0.05)
    # All paths identical → mean == CVaR == VaR.
    assert out.mean_eur_kwh == pytest.approx(0.20)
    assert out.cvar_eur_kwh == pytest.approx(0.20)
    assert out.var_eur_kwh == pytest.approx(0.20)


def test_cvar_at_least_mean_for_asymmetric_distribution() -> None:
    rng = np.random.default_rng(42)
    n_paths, n_hours = 500, 24
    # Right-skewed prices: lognormal mean 0.20, occasional spikes
    buy = rng.lognormal(mean=np.log(0.15), sigma=0.6, size=(n_paths, n_hours))
    sell = buy * 0.25
    pv = np.zeros_like(buy)
    cons = np.ones(n_hours)
    out = cost_distribution(buy, sell, pv, cons, alpha=0.05)
    # For a right-tailed cost distribution, CVaR_5% > mean.
    assert out.cvar_eur_kwh > out.mean_eur_kwh
    assert out.var_eur_kwh >= out.mean_eur_kwh


# ── Monotonicity ─────────────────────────────────────────────────────


def test_more_pv_uniformly_reduces_mean_cost() -> None:
    rng = np.random.default_rng(1)
    n_paths, n_hours = 200, 24
    buy = rng.uniform(0.05, 0.30, size=(n_paths, n_hours))
    sell = buy * 0.3
    cons = np.full(n_hours, 1.0)
    pv_low = np.full_like(buy, 0.2)
    pv_high = np.full_like(buy, 0.8)
    low = cost_distribution(buy, sell, pv_low, cons)
    high = cost_distribution(buy, sell, pv_high, cons)
    assert high.mean_eur_kwh < low.mean_eur_kwh


# ── PV self-consumption bookkeeping ──────────────────────────────────


def test_self_consumption_plus_export_equals_total_pv() -> None:
    rng = np.random.default_rng(2)
    n_paths, n_hours = 100, 24
    buy = np.full((n_paths, n_hours), 0.20)
    sell = np.full((n_paths, n_hours), 0.05)
    cons = rng.uniform(0.5, 2.5, size=n_hours)
    pv = rng.uniform(0.0, 3.0, size=(n_paths, n_hours))
    out = cost_distribution(buy, sell, pv, cons)
    total_pv = pv.sum(axis=1).mean()
    assert out.pv_self_consumed_kwh_mean + out.pv_exported_kwh_mean == pytest.approx(
        total_pv, rel=1e-10
    )
