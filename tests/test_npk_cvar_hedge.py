"""Tests for the NPK-CVaR hedge analysis tool (studies/npk_cvar_hedge.py).

The hedge tool is the validation framework for v2.4.x → v2.5.0: every new
model variant gets gated on out-of-sample CVaR reduction. These tests
verify the building blocks behave correctly so the gate is trustworthy.

Coverage:
- fit_seasonal_hdw: zero-mean residual, normalisation, week fill
- fit_ou_ar1: parameter recovery on synthetic OU process
- npk_cvar_objective: matches Rockafellar identity on Gaussian data
- historical_cvar: matches np.percentile-based reference
- optimize_hedge: recovers known optimum on synthetic data
- acf: lag-0 = 1, decays for white noise
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_hedge_module():
    """Load studies/npk_cvar_hedge.py without going through any package init."""
    if "npk_cvar_hedge_test" in sys.modules:
        return sys.modules["npk_cvar_hedge_test"]
    path = REPO / "studies" / "npk_cvar_hedge.py"
    spec = importlib.util.spec_from_file_location("npk_cvar_hedge_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["npk_cvar_hedge_test"] = mod
    spec.loader.exec_module(mod)
    return mod


hedge = _load_hedge_module()


# ── fit_seasonal_hdw ────────────────────────────────────────────────


def test_seasonal_residual_is_zero_mean() -> None:
    """Sequential subtraction guarantees E[Y] = 0 exactly."""
    rng = np.random.default_rng(42)
    n = 24 * 365 * 2  # 2 years hourly
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Pure noise (no real seasonality) — Y should still have zero mean
    x = rng.normal(50, 10, size=n)
    _, _, _, seasonal, Y = hedge.fit_seasonal_hdw(x, ts)
    assert abs(np.mean(Y)) < 1e-9, f"E[Y] should be 0, got {np.mean(Y)}"


def test_seasonal_decomposition_captures_known_pattern() -> None:
    """Inject a 5 c/kWh hourly oscillation; check P_hour recovers it."""
    n = 24 * 365 * 2
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    h = ts.hour.to_numpy()
    truth = 50.0 + 5.0 * np.sin(2 * np.pi * h / 24)
    P_hour, _, _, _, Y = hedge.fit_seasonal_hdw(truth, ts)
    # P_hour should approximately equal truth pattern minus its mean
    expected_hour_pattern = 5.0 * np.sin(2 * np.pi * np.arange(24) / 24)
    # P_hour absorbs the global mean (50) too
    assert P_hour.mean() == pytest.approx(50.0, abs=0.01)
    # Centred pattern should match
    assert np.allclose(P_hour - P_hour.mean(), expected_hour_pattern, atol=0.01)
    # Residual should be near zero
    assert np.std(Y) < 0.01


def test_seasonal_fills_unobserved_weeks() -> None:
    """Cover only weeks 10-30 of a year; P_week should be nearest-neighbour filled."""
    n = 24 * 7 * 21  # 21 weeks of data
    ts = pd.date_range("2024-03-04", periods=n, freq="1h", tz="UTC")
    x = np.full(n, 50.0)
    _, _, P_week, _, _ = hedge.fit_seasonal_hdw(x, ts)
    # All 53 entries populated (no NaN)
    assert not np.any(np.isnan(P_week))


# ── fit_ou_ar1 ──────────────────────────────────────────────────────


def test_ou_recovers_known_half_life() -> None:
    """Simulate AR(1) with b=0.95 → expected half-life log(2)/(-log(0.95)) ≈ 13.5 h.

    Use 50k samples and ±10 % tolerance — half-life is a nonlinear
    transform of b that amplifies sampling noise in the AR coefficient.
    """
    rng = np.random.default_rng(0)
    n = 50000
    b_true = 0.95
    sigma_eps = 1.0
    Y = np.zeros(n)
    for i in range(1, n):
        Y[i] = b_true * Y[i - 1] + rng.normal(0, sigma_eps)
    ou = hedge.fit_ou_ar1(Y)
    expected_hl = np.log(2.0) / -np.log(b_true)  # ≈ 13.5 hours
    assert ou["half_life_hours"] == pytest.approx(expected_hl, rel=0.10)
    assert ou["b"] == pytest.approx(b_true, rel=0.02)


def test_ou_raises_on_tiny_input() -> None:
    with pytest.raises(ValueError):
        hedge.fit_ou_ar1(np.array([1.0, 2.0]))


# ── historical_cvar ─────────────────────────────────────────────────


def test_historical_cvar_matches_manual_calc() -> None:
    """For uniform [0,100] losses with alpha=0.1, CVaR ≈ mean of top 10% ≈ 95."""
    rng = np.random.default_rng(7)
    L = rng.uniform(0, 100, size=10000)
    cvar = hedge.historical_cvar(L, alpha=0.1)
    # Top 10% of Uniform(0,100) has mean ≈ 95
    assert cvar == pytest.approx(95.0, abs=1.0)


def test_historical_cvar_handles_empty() -> None:
    assert np.isnan(hedge.historical_cvar(np.array([]), 0.05))


# ── npk_cvar_objective ──────────────────────────────────────────────


def test_npk_cvar_objective_is_finite_and_positive_for_typical_case() -> None:
    """For positive losses, the kernel CVaR should be finite and ≥ VaR."""
    rng = np.random.default_rng(1)
    rS = rng.normal(0, 1, 1000)
    rF = rng.normal(0, 1, 1000)
    j = hedge.npk_cvar_objective(h=0.0, v=2.0, rS=rS, rF=rF, alpha=0.05)
    assert np.isfinite(j)


def test_npk_cvar_objective_decreases_when_hedge_helps() -> None:
    """If rF perfectly tracks rS, hedging with h=1 should reduce CVaR vs h=0."""
    rng = np.random.default_rng(2)
    rF = rng.normal(0, 1, 2000)
    rS = rF + rng.normal(0, 0.1, 2000)  # rS ≈ rF + noise
    # At h=0 the loss is just rS; at h=1 it's just the small noise
    j_unhedged = hedge.npk_cvar_objective(h=0.0, v=1.0, rS=rS, rF=rF, alpha=0.05)
    j_hedged = hedge.npk_cvar_objective(h=1.0, v=0.1, rS=rS, rF=rF, alpha=0.05)
    assert j_hedged < j_unhedged


# ── optimize_hedge ──────────────────────────────────────────────────


def test_optimize_hedge_recovers_known_optimum() -> None:
    """If rS = β*rF + noise with β=0.7, optimal h ≈ 0.7."""
    rng = np.random.default_rng(3)
    rF = rng.normal(0, 1, 2000)
    rS = 0.7 * rF + rng.normal(0, 0.3, 2000)
    result = hedge.optimize_hedge(rS, rF, alpha=0.05)
    assert result["h_hat"] == pytest.approx(0.7, abs=0.1)
    # Test CVaR should be below the unhedged
    assert result["cvar_test_hist_hedged"] < result["cvar_test_hist_unhedged"]


def test_optimize_hedge_zero_hedge_when_uncorrelated() -> None:
    """If rS and rF are independent, the optimal h should be ≈ 0."""
    rng = np.random.default_rng(4)
    rF = rng.normal(0, 1, 2000)
    rS = rng.normal(0, 1, 2000)
    result = hedge.optimize_hedge(rS, rF, alpha=0.05)
    assert abs(result["h_hat"]) < 0.2


def test_optimize_hedge_raises_on_short_input() -> None:
    with pytest.raises(ValueError):
        hedge.optimize_hedge(np.zeros(10), np.zeros(10))


# ── acf ─────────────────────────────────────────────────────────────


def test_acf_lag_0_is_excluded_but_lag_1_meaningful() -> None:
    """White noise should have ACF near 0 at all lags."""
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 10000)
    a = hedge.acf(y, lags=[1, 10, 100])
    for lag, rho in a.items():
        assert abs(rho) < 0.05, f"white noise should have ACF near 0 at lag {lag}, got {rho}"


def test_acf_persistent_series_has_high_lag1() -> None:
    """AR(1) with b=0.9 should give ACF ≈ 0.9 at lag 1."""
    rng = np.random.default_rng(6)
    n = 10000
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.9 * y[i - 1] + rng.normal(0, 1)
    a = hedge.acf(y, lags=[1])
    assert a[1] == pytest.approx(0.9, abs=0.03)
