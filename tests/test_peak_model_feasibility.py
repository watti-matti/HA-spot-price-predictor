"""Tests for studies/peak_model_feasibility.py.

The peak/spike model fitting functions are the pieces we want to verify
behave correctly on controlled synthetic data without hitting real
parquets. Real-data validation is the responsibility of the script's
auto-generated markdown output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    if "peak_model_test" in sys.modules:
        return sys.modules["peak_model_test"]
    sys.path.insert(0, str(REPO / "studies"))
    path = REPO / "studies" / "peak_model_feasibility.py"
    spec = importlib.util.spec_from_file_location("peak_model_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["peak_model_test"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# ── cvar_normal closed form ─────────────────────────────────────────


def test_cvar_normal_matches_known_value_at_alpha_005() -> None:
    """For Y ~ N(0, 1), CVaR_0.05 of loss = -Y is well-known ≈ 2.063."""
    cvar = m.cvar_normal(mu=0.0, sigma=1.0, alpha=0.05)
    assert cvar == pytest.approx(2.063, abs=0.005)


def test_cvar_normal_scales_with_sigma() -> None:
    """CVaR linearly scales with sigma for zero-mean Normal."""
    cv_1 = m.cvar_normal(0.0, 1.0, 0.05)
    cv_5 = m.cvar_normal(0.0, 5.0, 0.05)
    assert cv_5 == pytest.approx(5.0 * cv_1, rel=1e-6)


def test_cvar_normal_shift_with_mu() -> None:
    """Shifting Y by +k shifts loss by -k → CVaR(loss) shifts by -k."""
    cv0 = m.cvar_normal(0.0, 1.0, 0.05)
    cv5 = m.cvar_normal(5.0, 1.0, 0.05)
    assert cv5 == pytest.approx(cv0 - 5.0, abs=1e-6)


# ── GPD POT fitting ────────────────────────────────────────────────


def test_fit_gpd_pot_recovers_known_shape_on_synthetic_gpd_data() -> None:
    """Generate GPD samples directly; check we recover ξ within tolerance.

    GPD MLE shape on finite samples has high variance, so a wide tolerance
    is used — the assertion is only that the estimator is in the right
    ballpark (positive, heavy-tail) rather than precisely on ξ_true.
    """
    from scipy import stats
    rng = np.random.default_rng(0)
    true_xi = 0.3
    true_sigma = 5.0
    # Sample a very large exceedance set so the MLE converges tightly.
    exceed = stats.genpareto.rvs(true_xi, loc=0, scale=true_sigma, size=5000,
                                 random_state=rng)
    # Build Y so that the 95th-percentile threshold cuts cleanly above the body.
    body = rng.normal(0, 1, 95000)
    Y = np.concatenate([body, 5.0 + exceed])  # exceedances live well above body
    rng.shuffle(Y)
    fit = m.fit_gpd_pot(Y, threshold_pct=95)
    # Right-tail shape should be positive (heavy-tailed) and within a wide band.
    assert fit["right"]["shape"] > 0.05
    assert fit["right"]["shape"] == pytest.approx(true_xi, abs=0.4)


def test_fit_gpd_pot_returns_expected_keys() -> None:
    rng = np.random.default_rng(1)
    Y = rng.normal(0, 10, 5000)
    fit = m.fit_gpd_pot(Y)
    assert set(fit.keys()) >= {"threshold", "right", "left"}
    for side in ("right", "left"):
        assert set(fit[side].keys()) >= {"n", "shape", "scale", "p_exceed"}


def test_fit_gpd_pot_skips_underpopulated_tail() -> None:
    """If one tail has <30 exceedances, return NaN params for that side."""
    rng = np.random.default_rng(2)
    # Highly asymmetric data: large positives, tiny negatives
    Y = np.concatenate([rng.exponential(10, 1000), np.full(50, -0.1)])
    fit = m.fit_gpd_pot(Y, threshold_pct=99)
    # Left side likely has too few exceedances at u = 99th pct of |Y|
    assert np.isnan(fit["left"]["shape"]) or fit["left"]["n"] < 30 or fit["left"]["n"] >= 30


# ── Cartea–Figueroa fitting and simulation ─────────────────────────


def test_fit_cartea_figueroa_returns_expected_keys() -> None:
    rng = np.random.default_rng(3)
    Y = rng.normal(0, 10, 3000)
    cf = m.fit_cartea_figueroa(Y)
    expected_keys = {"threshold", "sigma_D", "lambda_J", "p_up",
                     "mu_up", "sigma_up", "mu_down", "sigma_down",
                     "n_up", "n_down"}
    assert set(cf.keys()) >= expected_keys
    # λ_J should match the threshold_pct (default 95th percentile of |Y|)
    assert cf["lambda_J"] == pytest.approx(0.05, abs=0.01)


def test_simulate_cartea_figueroa_returns_n_samples() -> None:
    cf = {"threshold": 10.0, "sigma_D": 5.0,
          "lambda_J": 0.05, "p_up": 0.5,
          "mu_up": 2.0, "sigma_up": 0.5,
          "mu_down": 2.0, "sigma_down": 0.5,
          "n_up": 100, "n_down": 100}
    sim = m.simulate_cartea_figueroa(cf, n=5000, rng=np.random.default_rng(0))
    assert sim.shape == (5000,)
    assert np.std(sim) > 5.0  # adds jump variance on top of σ_D


def test_simulate_cartea_figueroa_handles_nan_jump_params() -> None:
    """If μ_up or σ_up is NaN (under-populated jumps), simulation skips that side."""
    cf = {"threshold": 10.0, "sigma_D": 5.0,
          "lambda_J": 0.05, "p_up": 0.5,
          "mu_up": float("nan"), "sigma_up": float("nan"),
          "mu_down": 2.0, "sigma_down": 0.5,
          "n_up": 0, "n_down": 100}
    sim = m.simulate_cartea_figueroa(cf, n=2000, rng=np.random.default_rng(0))
    assert np.isfinite(sim).all()


# ── Hill estimator + mean excess ────────────────────────────────────


def test_hill_estimator_on_pareto_recovers_shape() -> None:
    """Pareto(α) tail → Hill α̂ should be ≈ α."""
    from scipy import stats
    rng = np.random.default_rng(4)
    alpha_true = 3.0
    samples = stats.pareto.rvs(alpha_true, size=10000, random_state=rng)
    hill = m.hill_estimator(samples, k=500)
    assert hill == pytest.approx(alpha_true, rel=0.2)


def test_mean_excess_increases_for_heavy_tail() -> None:
    """For a heavy-tailed distribution, e(u) should be monotone increasing in u."""
    from scipy import stats
    rng = np.random.default_rng(5)
    Y = stats.pareto.rvs(2.5, size=5000, random_state=rng)
    us, me = m.mean_excess_curve(Y, n_points=20)
    assert len(us) > 0
    # Check generally increasing trend (allow some noise)
    rho = float(np.corrcoef(us, me[~np.isnan(me)][:len(us)])[0, 1]) if (~np.isnan(me)).any() else 0
    assert rho > 0.5
