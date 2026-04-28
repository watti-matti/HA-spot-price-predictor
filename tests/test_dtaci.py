"""Tests for DtACI online conformal-inference layer (Phase B).

Covers the contracts that make DtACI safe to drop into the coordinator:

* Quantile / pinball-loss numerical correctness
* ACI alpha update direction (miss → wider, cover → narrower)
* Realised coverage converges to target on stationary noise
* Realised coverage tracks target through abrupt regime shift
* State serialisation round-trip preserves intervals exactly
* Cold-start safety: no spurious intervals before warmup
* Asymmetric scoring is symmetric by design (sanity)
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest


# Load the modules without going through the HA-component package
# initialiser (which imports homeassistant.*).
_PKG = Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bias_corrector = _load(
    "custom_components.spot_price_predictor.bias_corrector",
    _PKG / "bias_corrector.py",
)
dtaci = _load(
    "custom_components.spot_price_predictor.dtaci",
    _PKG / "dtaci.py",
)
DtACI = dtaci.DtACI
OnlineBiasCorrector = bias_corrector.OnlineBiasCorrector


# ── Building blocks ─────────────────────────────────────────────────


def test_empirical_quantile_matches_numpy_linear():
    """Reference test: our quantile fn matches numpy `linear` interpolation."""
    vs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    # Manual numpy-linear at 0.9 of sorted [1,1,2,3,4,5,6,9] (n=8):
    # pos = 0.9 * 7 = 6.3 → between sorted[6]=6 and sorted[7]=9, frac=0.3
    # → 6 + 0.3 * (9 - 6) = 6.9
    assert math.isclose(dtaci._empirical_quantile(vs, 0.9), 6.9)
    # At 0.0 → minimum
    assert dtaci._empirical_quantile(vs, 0.0) == 1.0
    # At 1.0 → maximum
    assert dtaci._empirical_quantile(vs, 1.0) == 9.0
    # Empty → 0
    assert dtaci._empirical_quantile([], 0.5) == 0.0
    # Singleton
    assert dtaci._empirical_quantile([7.0], 0.5) == 7.0


def test_pinball_loss_is_zero_at_quantile():
    """Pinball loss is 0 when score == quantile; positive otherwise."""
    assert dtaci._pinball_loss(5.0, 5.0, 0.1) == 0.0
    # score > q → underprediction of upper tail → loss = alpha * (s - q)
    assert math.isclose(dtaci._pinball_loss(7.0, 5.0, 0.1), 0.1 * 2.0)
    # score < q → overprediction → loss = (1 - alpha) * (q - s)
    assert math.isclose(dtaci._pinball_loss(3.0, 5.0, 0.1), 0.9 * 2.0)


# ── ACI update direction ───────────────────────────────────────────


def test_alpha_decreases_on_miss_increases_on_cover():
    """Single-expert ACI: a miss should decrease alpha (widening),
    a cover should increase alpha (narrowing)."""
    inst = DtACI(target_coverage=0.9, gammas=[0.1], window=10, min_warmup=0)
    # Seed with some scores so quantile is well-defined
    for s in [1.0, 2.0, 3.0, 4.0, 5.0]:
        inst.score_window.append(s)
    inst.alphas = [0.1]
    initial = inst.alphas[0]
    # Force a miss: actual far from forecast (huge residual relative to scores)
    inst.update(forecast=0.0, actual=100.0)
    assert inst.alphas[0] < initial - 1e-6, "miss should decrease alpha"

    # Reset state, force a cover: very small residual
    inst2 = DtACI(target_coverage=0.9, gammas=[0.1], window=10, min_warmup=0)
    for s in [1.0, 2.0, 3.0, 4.0, 5.0]:
        inst2.score_window.append(s)
    inst2.alphas = [0.1]
    initial2 = inst2.alphas[0]
    inst2.update(forecast=0.0, actual=0.001)
    assert inst2.alphas[0] > initial2 - 1e-6, "cover should not decrease alpha"


# ── Coverage convergence on stationary noise ───────────────────────


def test_realised_coverage_converges_on_gaussian_noise():
    """On i.i.d. Gaussian residuals, realised coverage should be within
    a few % of the target after enough steps."""
    rng = random.Random(20260427)
    inst = DtACI(target_coverage=0.9, window=500, min_warmup=50)
    n = 2000
    covered_in_eval = 0
    eval_start = 500  # skip warmup region
    for t in range(n):
        forecast = 100.0
        actual = forecast + rng.gauss(0, 10.0)
        if t >= eval_start:
            low, _, high = inst.predict_interval(forecast)
            if low <= actual <= high:
                covered_in_eval += 1
        inst.update(forecast, actual)
    realised = covered_in_eval / (n - eval_start)
    assert 0.85 <= realised <= 0.95, (
        f"realised coverage {realised:.3f} outside [0.85, 0.95]"
    )


def test_realised_coverage_tracks_through_regime_shift():
    """Abrupt 5x noise scale change at midpoint: DtACI should adapt and
    long-run coverage stays within target ± 0.07."""
    rng = random.Random(7)
    inst = DtACI(target_coverage=0.9, window=300, min_warmup=50)
    n = 4000
    covered = 0
    eval_start = 1000
    eval_count = 0
    for t in range(n):
        sigma = 10.0 if t < n // 2 else 50.0
        forecast = 0.0
        actual = rng.gauss(0, sigma)
        if t >= eval_start:
            low, _, high = inst.predict_interval(forecast)
            if low <= actual <= high:
                covered += 1
            eval_count += 1
        inst.update(forecast, actual)
    realised = covered / eval_count
    # Allow a wider tolerance through the shift; the *long-run*
    # coverage must still be within a healthy margin of target.
    assert 0.83 <= realised <= 0.97, (
        f"post-shift realised coverage {realised:.3f} outside [0.83, 0.97]"
    )


# ── Bias-corrected DtACI ───────────────────────────────────────────


def test_bias_corrector_converges_toward_true_bias():
    """Bias EMA should pull the estimate toward the true bias on
    stationary biased data."""
    rng = random.Random(3)
    bc = OnlineBiasCorrector(halflife_days=5.0, warmup_steps=24,
                             cadence_per_day=24)
    true_bias = -7.0
    for _ in range(2000):
        actual = rng.gauss(0, 2)
        forecast = actual - true_bias  # forecast is biased high by 7
        bc.update(forecast, actual)
    # After 2000 steps with halflife of 5*24=120, the estimate should
    # be tightly converged
    assert bc.bias_estimate == pytest.approx(true_bias, abs=0.5)


def test_combined_corrects_bias_in_intervals():
    """Combined bias_corrector + DtACI: realised coverage on biased,
    noisy data should be near target. Forecast = actual + bias + noise
    (independent noise term so DtACI's intervals have something
    non-trivial to track)."""
    rng = random.Random(99)
    bc = OnlineBiasCorrector(halflife_days=5.0, warmup_steps=24,
                             cadence_per_day=24)
    inst = DtACI(target_coverage=0.9, window=400, min_warmup=50,
                 bias_corrector=bc)
    true_bias = -5.0
    forecast_noise_sigma = 4.0
    n = 1500
    covered = 0
    eval_start = 500
    eval_count = 0
    for t in range(n):
        actual = rng.gauss(0, 3)
        # Forecast carries a constant bias plus independent noise so the
        # post-correction residual has σ ~= forecast_noise_sigma.
        forecast = actual - true_bias + rng.gauss(0, forecast_noise_sigma)
        if t >= eval_start:
            low, point, high = inst.predict_interval(forecast)
            if low <= actual <= high:
                covered += 1
            eval_count += 1
            if t == n - 1:
                # Bias estimate should point in the right direction
                assert bc.bias_estimate == pytest.approx(true_bias, abs=1.0)
        inst.update(forecast, actual)
    realised = covered / eval_count
    assert 0.85 <= realised <= 0.95, (
        f"realised coverage {realised:.3f} outside [0.85, 0.95]"
    )


# ── Persistence ────────────────────────────────────────────────────


def test_to_from_dict_preserves_intervals_exactly():
    """Round-tripping through to_dict / from_dict must produce a
    DtACI that yields *identical* intervals on the next forecast."""
    rng = random.Random(5)
    inst = DtACI(target_coverage=0.9, window=200, min_warmup=20)
    for _ in range(300):
        actual = rng.gauss(50, 10)
        forecast = actual + rng.gauss(2, 1)
        inst.update(forecast, actual)
    f_test = 75.0
    low_a, p_a, high_a = inst.predict_interval(f_test)

    state = inst.to_dict()
    inst2 = DtACI.from_dict(state)
    low_b, p_b, high_b = inst2.predict_interval(f_test)

    assert low_a == pytest.approx(low_b, abs=1e-12)
    assert p_a == pytest.approx(p_b, abs=1e-12)
    assert high_a == pytest.approx(high_b, abs=1e-12)


def test_to_from_dict_with_bias_corrector_round_trip():
    """Persistence preserves bias_corrector state too."""
    rng = random.Random(2)
    bc = OnlineBiasCorrector(halflife_days=10, warmup_steps=24,
                             cadence_per_day=24)
    inst = DtACI(target_coverage=0.9, window=100, min_warmup=10,
                 bias_corrector=bc)
    for _ in range(200):
        actual = rng.gauss(50, 5)
        forecast = actual + 3.0  # constant bias
        inst.update(forecast, actual)

    state = inst.to_dict()
    assert "bias_corrector" in state
    inst2 = DtACI.from_dict(state)
    assert inst2.bias_corrector is not None
    assert inst2.bias_corrector.bias_estimate == pytest.approx(
        bc.bias_estimate, abs=1e-12
    )
    # And the corrected point is preserved
    p1 = inst.predict_interval(100.0)
    p2 = inst2.predict_interval(100.0)
    assert p1 == pytest.approx(p2, abs=1e-12)


# ── Cold-start safety ─────────────────────────────────────────────


def test_cold_start_returns_point_only():
    """Before min_warmup, predict_interval must return point-only intervals."""
    inst = DtACI(target_coverage=0.9, min_warmup=10)
    low, point, high = inst.predict_interval(42.0)
    assert low == point == high == 42.0


def test_warmup_gating_holds_through_few_updates():
    """Even after several updates, intervals stay collapsed until min_warmup."""
    inst = DtACI(target_coverage=0.9, min_warmup=30)
    rng = random.Random(11)
    for _ in range(20):
        inst.update(0.0, rng.gauss(0, 10))
    low, point, high = inst.predict_interval(0.0)
    # We've done 20 updates, threshold is 30 → still cold
    assert low == point == high == 0.0


def test_bias_corrector_cold_start_no_correction():
    """Bias correction is disabled below warmup_steps."""
    bc = OnlineBiasCorrector(halflife_days=10, warmup_steps=50)
    bc.bias_estimate = -100.0  # large fake bias, but n_updates = 0
    assert bc.correct(50.0) == 50.0  # no correction yet
    bc.n_updates = 50
    assert bc.correct(50.0) == 50.0 - 100.0  # now applied


# ── Configuration validation ──────────────────────────────────────


def test_target_coverage_bounds():
    with pytest.raises(ValueError):
        DtACI(target_coverage=0.0)
    with pytest.raises(ValueError):
        DtACI(target_coverage=1.0)


def test_empty_gammas_rejected():
    with pytest.raises(ValueError):
        DtACI(target_coverage=0.9, gammas=[])


def test_window_too_small():
    with pytest.raises(ValueError):
        DtACI(target_coverage=0.9, window=1)


# ── Diagnostics sanity ────────────────────────────────────────────


def test_dominant_expert_tracks_best_gamma_for_regime():
    """In a slowly-shifting regime, the largest-gamma expert should
    get most of the weight long-term (faster adaptation wins on
    pinball loss)."""
    rng = random.Random(13)
    inst = DtACI(target_coverage=0.9, gammas=[0.001, 0.01, 0.1],
                 window=200, min_warmup=20)
    # Slowly shifting bias: drifts linearly through the run
    for t in range(2000):
        bias_t = 0.05 * t  # linear drift
        actual = rng.gauss(0, 5)
        forecast = actual + bias_t
        inst.update(forecast, actual)
    # The fastest expert (gamma=0.1) should be at least as weighted
    # as the slowest. We don't require strict dominance because the
    # weights are noisy.
    assert inst.weights[2] > inst.weights[0]
