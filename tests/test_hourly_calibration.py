"""Tests for custom_components/spot_price_predictor/hourly_calibration.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

# The package __init__.py imports homeassistant; bypass that by adding
# the inner directory to sys.path so we can import the module directly.
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

# Pre-import the dependencies the module needs so the
# `from . import dtaci as _dtaci_mod` lines resolve.
import importlib
import dtaci as _dtaci_mod                   # noqa: F401, E402
import bias_corrector as _bias_mod           # noqa: F401, E402

# Manually load hourly_calibration as a top-level module so the
# relative `from .` imports work — we trick it by injecting the
# package context.
import types
pkg = types.ModuleType("spot_price_predictor")
pkg.__path__ = [str(REPO / "custom_components" / "spot_price_predictor")]
sys.modules["spot_price_predictor"] = pkg
sys.modules["spot_price_predictor.dtaci"] = _dtaci_mod
sys.modules["spot_price_predictor.bias_corrector"] = _bias_mod
spec = importlib.util.spec_from_file_location(
    "spot_price_predictor.hourly_calibration",
    REPO / "custom_components" / "spot_price_predictor" / "hourly_calibration.py",
)
hc = importlib.util.module_from_spec(spec)
sys.modules["spot_price_predictor.hourly_calibration"] = hc
spec.loader.exec_module(hc)


# ── HourlyBiasCorrector ────────────────────────────────────────────


def test_bias_corrector_returns_forecast_during_warmup() -> None:
    bc = hc.HourlyBiasCorrector(warmup_hours=24)
    # Feed a few biased observations; correction should still be 0
    for _ in range(10):
        bc.update(forecast=10.0, actual=12.0)
    assert bc.correct(20.0) == pytest.approx(20.0, abs=1e-9)


def test_bias_corrector_converges_to_persistent_bias() -> None:
    """If the forecast is consistently 3 EUR/MWh too low, the EMA of
    residuals (actual − forecast) → +3, so correct(forecast) → forecast + 3."""
    bc = hc.HourlyBiasCorrector(halflife_days=2.0, warmup_hours=24)
    # 10 days of biased observations (240 hours) — plenty to converge
    for _ in range(240):
        bc.update(forecast=10.0, actual=13.0)
    assert bc.warm
    assert bc.bias_estimate == pytest.approx(3.0, abs=0.3)
    assert bc.correct(50.0) == pytest.approx(53.0, abs=0.3)


def test_bias_corrector_winsorises_extreme_residuals() -> None:
    """A single 1000 EUR/MWh outlier should not blow up the bias estimate."""
    bc = hc.HourlyBiasCorrector(halflife_days=2.0, warmup_hours=24,
                                  winsor_limit=5.0)
    for _ in range(100):
        bc.update(forecast=10.0, actual=10.5)   # gentle bias to seed scale
    spike_bias_before = bc.bias_estimate
    bc.update(forecast=10.0, actual=1010.0)     # absurd outlier
    assert abs(bc.bias_estimate - spike_bias_before) < 5.0


def test_bias_corrector_roundtrips_through_dict() -> None:
    bc = hc.HourlyBiasCorrector(halflife_days=10.0, warmup_hours=48)
    for _ in range(80):
        bc.update(20.0, 21.5)
    d = bc.to_dict()
    bc2 = hc.HourlyBiasCorrector.from_dict(d)
    assert bc2.bias_estimate == pytest.approx(bc.bias_estimate, abs=1e-9)
    assert bc2.correct(100.0) == pytest.approx(bc.correct(100.0), abs=1e-9)


# ── Warm-up: CMA -> EMA hand-over (v2.18.0) ───────────────────────


def test_zero_init_reaches_only_half_the_bias_at_one_halflife() -> None:
    """Documents the defect the adaptive init fixes.

    A constant-gain EMA started at zero has expectation
    `1 - (1 - lambda)^n` of the truth, so at n = halflife it is at
    exactly 50 %. This is why a 14-day half-life behind a 14-update
    warm-up gate switched on at half strength.
    """
    bc = _bias_mod.OnlineBiasCorrector(halflife_days=14.0, warmup_steps=0,
                                       cadence_per_day=1, winsor_limit=None,
                                       adaptive_init=False)
    for _ in range(14):
        bc.update(forecast=10.0, actual=30.0)     # true bias +20
    assert bc.bias_estimate == pytest.approx(10.0, rel=1e-6)


def test_adaptive_init_is_exact_after_a_single_observation() -> None:
    """CMA property: with gain 1/n the first update sets m_1 = x_1."""
    bc = _bias_mod.OnlineBiasCorrector(halflife_days=14.0, warmup_steps=0,
                                       cadence_per_day=1, winsor_limit=None,
                                       adaptive_init=True)
    bc.update(forecast=10.0, actual=30.0)
    assert bc.bias_estimate == pytest.approx(20.0, rel=1e-9)


def test_adaptive_init_tracks_the_truth_where_zero_init_lags() -> None:
    """Same data, same half-life — only the gain schedule differs."""
    kw = dict(halflife_days=14.0, warmup_steps=0, cadence_per_day=1,
              winsor_limit=None)
    fast = _bias_mod.OnlineBiasCorrector(adaptive_init=True, **kw)
    slow = _bias_mod.OnlineBiasCorrector(adaptive_init=False, **kw)
    for _ in range(14):
        fast.update(10.0, 30.0)
        slow.update(10.0, 30.0)
    assert fast.bias_estimate == pytest.approx(20.0, rel=1e-6)
    assert slow.bias_estimate < 0.6 * fast.bias_estimate


def test_adaptive_gain_hands_over_to_lambda_at_one_over_lambda() -> None:
    """Gain is `max(1/n, lambda)`; the hand-over must be at n* = 1/lambda
    and continuous (no step change in the applied gain)."""
    bc = _bias_mod.OnlineBiasCorrector(halflife_days=3.0, warmup_steps=0,
                                       cadence_per_day=1, winsor_limit=None,
                                       adaptive_init=True)
    lam = bc.lambda_
    n_star = 1.0 / lam
    assert 4.0 < n_star < 5.0            # halflife 3 -> lambda ~= 0.2063
    for k in range(8):
        probe = _bias_mod.OnlineBiasCorrector(
            halflife_days=3.0, warmup_steps=0, cadence_per_day=1,
            winsor_limit=None, adaptive_init=True)
        for _ in range(k):               # k zero-residual observations
            probe.update(10.0, 10.0)
        probe.update(10.0, 11.0)         # then a unit residual
        assert probe.bias_estimate == pytest.approx(max(1.0 / (k + 1), lam),
                                                    rel=1e-9)


def test_adaptive_init_does_not_crush_early_residuals_by_winsorising() -> None:
    """Winsorisation starts early under adaptive_init; it must clip
    outliers, not legitimate residuals seen while the scale is young."""
    bc = _bias_mod.OnlineBiasCorrector(halflife_days=3.0, warmup_steps=0,
                                       cadence_per_day=1, winsor_limit=5.0,
                                       adaptive_init=True)
    for _ in range(6):
        bc.update(forecast=10.0, actual=30.0)     # steady +20 bias
    assert bc.bias_estimate == pytest.approx(20.0, abs=1.0)


def test_dtaci_style_correctors_keep_the_constant_gain_by_default() -> None:
    """The v2.18.0 retune is scoped to the hourly bias corrector; the
    D(k) bundles must be untouched until measured separately."""
    bc = _bias_mod.OnlineBiasCorrector(halflife_days=21.0, warmup_steps=7,
                                       cadence_per_day=1)
    assert bc.adaptive_init is False


# ── PerHourBiasCorrector retune (v2.18.0) ─────────────────────────


def test_per_hour_defaults_are_the_retuned_values() -> None:
    ph = hc.PerHourBiasCorrector()
    assert ph.halflife_days == pytest.approx(3.0)
    assert ph.warmup_updates == 2
    assert all(b.adaptive_init for b in ph._inner.values())


def test_per_hour_corrects_within_days_not_weeks() -> None:
    """The user-visible fix: after a state wipe the corrector must start
    correcting in days. Previously it was silent for 14 daily
    observations and then applied half the true bias."""
    ph = hc.PerHourBiasCorrector()
    for _ in range(3):                              # three days
        ph.update(forecast=40.0, actual=20.0, hour=9)
    assert ph.correct(40.0, hour=9) == pytest.approx(20.0, abs=3.0)


def test_per_hour_from_dict_ignores_a_persisted_halflife() -> None:
    """An upgrade must adopt the retuned tuning, not resurrect the old
    values from the state file, while keeping the learned history."""
    ph = hc.PerHourBiasCorrector()
    for _ in range(10):
        ph.update(forecast=40.0, actual=20.0, hour=9)
    d = ph.to_dict()
    d["halflife_days"] = 14.0            # as written by v2.17.x
    d["warmup_updates"] = 14
    for ser in d["inner"].values():
        ser["halflife_days"] = 14.0
        ser["warmup_steps"] = 14
        ser["adaptive_init"] = False
    back = hc.PerHourBiasCorrector.from_dict(d)
    assert back.halflife_days == pytest.approx(3.0)
    assert back.warmup_updates == 2
    assert all(b.adaptive_init for b in back._inner.values())
    # learned state survives
    assert back._inner[9].n_updates == 10
    assert back._inner[9].bias_estimate == pytest.approx(
        ph._inner[9].bias_estimate, abs=1e-9)


def test_per_hour_state_roundtrip_is_behaviour_preserving() -> None:
    ph = hc.PerHourBiasCorrector()
    for _ in range(8):
        ph.update(forecast=40.0, actual=20.0, hour=9)
    back = hc.PerHourBiasCorrector.from_dict(ph.to_dict())
    assert back.correct(40.0, hour=9) == pytest.approx(
        ph.correct(40.0, hour=9), abs=1e-9)


# ── HourlyFanChartCalibrator ──────────────────────────────────────


def test_fan_chart_calibrator_initialises_one_dtaci_per_band() -> None:
    fc = hc.HourlyFanChartCalibrator(target_coverages=(0.5, 0.8, 0.95))
    assert set(fc.instances.keys()) == {0.5, 0.8, 0.95}


def test_fan_chart_calibrator_bands_widen_with_higher_coverage() -> None:
    """After warmup, the 95 % band must be wider than the 50 % band."""
    rng = np.random.default_rng(0)
    fc = hc.HourlyFanChartCalibrator(target_coverages=(0.5, 0.95),
                                       window=2000)
    # 1500 obs of Gaussian noise around a constant forecast
    for _ in range(1500):
        forecast = 10.0
        actual   = 10.0 + rng.normal(0, 5)
        fc.update(forecast, actual)
    bands = fc.predict_bands(forecast=10.0)
    width_50 = bands[0.5][1] - bands[0.5][0]
    width_95 = bands[0.95][1] - bands[0.95][0]
    assert width_95 > width_50


def test_fan_chart_realised_coverage_approaches_target() -> None:
    """Feed Gaussian-noise observations; the 90 % band's realised
    coverage on a held-out batch should approach 0.9 within ±10 pp."""
    rng = np.random.default_rng(42)
    fc = hc.HourlyFanChartCalibrator(target_coverages=(0.9,), window=2000)
    # Warm up on 1000 obs
    for _ in range(1000):
        actual = rng.normal(0, 3)
        fc.update(forecast=0.0, actual=actual)
    # Evaluate 500 obs
    hits = 0
    for _ in range(500):
        actual = rng.normal(0, 3)
        lo, hi = fc.predict_bands(0.0)[0.9]
        if lo <= actual <= hi:
            hits += 1
        fc.update(forecast=0.0, actual=actual)
    realised = hits / 500
    assert abs(realised - 0.9) < 0.1


def test_fan_chart_roundtrips_through_dict() -> None:
    fc = hc.HourlyFanChartCalibrator(target_coverages=(0.5, 0.9))
    rng = np.random.default_rng(0)
    for _ in range(200):
        fc.update(0.0, rng.normal(0, 1))
    d = fc.to_dict()
    fc2 = hc.HourlyFanChartCalibrator.from_dict(d)
    b1 = fc.predict_bands(0.0)
    b2 = fc2.predict_bands(0.0)
    for tc in (0.5, 0.9):
        assert b2[tc][0] == pytest.approx(b1[tc][0], abs=1e-9)
        assert b2[tc][1] == pytest.approx(b1[tc][1], abs=1e-9)


# ── RefitMonitor ───────────────────────────────────────────────────


def test_refit_monitor_does_not_trigger_below_drift() -> None:
    mon = hc.RefitMonitor(target_coverage=0.9, drift_pp=0.05,
                            persistence_steps=10)
    for _ in range(50):
        mon.observe(realised_coverage=0.88)  # only 2 pp drift
    assert not mon.refit_recommended


def test_refit_monitor_triggers_after_persistent_drift() -> None:
    mon = hc.RefitMonitor(target_coverage=0.9, drift_pp=0.05,
                            persistence_steps=10)
    for _ in range(20):
        mon.observe(realised_coverage=0.70)  # 20 pp drift, way above threshold
    assert mon.refit_recommended
    assert len(mon.trigger_history) == 1


def test_refit_monitor_resets_when_drift_subsides() -> None:
    mon = hc.RefitMonitor(target_coverage=0.9, drift_pp=0.05,
                            persistence_steps=5)
    for _ in range(10):
        mon.observe(0.70)
    assert mon.refit_recommended
    # Recovery: coverage back near target
    for _ in range(10):
        mon.observe(0.90)
    assert not mon.refit_recommended
    assert mon.consecutive_drift_hours == 0


def test_refit_monitor_persists_history_via_roundtrip() -> None:
    mon = hc.RefitMonitor(target_coverage=0.9, drift_pp=0.05,
                            persistence_steps=5)
    for _ in range(10):
        mon.observe(0.70, timestamp_iso="2025-01-15T10:00:00+00:00")
    d = mon.to_dict()
    mon2 = hc.RefitMonitor.from_dict(d)
    assert mon2.refit_recommended is True
    assert mon2.trigger_history == mon.trigger_history
