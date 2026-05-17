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
