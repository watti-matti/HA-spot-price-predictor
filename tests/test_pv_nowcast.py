"""Unit tests for the intraday PV nowcast correction (pv_nowcast.py).

Covers the clear-sky-index primitive, EMA smoothing, forecast
correction (future hours only), realized-PV-energy fraction, and the
confidence label. All functions are pure — no HA, fully deterministic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load pv_nowcast.py directly to bypass the package __init__ (which
# imports homeassistant). The module is pure-Python.
_NC_PATH = (
    Path(__file__).resolve().parent.parent
    / "custom_components" / "spot_price_predictor" / "pv_nowcast.py"
)
_spec = importlib.util.spec_from_file_location("pv_nowcast", _NC_PATH)
pv_nowcast = importlib.util.module_from_spec(_spec)
sys.modules["pv_nowcast"] = pv_nowcast
_spec.loader.exec_module(pv_nowcast)

clear_sky_index = pv_nowcast.clear_sky_index
smooth_index = pv_nowcast.smooth_index
apply_nowcast = pv_nowcast.apply_nowcast
realized_pv_fraction = pv_nowcast.realized_pv_fraction
nowcast_confidence = pv_nowcast.nowcast_confidence
should_trigger_refresh = pv_nowcast.should_trigger_refresh


# ---------------------------------------------------------------------------
# clear_sky_index
# ---------------------------------------------------------------------------


def test_index_sunny_day_above_one():
    # Producing 4.5 kW where 3.0 was forecast → today is clearer.
    assert clear_sky_index(4.5, 3.0) == pytest.approx(1.5)


def test_index_cloudy_day_below_one():
    assert clear_sky_index(1.5, 3.0) == pytest.approx(0.5)


def test_index_none_when_sun_effectively_down():
    # Forecast near zero → no meaningful ratio (dawn/dusk/night).
    assert clear_sky_index(0.0, 0.0) is None
    assert clear_sky_index(0.5, 0.01) is None


def test_index_none_on_missing_or_negative_measurement():
    assert clear_sky_index(None, 3.0) is None
    assert clear_sky_index(-1.0, 3.0) is None


def test_index_clamped_both_ends():
    # A glitchy 30 kW reading against 3 kW forecast is clamped, not 10x.
    assert clear_sky_index(30.0, 3.0) == pytest.approx(3.0)
    # Near-zero measurement against real forecast clamps at the floor.
    assert clear_sky_index(0.01, 3.0) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# smooth_index
# ---------------------------------------------------------------------------


def test_smooth_seeds_on_first_sample():
    assert smooth_index(None, 1.4) == pytest.approx(1.4)


def test_smooth_none_keeps_previous():
    assert smooth_index(1.3, None) == pytest.approx(1.3)


def test_smooth_ema_blend():
    # alpha 0.3: 0.7*1.0 + 0.3*2.0 = 1.3
    assert smooth_index(1.0, 2.0, alpha=0.3) == pytest.approx(1.3)


def test_smooth_rejects_single_transient_over_cycles():
    # Steady k≈1.5 with one spike to 3.0 — smoothed stays well below 3.
    k = None
    for sample in (1.5, 1.5, 3.0, 1.5, 1.5):
        k = smooth_index(k, sample, alpha=0.3)
    assert k < 1.8


# ---------------------------------------------------------------------------
# apply_nowcast — future hours only
# ---------------------------------------------------------------------------


def test_apply_scales_future_leaves_past():
    forecast = [2.0, 2.0, 2.0, 2.0]
    hours = [-1.0, 0.0, 1.0, 2.0]  # first two are past/current
    out = apply_nowcast(forecast, 1.5, hours)
    assert out[0] == pytest.approx(2.0)  # past untouched
    assert out[1] == pytest.approx(2.0)  # current (h=0) untouched
    assert out[2] == pytest.approx(3.0)  # future scaled
    assert out[3] == pytest.approx(3.0)


def test_apply_none_or_unity_is_copy():
    forecast = [1.0, 2.0, 3.0]
    hours = [1.0, 2.0, 3.0]
    assert apply_nowcast(forecast, None, hours) == forecast
    assert apply_nowcast(forecast, 1.0, hours) == forecast
    # returns a new list, not the same object
    assert apply_nowcast(forecast, None, hours) is not forecast


def test_apply_floors_at_zero():
    out = apply_nowcast([1.0], 0.15, [1.0])
    assert out[0] >= 0.0


def test_apply_does_not_mutate_input():
    forecast = [2.0, 2.0]
    apply_nowcast(forecast, 2.0, [1.0, 1.0])
    assert forecast == [2.0, 2.0]


# ---------------------------------------------------------------------------
# realized_pv_fraction — PV-energy-weighted, not clock
# ---------------------------------------------------------------------------


def _bell_day():
    """A 24-h forecast with a midday solar bell (kWh/h)."""
    prof = [0.0] * 24
    for h in range(6, 20):
        prof[h] = max(0.0, 4.0 - abs(h - 13) * 0.5)
    return prof


def test_fraction_zero_before_dawn():
    day = _bell_day()
    # now = 04:00, all hours ahead (offsets +1..+20 for the sunlit ones)
    hours = [h - 4 for h in range(24)]
    assert realized_pv_fraction(day, hours) == pytest.approx(0.0)


def test_fraction_near_one_after_sunset():
    day = _bell_day()
    # now = 22:00, all production hours in the past
    hours = [h - 22 for h in range(24)]
    assert realized_pv_fraction(day, hours) == pytest.approx(1.0)


def test_fraction_midday_between():
    day = _bell_day()
    hours = [h - 13 for h in range(24)]  # now = 13:00 (peak)
    f = realized_pv_fraction(day, hours)
    assert 0.3 < f < 0.7


def test_fraction_beats_clock_in_the_evening():
    """At 18:00 the clock says 0.75 of the day is gone, but nearly all
    solar energy is already banked — the PV fraction is higher."""
    day = _bell_day()
    hours = [h - 18 for h in range(24)]
    assert realized_pv_fraction(day, hours) > 0.75


def test_fraction_zero_energy_day():
    assert realized_pv_fraction([0.0] * 24, [float(h) for h in range(24)]) == 0.0


# ---------------------------------------------------------------------------
# nowcast_confidence
# ---------------------------------------------------------------------------


def test_confidence_low_without_live_measurement():
    assert nowcast_confidence(0.9, measurement_live=False) == "low"


def test_confidence_scales_with_realized_fraction():
    assert nowcast_confidence(0.05, measurement_live=True) == "low"
    assert nowcast_confidence(0.3, measurement_live=True) == "medium"
    assert nowcast_confidence(0.8, measurement_live=True) == "high"


# ---------------------------------------------------------------------------
# should_trigger_refresh — fast-tick gating
# ---------------------------------------------------------------------------


def test_refresh_none_index_never_fires():
    assert should_trigger_refresh(
        None, 1.0, 99999, k_delta=0.15, min_gap_seconds=1800,
    ) is False


def test_refresh_fires_on_large_drift_after_gap():
    # k drifted 1.0 → 1.4 vs applied 1.0, and 40 min since last refresh.
    assert should_trigger_refresh(
        1.4, 1.0, 2400, k_delta=0.15, min_gap_seconds=1800,
    ) is True


def test_refresh_suppressed_within_rate_limit():
    # Big drift but only 10 min since last refresh → hold.
    assert should_trigger_refresh(
        1.4, 1.0, 600, k_delta=0.15, min_gap_seconds=1800,
    ) is False


def test_refresh_suppressed_on_small_drift():
    assert should_trigger_refresh(
        1.05, 1.0, 9999, k_delta=0.15, min_gap_seconds=1800,
    ) is False


def test_refresh_baseline_is_one_when_nothing_applied():
    # No prior applied k → compare against 1.0. k=1.3 is a 0.3 drift.
    assert should_trigger_refresh(
        1.3, None, None, k_delta=0.15, min_gap_seconds=1800,
    ) is True


def test_refresh_none_gap_treated_as_elapsed():
    # No prior refresh timestamp → not rate-limited.
    assert should_trigger_refresh(
        0.7, 1.0, None, k_delta=0.15, min_gap_seconds=1800,
    ) is True


# ---------------------------------------------------------------------------
# End-to-end: the incident regime
# ---------------------------------------------------------------------------


def test_incident_sunny_afternoon_lifts_remaining_forecast():
    """The 2026-07 incident in miniature: at 11:00 the panels are
    producing 1.7x the forecast (clear day the forecast missed). The
    remaining daylight hours should be lifted so the effective-price
    curve downstream reflects the surplus."""
    # 24-h forecast bell, now = 11:00.
    day = _bell_day()
    hours = [h - 11 for h in range(24)]
    # Measured 5.1 kW where 3.0 forecast at this hour.
    k_raw = clear_sky_index(5.1, 3.0)
    k = smooth_index(None, k_raw)
    corrected = apply_nowcast(day, k, hours)
    # Past mornings unchanged, afternoons lifted by ~1.7x.
    assert corrected[9] == pytest.approx(day[9])       # 09:00 past
    assert corrected[15] > day[15] * 1.5               # 15:00 lifted
    # Total remaining energy is now materially higher.
    rem_before = sum(day[h] for h in range(12, 24))
    rem_after = sum(corrected[h] for h in range(12, 24))
    assert rem_after > rem_before * 1.5
