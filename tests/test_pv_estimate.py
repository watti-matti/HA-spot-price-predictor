"""Unit tests for the PV production estimator and effective-cost helpers.

Covers:
- estimate_pv_kwh_per_hour physics: zero-irradiance, capacity ceiling,
  tilt/azimuth corrections, efficiency parameter.
- marginal_effective_eur_kwh boundary cases (no PV, partial cover, full
  self-consumption, net export, negative sell price → liability).
- net_household_cost_eur sign convention (positive = pay grid; negative
  = paid).

These functions are pure (no I/O, no HA dependencies), so the tests are
entirely deterministic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# Load pv_estimate.py directly to bypass the custom_components package
# __init__.py (which imports homeassistant). The module is pure-Python
# with no HA dependency, so direct loading is safe.
_PV_PATH = (
    Path(__file__).resolve().parent.parent
    / "custom_components" / "spot_price_predictor" / "pv_estimate.py"
)
_spec = importlib.util.spec_from_file_location("pv_estimate", _PV_PATH)
pv_estimate = importlib.util.module_from_spec(_spec)
sys.modules["pv_estimate"] = pv_estimate
_spec.loader.exec_module(pv_estimate)

estimate_pv_kwh_per_hour = pv_estimate.estimate_pv_kwh_per_hour
marginal_effective_eur_kwh = pv_estimate.marginal_effective_eur_kwh
net_household_cost_eur = pv_estimate.net_household_cost_eur


# ── estimate_pv_kwh_per_hour ────────────────────────────────────────


def test_estimate_zero_when_disabled() -> None:
    assert estimate_pv_kwh_per_hour(800.0, capacity_kwp=0.0) == 0.0


def test_estimate_zero_irradiance_returns_zero() -> None:
    assert estimate_pv_kwh_per_hour(0.0, capacity_kwp=5.0) == 0.0


def test_estimate_negative_irradiance_clamped_to_zero() -> None:
    assert estimate_pv_kwh_per_hour(-100.0, capacity_kwp=5.0) == 0.0


def test_estimate_reference_case_at_45_south() -> None:
    """1000 W/m² × 5 kWp × 0.85 efficiency at 45°/S = 4.25 kWh."""
    pv = estimate_pv_kwh_per_hour(
        irradiance_w_m2=1000.0,
        capacity_kwp=5.0,
        tilt_deg=45.0,
        azimuth_deg=180.0,
        efficiency=0.85,
    )
    assert pv == pytest.approx(4.25, rel=0.01)


def test_estimate_capped_at_physical_ceiling() -> None:
    """No matter the irradiance, output ≤ capacity_kwp · efficiency."""
    pv = estimate_pv_kwh_per_hour(
        irradiance_w_m2=2000.0,  # Unrealistic, but tests the clamp
        capacity_kwp=5.0,
        efficiency=0.85,
    )
    assert pv <= 5.0 * 0.85 + 1e-9
    assert pv == pytest.approx(4.25, abs=0.01)


def test_estimate_tilt_correction_centered_at_45() -> None:
    """At tilt=45°, output should equal raw irradiance × area × eff."""
    base = estimate_pv_kwh_per_hour(500.0, 5.0, tilt_deg=45.0, efficiency=1.0)
    expect = 0.5 * 5.0 * 1.0  # irradiance(kW/m²) · capacity · efficiency at S
    assert base == pytest.approx(expect, rel=0.001)


def test_estimate_tilt_falls_off_far_from_45() -> None:
    """Output at flat (0°) or very steep (90°) tilt is less than at 45°."""
    base = estimate_pv_kwh_per_hour(500.0, 5.0, tilt_deg=45.0)
    flat = estimate_pv_kwh_per_hour(500.0, 5.0, tilt_deg=0.0)
    steep = estimate_pv_kwh_per_hour(500.0, 5.0, tilt_deg=90.0)
    assert flat < base
    assert steep < base


def test_estimate_azimuth_north_loses_more_than_south() -> None:
    south = estimate_pv_kwh_per_hour(500.0, 5.0, azimuth_deg=180.0)
    east = estimate_pv_kwh_per_hour(500.0, 5.0, azimuth_deg=90.0)
    west = estimate_pv_kwh_per_hour(500.0, 5.0, azimuth_deg=270.0)
    north = estimate_pv_kwh_per_hour(500.0, 5.0, azimuth_deg=0.0)
    assert south > east
    assert south > west
    assert east > north
    assert west > north
    # East/west should be roughly symmetric around south
    assert abs(east - west) < 0.05


def test_estimate_efficiency_scales_linearly() -> None:
    pv_full = estimate_pv_kwh_per_hour(1000.0, 5.0, efficiency=1.0)
    pv_half = estimate_pv_kwh_per_hour(1000.0, 5.0, efficiency=0.5)
    assert pv_half == pytest.approx(pv_full * 0.5, rel=0.001)


def test_estimate_invalid_efficiency_raises() -> None:
    with pytest.raises(ValueError):
        estimate_pv_kwh_per_hour(500.0, 5.0, efficiency=-0.1)
    with pytest.raises(ValueError):
        estimate_pv_kwh_per_hour(500.0, 5.0, efficiency=1.5)


# ── marginal_effective_eur_kwh ──────────────────────────────────────


def test_marginal_no_pv_returns_buy_price() -> None:
    """When p_h = 0, m_h = b_h regardless of sell price."""
    m = marginal_effective_eur_kwh(
        buy_eur_kwh=0.16, sell_eur_kwh=0.04, pv_kwh=0.0, baseload_kwh=1.0)
    assert m == pytest.approx(0.16, rel=0.001)


def test_marginal_pv_exactly_meets_baseload_returns_buy_price() -> None:
    """No surplus PV available for the new load → still buy from grid."""
    m = marginal_effective_eur_kwh(0.16, 0.04, pv_kwh=1.0, baseload_kwh=1.0)
    assert m == pytest.approx(0.16, rel=0.001)


def test_marginal_pv_covers_full_extra_load_returns_sell_price() -> None:
    """PV surplus ≥ 1 kWh → marginal cost = sell price (opportunity cost)."""
    m = marginal_effective_eur_kwh(
        buy_eur_kwh=0.16, sell_eur_kwh=0.04, pv_kwh=3.5, baseload_kwh=1.0)
    assert m == pytest.approx(0.04, rel=0.001)


def test_marginal_partial_pv_cover_interpolates() -> None:
    """PV surplus 0.5 kWh → 50/50 mix of sell & buy prices."""
    m = marginal_effective_eur_kwh(
        buy_eur_kwh=0.16, sell_eur_kwh=0.04, pv_kwh=1.5, baseload_kwh=1.0)
    expected = 0.5 * 0.04 + 0.5 * 0.16
    assert m == pytest.approx(expected, rel=0.001)


def test_marginal_negative_sell_price_propagates() -> None:
    """When s_h < 0 (deep oversupply) marginal cost can be slightly negative."""
    m = marginal_effective_eur_kwh(
        buy_eur_kwh=0.16, sell_eur_kwh=-0.05, pv_kwh=3.5, baseload_kwh=1.0)
    assert m == pytest.approx(-0.05, rel=0.001)


def test_marginal_bounded_by_sell_and_buy() -> None:
    """For ANY (b, s, p, c) with b ≥ s, m_h ∈ [s, b]."""
    cases = [
        (0.16, 0.04, 0.0, 1.0),
        (0.16, 0.04, 0.5, 1.0),
        (0.16, 0.04, 1.0, 1.0),
        (0.16, 0.04, 2.5, 1.0),
        (0.16, 0.04, 10.0, 1.0),
        (0.10, -0.05, 5.0, 1.0),
    ]
    for b, s, p, c in cases:
        m = marginal_effective_eur_kwh(b, s, p, c)
        lo, hi = (s, b) if s <= b else (b, s)
        assert lo - 1e-9 <= m <= hi + 1e-9, f"{m} not in [{lo},{hi}] for {(b,s,p,c)}"


# ── net_household_cost_eur ──────────────────────────────────────────


def test_net_cost_no_pv() -> None:
    """No PV → cost = c · b."""
    n = net_household_cost_eur(
        buy_eur_kwh=0.16, sell_eur_kwh=0.04, pv_kwh=0.0, consumption_kwh=1.0)
    assert n == pytest.approx(0.16, rel=0.001)


def test_net_cost_self_sufficient() -> None:
    """PV exactly equals consumption → zero net cost."""
    n = net_household_cost_eur(0.16, 0.04, pv_kwh=1.0, consumption_kwh=1.0)
    assert n == pytest.approx(0.0, abs=1e-9)


def test_net_cost_export_earns_revenue() -> None:
    """PV > consumption with positive sell price → user is paid (negative cost)."""
    n = net_household_cost_eur(0.16, 0.04, pv_kwh=3.0, consumption_kwh=1.0)
    # Export 2 kWh × 0.04 = 0.08 EUR earned
    assert n == pytest.approx(-0.08, rel=0.001)


def test_net_cost_negative_sell_creates_liability() -> None:
    """PV > consumption + sell price negative → user pays to export."""
    n = net_household_cost_eur(0.16, -0.05, pv_kwh=3.0, consumption_kwh=1.0)
    # Export 2 kWh and pay 0.05 each = 0.10 EUR cost from export
    assert n == pytest.approx(0.10, rel=0.001)
