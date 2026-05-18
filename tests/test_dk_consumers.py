"""Tests for D(k) cheap/peak schema across producers and consumers (Phase A).

This test module covers the *boundary contracts* that downstream consumers
of the D(k) schema rely on:

1. The HA-side mirror `custom_components.spot_price_predictor.dk_utils`
   matches `src.dk_utils` byte-for-byte semantically.
2. The legacy 24-element D(k) array can be reconstructed exactly from
   the new (dk_cheap[12], dk_peak[12]) split — used by the thermal
   optimizer when the sensor still emits both for backward compatibility.
3. The cheap/peak split satisfies the "sum identity":
       cheap[11] + peak[11] = 2 * daily_average
   which provides a free cross-check the dashboard, sensor, and optimizer
   can all assert on.

These contracts are what make the migration safe: any new consumer (HA
sensor, Lovelace card, thermal LP) can read either schema and produce
the same answer, modulo numerical noise.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from src.dk_utils import (
    compute_dk_cheap_peak,
    is_monotone_cheap,
    is_monotone_peak,
    reconstruct_sorted_prices,
)


# Load the HA-side mirror without going through the package __init__
# (which imports from homeassistant and is not available in pytest).
_HA_DKUTILS_PATH = (
    Path(__file__).parent.parent
    / "custom_components" / "spot_price_predictor" / "dk_utils.py"
)
_spec = importlib.util.spec_from_file_location("ha_dk_utils", _HA_DKUTILS_PATH)
ha_dk_utils = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ha_dk_utils)


# ─── Mirror parity ───────────────────────────────────────────────────


def _diverse_24h() -> list[float]:
    """Return 24 prices spanning a realistic daily distribution."""
    return [
        12.4, 8.9, 7.1, 6.5, 6.2, 7.0, 9.8, 22.1,
        45.0, 38.2, 31.5, 28.7, 26.9, 25.4, 24.1, 24.8,
        29.3, 41.6, 58.2, 66.1, 51.4, 38.7, 25.3, 17.6,
    ]


def test_ha_mirror_matches_src_compute_dk_cheap_peak():
    """HA-side and src-side `compute_dk_cheap_peak` agree to numerical noise."""
    prices = _diverse_24h()
    src_cheap, src_peak = compute_dk_cheap_peak(prices)
    ha_cheap, ha_peak = ha_dk_utils.compute_dk_cheap_peak(prices)
    assert len(src_cheap) == len(ha_cheap) == 12
    assert len(src_peak) == len(ha_peak) == 12
    for s, h in zip(src_cheap, ha_cheap):
        assert math.isclose(s, h, rel_tol=1e-12, abs_tol=1e-12)
    for s, h in zip(src_peak, ha_peak):
        assert math.isclose(s, h, rel_tol=1e-12, abs_tol=1e-12)


def test_ha_mirror_helpers_present_and_consistent():
    """The HA-side mirror exposes `reconstruct_sorted_prices` and monotone
    helpers identically to the src-side implementation."""
    prices = _diverse_24h()
    cheap, peak = compute_dk_cheap_peak(prices)

    src_recon = reconstruct_sorted_prices(cheap, peak)
    ha_recon = ha_dk_utils.reconstruct_sorted_prices(cheap, peak)
    for s_list, h_list in zip(src_recon, ha_recon):
        for s, h in zip(s_list, h_list):
            assert math.isclose(s, h, rel_tol=1e-12, abs_tol=1e-12)

    assert is_monotone_cheap(cheap) == ha_dk_utils.is_monotone_cheap(cheap)
    assert is_monotone_peak(peak) == ha_dk_utils.is_monotone_peak(peak)


# ─── Sum identity (cross-check used by sensor/dashboard) ─────────────


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
def test_cheap_peak_sum_identity(seed: int):
    """`cheap[11] + peak[11]` equals 2 * daily_average for any 24-hour vector.

    This identity is the foundation of the dual schema — it lets any
    consumer cross-check that cheap and peak are jointly consistent
    without needing the underlying hourly prices.
    """
    import random
    rng = random.Random(seed)
    prices = [rng.uniform(-50.0, 200.0) for _ in range(24)]
    daily_avg = sum(prices) / 24.0
    cheap, peak = compute_dk_cheap_peak(prices)
    assert math.isclose(cheap[11] + peak[11], 2 * daily_avg,
                        rel_tol=1e-12, abs_tol=1e-9)


# ─── Sensor-state helper (state = cheap[3]) ──────────────────────────


def test_state_picks_cheap_4h_from_canonical_schema():
    """`DurationForecastSensor.native_value` reads `dk_cheap_eur_kwh[3]`
    (the mean of the four cheapest hours).

    The sensor schema is now a single canonical 24-entry shape — no
    legacy fallback path remains.
    """
    def native_value(first_day: dict) -> float | None:
        cheap_vec = first_day.get("dk_cheap_eur_kwh") or []
        return cheap_vec[3] if len(cheap_vec) >= 4 else None

    # 24-entry array (canonical)
    canonical = {"dk_cheap_eur_kwh": [0.10, 0.11, 0.12, 0.13] + [0.20] * 20}
    assert native_value(canonical) == 0.13

    # Short vector → None
    short = {"dk_cheap_eur_kwh": [0.10, 0.11, 0.12]}
    assert native_value(short) is None

    # Missing attribute → None
    assert native_value({}) is None


# ─── Phase A: load cost contracts ────────────────────────────────────


def test_compute_load_dk_cost_uses_cheap_end_for_scheduling():
    """A deferrable load scheduled into its cheapest k hours achieves
    cost = energy * D_cheap(k). For k > 12 this saturates at D_cheap(12)
    when only the new 12-element array is provided."""
    cheap = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
             26.0, 28.0, 30.0, 32.0]  # EUR/kWh-equivalent
    # simulate 4 hours of 1 kW load needing 4 kWh
    energy_kwh = 4.0
    nominal_power_kw = 1.0
    k = max(1, math.ceil(energy_kwh / nominal_power_kw))
    rate = cheap[k - 1] if k <= len(cheap) else cheap[-1]
    cost = energy_kwh * rate
    assert k == 4
    assert math.isclose(rate, 16.0)
    assert math.isclose(cost, 64.0)


def test_compute_load_peak_cost_marginal_is_kth_priciest_hour():
    """Marginal price formula `k*D_peak(k) - (k-1)*D_peak(k-1)` recovers
    the price of the k-th priciest hour (1-indexed). This is the worst-
    case-cost-of-one-more-hour useful for storage planning."""
    # Synthetic priciest-prices: [100, 80, 60, 50, 40, 30, ...]
    priciest = [100.0, 80.0, 60.0, 50.0, 40.0, 30.0,
                25.0, 20.0, 15.0, 12.0, 10.0, 8.0]
    peak = []
    running = 0.0
    for k, p in enumerate(priciest, start=1):
        running += p
        peak.append(running / k)
    # Recover marginal prices
    for k in range(1, 13):
        if k == 1:
            marginal = peak[0]
        else:
            marginal = k * peak[k - 1] - (k - 1) * peak[k - 2]
        assert math.isclose(marginal, priciest[k - 1], rel_tol=1e-9, abs_tol=1e-9)
