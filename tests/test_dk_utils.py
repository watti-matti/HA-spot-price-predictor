"""Tests for src/dk_utils.py — cheap/peak D(k) computation."""
from __future__ import annotations

import math

import pytest

from src.dk_utils import (
    compute_dk_cheap_peak,
    is_monotone_cheap,
    is_monotone_peak,
    reconstruct_sorted_prices,
)


def test_uniform_prices_yield_constant_dk():
    """When all 24 hours have the same price, both dk_cheap and dk_peak are flat."""
    prices = [50.0] * 24
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
    assert len(dk_cheap) == 12
    assert len(dk_peak) == 12
    assert all(abs(v - 50.0) < 1e-9 for v in dk_cheap)
    assert all(abs(v - 50.0) < 1e-9 for v in dk_peak)


def test_known_arithmetic_sequence():
    """Prices 1..24 produce known cumulative averages."""
    prices = list(range(1, 25))  # [1, 2, ..., 24]
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices)

    # cheap[0] = 1, cheap[1] = (1+2)/2 = 1.5, ..., cheap[11] = sum(1..12)/12 = 78/12 = 6.5
    expected_cheap = [sum(range(1, k + 1)) / k for k in range(1, 13)]
    for got, want in zip(dk_cheap, expected_cheap):
        assert math.isclose(got, want)

    # peak[0] = 24, peak[1] = (24+23)/2 = 23.5, ..., peak[11] = sum(13..24)/12 = 222/12 = 18.5
    expected_peak = [sum(range(24 - k + 1, 25)) / k for k in range(1, 13)]
    for got, want in zip(dk_peak, expected_peak):
        assert math.isclose(got, want)


def test_monotonicity_invariants():
    """dk_cheap must be non-decreasing; dk_peak must be non-increasing."""
    import random
    rng = random.Random(42)
    for _ in range(100):
        prices = [rng.uniform(-50.0, 200.0) for _ in range(24)]
        dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
        assert is_monotone_cheap(dk_cheap)
        assert is_monotone_peak(dk_peak)


def test_cheap_le_peak():
    """For any single index k, dk_cheap[k] <= dk_peak[k] (cheapest k <= priciest k)."""
    import random
    rng = random.Random(123)
    for _ in range(50):
        prices = [rng.uniform(0.0, 100.0) for _ in range(24)]
        dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
        for k in range(12):
            assert dk_cheap[k] <= dk_peak[k] + 1e-9, (
                f"dk_cheap[{k}]={dk_cheap[k]} > dk_peak[{k}]={dk_peak[k]}"
            )


def test_cheap_12_plus_peak_12_equals_daily_average_times_2():
    """Sum of cheapest 12 + priciest 12 = sum of all 24, so means satisfy:
    dk_cheap[11] + dk_peak[11] = 2 * daily_average
    """
    import random
    rng = random.Random(7)
    for _ in range(50):
        prices = [rng.uniform(0.0, 100.0) for _ in range(24)]
        dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
        daily_avg = sum(prices) / 24
        assert math.isclose(dk_cheap[11] + dk_peak[11], 2 * daily_avg, abs_tol=1e-9)


def test_invalid_input_lengths():
    with pytest.raises(ValueError, match="exactly 24"):
        compute_dk_cheap_peak([1.0] * 23)
    with pytest.raises(ValueError, match="exactly 24"):
        compute_dk_cheap_peak([1.0] * 25)
    with pytest.raises(ValueError, match="exactly 24"):
        compute_dk_cheap_peak([])


def test_half_horizon_bounds():
    prices = list(range(1, 25))
    # Lower half_horizon works
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices, half_horizon=6)
    assert len(dk_cheap) == 6 == len(dk_peak)
    # Out-of-range
    with pytest.raises(ValueError):
        compute_dk_cheap_peak(prices, half_horizon=0)
    with pytest.raises(ValueError):
        compute_dk_cheap_peak(prices, half_horizon=13)


def test_reconstruct_sorted_roundtrip():
    """compute_dk_cheap_peak then reconstruct should recover sorted price vector."""
    import random
    rng = random.Random(11)
    prices = [rng.uniform(10.0, 80.0) for _ in range(24)]
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
    cheap_sorted, peak_sorted = reconstruct_sorted_prices(dk_cheap, dk_peak)
    expected_asc = sorted(prices)
    expected_desc = sorted(prices, reverse=True)
    for i in range(12):
        assert math.isclose(cheap_sorted[i], expected_asc[i], abs_tol=1e-9)
        assert math.isclose(peak_sorted[i], expected_desc[i], abs_tol=1e-9)


def test_negative_prices_handled():
    """Nordpool prices can be negative during high-renewables periods."""
    prices = [-10.0] * 6 + [20.0] * 12 + [80.0] * 6
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices)
    # dk_cheap[0..5] should all be -10
    for k in range(6):
        assert dk_cheap[k] == -10.0
    # dk_peak[0..5] should all be 80
    for k in range(6):
        assert dk_peak[k] == 80.0
    # Monotone invariants
    assert is_monotone_cheap(dk_cheap)
    assert is_monotone_peak(dk_peak)


def test_realistic_double_peak_day():
    """Typical Nordic day: cheap night, morning peak, midday valley, evening peak."""
    prices = [
        # 0-5 night (cheap)
        15.0, 12.0, 10.0, 11.0, 12.0, 18.0,
        # 6-9 morning ramp (peak)
        45.0, 75.0, 95.0, 85.0,
        # 10-15 midday (medium)
        55.0, 45.0, 40.0, 38.0, 42.0, 50.0,
        # 16-21 evening (peak)
        65.0, 80.0, 110.0, 105.0, 75.0, 50.0,
        # 22-23 late evening
        30.0, 22.0,
    ]
    dk_cheap, dk_peak = compute_dk_cheap_peak(prices)

    # Cheapest single hour should be hour 2: 10 EUR/MWh
    assert dk_cheap[0] == 10.0
    # Single most expensive hour should be hour 18: 110 EUR/MWh
    assert dk_peak[0] == 110.0

    # Cheapest 4h average — hours [2, 3, 1, 4] sorted = [10, 11, 12, 12]
    expected_cheap_4 = (10 + 11 + 12 + 12) / 4
    assert math.isclose(dk_cheap[3], expected_cheap_4)

    # Most expensive 4h average — top 4 sorted desc = [110, 105, 95, 85]
    expected_peak_4 = (110 + 105 + 95 + 85) / 4
    assert math.isclose(dk_peak[3], expected_peak_4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
