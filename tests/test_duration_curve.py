"""Tests for duration curve computation and reconstruction.

Verifies:
  - D(k) = (1/k) * sum of k cheapest prices
  - Inverse: p_(k) = (k+1)*D(k) - k*D(k-1) recovers sorted prices
  - Segment merge → re-sort → full-day D(k) roundtrip
  - Edge cases: negative prices, all-equal prices, single price
"""
import math
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"))
from model import DurationModel, _pava_increasing


def compute_duration_curve(prices: list[float]) -> list[float]:
    """Reference implementation: D(k) from sorted prices."""
    sorted_p = sorted(prices)
    running = 0.0
    curve = []
    for i, p in enumerate(sorted_p):
        running += p
        curve.append(running / (i + 1))
    return curve


def extract_sorted_prices(duration_curve: list[float]) -> list[float]:
    """Reference implementation: recover sorted prices from D(k)."""
    prices = []
    for i in range(len(duration_curve)):
        if i == 0:
            prices.append(duration_curve[0])
        else:
            p = (i + 1) * duration_curve[i] - i * duration_curve[i - 1]
            prices.append(max(0.0, p))
    return prices


class TestDurationCurveComputation:
    """D(k) = cumulative mean of ascending-sorted prices."""

    def test_simple_3_prices(self):
        prices = [30.0, 10.0, 20.0]
        dk = compute_duration_curve(prices)
        assert dk[0] == pytest.approx(10.0)   # cheapest 1
        assert dk[1] == pytest.approx(15.0)   # avg cheapest 2
        assert dk[2] == pytest.approx(20.0)   # avg all 3

    def test_already_sorted(self):
        prices = [5.0, 10.0, 15.0, 20.0]
        dk = compute_duration_curve(prices)
        assert dk[0] == pytest.approx(5.0)
        assert dk[1] == pytest.approx(7.5)
        assert dk[2] == pytest.approx(10.0)
        assert dk[3] == pytest.approx(12.5)

    def test_all_equal(self):
        prices = [10.0] * 6
        dk = compute_duration_curve(prices)
        assert all(d == pytest.approx(10.0) for d in dk)

    def test_monotonicity(self):
        """D(k) must be non-decreasing by construction."""
        prices = [50, 10, 30, 5, 45, 20, 35, 15]
        dk = compute_duration_curve([float(p) for p in prices])
        for i in range(1, len(dk)):
            assert dk[i] >= dk[i - 1]

    def test_d24_is_daily_average(self):
        """D(24) = average of all 24 prices."""
        np.random.seed(42)
        prices = np.random.uniform(5, 50, 24).tolist()
        dk = compute_duration_curve(prices)
        assert dk[23] == pytest.approx(np.mean(prices), abs=1e-10)

    def test_d1_is_minimum(self):
        """D(1) = minimum price."""
        prices = [30.0, 10.0, 20.0, 40.0, 5.0]
        dk = compute_duration_curve(prices)
        assert dk[0] == pytest.approx(5.0)

    def test_negative_prices(self):
        """Duration curves handle negative spot prices."""
        prices = [-10.0, 5.0, 20.0, -5.0]
        dk = compute_duration_curve(prices)
        assert dk[0] == pytest.approx(-10.0)
        assert dk[1] == pytest.approx(-7.5)  # avg(-10, -5)
        assert dk[3] == pytest.approx(2.5)   # avg all


class TestSortedPriceRecovery:
    """Inverse: p_(k) = (k+1)*D(k) - k*D(k-1) recovers sorted prices."""

    @pytest.mark.parametrize("prices", [
        [10.0, 20.0, 30.0],
        [5.0, 5.0, 5.0, 5.0],
        [50.0, 10.0, 30.0, 5.0, 45.0, 20.0],
        [1.0, 100.0],
    ])
    def test_roundtrip(self, prices):
        """D(k) -> extract p_(k) should recover the original sorted prices."""
        dk = compute_duration_curve(prices)
        recovered = extract_sorted_prices(dk)
        expected = sorted(prices)
        assert recovered == pytest.approx(expected, abs=1e-10)

    def test_roundtrip_24h(self):
        """Full 24-hour roundtrip."""
        np.random.seed(123)
        prices = np.random.uniform(0, 60, 24).tolist()
        dk = compute_duration_curve(prices)
        recovered = extract_sorted_prices(dk)
        expected = sorted(prices)
        assert recovered == pytest.approx(expected, abs=1e-10)

    def test_negative_price_recovery(self):
        """First extracted price equals D(0) directly (no clamping at k=0).

        The max(0, p) in extraction only applies for k >= 1 marginal prices.
        For k=0, p_(0) = D(0), which preserves negatives.
        """
        prices = [-10.0, 5.0, 20.0]
        dk = compute_duration_curve(prices)
        recovered = extract_sorted_prices(dk)
        # p_(0) = D(0) = -10.0 (the minimum price, preserved exactly)
        assert recovered[0] == pytest.approx(-10.0)
        # Higher-order marginals are clamped to 0 if negative
        # For this data they're positive, so roundtrip is exact
        assert recovered == pytest.approx(sorted(prices), abs=1e-10)


class TestSegmentReconstruction:
    """4 segments → extract sorted prices → merge → re-sort → full-day D(k)."""

    def _make_segment_curves(self, all_prices, segments):
        """Helper: compute D(k) per segment from 24 hourly prices."""
        curves = {}
        for seg_name, hours in segments.items():
            seg_prices = [all_prices[h] for h in hours]
            curves[seg_name] = compute_duration_curve(seg_prices)
        return curves

    def test_reconstruction_matches_direct(self):
        """Segment reconstruction should approximate direct D(k)."""
        np.random.seed(42)
        # 24 hourly prices (non-negative for clean roundtrip)
        prices_by_hour = np.random.uniform(5, 50, 24).tolist()

        segments = {
            "night":   [22, 23, 0, 1, 2, 3, 4, 5],
            "morning": [6, 7, 8, 9],
            "midday":  [10, 11, 12, 13, 14, 15],
            "evening": [16, 17, 18, 19, 20, 21],
        }

        # Direct full-day D(k)
        dk_direct = compute_duration_curve(prices_by_hour)

        # Segment reconstruction
        seg_curves = self._make_segment_curves(prices_by_hour, segments)
        all_extracted = []
        for seg_name, curve in seg_curves.items():
            extracted = extract_sorted_prices(curve)
            all_extracted.extend(extracted)

        all_extracted.sort()
        running = 0.0
        dk_reconstructed = []
        for i, p in enumerate(all_extracted):
            running += p
            dk_reconstructed.append(running / (i + 1))

        # Should match exactly when prices are non-negative
        assert dk_reconstructed == pytest.approx(dk_direct, abs=1e-10)

    def test_reconstruction_24_prices(self):
        """Reconstruction produces exactly 24 price levels."""
        segments = {
            "night":   [22, 23, 0, 1, 2, 3, 4, 5],    # 8
            "morning": [6, 7, 8, 9],                     # 4
            "midday":  [10, 11, 12, 13, 14, 15],         # 6
            "evening": [16, 17, 18, 19, 20, 21],         # 6
        }
        prices = list(range(1, 25))  # [1, 2, ..., 24]
        seg_curves = self._make_segment_curves(prices, segments)

        total_prices = 0
        for curve in seg_curves.values():
            extracted = extract_sorted_prices(curve)
            total_prices += len(extracted)

        assert total_prices == 24


class TestLogTransformRoundtrip:
    """Log transform: y = log(D(k) + offset) <-> D(k) = exp(y) - offset."""

    LOG_OFFSET = 55

    @pytest.mark.parametrize("dk", [0.0, 5.0, 20.0, 50.0, 100.0])
    def test_forward_backward(self, dk):
        y = math.log(dk + self.LOG_OFFSET)
        recovered = math.exp(y) - self.LOG_OFFSET
        assert recovered == pytest.approx(dk, abs=1e-10)

    def test_negative_price_stays_positive_in_log(self):
        """log(D(k) + 55) is positive even for D(k) = -50."""
        dk = -50.0
        y = math.log(dk + self.LOG_OFFSET)
        assert y > 0  # log(5) = 1.61

    def test_offset_55_covers_finnish_range(self):
        """Finnish prices rarely go below -50 EUR/MWh.
        Offset 55 keeps log argument > 0 for prices >= -54.99."""
        min_price = -54.99
        assert min_price + self.LOG_OFFSET > 0
