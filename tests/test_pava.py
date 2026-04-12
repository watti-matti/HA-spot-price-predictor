"""Tests for Pool Adjacent Violators Algorithm (PAVA).

Verifies the pure-Python PAVA implementation in model.py matches
sklearn.isotonic.IsotonicRegression and produces valid monotone sequences.
"""
import sys
from pathlib import Path

import pytest
import numpy as np

# Add HA component to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"))
from model import _pava_increasing


class TestPavaBasic:
    """Basic PAVA correctness tests."""

    def test_already_monotone(self):
        assert _pava_increasing([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_single_element(self):
        assert _pava_increasing([5.0]) == [5.0]

    def test_empty(self):
        assert _pava_increasing([]) == []

    def test_two_elements_ordered(self):
        assert _pava_increasing([1.0, 3.0]) == [1.0, 3.0]

    def test_two_elements_reversed(self):
        result = _pava_increasing([3.0, 1.0])
        assert result == [2.0, 2.0]

    def test_all_equal(self):
        assert _pava_increasing([5.0, 5.0, 5.0]) == [5.0, 5.0, 5.0]

    def test_single_violation(self):
        """[3, 1, 2] -> pool all three to mean."""
        result = _pava_increasing([3.0, 1.0, 2.0])
        assert result == [2.0, 2.0, 2.0]

    def test_full_reversal(self):
        """[5, 3, 1] -> all pooled to mean = 3."""
        result = _pava_increasing([5.0, 3.0, 1.0])
        assert result == [3.0, 3.0, 3.0]

    def test_partial_violation(self):
        """[1, 5, 2, 4] -> [1, 3.5, 3.5, 4] (middle two pooled)."""
        result = _pava_increasing([1.0, 5.0, 2.0, 4.0])
        # 5 > 2 violation: pool to 3.5, then 3.5 < 4 OK, 1 < 3.5 OK
        assert result == pytest.approx([1.0, 3.5, 3.5, 4.0])


class TestPavaMonotonicity:
    """Property tests: output must always be non-decreasing."""

    @pytest.mark.parametrize("values", [
        [10, 5, 8, 3, 9, 1],
        [1, 1, 1, 1],
        [100, 1],
        [3, 3, 1, 1, 5, 5],
        [8, 6, 7, 5, 3, 4, 2, 1],
    ])
    def test_output_non_decreasing(self, values):
        result = _pava_increasing([float(v) for v in values])
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1], (
                f"Violation at index {i}: {result[i]} < {result[i-1]}"
            )

    @pytest.mark.parametrize("values", [
        [10, 5, 8, 3, 9, 1],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 1, 4, 1, 5, 9, 2, 6],
    ])
    def test_preserves_mean(self, values):
        """PAVA preserves the overall mean of the sequence."""
        fv = [float(v) for v in values]
        result = _pava_increasing(fv)
        assert sum(result) == pytest.approx(sum(fv), abs=1e-10)

    def test_length_preserved(self):
        values = [5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0]
        result = _pava_increasing(values)
        assert len(result) == len(values)


class TestPavaMatchesSklearn:
    """Verify pure-Python PAVA matches sklearn IsotonicRegression."""

    @pytest.fixture
    def sklearn_pava(self):
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True)
        def _fit(values):
            x = np.arange(len(values), dtype=float)
            return iso.fit_transform(x, values).tolist()
        return _fit

    @pytest.mark.parametrize("values", [
        [3, 1, 2],
        [5, 3, 1],
        [1, 2, 3],
        [10, 5, 8, 3, 9, 1],
        [3, 3, 1, 1, 5, 5],
        [1, 5, 2, 4],
        [8, 6, 7, 5, 3, 4, 2, 1],
        [4.1, 3.9, 4.2, 3.8, 4.3],
    ])
    def test_matches_sklearn(self, values, sklearn_pava):
        fv = [float(v) for v in values]
        ours = _pava_increasing(fv)
        theirs = sklearn_pava(fv)
        assert ours == pytest.approx(theirs, abs=1e-10), (
            f"Mismatch:\n  ours:   {ours}\n  sklearn: {theirs}"
        )

    def test_duration_curve_sized_input(self, sklearn_pava):
        """Test with realistic 8-element input (night segment size)."""
        values = [4.05, 4.02, 4.08, 4.01, 4.10, 4.03, 4.15, 4.20]
        ours = _pava_increasing(values)
        theirs = sklearn_pava(values)
        assert ours == pytest.approx(theirs, abs=1e-10)
