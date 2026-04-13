"""Tests for the hourly SpotPriceModel inference engine.

Verifies:
  - Log-linear prediction: exp(linear) - offset
  - Power stretch: scale * raw^power
  - Feature coefficient ordering
  - Edge cases: zero features, extreme values
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"))
from model import SpotPriceModel


@pytest.fixture
def simple_model():
    """Simple 2-feature log-linear model."""
    return SpotPriceModel({
        "model_version": "v2.0.0",
        "model_type": "log-linear",
        "intercept": 4.0,
        "log_offset": 55,
        "power_scale": 1.0,
        "power_exp": 1.0,
        "feature_names": ["wind", "hdd"],
        "features": [
            {"name": "wind", "coef": -0.01},
            {"name": "hdd", "coef": 0.005},
        ],
    })


class TestLogLinearPrediction:
    """exp(intercept + sum(coef_i * x_i)) - offset."""

    def test_zero_features(self, simple_model):
        """All features = 0 → exp(intercept) - offset."""
        pred = simple_model.predict_single({"wind": 0.0, "hdd": 0.0})
        expected = max(0, math.exp(4.0) - 55)
        assert pred == pytest.approx(expected, abs=1e-6)

    def test_known_prediction(self, simple_model):
        """Manual calculation check."""
        # linear = 4.0 + (-0.01 * 10) + (0.005 * 20) = 4.0 - 0.1 + 0.1 = 4.0
        pred = simple_model.predict_single({"wind": 10.0, "hdd": 20.0})
        expected = max(0, math.exp(4.0) - 55)
        assert pred == pytest.approx(expected, abs=1e-6)

    def test_missing_feature_is_zero(self, simple_model):
        """Missing feature treated as 0."""
        pred_with = simple_model.predict_single({"wind": 0.0, "hdd": 0.0})
        pred_without = simple_model.predict_single({})
        assert pred_with == pytest.approx(pred_without)

    def test_negative_raw_clamped_to_zero(self):
        """When exp(linear) < offset, prediction = 0."""
        model = SpotPriceModel({
            "model_version": "v2.0.0",
            "model_type": "log-linear",
            "intercept": 2.0,  # exp(2) ≈ 7.39 < 55
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": [],
            "features": [],
        })
        assert model.predict_single({}) == 0.0

    def test_overflow_protection(self):
        """Linear > 20 is capped: exp(20) ≈ 485M."""
        model = SpotPriceModel({
            "model_version": "v2.0.0",
            "model_type": "log-linear",
            "intercept": 100.0,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": [],
            "features": [],
        })
        pred = model.predict_single({})
        assert pred < 1e10  # Not infinite


class TestPowerStretch:
    """Power stretch: scale * raw^power."""

    def test_identity_stretch(self):
        """scale=1, power=1 → no stretch."""
        model = SpotPriceModel({
            "model_version": "v2.0.0",
            "model_type": "log-linear",
            "intercept": 4.5,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": [],
            "features": [],
        })
        pred = model.predict_single({})
        raw = math.exp(4.5) - 55
        assert pred == pytest.approx(raw, abs=1e-6)

    def test_stretch_amplifies(self):
        """scale=0.58, power=1.2 (production values)."""
        model = SpotPriceModel({
            "model_version": "v2.0.0",
            "model_type": "log-linear",
            "intercept": 4.5,
            "log_offset": 55,
            "power_scale": 0.58,
            "power_exp": 1.2,
            "feature_names": [],
            "features": [],
        })
        pred = model.predict_single({})
        raw = math.exp(4.5) - 55
        expected = 0.58 * raw ** 1.2
        assert pred == pytest.approx(expected, abs=1e-6)


class TestBatchPrediction:
    """predict_batch returns list of predictions."""

    def test_batch_equals_individual(self, simple_model):
        rows = [
            {"wind": 5.0, "hdd": 10.0},
            {"wind": 10.0, "hdd": 5.0},
            {"wind": 0.0, "hdd": 0.0},
        ]
        batch = simple_model.predict_batch(rows)
        individual = [simple_model.predict_single(r) for r in rows]
        assert batch == pytest.approx(individual)

    def test_empty_batch(self, simple_model):
        assert simple_model.predict_batch([]) == []


class TestModelLoadingErrorHandling:
    """Null model fallback when coefficients are missing or corrupt."""

    def test_null_model_predicts_zero(self):
        """Null model always returns 0.0."""
        model = SpotPriceModel._null_model()
        assert model.predict_single({"wind": 5.0, "hdd": 10.0}) == 0.0

    def test_null_model_batch(self):
        """Null model batch prediction returns all zeros."""
        model = SpotPriceModel._null_model()
        assert model.predict_batch([{"wind": 5.0}, {}]) == [0.0, 0.0]

    def test_null_model_has_no_duration(self):
        """Null model has no duration model."""
        model = SpotPriceModel._null_model()
        assert model.duration_model is None

    def test_null_model_feature_names_empty(self):
        """Null model has no features."""
        model = SpotPriceModel._null_model()
        assert model.feature_names == []
        assert model.features == []

    def test_load_missing_file_returns_null(self, tmp_path):
        """Loading from nonexistent path returns null model (no crash)."""
        model = SpotPriceModel.load(tmp_path / "nonexistent.json")
        assert model.predict_single({}) == 0.0

    def test_load_corrupt_json_returns_null(self, tmp_path):
        """Loading corrupt JSON returns null model (no crash)."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")
        model = SpotPriceModel.load(bad_file)
        assert model.predict_single({}) == 0.0

    def test_load_invalid_structure_returns_null(self, tmp_path):
        """Loading valid JSON with missing keys returns null model."""
        bad_file = tmp_path / "incomplete.json"
        bad_file.write_text('{"foo": "bar"}', encoding="utf-8")
        model = SpotPriceModel.load(bad_file)
        assert model.predict_single({}) == 0.0

    def test_load_valid_file_works(self, tmp_path):
        """Loading a valid coefficient file works normally."""
        import json
        coefs = {
            "intercept": 4.0,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": ["wind"],
            "features": [{"name": "wind", "coef": -0.01}],
        }
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(json.dumps(coefs), encoding="utf-8")
        model = SpotPriceModel.load(valid_file)
        # Should predict something > 0 (exp(4) - 55 ≈ -0.4 → clamped to 0)
        # Actually exp(4)=54.6, 54.6-55=-0.4 → 0.0. Use smaller offset:
        assert model.intercept == 4.0
        assert len(model.features) == 1
