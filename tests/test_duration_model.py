"""Tests for the DurationModel inference engine.

Verifies:
  - Model loads from JSON correctly
  - predict_day() produces valid output structure
  - Segment predictions use correct coefficients
  - Full-day reconstruction from segments is monotone
  - Consumer price conversion
  - Backward compatibility (model absent)
"""
import json
import math
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"))
from model import SpotPriceModel, DurationModel, _pava_increasing


@pytest.fixture
def minimal_duration_config():
    """Minimal valid duration model config for testing."""
    return {
        "model_version": "v1.0",
        "log_offset": 55,
        "lambda": 0.990,
        "ridge_alpha": 1.0,
        "feature_names": ["wind_mean", "solar_mean", "hdd_mean"],
        "segments": {
            "night": {
                "hours": [22, 23, 0, 1, 2, 3, 4, 5],
                "n_levels": 3,   # reduced for testing
                "models": [
                    {"k": 0, "intercept": 4.0, "coefs": [-0.01, 0.0, 0.005]},
                    {"k": 1, "intercept": 4.05, "coefs": [-0.01, 0.0, 0.005]},
                    {"k": 2, "intercept": 4.1, "coefs": [-0.01, 0.0, 0.005]},
                ],
            },
            "morning": {
                "hours": [6, 7, 8, 9],
                "n_levels": 2,
                "models": [
                    {"k": 0, "intercept": 4.1, "coefs": [-0.01, 0.0, 0.005]},
                    {"k": 1, "intercept": 4.15, "coefs": [-0.01, 0.0, 0.005]},
                ],
            },
            "midday": {
                "hours": [10, 11, 12, 13, 14, 15],
                "n_levels": 2,
                "models": [
                    {"k": 0, "intercept": 4.0, "coefs": [-0.008, -0.0001, 0.004]},
                    {"k": 1, "intercept": 4.08, "coefs": [-0.008, -0.0001, 0.004]},
                ],
            },
            "evening": {
                "hours": [16, 17, 18, 19, 20, 21],
                "n_levels": 2,
                "models": [
                    {"k": 0, "intercept": 4.2, "coefs": [-0.012, 0.0, 0.006]},
                    {"k": 1, "intercept": 4.25, "coefs": [-0.012, 0.0, 0.006]},
                ],
            },
        },
    }


@pytest.fixture
def typical_features():
    """Typical segment features for testing."""
    base = {"wind_mean": 5.0, "solar_mean": 50.0, "hdd_mean": 10.0}
    return {
        "night": dict(base),
        "morning": dict(base),
        "midday": dict(base),
        "evening": dict(base),
    }


class TestDurationModelInit:
    """Model loading and initialization."""

    def test_loads_from_config(self, minimal_duration_config):
        model = DurationModel(minimal_duration_config)
        assert model.log_offset == 55
        assert len(model.feature_names) == 3
        assert len(model.segments) == 4

    def test_segment_level_counts(self, minimal_duration_config):
        model = DurationModel(minimal_duration_config)
        assert model.segments["night"]["n_levels"] == 3
        assert model.segments["morning"]["n_levels"] == 2


class TestDurationModelPredict:
    """Prediction produces valid output."""

    def test_output_structure(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)

        assert "duration_curve" in result
        assert "sorted_prices" in result
        assert "segment_curves" in result
        assert isinstance(result["duration_curve"], list)
        assert isinstance(result["sorted_prices"], list)

    def test_output_length(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)

        # 3 + 2 + 2 + 2 = 9 total prices
        assert len(result["sorted_prices"]) == 9
        assert len(result["duration_curve"]) == 9

    def test_duration_curve_monotone(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)
        dk = result["duration_curve"]

        for i in range(1, len(dk)):
            assert dk[i] >= dk[i - 1] - 1e-10, (
                f"D({i+1}) = {dk[i]:.4f} < D({i}) = {dk[i-1]:.4f}"
            )

    def test_sorted_prices_ascending(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)
        sp = result["sorted_prices"]

        for i in range(1, len(sp)):
            assert sp[i] >= sp[i - 1] - 1e-10

    def test_all_prices_non_negative(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)

        for p in result["sorted_prices"]:
            assert p >= 0.0

    def test_segment_curves_present(self, minimal_duration_config, typical_features):
        model = DurationModel(minimal_duration_config)
        result = model.predict_day(typical_features)

        for seg in ["night", "morning", "midday", "evening"]:
            assert seg in result["segment_curves"]

    def test_prediction_responds_to_wind(self, minimal_duration_config):
        """Higher wind should give lower prices (negative wind coefficient)."""
        model = DurationModel(minimal_duration_config)

        calm = {seg: {"wind_mean": 2.0, "solar_mean": 0.0, "hdd_mean": 10.0}
                for seg in ["night", "morning", "midday", "evening"]}
        windy = {seg: {"wind_mean": 15.0, "solar_mean": 0.0, "hdd_mean": 10.0}
                 for seg in ["night", "morning", "midday", "evening"]}

        r_calm = model.predict_day(calm)
        r_windy = model.predict_day(windy)

        # Wind coefficient is negative => windy should be cheaper
        assert r_windy["duration_curve"][-1] < r_calm["duration_curve"][-1]

    def test_missing_segment_graceful(self, minimal_duration_config):
        """Missing segment features don't crash, just produce fewer prices."""
        model = DurationModel(minimal_duration_config)
        partial = {
            "night": {"wind_mean": 5.0, "solar_mean": 0.0, "hdd_mean": 10.0},
            "morning": {"wind_mean": 5.0, "solar_mean": 0.0, "hdd_mean": 10.0},
            # midday and evening missing
        }
        result = model.predict_day(partial)
        # 3 + 2 = 5 prices from night + morning only
        assert len(result["sorted_prices"]) == 5


class TestLinearPrediction:
    """Verify linear combination: intercept + sum(coef_i * feature_i)."""

    def test_known_linear(self, minimal_duration_config):
        model = DurationModel(minimal_duration_config)
        # Night k=0: intercept=4.0, coefs=[-0.01, 0.0, 0.005]
        # wind=5: -0.01*5 = -0.05
        # solar=0: 0*0 = 0
        # hdd=10: 0.005*10 = 0.05
        # linear = 4.0 + (-0.05) + 0 + 0.05 = 4.0
        # D(k) = exp(4.0) - 55 = 54.598 - 55 = -0.402 -> max(0, -0.402) = 0

        features = {"wind_mean": 5.0, "solar_mean": 0.0, "hdd_mean": 10.0}
        curve = model._predict_segment("night", features)

        expected_linear = 4.0 + (-0.01 * 5.0) + (0.0 * 0.0) + (0.005 * 10.0)
        expected_dk = max(0.0, math.exp(expected_linear) - 55)
        assert curve[0] == pytest.approx(expected_dk, abs=1e-6)

    def test_missing_feature_defaults_zero(self, minimal_duration_config):
        """Features not in dict default to 0.0."""
        model = DurationModel(minimal_duration_config)
        features = {"wind_mean": 5.0}  # solar_mean, hdd_mean missing
        curve = model._predict_segment("night", features)

        expected_linear = 4.0 + (-0.01 * 5.0) + (0.0 * 0.0) + (0.005 * 0.0)
        expected_dk = max(0.0, math.exp(expected_linear) - 55)
        assert curve[0] == pytest.approx(expected_dk, abs=1e-6)

    def test_exp_overflow_capped(self, minimal_duration_config):
        """Linear values > 20 are capped to prevent overflow."""
        config = minimal_duration_config
        # Set extreme intercept
        config["segments"]["night"]["models"][0]["intercept"] = 25.0
        model = DurationModel(config)

        features = {"wind_mean": 0.0, "solar_mean": 0.0, "hdd_mean": 0.0}
        curve = model._predict_segment("night", features)
        # exp(20) - 55 ≈ 485 million - 55 ≈ 485 million
        assert curve[0] < 1e9  # not infinite


class TestSpotPriceModelIntegration:
    """SpotPriceModel with optional duration model."""

    def test_loads_without_duration(self):
        """Backward compatibility: no duration_model key."""
        coefs = {
            "model_version": "v6.1",
            "model_type": "log-linear",
            "intercept": 4.0,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": ["wind_speed_weighted"],
            "features": [{"name": "wind_speed_weighted", "coef": -0.003}],
        }
        model = SpotPriceModel(coefs)
        assert model.duration_model is None

    def test_loads_with_duration(self, minimal_duration_config):
        """Duration model loads when present."""
        coefs = {
            "model_version": "v6.1",
            "model_type": "log-linear",
            "intercept": 4.0,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": ["wind_speed_weighted"],
            "features": [{"name": "wind_speed_weighted", "coef": -0.003}],
            "duration_model": minimal_duration_config,
        }
        model = SpotPriceModel(coefs)
        assert model.duration_model is not None
        assert isinstance(model.duration_model, DurationModel)

    def test_hourly_prediction_unaffected(self, minimal_duration_config):
        """Adding duration model doesn't change hourly predictions."""
        base_coefs = {
            "model_version": "v6.1",
            "model_type": "log-linear",
            "intercept": 4.0,
            "log_offset": 55,
            "power_scale": 1.0,
            "power_exp": 1.0,
            "feature_names": ["wind_speed_weighted"],
            "features": [{"name": "wind_speed_weighted", "coef": -0.003}],
        }
        model_no_dur = SpotPriceModel(dict(base_coefs))

        coefs_with_dur = dict(base_coefs)
        coefs_with_dur["duration_model"] = minimal_duration_config
        model_with_dur = SpotPriceModel(coefs_with_dur)

        feat = {"wind_speed_weighted": 8.0}
        assert model_no_dur.predict_single(feat) == model_with_dur.predict_single(feat)


class TestProductionCoefficients:
    """Test with actual production model_coefs.json if available."""

    @pytest.fixture
    def prod_model(self):
        coefs_path = Path(__file__).parent.parent / "output" / "model_coefs.json"
        if not coefs_path.exists():
            pytest.skip("Production model_coefs.json not available")
        with open(coefs_path) as f:
            coefs = json.load(f)
        if "duration_model" not in coefs:
            pytest.skip("Duration model not in production coefficients")
        return SpotPriceModel(coefs)

    def test_production_model_loads(self, prod_model):
        assert prod_model.duration_model is not None

    def test_production_10_features(self, prod_model):
        assert len(prod_model.duration_model.feature_names) == 10

    def test_production_4_segments(self, prod_model):
        assert len(prod_model.duration_model.segments) == 4

    def test_production_segment_sizes(self, prod_model):
        dm = prod_model.duration_model
        # Tariff-aligned: night 22-06 (9h), morning 07-11 (5h),
        # midday 12-17 (6h), evening 18-21 (4h)
        assert dm.segments["night"]["n_levels"] == 9
        assert dm.segments["morning"]["n_levels"] == 5
        assert dm.segments["midday"]["n_levels"] == 6
        assert dm.segments["evening"]["n_levels"] == 4

    def test_production_realistic_output(self, prod_model):
        """Production model gives prices in 0-100 EUR/MWh range."""
        features = {seg: {
            "wind_mean": 6.0, "solar_mean": 30.0, "hdd_mean": 8.0,
            "se3_mean": 40.0, "se1_mean": 35.0, "nuclear_deficit": 0.1,
            "is_workday": 1.0, "month_sin": 0.5, "month_cos": 0.866,
            "wind_log_scarcity": 0.9,
        } for seg in ["night", "morning", "midday", "evening"]}

        result = prod_model.duration_model.predict_day(features)
        dk = result["duration_curve"]

        # Spot prices should be in reasonable range for Finland
        assert 0 <= dk[0] <= 100, f"D(1) = {dk[0]} out of range"
        assert 0 <= dk[-1] <= 200, f"D(24) = {dk[-1]} out of range"
        assert len(dk) == 24


class TestConsumerPrice:
    """Consumer price conversion: (max(0, spot)/1000 + transfer + tax) * VAT * 100."""

    TRANSFER = 0.0361   # EUR/kWh
    ENERGY_TAX = 0.02325  # EUR/kWh
    VAT = 1.255           # 25.5%

    def to_cons(self, spot_eur_mwh: float) -> float:
        return (max(0.0, spot_eur_mwh) / 1000 + self.TRANSFER + self.ENERGY_TAX) * self.VAT * 100

    def test_zero_spot(self):
        """Zero spot price → transfer + tax only."""
        c = self.to_cons(0.0)
        expected = (0.0 + 0.0361 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected)
        assert c == pytest.approx(7.448, abs=0.01)

    def test_50_eur_mwh(self):
        """50 EUR/MWh spot price."""
        c = self.to_cons(50.0)
        expected = (0.05 + 0.0361 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected)
        assert c == pytest.approx(13.723, abs=0.01)

    def test_negative_spot_clamped(self):
        """Negative spot price → treated as 0 for consumer."""
        c = self.to_cons(-20.0)
        assert c == pytest.approx(self.to_cons(0.0))
