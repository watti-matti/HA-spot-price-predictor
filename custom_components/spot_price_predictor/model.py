"""Pure Python two-stage Ridge regression inference. No numpy/sklearn."""

import json
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_COEFS_PATH = Path(__file__).parent / "data" / "model_coefs_default.json"


class SpotPriceModel:
    """Two-stage piecewise-linear Ridge regression model."""

    def __init__(self, coefs: dict[str, Any]) -> None:
        self.stage1_intercept: float = coefs["stage1"]["intercept"]
        self.stage1_features: list[dict] = coefs["stage1"]["features"]
        self.stage2_intercept: float = coefs["intercept"]
        self.stage2_features: list[dict] = coefs["features"]
        self.breakpoints: list[float] = coefs["piecewise_breakpoints"]
        self.feature_names: list[str] = coefs["feature_names"]

    @classmethod
    def load(cls, path: Path | None = None) -> "SpotPriceModel":
        """Load model from JSON coefficients file."""
        p = path or DEFAULT_COEFS_PATH
        with open(p, "r", encoding="utf-8") as f:
            coefs = json.load(f)
        _LOGGER.info("Loaded model %s with %d features", coefs.get("model_version"), coefs.get("feature_count"))
        return cls(coefs)

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict for a single hour given a feature dict.

        Steps:
        1. stage1_pred = sum(value * coef) + stage1_intercept
        2. Augment with stage1_pred and piecewise ReLU terms
        3. final = sum(augmented * coef) + stage2_intercept
        """
        # Stage 1
        stage1_pred = self.stage1_intercept
        for feat in self.stage1_features:
            val = features.get(feat["name"], 0.0)
            stage1_pred += val * feat["coef"]

        # Augmented features
        augmented = dict(features)
        augmented["stage1_pred"] = stage1_pred
        for bp in self.breakpoints:
            augmented[f"pw_relu_{bp}"] = max(0.0, stage1_pred - bp)

        # Stage 2
        final = self.stage2_intercept
        for feat in self.stage2_features:
            val = augmented.get(feat["name"], 0.0)
            final += val * feat["coef"]

        return final

    def predict_batch(self, feature_rows: list[dict[str, float]]) -> list[float]:
        """Predict for multiple hours."""
        return [self.predict_single(row) for row in feature_rows]
