"""Pure Python two-stage Ridge regression inference. No numpy/sklearn."""

from __future__ import annotations

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
    async def async_load(cls, path: Path | None = None) -> "SpotPriceModel":
        """Load model from JSON coefficients file (async-safe).

        Priority: explicit path > user-uploaded > bundled default.
        """
        import asyncio

        if path is not None:
            p = path
        else:
            user_path = DEFAULT_COEFS_PATH.parent / "model_coefs_user.json"
            if user_path.exists():
                p = user_path
                _LOGGER.info("Using user-uploaded coefficients: %s", p)
            else:
                p = DEFAULT_COEFS_PATH
                _LOGGER.info("Using bundled default coefficients: %s", p)

        def _read():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        coefs = await asyncio.get_event_loop().run_in_executor(None, _read)
        _LOGGER.info(
            "Loaded model %s with %d features (tiers: %s)",
            coefs.get("model_version"),
            coefs.get("feature_count"),
            coefs.get("tier_info", {}),
        )
        return cls(coefs)

    @classmethod
    def load(cls, path: Path | None = None) -> "SpotPriceModel":
        """Load model (sync fallback for non-async contexts like training)."""
        if path is not None:
            p = path
        else:
            user_path = DEFAULT_COEFS_PATH.parent / "model_coefs_user.json"
            if user_path.exists():
                p = user_path
            else:
                p = DEFAULT_COEFS_PATH

        with open(p, "r", encoding="utf-8") as f:
            coefs = json.load(f)
        _LOGGER.info(
            "Loaded model %s with %d features (tiers: %s)",
            coefs.get("model_version"),
            coefs.get("feature_count"),
            coefs.get("tier_info", {}),
        )
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
