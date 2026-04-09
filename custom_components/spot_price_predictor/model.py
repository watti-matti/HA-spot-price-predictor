"""Pure Python log-linear Ridge regression inference. No numpy/sklearn."""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_COEFS_PATH = Path(__file__).parent / "data" / "model_coefs_default.json"


class SpotPriceModel:
    """Log-linear Ridge regression model.

    Prediction: exp(sum(coef_i * feature_i) + intercept) - log_offset

    The log transform naturally handles the nonlinear price-scarcity
    relationship: nearly linear at low prices, exponential at high prices.
    """

    def __init__(self, coefs: dict[str, Any]) -> None:
        self.intercept: float = coefs["intercept"]
        self.features: list[dict] = coefs["features"]
        self.feature_names: list[str] = coefs["feature_names"]
        self.log_offset: float = coefs.get("log_offset", 55)
        self.model_type: str = coefs.get("model_type", "linear")
        self.ar_models: dict | None = coefs.get("ar_models")

    @classmethod
    async def async_load(cls, path: Path | None = None) -> "SpotPriceModel":
        """Load model from JSON coefficients file (async-safe)."""
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
            "Loaded model %s (%s) with %d features",
            coefs.get("model_version"),
            coefs.get("model_type", "linear"),
            coefs.get("feature_count"),
        )
        return cls(coefs)

    @classmethod
    def load(cls, path: Path | None = None) -> "SpotPriceModel":
        """Load model (sync fallback for non-async contexts)."""
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
        return cls(coefs)

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict spot price for a single hour.

        For log-linear model: exp(sum(coef * feature) + intercept) - offset
        For linear model (backward compat): sum(coef * feature) + intercept
        """
        linear = self.intercept
        for feat in self.features:
            linear += features.get(feat["name"], 0.0) * feat["coef"]

        if self.model_type == "log-linear":
            return math.exp(linear) - self.log_offset
        return linear

    def predict_batch(self, feature_rows: list[dict[str, float]]) -> list[float]:
        """Predict for multiple hours."""
        return [self.predict_single(row) for row in feature_rows]
