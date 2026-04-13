"""Pure Python log-linear Ridge regression inference. No numpy/sklearn."""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_COEFS_PATH = Path(__file__).parent / "data" / "model_coefs_default.json"


# ---------------------------------------------------------------------------
# PAVA: Pool Adjacent Violators Algorithm (pure Python)
# ---------------------------------------------------------------------------

def _pava_increasing(values: list[float]) -> list[float]:
    """Enforce non-decreasing sequence via Pool Adjacent Violators.

    Returns the monotone non-decreasing sequence that minimises
    the sum of squared deviations from the input (equal weights).

    Time complexity: O(n) amortised — trivial for n <= 8.
    """
    n = len(values)
    if n <= 1:
        return list(values)

    # blocks: list of [sum, count]
    blocks: list[list[float]] = [[v, 1] for v in values]

    merged = True
    while merged:
        merged = False
        new_blocks: list[list[float]] = [blocks[0]]
        for j in range(1, len(blocks)):
            prev_avg = new_blocks[-1][0] / new_blocks[-1][1]
            curr_avg = blocks[j][0] / blocks[j][1]
            if prev_avg > curr_avg:
                # Violation — pool blocks
                new_blocks[-1][0] += blocks[j][0]
                new_blocks[-1][1] += blocks[j][1]
                merged = True
            else:
                new_blocks.append(blocks[j])
        blocks = new_blocks

    # Expand blocks to full sequence
    result: list[float] = []
    for block_sum, block_count in blocks:
        avg = block_sum / block_count
        result.extend([avg] * int(block_count))
    return result


# ---------------------------------------------------------------------------
# Duration model: D(k) = avg price for cheapest k hours
# ---------------------------------------------------------------------------

class DurationModel:
    """Duration curve model: Ridge per (segment, k) + PAVA.

    Predicts D(k) = average spot price for the cheapest k hours in a day.
    Each day-segment (night, morning, midday, evening) has independent
    Ridge models per duration level, combined via PAVA isotonic correction.
    Full-day D(k) is reconstructed by merging all segment sorted prices.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.log_offset: float = config.get("log_offset", 55)
        self.exp_cap: float = config.get("exp_cap", 20.0)
        self.feature_names: list[str] = config["feature_names"]
        self.segments: dict[str, dict] = config["segments"]
        _LOGGER.info(
            "Duration model loaded: %d segments, %d features",
            len(self.segments), len(self.feature_names),
        )

    def _predict_segment(
        self, seg_name: str, features: dict[str, float]
    ) -> list[float]:
        """Predict raw D(k) for one segment, then apply PAVA."""
        seg = self.segments[seg_name]
        raw: list[float] = []

        for model in seg["models"]:
            linear = model["intercept"]
            for i, fname in enumerate(self.feature_names):
                linear += model["coefs"][i] * features.get(fname, 0.0)
            # Back-transform from log; cap to prevent overflow
            dk = max(0.0, math.exp(min(linear, self.exp_cap)) - self.log_offset)
            raw.append(dk)

        # PAVA: enforce D(1) <= D(2) <= ... <= D(n)
        return _pava_increasing(raw)

    def predict_day(
        self, segment_features: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Predict full-day D(k) from segment feature dicts.

        Args:
            segment_features: {"night": {feat: val, ...}, "morning": {...}, ...}

        Returns:
            {"duration_curve": [N floats], "sorted_prices": [N floats],
             "segment_curves": {"night": [...], ...}}
        """
        segment_curves: dict[str, list[float]] = {}
        all_prices: list[float] = []

        for seg_name in self.segments:
            if seg_name not in segment_features:
                continue

            curve = self._predict_segment(seg_name, segment_features[seg_name])
            segment_curves[seg_name] = curve

            # Extract sorted prices: p(0)=D(0), p(k)=(k+1)*D(k)-k*D(k-1)
            for i in range(len(curve)):
                if i == 0:
                    all_prices.append(curve[0])
                else:
                    p = (i + 1) * curve[i] - i * curve[i - 1]
                    all_prices.append(max(0.0, p))

        if not all_prices:
            return {"duration_curve": [], "sorted_prices": [], "segment_curves": {}}

        # Sort all extracted prices, compute full-day D(k)
        all_prices.sort()
        running_sum = 0.0
        duration_curve: list[float] = []
        for i, price in enumerate(all_prices):
            running_sum += price
            duration_curve.append(running_sum / (i + 1))

        return {
            "duration_curve": duration_curve,
            "sorted_prices": all_prices,
            "segment_curves": segment_curves,
        }


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
        self.power_scale: float = coefs.get("power_scale", 1.0)
        self.power_exp: float = coefs.get("power_exp", 1.0)

        # Duration model (optional)
        dur_data = coefs.get("duration_model")
        self.duration_model: DurationModel | None = (
            DurationModel(dur_data) if dur_data else None
        )

    @classmethod
    def _null_model(cls) -> "SpotPriceModel":
        """Return a minimal model that predicts 0.0 for all hours.

        Used as a safe fallback when model coefficients cannot be loaded.
        The integration remains functional and can be fixed by uploading
        valid coefficients via the upload_coefficients service.
        """
        return cls({
            "intercept": 0.0,
            "features": [],
            "feature_names": [],
            "log_offset": 0.0,
            "power_scale": 0.0,
            "power_exp": 1.0,
        })

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

        try:
            coefs = await asyncio.get_event_loop().run_in_executor(None, _read)
        except FileNotFoundError:
            _LOGGER.error(
                "Model coefficients file not found: %s. "
                "Using null model (predictions will be 0). "
                "Upload coefficients via the upload_coefficients service.",
                p,
            )
            return cls._null_model()
        except (json.JSONDecodeError, OSError) as err:
            _LOGGER.error(
                "Failed to read model coefficients from %s: %s. "
                "Using null model.", p, err,
            )
            return cls._null_model()

        try:
            _LOGGER.info(
                "Loaded model %s (%s) with %d features",
                coefs.get("model_version"),
                coefs.get("model_type", "linear"),
                coefs.get("feature_count"),
            )
            return cls(coefs)
        except (KeyError, TypeError) as err:
            _LOGGER.error(
                "Invalid model coefficients structure in %s: %s. "
                "Using null model.", p, err,
            )
            return cls._null_model()

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

        try:
            with open(p, "r", encoding="utf-8") as f:
                coefs = json.load(f)
            return cls(coefs)
        except (FileNotFoundError, json.JSONDecodeError, OSError,
                KeyError, TypeError) as err:
            _LOGGER.error(
                "Failed to load model from %s: %s. Using null model.",
                p, err,
            )
            return cls._null_model()

    def predict_single(self, features: dict[str, float]) -> float:
        """Predict spot price for a single hour.

        Log-linear: scale * max(0, exp(linear) - offset) ^ power
        """
        linear = self.intercept
        for feat in self.features:
            linear += features.get(feat["name"], 0.0) * feat["coef"]

        raw = math.exp(min(linear, 20.0)) - self.log_offset
        raw = max(0.0, raw)
        if raw > 0:
            return self.power_scale * raw ** self.power_exp
        return 0.0

    def predict_batch(self, feature_rows: list[dict[str, float]]) -> list[float]:
        """Predict for multiple hours."""
        return [self.predict_single(row) for row in feature_rows]
