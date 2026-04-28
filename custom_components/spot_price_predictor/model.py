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


def _pava_decreasing(values: list[float]) -> list[float]:
    """Enforce non-increasing sequence via PAVA on negated values.

    Used for the peak-end direction:  D_peak(k) is non-increasing in k
    (averaging in lower prices as more hours are included).
    """
    return [-v for v in _pava_increasing([-v for v in values])]


# ---------------------------------------------------------------------------
# Duration model: D(k) = avg price for cheapest k hours
# ---------------------------------------------------------------------------

class DurationModel:
    """Duration curve model: dual cheap/peak Ridge per (segment, k) + PAVA.

    Predicts two complementary duration curves per day:

      * `dk_cheap[k-1]` = mean spot price of the cheapest k hours, k=1..12
                          (monotone non-decreasing)
      * `dk_peak[k-1]`  = mean spot price of the priciest k hours, k=1..12
                          (monotone non-increasing)

    Each day-segment (night, morning, midday, evening) has independent
    cheap-end and peak-end Ridge models per duration level. PAVA is
    applied per direction. Full-day D(k) is reconstructed by extracting
    the underlying sorted prices from each segment and recomputing the
    cheap/peak curves on the merged 24 hourly forecasts.

    Backward compatibility: when the JSON only contains a single
    `models` key per segment (legacy schema), the cheap-end models are
    used and `dk_peak_12` is derived by sorting the merged hourly
    forecasts descending — identical to the coordinator's pre-Phase-A.3
    behaviour.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.log_offset: float = config.get("log_offset", 55)
        self.exp_cap: float = config.get("exp_cap", 20.0)
        self.feature_names: list[str] = config["feature_names"]
        self.segments: dict[str, dict] = config["segments"]
        # Phase A.3 — detect dual-model JSON (cheap_models AND peak_models
        # present per segment). If only `models` is present, we operate in
        # legacy mode: cheap_models = models, peak_models derived by sort.
        self.has_dual: bool = all(
            "cheap_models" in seg and "peak_models" in seg
            for seg in self.segments.values()
        )
        _LOGGER.info(
            "Duration model loaded: %d segments, %d features, "
            "schema=%s",
            len(self.segments), len(self.feature_names),
            "dual cheap/peak" if self.has_dual else "legacy cheap-only",
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _predict_one(
        self, sub_models: list[dict], features: dict[str, float],
    ) -> list[float]:
        """Compute raw `exp(linear) - offset` for each k, no monotonicity."""
        raw: list[float] = []
        for model in sub_models:
            linear = model["intercept"]
            for i, fname in enumerate(self.feature_names):
                linear += model["coefs"][i] * features.get(fname, 0.0)
            dk = max(0.0, math.exp(min(linear, self.exp_cap)) - self.log_offset)
            raw.append(dk)
        return raw

    def _segment_cheap_models(self, seg: dict) -> list[dict]:
        """Return the cheap-end sub-models for a segment, with legacy fallback."""
        return seg.get("cheap_models") or seg["models"]

    def _segment_peak_models(self, seg: dict) -> list[dict] | None:
        """Return the peak-end sub-models for a segment, or None if legacy."""
        return seg.get("peak_models")

    def _predict_segment(
        self, seg_name: str, features: dict[str, float],
    ) -> list[float]:
        """Legacy: predict cheap-end PAVA curve only (alias retained
        for any downstream code still calling this directly)."""
        seg = self.segments[seg_name]
        raw = self._predict_one(self._segment_cheap_models(seg), features)
        return _pava_increasing(raw)

    @staticmethod
    def _curve_to_hourly(curve: list[float], descending: bool = False
                         ) -> list[float]:
        """Inverse-cumulative: recover the underlying sorted hourly prices
        from a cumulative-mean curve.

        For an ascending cheap-end curve, returns the prices in ascending
        order. For a descending peak-end curve, returns them in descending
        order (cheapest → priciest if descending=False, priciest → cheapest
        if descending=True). The caller is responsible for re-sorting if
        a particular ordering is needed.
        """
        out: list[float] = []
        for i, dk in enumerate(curve):
            if i == 0:
                out.append(dk)
            else:
                p = (i + 1) * dk - i * curve[i - 1]
                out.append(max(0.0, p))
        return out

    # ── Public ──────────────────────────────────────────────────────

    def predict_day(
        self, segment_features: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Predict full-day duration curves (cheap + peak) from segments.

        Args:
            segment_features: {"night": {feat: val, ...}, "morning": {...}, ...}

        Returns:
            {
              "duration_curve":         [24 floats],   # legacy cumulative cheap-end
              "sorted_prices":          [24 floats],   # ascending hourly forecasts
              "segment_curves":         {seg: [...]},  # legacy cheap-end per seg
              "segment_cheap_curves":   {seg: [...]},  # explicit alias
              "segment_peak_curves":    {seg: [...]},  # only when has_dual
              "dk_cheap_12":            [12 floats],   # mean cheapest k hours
              "dk_peak_12":             [12 floats],   # mean priciest k hours
              "schema":                 "dual" | "legacy",
            }
        """
        segment_cheap_curves: dict[str, list[float]] = {}
        segment_peak_curves: dict[str, list[float]] = {}
        cheap_hourly: list[float] = []
        peak_hourly: list[float] = []

        for seg_name, seg in self.segments.items():
            if seg_name not in segment_features:
                continue
            feats = segment_features[seg_name]

            # Cheap-end Ridge → PAVA non-decreasing
            cheap_raw = self._predict_one(
                self._segment_cheap_models(seg), feats)
            cheap_curve = _pava_increasing(cheap_raw)
            segment_cheap_curves[seg_name] = cheap_curve
            cheap_hourly.extend(self._curve_to_hourly(cheap_curve))

            # Peak-end (when present): Ridge → PAVA non-increasing
            peak_models = self._segment_peak_models(seg)
            if peak_models is not None:
                peak_raw = self._predict_one(peak_models, feats)
                peak_curve = _pava_decreasing(peak_raw)
                segment_peak_curves[seg_name] = peak_curve
                peak_hourly.extend(self._curve_to_hourly(peak_curve))

        if not cheap_hourly:
            return {
                "duration_curve": [],
                "sorted_prices": [],
                "segment_curves": {},
                "segment_cheap_curves": {},
                "segment_peak_curves": {},
                "dk_cheap_12": [],
                "dk_peak_12": [],
                "schema": "dual" if self.has_dual else "legacy",
            }

        # Legacy duration_curve: ascending cumulative on the cheap-end
        # hourly reconstruction (matches pre-Phase-A.3 behaviour).
        cheap_hourly_sorted = sorted(cheap_hourly)
        running_sum = 0.0
        duration_curve: list[float] = []
        for i, price in enumerate(cheap_hourly_sorted):
            running_sum += price
            duration_curve.append(running_sum / (i + 1))

        # Phase A: dk_cheap_12, dk_peak_12.
        # Cheap-end curve always derives from the cheap-end hourly
        # reconstruction. Peak-end derives from the peak-end hourly
        # reconstruction when has_dual; otherwise from sorting the
        # cheap-end reconstruction descending (legacy fallback — what
        # the coordinator did before this refactor).
        dk_cheap_12: list[float] = []
        if len(cheap_hourly_sorted) >= 12:
            running = 0.0
            for k in range(12):
                running += cheap_hourly_sorted[k]
                dk_cheap_12.append(running / (k + 1))

        dk_peak_12: list[float] = []
        if peak_hourly:
            # Use the peak-end model's reconstruction
            peak_hourly_sorted_desc = sorted(peak_hourly, reverse=True)
            running = 0.0
            for k in range(min(12, len(peak_hourly_sorted_desc))):
                running += peak_hourly_sorted_desc[k]
                dk_peak_12.append(running / (k + 1))
        elif len(cheap_hourly_sorted) >= 12:
            # Legacy fallback: derive peak by reversing cheap-end sort
            cheap_desc = list(reversed(cheap_hourly_sorted))
            running = 0.0
            for k in range(12):
                running += cheap_desc[k]
                dk_peak_12.append(running / (k + 1))

        return {
            "duration_curve": duration_curve,
            "sorted_prices": cheap_hourly_sorted,
            "segment_curves": segment_cheap_curves,
            "segment_cheap_curves": segment_cheap_curves,
            "segment_peak_curves": segment_peak_curves,
            "dk_cheap_12": dk_cheap_12,
            "dk_peak_12": dk_peak_12,
            "schema": "dual" if self.has_dual else "legacy",
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
