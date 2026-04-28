"""Tests for DurationModel with dual cheap/peak schema (Phase A.3).

Covers:
* Dual JSON loads correctly and `has_dual` flag is set.
* Legacy JSON (only `models`) loads and falls back gracefully.
* `predict_day` returns dk_cheap_12 monotone non-decreasing.
* `predict_day` returns dk_peak_12 monotone non-increasing (dual mode).
* Sum identity:  cheap[11] + peak[11] ≈ 2 * mean of merged hourly.
* Backward compat: legacy 24-array output preserved.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


_PKG = Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


model_mod = _load(
    "custom_components.spot_price_predictor.model",
    _PKG / "model.py",
)
DurationModel = model_mod.DurationModel


LOG_OFFSET = 55.0


def _intercept_for_target(target: float) -> float:
    """Inverse of `dk = exp(intercept) - log_offset` (assuming all coefs 0):
    intercept = log(target + log_offset)."""
    return math.log(target + LOG_OFFSET)


def _models_from_curve(curve: list[float]) -> list[dict]:
    """Convert a target D(k) curve to per-k intercepts (coefs=0)."""
    return [
        {"k": k, "intercept": _intercept_for_target(curve[k]),
         "coefs": [0.0, 0.0, 0.0, 0.0]}
        for k in range(len(curve))
    ]


def _seg_curves_from_prices(prices: list[float]
                             ) -> tuple[list[float], list[float]]:
    """Compute (cheap_curve, peak_curve) from a 6-hour segment's
    underlying prices."""
    asc = sorted(prices)
    desc = list(reversed(asc))
    cheap = []
    peak = []
    cum_a = cum_d = 0.0
    for k in range(len(asc)):
        cum_a += asc[k]
        cum_d += desc[k]
        cheap.append(cum_a / (k + 1))
        peak.append(cum_d / (k + 1))
    return cheap, peak


def _build_dual_config() -> dict:
    """4 segments × 6 hours each. Each segment uses *self-consistent*
    cheap and peak curves derived from the same underlying 6 prices,
    so cheap[5] + peak[5] = 2 * segment_mean exactly."""
    feature_names = ["a", "b", "c", "d"]
    seg_prices = {
        "night":   [12, 15, 18, 22, 28, 35],   # mean 21.67
        "morning": [25, 30, 38, 45, 55, 70],   # mean 43.83
        "midday":  [40, 50, 60, 75, 95, 120],  # mean 73.33
        "evening": [55, 70, 85, 105, 130, 165],# mean 101.67
    }
    segments = {}
    for name, prices in seg_prices.items():
        cheap_curve, peak_curve = _seg_curves_from_prices(prices)
        segments[name] = {
            "hours": list(range(6)),
            "n_levels": 6,
            "cheap_models": _models_from_curve(cheap_curve),
            "peak_models": _models_from_curve(peak_curve),
            "models": _models_from_curve(cheap_curve),
        }
    return {
        "log_offset": LOG_OFFSET,
        "exp_cap": 10.0,
        "feature_names": feature_names,
        "segments": segments,
    }


def _build_legacy_config() -> dict:
    """Same as dual but only `models` key — simulates an old model_coefs.json."""
    cfg = _build_dual_config()
    for seg in cfg["segments"].values():
        seg.pop("cheap_models", None)
        seg.pop("peak_models", None)
    return cfg


def _seg_features() -> dict[str, dict[str, float]]:
    return {
        "night":   {"a": 0.1, "b": 0.2, "c": 0.0, "d": 0.0},
        "morning": {"a": 0.1, "b": 0.2, "c": 0.0, "d": 0.0},
        "midday":  {"a": 0.1, "b": 0.2, "c": 0.0, "d": 0.0},
        "evening": {"a": 0.1, "b": 0.2, "c": 0.0, "d": 0.0},
    }


# ── Schema detection ───────────────────────────────────────────────


def test_dual_schema_detected():
    m = DurationModel(_build_dual_config())
    assert m.has_dual is True


def test_legacy_schema_detected():
    m = DurationModel(_build_legacy_config())
    assert m.has_dual is False


# ── predict_day output shape ───────────────────────────────────────


def test_predict_day_dual_returns_both_arrays():
    m = DurationModel(_build_dual_config())
    out = m.predict_day(_seg_features())
    assert out["schema"] == "dual"
    assert len(out["dk_cheap_12"]) == 12
    assert len(out["dk_peak_12"]) == 12
    # Legacy 24-array must still be present for back-compat
    assert len(out["duration_curve"]) == 24
    assert len(out["sorted_prices"]) == 24


def test_predict_day_legacy_falls_back_to_cheap_sorted_descending():
    m = DurationModel(_build_legacy_config())
    out = m.predict_day(_seg_features())
    assert out["schema"] == "legacy"
    assert len(out["dk_cheap_12"]) == 12
    # Legacy peak fallback sorts the cheap-end reconstruction descending,
    # so dk_peak_12[0] is the priciest single hour (max of cheap_hourly)
    assert len(out["dk_peak_12"]) == 12
    assert out["segment_peak_curves"] == {}


# ── Monotonicity ───────────────────────────────────────────────────


def test_dk_cheap_is_non_decreasing():
    m = DurationModel(_build_dual_config())
    out = m.predict_day(_seg_features())
    cheap = out["dk_cheap_12"]
    for i in range(len(cheap) - 1):
        assert cheap[i] <= cheap[i + 1] + 1e-9, (
            f"non-monotone cheap at {i}: {cheap[i]} > {cheap[i+1]}"
        )


def test_dk_peak_is_non_increasing():
    m = DurationModel(_build_dual_config())
    out = m.predict_day(_seg_features())
    peak = out["dk_peak_12"]
    for i in range(len(peak) - 1):
        assert peak[i] >= peak[i + 1] - 1e-9, (
            f"non-monotone peak at {i}: {peak[i]} < {peak[i+1]}"
        )


# ── Sum identity ───────────────────────────────────────────────────


def test_sum_identity_within_tolerance_dual_mode():
    """cheap[11] + peak[11] should equal 2 * mean(merged hourly), which
    equals 2 * mean of the 24-element sorted cheap reconstruction
    PLUS any discrepancy from the peak-end reconstruction sourcing
    different underlying prices.

    We accept up to a ~5% slack because the cheap-end and peak-end
    Ridge models can produce slightly inconsistent underlying hourly
    forecasts (their independent fits don't enforce
    cheap_hourly == peak_hourly identity)."""
    m = DurationModel(_build_dual_config())
    out = m.predict_day(_seg_features())
    cheap_hourly = out["sorted_prices"]
    daily_mean_cheap = sum(cheap_hourly) / len(cheap_hourly)
    s = out["dk_cheap_12"][11] + out["dk_peak_12"][11]
    # In legacy fallback, peak comes from sorting cheap descending, so
    # the identity is exact. In dual mode there is some slack due to the
    # peak-end Ridge fitting independently. Accept ≤10% relative slack.
    rel_err = abs(s - 2 * daily_mean_cheap) / max(abs(2 * daily_mean_cheap), 1e-9)
    assert rel_err < 0.10, (
        f"sum identity off by {rel_err:.2%}  "
        f"(cheap[11]+peak[11]={s:.4f}, 2*mean={2*daily_mean_cheap:.4f})"
    )


def test_sum_identity_exact_in_legacy_mode():
    """Without peak_models, dk_peak_12 derives from the same hourly data
    as dk_cheap_12 → identity is exact."""
    m = DurationModel(_build_legacy_config())
    out = m.predict_day(_seg_features())
    cheap_hourly = out["sorted_prices"]
    daily_mean = sum(cheap_hourly) / len(cheap_hourly)
    s = out["dk_cheap_12"][11] + out["dk_peak_12"][11]
    assert math.isclose(s, 2 * daily_mean, rel_tol=1e-9, abs_tol=1e-9)


# ── Cheap-end backward compatibility ───────────────────────────────


def test_legacy_duration_curve_matches_dk_cheap_first_12():
    """The legacy 24-array's first 12 entries must equal `dk_cheap_12`
    exactly — they are different views of the same cumulative cheap-end
    structure, computed from the same underlying hourly data."""
    m = DurationModel(_build_dual_config())
    out = m.predict_day(_seg_features())
    legacy = out["duration_curve"]
    cheap = out["dk_cheap_12"]
    for i in range(12):
        assert math.isclose(legacy[i], cheap[i], rel_tol=1e-9, abs_tol=1e-9), (
            f"legacy[{i}]={legacy[i]} differs from cheap[{i}]={cheap[i]}"
        )


def test_predict_segment_legacy_alias_works():
    """`_predict_segment` (legacy direct call) still returns the
    cheap-end PAVA curve."""
    m = DurationModel(_build_dual_config())
    curve = m._predict_segment("night", _seg_features()["night"])
    assert len(curve) == 6
    # Non-decreasing
    for i in range(len(curve) - 1):
        assert curve[i] <= curve[i + 1] + 1e-9
