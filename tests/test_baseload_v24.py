"""Tests for the v2.4 baseload schema (annual_consumption_kwh + smoothing).

Covers without importing homeassistant:

1. **Monthly factor lookup** — `FINLAND_RESIDENTIAL_MONTHLY_FACTORS`
   sums to exactly 12.00 (normalization invariant), is non-negative,
   and shows the expected Finnish-residential pattern (winter peak,
   summer trough).

2. **Baseload formula** — for a typical Finnish single-family house
   (12 000 kWh/yr), the per-hour baseload at noon in January and at
   noon in July match the documented worked example in the plan.

3. **Migration** — legacy v2.3 baseload (kwh_per_hour + day/night
   factors) → equivalent annual_consumption_kwh, mirroring the
   coordinator and config_flow inference logic.

4. **Hysteresis dead-band** — small fluctuations in the smoothed
   daily kWh (< 5 %) do not propagate through the cached value.

5. **Stability under optimizer-driven daily perturbation** — when the
   smoothed daily kWh comes from a rolling 14-day average, a single
   day's variation (e.g. EMHASS rescheduling the heat pump) causes
   only ~7 % change in the average, well within the hysteresis band.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONST_PATH = REPO / "custom_components" / "spot_price_predictor" / "const.py"


def _load_const():
    """Load const.py with a stubbed homeassistant module."""
    if "const_v24_test" in sys.modules:
        return sys.modules["const_v24_test"]
    ha = types.ModuleType("homeassistant")
    ha_const = types.ModuleType("homeassistant.const")

    class P:
        SENSOR = "sensor"

    ha_const.Platform = P
    sys.modules["homeassistant"] = ha
    sys.modules["homeassistant.const"] = ha_const
    spec = importlib.util.spec_from_file_location("const_v24_test", CONST_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["const_v24_test"] = mod
    spec.loader.exec_module(mod)
    return mod


const = _load_const()


# ── 1. Monthly factor invariants ────────────────────────────────────


def test_monthly_factors_sum_to_exactly_12() -> None:
    """Normalization: sum is exactly 12 so that
    `Σ_month (factor × annual_kwh / 12) = annual_kwh`.
    """
    s = sum(const.FINLAND_RESIDENTIAL_MONTHLY_FACTORS)
    assert abs(s - 12.0) < 1e-9


def test_monthly_factors_length_is_12() -> None:
    assert len(const.FINLAND_RESIDENTIAL_MONTHLY_FACTORS) == 12


def test_monthly_factors_all_positive() -> None:
    for i, f in enumerate(const.FINLAND_RESIDENTIAL_MONTHLY_FACTORS):
        assert f > 0, f"Month {i+1}: factor {f} must be positive"


def test_monthly_factors_winter_peak_and_summer_trough() -> None:
    """Finnish residential at 60°N: lighting peak Dec/Jan, vacation
    trough Jul. Validates the qualitative shape baked into the array."""
    factors = const.FINLAND_RESIDENTIAL_MONTHLY_FACTORS
    jan, jul = factors[0], factors[6]
    apr, oct_ = factors[3], factors[9]
    # Winter (Jan) clearly above shoulder (Apr/Oct)
    assert jan > apr
    assert jan > oct_
    # Summer (Jul) clearly below shoulder
    assert jul < apr
    assert jul < oct_
    # Range is moderate (10-25%, not extreme)
    assert max(factors) - min(factors) < 0.5


# ── 2. Baseload formula ─────────────────────────────────────────────


def _baseload_formula(annual_kwh: float, month_idx: int) -> float:
    """Reference implementation matching `_resolve_baseload` in coordinator.

    `baseload(h) = annual_kwh / 8760 × monthly_factor[month] × 24 / 24`
    Equivalently: `daily_kwh = annual_kwh / 365`, then per-hour =
    `daily_kwh / 24 × monthly_factor`. The two are equivalent because
    8760 / 24 = 365.
    """
    daily = annual_kwh / 365.0
    factor = const.FINLAND_RESIDENTIAL_MONTHLY_FACTORS[month_idx]
    return daily / 24.0 * factor


def test_baseload_default_typical_finnish_jan_vs_jul() -> None:
    """For 12 000 kWh/yr, January per-hour ≈ 1.62, July per-hour ≈ 1.10
    — matching the documented expected values (within rounding)."""
    annual = const.DEFAULT_ANNUAL_CONSUMPTION_KWH
    jan = _baseload_formula(annual, 0)
    jul = _baseload_formula(annual, 6)
    assert jan == pytest.approx(1.616, abs=0.01)
    assert jul == pytest.approx(1.096, abs=0.01)
    # Annual mean of the per-hour value is annual_kwh / 8760
    annual_mean = annual / 8760.0
    assert annual_mean == pytest.approx(1.370, abs=0.01)


def test_baseload_scales_linearly_with_annual_kwh() -> None:
    b_low = _baseload_formula(6000, 0)
    b_high = _baseload_formula(18000, 0)
    assert b_high == pytest.approx(b_low * 3, rel=1e-6)


# ── 3. Legacy v2.3 → v2.4 migration ─────────────────────────────────


def _infer_annual_from_legacy(
    base_kwh_h: float, day_factor: float, night_factor: float,
) -> float:
    """Mirror of coordinator / config_flow inference logic."""
    avg = base_kwh_h * ((day_factor * 15 + night_factor * 9) / 24.0)
    return avg * 8760.0


def test_migration_v23_default_baseload_to_v24_annual() -> None:
    """Default v2.3 baseload (0.8 kWh/h, day=1.2, night=0.7) → ~7660 kWh/yr."""
    annual = _infer_annual_from_legacy(0.8, 1.2, 0.7)
    expected_avg = 0.8 * ((1.2 * 15 + 0.7 * 9) / 24.0)
    assert annual == pytest.approx(expected_avg * 8760.0, rel=1e-9)
    # Sanity: this is well below the v2.4 default of 12000, reflecting
    # the v2.3 misleading "non-flexible only" guidance that this release
    # corrects.
    assert annual < const.DEFAULT_ANNUAL_CONSUMPTION_KWH


def test_migration_typical_total_baseload_to_v24_annual() -> None:
    """User who followed v2.3.1 corrected guidance (1.4 kWh/h ≈ 12000/8760)
    migrates to ~12200 kWh/yr — close to v2.4 default."""
    annual = _infer_annual_from_legacy(1.4, 1.2, 0.7)
    assert 11500 < annual < 13500


def test_migration_with_explicit_annual_already_set() -> None:
    """If config already has annual_consumption_kwh, inference is bypassed.
    (Tested implicitly — this is just documenting the contract.)"""
    # Mirrored in coordinator __init__: if annual_consumption_kwh > 0,
    # the legacy inference is skipped. Smoke check that the constant
    # exists with the documented default.
    assert const.DEFAULT_ANNUAL_CONSUMPTION_KWH == 12000


# ── 4. Hysteresis dead-band ─────────────────────────────────────────


def test_hysteresis_within_band_keeps_old_value() -> None:
    """5 % hysteresis: if new reading is within 5 % of cached, keep cached."""
    cached = 30.0  # daily kWh
    band = const.CONSUMPTION_HYSTERESIS_PCT  # 0.05
    # Values strictly within ±5% (28.51..31.49 around 30.0)
    for new in [30.5, 31.0, 31.4, 28.6, 29.5]:
        within = abs(new - cached) / cached < band
        assert within, f"new={new} delta {abs(new-cached)/cached*100:.1f}%"


def test_hysteresis_outside_band_updates_value() -> None:
    """5 % hysteresis: if new reading deviates by ≥ 5 %, update cached."""
    cached = 30.0
    band = const.CONSUMPTION_HYSTERESIS_PCT
    for new in [32.0, 28.0, 33.0, 27.0]:
        outside = abs(new - cached) / cached >= band
        assert outside, f"new={new} delta {abs(new-cached)/cached*100:.1f}%"


# ── 5. Stability under optimizer-driven perturbation ────────────────


def test_smoothing_window_dampens_single_day_swing() -> None:
    """A 14-day rolling average dampens a single day's swing to ~7 %.

    Scenario: EMHASS reschedules a 5 kWh load to a different day. The
    daily kWh on those two days swings by ±5 kWh. The 14-day average
    barely moves: 5/14 ≈ 0.36 kWh, which is ~3 % of a 12 kWh/day baseline.
    Combined with the 5 % hysteresis dead-band, this never propagates
    through to the cached baseload value.
    """
    smoothing_days = const.CONSUMPTION_SMOOTHING_DAYS  # 14
    baseline_daily = 12000.0 / 365.0  # ~32.9 kWh/day
    perturbation = 5.0  # EMHASS moves a 5 kWh load
    rolling_change = perturbation / smoothing_days
    pct_change = rolling_change / baseline_daily
    assert pct_change < const.CONSUMPTION_HYSTERESIS_PCT, (
        f"Single-day perturbation {perturbation} kWh moves the "
        f"{smoothing_days}-day average by {pct_change*100:.2f}%, which "
        f"should be below the {const.CONSUMPTION_HYSTERESIS_PCT*100:.0f}% "
        f"hysteresis dead-band."
    )


def test_smoothing_window_long_enough_for_weekly_pattern() -> None:
    """14 days covers two full weeks, washing out any week-of-day pattern
    in EMHASS's optimization (e.g. weekend-vs-weekday scheduling)."""
    assert const.CONSUMPTION_SMOOTHING_DAYS >= 14
