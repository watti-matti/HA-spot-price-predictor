"""Tests for `consumption_profile_loader`."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from consumption_profile_loader import (  # noqa: E402
    ConsumptionProfile,
    load_profile_from_entity_attrs,
    synthetic_profile,
)


# ── Synthetic profile ───────────────────────────────────────────────


def test_synthetic_profile_mean_kwh_matches_annual() -> None:
    p = synthetic_profile(annual_kwh=8760)
    assert p.mean_kwh_per_hour == pytest.approx(1.0)
    assert p.data_provenance == "synthetic_cold_start"


def test_synthetic_profile_shape_normalised_to_mean_one() -> None:
    p = synthetic_profile(annual_kwh=20000)
    # Shape must average to 1.0 across all 168 (weekday, hour) cells.
    assert p.shape_hour_weekday.mean() == pytest.approx(1.0)
    assert p.shape_hour_weekday.shape == (7, 24)


def test_synthetic_profile_monthly_factor_mean_one() -> None:
    p = synthetic_profile(annual_kwh=15000)
    assert p.monthly_factor.mean() == pytest.approx(1.0)
    assert p.monthly_factor.shape == (12,)


def test_synthetic_profile_winter_higher_than_summer() -> None:
    """Sanity: Finnish climate has heating-dominated winter peak."""
    p = synthetic_profile(annual_kwh=12000)
    jan_factor = p.monthly_factor[0]
    jul_factor = p.monthly_factor[6]
    assert jan_factor > 1.0 > jul_factor
    # Empirically observed for FI: peak/trough ratio ≥ 2x.
    assert jan_factor / jul_factor >= 2.0


def test_synthetic_profile_evening_higher_than_night() -> None:
    """Sanity: non-optimised household has evening peak."""
    p = synthetic_profile(annual_kwh=12000)
    # Average across weekdays at the relevant hours.
    night_kwh = p.shape_hour_weekday[:5, 3].mean()    # 03:00 weekday
    evening_kwh = p.shape_hour_weekday[:5, 18].mean()  # 18:00 weekday
    assert evening_kwh > night_kwh


def test_synthetic_profile_zero_annual_kwh() -> None:
    p = synthetic_profile(annual_kwh=0)
    assert p.mean_kwh_per_hour == 0.0
    # Shape and monthly factor still defined.
    assert p.shape_hour_weekday.shape == (7, 24)


def test_synthetic_profile_negative_annual_raises() -> None:
    with pytest.raises(ValueError):
        synthetic_profile(annual_kwh=-100)


# ── ConsumptionProfile.consumption_for_timestamps ───────────────────


def test_consumption_for_timestamps_scales_by_shape() -> None:
    p = synthetic_profile(annual_kwh=8760)
    # Single Wednesday at 18:00 in July → mean=1, weekday=Wed (idx 2),
    # hour=18 (peak), month=Jul (low factor).
    ts = [datetime(2026, 7, 15, 18, 0)]  # naive local
    out = p.consumption_for_timestamps(ts)
    assert out.shape == (1,)
    expected = (
        p.shape_hour_weekday[2, 18]
        * 1.0
        * p.monthly_factor[6]
    )
    assert out[0] == pytest.approx(expected)


def test_consumption_for_timestamps_full_year_averages_to_mean() -> None:
    """Over a full year, hourly consumption averages to mean_kwh_per_hour."""
    p = synthetic_profile(annual_kwh=8760)
    # Build all hours of 2025 (not a leap year)
    from datetime import timedelta
    start = datetime(2025, 1, 1, 0, 0)
    ts = [start + timedelta(hours=h) for h in range(8760)]
    out = p.consumption_for_timestamps(ts)
    # Average across the whole year should equal mean_kwh_per_hour
    # within numerical noise (shape mean=1, monthly factor mean=1).
    assert out.mean() == pytest.approx(p.mean_kwh_per_hour, rel=1e-2)


# ── load_profile_from_entity_attrs ─────────────────────────────────


def test_load_profile_returns_synthetic_when_attrs_none() -> None:
    p = load_profile_from_entity_attrs(None, fallback_annual_kwh=15000)
    assert p.data_provenance == "synthetic_cold_start"
    assert p.mean_kwh_per_hour == pytest.approx(15000 / 8760)


def test_load_profile_returns_synthetic_when_attrs_empty() -> None:
    p = load_profile_from_entity_attrs({}, fallback_annual_kwh=10000)
    assert p.data_provenance == "synthetic_cold_start"


def test_load_profile_uses_provided_attrs_when_valid() -> None:
    shape = [[0.5] * 24 for _ in range(7)]
    monthly = [1.0] * 12
    attrs = {
        "mean_kwh_per_hour": 2.0,
        "shape_hour_weekday": shape,
        "monthly_factor": monthly,
        "data_provenance": "ema_warm",
    }
    p = load_profile_from_entity_attrs(attrs, fallback_annual_kwh=99999)
    assert p.data_provenance == "ema_warm"
    assert p.mean_kwh_per_hour == 2.0
    assert p.shape_hour_weekday.shape == (7, 24)
    assert p.shape_hour_weekday[0, 0] == 0.5


def test_load_profile_fallback_on_malformed_shape() -> None:
    attrs = {
        "mean_kwh_per_hour": 2.0,
        "shape_hour_weekday": [[0.5] * 5 for _ in range(7)],  # wrong width
        "monthly_factor": [1.0] * 12,
    }
    p = load_profile_from_entity_attrs(attrs, fallback_annual_kwh=10000)
    assert p.data_provenance == "synthetic_cold_start"


def test_load_profile_fallback_on_malformed_monthly() -> None:
    attrs = {
        "mean_kwh_per_hour": 2.0,
        "shape_hour_weekday": [[1.0] * 24 for _ in range(7)],
        "monthly_factor": [1.0] * 8,  # too few months
    }
    p = load_profile_from_entity_attrs(attrs, fallback_annual_kwh=10000)
    assert p.data_provenance == "synthetic_cold_start"


def test_load_profile_handles_none_cells_in_shape() -> None:
    """Sparse profiles from a partially-warm EMA may have None cells."""
    shape = [[1.0] * 24 for _ in range(7)]
    shape[3][12] = None   # gap on Thursday at noon
    monthly = [None] * 12
    monthly[0] = 1.5
    monthly[6] = 0.6
    for i in range(12):
        if monthly[i] is None:
            monthly[i] = 1.0
    attrs = {
        "mean_kwh_per_hour": 1.5,
        "shape_hour_weekday": shape,
        "monthly_factor": monthly,
        "data_provenance": "ema_blended",
    }
    # Need real-list-with-None for the shape — re-inject None:
    shape[3][12] = None
    p = load_profile_from_entity_attrs(attrs, fallback_annual_kwh=10000)
    assert p.data_provenance == "ema_blended"
    # The None cell defaulted to 1.0
    assert p.shape_hour_weekday[3, 12] == 1.0


def test_load_profile_default_provenance_when_missing() -> None:
    attrs = {
        "mean_kwh_per_hour": 1.0,
        "shape_hour_weekday": [[1.0] * 24 for _ in range(7)],
        "monthly_factor": [1.0] * 12,
    }
    p = load_profile_from_entity_attrs(attrs, fallback_annual_kwh=10000)
    assert p.data_provenance == "ema_unknown"


# ── ConsumptionProfile validation ──────────────────────────────────


def test_consumption_profile_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        ConsumptionProfile(
            mean_kwh_per_hour=1.0,
            shape_hour_weekday=np.ones((7, 23)),
            monthly_factor=np.ones(12),
            data_provenance="test",
        )


def test_consumption_profile_rejects_wrong_monthly_length() -> None:
    with pytest.raises(ValueError):
        ConsumptionProfile(
            mean_kwh_per_hour=1.0,
            shape_hour_weekday=np.ones((7, 24)),
            monthly_factor=np.ones(11),
            data_provenance="test",
        )


def test_consumption_profile_rejects_negative_mean() -> None:
    with pytest.raises(ValueError):
        ConsumptionProfile(
            mean_kwh_per_hour=-0.5,
            shape_hour_weekday=np.ones((7, 24)),
            monthly_factor=np.ones(12),
            data_provenance="test",
        )
