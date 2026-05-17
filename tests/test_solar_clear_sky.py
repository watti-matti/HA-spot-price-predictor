"""Tests for custom_components/spot_price_predictor/solar_clear_sky.py.

The clear-sky module is a pure deterministic function of (lat, lon, t).
We test:
  - solar geometry (zenith cosine at known sun-up / sun-down conditions)
  - Haurwitz GHI bounds and zero-night invariant
  - Ineichen-Perez GHI uses turbidity correctly
  - vectorised series helpers shape + dtype
  - cloudiness modulator monotonicity + bounds
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import solar_clear_sky as scs  # noqa: E402


# ── Solar geometry ─────────────────────────────────────────────────


def test_zenith_cos_is_zero_at_night() -> None:
    """At Helsinki midnight in December the sun is well below horizon."""
    ts = datetime(2024, 12, 21, 0, 0, tzinfo=timezone.utc)
    cz = scs.solar_zenith_cos(ts, lat_deg=60.17, lon_deg=24.94)
    assert cz == 0.0


def test_zenith_cos_is_positive_at_local_noon_summer() -> None:
    """At Helsinki summer-solstice solar noon (~10 UTC) the sun is high."""
    ts = datetime(2024, 6, 21, 10, 0, tzinfo=timezone.utc)
    cz = scs.solar_zenith_cos(ts, lat_deg=60.17, lon_deg=24.94)
    # Max possible cos(z) at lat=60.17 with decl=+23.45 is cos(60.17 - 23.45)
    # ≈ 0.802. Allow noon to be near this.
    assert cz == pytest.approx(0.80, abs=0.05)


def test_zenith_cos_higher_in_summer_than_winter_at_helsinki_noon() -> None:
    summer = scs.solar_zenith_cos(
        datetime(2024, 6, 21, 10, 0, tzinfo=timezone.utc), 60.17, 24.94)
    winter = scs.solar_zenith_cos(
        datetime(2024, 12, 21, 10, 0, tzinfo=timezone.utc), 60.17, 24.94)
    assert summer > winter
    assert winter < 0.2  # Helsinki winter noon: sun barely above horizon


# ── Haurwitz GHI ────────────────────────────────────────────────────


def test_haurwitz_is_zero_when_sun_below_horizon() -> None:
    assert scs.haurwitz_ghi(0.0) == 0.0
    assert scs.haurwitz_ghi(-0.5) == 0.0


def test_haurwitz_is_within_physical_bounds_at_zenith() -> None:
    """At cos(z) = 1 (sun directly overhead) GHI is ~1037 W/m² (well below
    extraterrestrial 1367 W/m² due to the atmospheric attenuation term)."""
    g = scs.haurwitz_ghi(1.0)
    assert 900.0 < g < 1100.0


def test_haurwitz_monotone_in_cos_zenith() -> None:
    """GHI should monotonically increase as the sun gets higher."""
    samples = [scs.haurwitz_ghi(c) for c in np.linspace(0.05, 1.0, 20)]
    diffs = np.diff(samples)
    assert (diffs > 0).all()


# ── Ineichen GHI ────────────────────────────────────────────────────


def test_ineichen_is_zero_at_night() -> None:
    e0 = 1367.0
    assert scs.ineichen_perez_ghi(0.0, e0, linke_turbidity=3.0) == 0.0


def test_ineichen_decreases_with_higher_turbidity() -> None:
    """More turbid atmosphere → less GHI at same sun angle."""
    e0 = 1367.0
    cz = 0.7
    clear = scs.ineichen_perez_ghi(cz, e0, linke_turbidity=2.0)
    hazy  = scs.ineichen_perez_ghi(cz, e0, linke_turbidity=5.0)
    assert hazy < clear


def test_ineichen_perez_at_helper_uses_monthly_climatology() -> None:
    """The at-helper looks up the bundled Linke climatology by month."""
    ts_jan = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    ts_jul = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    g_jan = scs.ineichen_perez_ghi_at(ts_jan, 60.17, 24.94)
    g_jul = scs.ineichen_perez_ghi_at(ts_jul, 60.17, 24.94)
    # Even though July has higher turbidity, sun is much higher → bigger GHI.
    assert g_jul > g_jan
    # And winter Helsinki noon GHI should be very small (~few tens of W/m²)
    assert g_jan < 200.0


# ── Vectorised helpers ────────────────────────────────────────────


def test_clear_sky_series_shape_and_dtype() -> None:
    ts = np.array(
        [np.datetime64("2024-06-21T10:00"),
         np.datetime64("2024-06-21T13:00"),
         np.datetime64("2024-12-21T10:00")],
        dtype="datetime64[ns]",
    )
    out = scs.clear_sky_series(ts, lat_deg=60.17, lon_deg=24.94,
                               model="haurwitz")
    assert out.shape == (3,)
    assert out.dtype == float
    assert out[0] > 100.0  # summer morning has plenty of sun
    assert out[2] < 100.0  # winter morning Helsinki has very little


def test_clear_sky_series_ineichen_runs() -> None:
    ts = np.array([np.datetime64("2024-06-21T12:00")], dtype="datetime64[ns]")
    out = scs.clear_sky_series(ts, 60.17, 24.94, model="ineichen")
    assert out.shape == (1,)
    assert out[0] > 0.0


def test_clear_sky_series_rejects_unknown_model() -> None:
    ts = np.array([np.datetime64("2024-06-21T12:00")], dtype="datetime64[ns]")
    with pytest.raises(ValueError):
        scs.clear_sky_series(ts, 60.17, 24.94, model="bogus")


# ── Cloudiness modulator ────────────────────────────────────────────


def test_modulator_kasten_czeplak_monotone_decreasing() -> None:
    c = np.linspace(0, 100, 11)
    f = scs.cloudiness_modulator(c, form="kasten_czeplak")
    assert (np.diff(f) <= 0).all()
    assert f[0] == pytest.approx(1.0, abs=1e-6)
    # At 100 % cloud, factor = 1 − 0.75 = 0.25
    assert f[-1] == pytest.approx(0.25, abs=1e-6)


def test_modulator_linear_monotone_and_bounded() -> None:
    c = np.linspace(0, 100, 11)
    f = scs.cloudiness_modulator(c, form="linear", params=(0.8,))
    assert (np.diff(f) <= 0).all()
    assert (f >= 0.0).all() and (f <= 1.0).all()


def test_modulator_affine_floor_respects_diffuse_floor() -> None:
    c = np.array([100.0])
    f = scs.cloudiness_modulator(c, form="affine_floor", params=(1.5, 0.1))
    # 1 - 1.5 = -0.5 -> clipped to 0; plus diffuse floor 0.1
    assert f[0] == pytest.approx(0.1, abs=1e-6)


def test_modulator_clips_out_of_range_input() -> None:
    f = scs.cloudiness_modulator(np.array([-5.0, 120.0]), form="kasten_czeplak")
    assert 0.0 <= f[0] <= 1.0
    assert 0.0 <= f[1] <= 1.0
