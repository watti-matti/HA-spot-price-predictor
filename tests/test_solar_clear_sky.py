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


# ── Artifact + inference helpers ────────────────────────────────────


def _toy_artifact() -> dict:
    return scs.build_artifact(
        clear_sky_model="haurwitz",
        modulator_form="kasten_czeplak",
        alpha=10.0,
        gain=0.001,         # MW per (W/m²) of capacity-weighted GHI
        capacity_ref_mw=1500.0,
        sites=[
            {"name": "Helsinki", "lat": 60.17, "lon": 24.94,
             "solar_weight": 0.5},
            {"name": "Lapland",  "lat": 67.30, "lon": 23.80,
             "solar_weight": 0.5},
        ],
    )


def test_build_artifact_collapses_capacity_into_K() -> None:
    """The runtime contract: K = gain · capacity_ref so inference does
    not need to know the original capacity series."""
    art = _toy_artifact()
    assert art["K"] == pytest.approx(0.001 * 1500.0)
    assert art["version"] == "2.5.3"
    assert art["clear_sky_model"] == "haurwitz"
    assert art["modulator_form"] == "kasten_czeplak"
    assert len(art["sites"]) == 2


def test_build_artifact_drops_zero_weight_sites() -> None:
    art = scs.build_artifact(
        clear_sky_model="haurwitz", modulator_form="kasten_czeplak",
        alpha=0.0, gain=1.0, capacity_ref_mw=100.0,
        sites=[
            {"name": "K", "lat": 60.0, "lon": 24.0, "solar_weight": 1.0},
            {"name": "Z", "lat": 60.0, "lon": 24.0, "solar_weight": 0.0},
        ],
    )
    assert [s["name"] for s in art["sites"]] == ["K"]


def test_predict_solar_mw_is_zero_at_night() -> None:
    """No matter the cloud cover, predicted production at midnight is just
    the intercept (clamped at 0)."""
    art = _toy_artifact()
    ts = np.array(
        [np.datetime64("2024-12-21T00:00"),
         np.datetime64("2024-12-21T01:00")],
        dtype="datetime64[ns]",
    )
    pred = scs.predict_solar_mw(ts, cloud_cover_pct=np.array([10.0, 80.0]),
                                artifact=art)
    # Intercept is +10 (positive); both predictions sit at the intercept
    # because the sun is below the horizon at both Helsinki and Lapland.
    assert pred[0] == pytest.approx(10.0, abs=1e-6)
    assert pred[1] == pytest.approx(10.0, abs=1e-6)


def test_predict_solar_mw_clips_negative_intercept() -> None:
    art = _toy_artifact()
    art["alpha"] = -50.0   # synthetic negative intercept
    ts = np.array([np.datetime64("2024-12-21T00:00")], dtype="datetime64[ns]")
    pred = scs.predict_solar_mw(ts, np.array([0.0]), art)
    assert pred[0] == 0.0   # negative production is clipped


def test_predict_solar_mw_higher_with_clear_sky_than_full_cloud() -> None:
    """Same daytime hour, varying cloud cover: clear weather ⇒ more
    predicted production than overcast."""
    art = _toy_artifact()
    ts = np.array([np.datetime64("2024-06-21T10:00")],
                  dtype="datetime64[ns]")
    clear = scs.predict_solar_mw(ts, np.array([0.0]), art)[0]
    overcast = scs.predict_solar_mw(ts, np.array([100.0]), art)[0]
    assert clear > overcast


def test_predict_solar_mw_uses_artifact_sites_when_sites_arg_omitted() -> None:
    art = _toy_artifact()
    ts = np.array([np.datetime64("2024-06-21T10:00")],
                  dtype="datetime64[ns]")
    cloud = np.array([20.0])
    p_default = scs.predict_solar_mw(ts, cloud, art)
    p_override = scs.predict_solar_mw(ts, cloud, art, sites=art["sites"])
    assert p_default[0] == pytest.approx(p_override[0], abs=1e-9)


def test_artifact_roundtrips_through_json() -> None:
    """Serialising and reloading the artifact preserves inference output."""
    import json
    art = _toy_artifact()
    s = json.dumps(art)
    art2 = json.loads(s)
    ts = np.array([np.datetime64("2024-06-21T10:00")],
                  dtype="datetime64[ns]")
    p1 = scs.predict_solar_mw(ts, np.array([50.0]), art)
    p2 = scs.predict_solar_mw(ts, np.array([50.0]), art2)
    assert p1[0] == pytest.approx(p2[0], abs=1e-9)
