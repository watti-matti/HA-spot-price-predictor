"""Tests for studies/se3_model_v242.py — the v2.4.2 SE3 model script.

These tests exercise the pure-function helpers without hitting Statnett
(network-isolation) so they're CI-friendly. The fetch path is tested
implicitly via the v2.4.1 statnett_client tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_module():
    if "se3_model_v242_test" in sys.modules:
        return sys.modules["se3_model_v242_test"]
    # Ensure studies/ is on sys.path so npk_cvar_hedge can be imported by the script
    sys.path.insert(0, str(REPO / "studies"))
    path = REPO / "studies" / "se3_model_v242.py"
    spec = importlib.util.spec_from_file_location("se3_model_v242_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["se3_model_v242_test"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


# ── build_hydro_offset ──────────────────────────────────────────────


def test_build_hydro_offset_zero_for_single_year() -> None:
    """With only one year of data per week, each week's offset is 0
    (because the baseline = the only observation)."""
    weeks = [
        {"year": 2025, "week": 1, "total_pct": 45.0},
        {"year": 2025, "week": 2, "total_pct": 50.0},
        {"year": 2025, "week": 3, "total_pct": 55.0},
    ]
    offset = m.build_hydro_offset(weeks)
    for k, v in offset.items():
        assert v == 0.0, f"week {k} should have zero offset, got {v}"


def test_build_hydro_offset_two_years_recovers_anomaly() -> None:
    """Same week-of-year, different years: offset = observation − cross-year mean."""
    weeks = [
        {"year": 2024, "week": 10, "total_pct": 60.0},
        {"year": 2025, "week": 10, "total_pct": 40.0},  # 20% below 2024
    ]
    offset = m.build_hydro_offset(weeks)
    baseline = (60.0 + 40.0) / 2  # 50
    assert offset[(2024, 10)] == pytest.approx(60.0 - baseline)
    assert offset[(2025, 10)] == pytest.approx(40.0 - baseline)
    assert offset[(2024, 10)] + offset[(2025, 10)] == pytest.approx(0)


def test_build_hydro_offset_preserves_keys() -> None:
    weeks = [
        {"year": 2024, "week": 5, "total_pct": 70.0},
        {"year": 2025, "week": 5, "total_pct": 30.0},
    ]
    offset = m.build_hydro_offset(weeks)
    assert set(offset.keys()) == {(2024, 5), (2025, 5)}


# ── fit_se3_v242 ────────────────────────────────────────────────────


def test_fit_se3_v242_returns_expected_keys() -> None:
    """Synthetic SE3 series — verify the model fits and exposes expected fields."""
    rng = np.random.default_rng(0)
    n = 24 * 90  # 90 days hourly
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    # Synthetic SE3: small sinusoid + AR(1) residual
    h = ts.hour.to_numpy()
    seasonal = 5.0 * np.sin(2 * np.pi * h / 24)
    Y = np.zeros(n)
    for i in range(1, n):
        Y[i] = 0.9 * Y[i - 1] + rng.normal(0, 1)
    SE3 = 50.0 + seasonal + Y

    # Hydro offset is zero across this window (no inter-year variation)
    iso = ts.isocalendar()
    hydro_map = {(int(y), int(w)): 0.0
                 for y, w in zip(iso.year.to_numpy(), iso.week.to_numpy())}

    result = m.fit_se3_v242(SE3, ts, hydro_map)
    assert "coefs" in result
    assert set(result["coefs"].keys()) == {"const", "hydro", "workday", "ar1"}
    assert "model_prediction" in result
    assert result["model_prediction"].shape == SE3.shape
    # AR(1) coefficient should be ≈ 0.9
    assert result["coefs"]["ar1"] == pytest.approx(0.9, abs=0.05)


def test_fit_se3_v242_hydro_sign_correct_for_synthetic() -> None:
    """If we inject a negative dependency on hydro_offset, the coefficient
    should come out negative — matching the production result."""
    rng = np.random.default_rng(1)
    n = 24 * 365 * 2
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    iso = ts.isocalendar()
    iso_year = iso.year.to_numpy()
    iso_week = iso.week.to_numpy()

    # Inject inter-year hydro variation: 2024 = +5%, 2025 = -5%
    hydro_offset_synth = np.where(iso_year == 2024, 5.0, -5.0)
    # SE3 has a -2 EUR/MWh per +1% offset linear effect (so 2024 should be lower)
    SE3 = 40.0 - 2.0 * hydro_offset_synth + rng.normal(0, 5, n)

    hydro_map = {(int(iy), int(iw)): float(off)
                 for iy, iw, off in zip(iso_year, iso_week, hydro_offset_synth)}

    result = m.fit_se3_v242(SE3, ts, hydro_map)
    # Coefficient should be negative (synthetic mechanism: SE3 decreases as offset increases)
    assert result["coefs"]["hydro"] < 0


# ── hedge_reduction ─────────────────────────────────────────────────


def test_hedge_reduction_returns_pct_field() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    actual = np.cumsum(rng.normal(0, 1, n)) + 50
    model = actual + rng.normal(0, 0.5, n)  # almost-perfect tracker
    r = m.hedge_reduction(actual, model, lag=24, alpha=0.05)
    assert "pct_reduction" in r
    assert "h_hat" in r
    assert np.isfinite(r["pct_reduction"])
