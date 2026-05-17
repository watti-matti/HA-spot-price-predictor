"""Tests for custom_components/spot_price_predictor/seasonal_decomposition.py.

Cover:
  - fit_components correctly recovers a synthetic seasonal pattern
  - compute_residual zeroes a pure-seasonal series
  - depth honouring: absent components are not subtracted
  - per-input depth specification via DEFAULT_DEPTHS
  - artifact round-trips through JSON
  - load_components returns None for missing file
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import seasonal_decomposition as sd  # noqa: E402


def _hourly_grid(n_days: int = 730) -> np.ndarray:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = np.array(
        [(start + timedelta(hours=h)).replace(tzinfo=None)
         for h in range(n_days * 24)],
        dtype="datetime64[ns]",
    )
    return ts


# ── Index helpers ───────────────────────────────────────────────────


def test_hour_index_covers_full_diurnal_cycle() -> None:
    ts = _hourly_grid(2)
    h = sd._hour_index(ts)
    assert set(np.unique(h).tolist()) == set(range(24))


def test_weekday_index_jan1_2024_is_monday() -> None:
    ts = np.array([np.datetime64("2024-01-01T00:00")],
                  dtype="datetime64[ns]")
    assert sd._weekday_index(ts)[0] == 0  # Monday = 0


def test_weekday_index_progresses_correctly() -> None:
    ts = np.array(
        [np.datetime64(f"2024-01-0{i+1}T12:00") for i in range(7)],
        dtype="datetime64[ns]",
    )
    assert sd._weekday_index(ts).tolist() == [0, 1, 2, 3, 4, 5, 6]


# ── Fit on synthetic data ───────────────────────────────────────────


def test_fit_recovers_pure_diurnal_pattern() -> None:
    """X = 10 + 5·sin(2π·h/24) → P_hour should match the sine values."""
    ts = _hourly_grid(60)
    hours = sd._hour_index(ts).astype(float)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * hours / 24)
    comp = sd.fit_components(x, ts, depth=("P_hour",))
    p_hour = np.array(comp["P_hour"])
    assert p_hour.shape == (24,)
    # Mean of the sine over 24 hours is 0, so P_hour mean should be 10.
    assert p_hour.mean() == pytest.approx(10.0, abs=1e-6)
    # Trough at hour 18 (sin(2π·18/24) = -1) → P_hour[18] ≈ 10 - 5 = 5
    assert p_hour[18] == pytest.approx(5.0, abs=1e-6)
    # Peak at hour 6 → P_hour[6] ≈ 15
    assert p_hour[6] == pytest.approx(15.0, abs=1e-6)


def test_fit_strips_in_sequence_so_each_component_sees_residual() -> None:
    """A pure diurnal pattern should leave P_day ≈ 0 after subtracting P_hour."""
    ts = _hourly_grid(60)
    h = sd._hour_index(ts).astype(float)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * h / 24)
    comp = sd.fit_components(x, ts, depth=("P_hour", "P_day"))
    p_day = np.array(comp["P_day"])
    assert np.allclose(p_day, 0.0, atol=1e-10)


def test_fit_skips_components_not_in_depth() -> None:
    ts = _hourly_grid(60)
    x = np.random.RandomState(0).normal(size=len(ts))
    comp = sd.fit_components(x, ts, depth=("P_week",))
    assert set(comp.keys()) == {"P_week"}


# ── compute_residual ───────────────────────────────────────────────


def test_compute_residual_zeroes_pure_seasonal_series() -> None:
    ts = _hourly_grid(60)
    h = sd._hour_index(ts).astype(float)
    x = 10.0 + 5.0 * np.sin(2 * np.pi * h / 24)
    comp = sd.fit_components(x, ts, depth=("P_hour",))
    y = sd.compute_residual(x, ts, comp)
    assert np.allclose(y, 0.0, atol=1e-10)


def test_compute_residual_only_subtracts_components_present() -> None:
    """If only P_hour is in the artifact, P_day-shaped variation passes
    through to the residual unchanged."""
    ts = _hourly_grid(28)
    d = sd._weekday_index(ts).astype(float)
    # Pure weekday-vs-weekend pattern
    x = 100.0 + np.where(d >= 5, -20.0, +5.0)
    # Fit ONLY P_hour
    comp_hour_only = sd.fit_components(x, ts, depth=("P_hour",))
    y = sd.compute_residual(x, ts, comp_hour_only)
    # The weekly shape should still be visible in y
    assert abs(y[d == 0].mean() - y[d == 6].mean()) > 10.0
    # Now fit P_hour + P_day; residual should be ≈ 0
    comp_both = sd.fit_components(x, ts, depth=("P_hour", "P_day"))
    y2 = sd.compute_residual(x, ts, comp_both)
    assert np.allclose(y2, 0.0, atol=1e-9)


def test_compute_residual_raises_on_wrong_length_components() -> None:
    ts = _hourly_grid(1)
    x = np.zeros(len(ts))
    bad = {"P_hour": [0.0] * 23}  # should be length 24
    with pytest.raises(ValueError):
        sd.compute_residual(x, ts, bad)


# ── Artifact lifecycle ─────────────────────────────────────────────


def test_build_artifact_records_default_depths_when_omitted() -> None:
    inputs = {"wind": {"P_hour": [0.0]*24, "P_week": [0.0]*53}}
    art = sd.build_artifact(inputs)
    assert art["version"] == "2.5.5"
    assert art["depths"]["wind"] == list(sd.DEFAULT_DEPTHS["wind"])
    assert set(art["components"]["wind"].keys()) == {"P_hour", "P_week"}


def test_build_artifact_accepts_custom_depths() -> None:
    inputs = {"custom": {"P_hour": [0.0]*24}}
    art = sd.build_artifact(inputs, depths={"custom": ("P_hour",)})
    assert art["depths"]["custom"] == ["P_hour"]


def test_load_components_returns_none_for_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.json"
    assert sd.load_components(p) is None


def test_artifact_roundtrips_through_json(tmp_path: Path) -> None:
    """Save → load → use yields identical residual."""
    ts = _hourly_grid(28)
    rng = np.random.RandomState(42)
    x = (rng.normal(loc=10, scale=2, size=len(ts))
         + 3.0 * np.sin(2 * np.pi * sd._hour_index(ts) / 24))
    comp = sd.fit_components(x, ts, depth=("P_hour", "P_day", "P_week"))
    art = sd.build_artifact({"test": comp})

    out_path = tmp_path / "components.json"
    out_path.write_text(json.dumps(art))
    loaded = sd.load_components(out_path)
    assert loaded is not None
    y_direct = sd.compute_residual(x, ts, comp)
    y_via_artifact = sd.compute_residual(x, ts, loaded["components"]["test"])
    assert np.allclose(y_direct, y_via_artifact, atol=1e-12)


def test_default_depths_match_v2_5_4_audit() -> None:
    """The default-depths table matches the v2.5.4 verdict tabulated in
    studies/results/V2_5_4_RELEASE_NOTES.md."""
    assert sd.DEFAULT_DEPTHS["fi"]    == ("P_hour", "P_day", "P_week")
    assert sd.DEFAULT_DEPTHS["se3"]   == ("P_hour", "P_day", "P_week")
    assert sd.DEFAULT_DEPTHS["se1"]   == ("P_day",  "P_week")
    assert sd.DEFAULT_DEPTHS["ee"]    == ("P_hour", "P_day", "P_week")
    assert sd.DEFAULT_DEPTHS["wind"]  == ("P_hour", "P_week")  # user hint: day + month
    assert sd.DEFAULT_DEPTHS["solar"] == ("P_hour", "P_week")
    assert sd.DEFAULT_DEPTHS["temp"]  == ("P_hour", "P_week")
    assert sd.DEFAULT_DEPTHS["cloud"] == ("P_week",)
