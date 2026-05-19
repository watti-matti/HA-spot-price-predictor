"""Tests for custom_components/spot_price_predictor/pipeline.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

# Bypass package __init__ (which imports homeassistant) by injecting
# a fake parent package + loading dependencies manually, matching the
# pattern used by test_hourly_calibration.py.
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))
import dtaci as _dtaci_mod              # noqa: F401, E402
import bias_corrector as _bias_mod      # noqa: F401, E402
import seasonal_decomposition as _sd    # noqa: F401, E402
import solar_clear_sky as _scs          # noqa: F401, E402
import price_floor as _pf               # noqa: F401, E402

pkg = types.ModuleType("spot_price_predictor")
pkg.__path__ = [str(REPO / "custom_components" / "spot_price_predictor")]
sys.modules["spot_price_predictor"] = pkg
for mod_name, mod in [
    ("dtaci", _dtaci_mod),
    ("bias_corrector", _bias_mod),
    ("seasonal_decomposition", _sd),
    ("solar_clear_sky", _scs),
    ("price_floor", _pf),
]:
    sys.modules[f"spot_price_predictor.{mod_name}"] = mod

# Load hourly_calibration
_hc_spec = importlib.util.spec_from_file_location(
    "spot_price_predictor.hourly_calibration",
    REPO / "custom_components" / "spot_price_predictor" / "hourly_calibration.py",
)
_hc_mod = importlib.util.module_from_spec(_hc_spec)
sys.modules["spot_price_predictor.hourly_calibration"] = _hc_mod
_hc_spec.loader.exec_module(_hc_mod)

# Load pipeline
_pipeline_spec = importlib.util.spec_from_file_location(
    "spot_price_predictor.pipeline",
    REPO / "custom_components" / "spot_price_predictor" / "pipeline.py",
)
pipeline_mod = importlib.util.module_from_spec(_pipeline_spec)
sys.modules["spot_price_predictor.pipeline"] = pipeline_mod
_pipeline_spec.loader.exec_module(pipeline_mod)


# ── Fixtures ───────────────────────────────────────────────────────


def _make_pipeline(tmp_path: Path) -> "pipeline_mod.Pipeline":
    """Construct a Pipeline using the SHIPPED production artifacts plus
    a temp directory for calibrator state."""
    data_dir = (REPO / "custom_components" / "spot_price_predictor"
                / "data")
    storage = tmp_path / "pipeline_state"
    return pipeline_mod.Pipeline(data_dir=data_dir, storage_dir=storage)


def _hourly_timestamps(n: int = 48) -> np.ndarray:
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    return np.array(
        [(start + timedelta(hours=h)).replace(tzinfo=None)
         for h in range(n)],
        dtype="datetime64[ns]",
    )


# ── Construction ───────────────────────────────────────────────────


def test_pipeline_loads_shipped_artifacts(tmp_path: Path) -> None:
    """Construction must succeed against the production artifacts and
    populate Ridge coef / AR(1) phi / L4 GPD params."""
    p = _make_pipeline(tmp_path)
    # v2.9.0 — intercept + 8 Ridge features (5 core + Y_se1 + Y_se3 + Y_ee)
    assert p._ridge_coef.shape == (9,)
    assert tuple(p._features) == (
        "intercept", "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
        "Y_se1", "Y_se3", "Y_ee",
    )
    assert -1.0 < p._ar1_phi < 1.0
    assert isinstance(p._gpd_right, dict)
    assert p._eta_sigma > 0


def test_pipeline_initialises_calibrators_cold(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    # No state files present → calibrators are cold
    assert not p._bias.warm
    assert p._fan.target_coverages == (0.5, 0.9)
    assert not p._refit.refit_recommended


# ── compute_forecast ──────────────────────────────────────────────


def test_compute_forecast_returns_expected_shape(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(48)
    n = len(ts)
    wind  = np.full(n, 6.0)
    solar = np.zeros(n)
    temp  = np.full(n, 5.0)
    out = p.compute_forecast(ts, wind, solar, temp,
                              enable_fan_chart=True)
    assert out["mean_eur_mwh"].shape == (n,)
    for k in ("P5", "P25", "P50", "P75", "P95"):
        assert out[f"{k}_eur_mwh"].shape == (n,)


def test_compute_forecast_accepts_neighbour_prices(tmp_path: Path) -> None:
    """v2.9.0 — supplying SE1/SE3/EE neighbour prices shifts the mean
    relative to the no-neighbour fallback, in line with the V_xb
    cross-border coefficients (which are positive)."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(48)
    n = len(ts)
    wind  = np.full(n, 6.0)
    solar = np.zeros(n)
    temp  = np.full(n, 5.0)
    out_no = p.compute_forecast(ts, wind, solar, temp,
                                 enable_fan_chart=False)
    # Synthetic neighbour prices: high relative to the shipped seasonal
    # climatology → positive Y_se deviations → predicted FI mean shifts
    # upward.
    neigh = {
        "se1": np.full(n, 120.0),
        "se3": np.full(n, 120.0),
        "ee":  np.full(n, 120.0),
    }
    out_high = p.compute_forecast(
        ts, wind, solar, temp,
        recent_neighbour_prices=neigh,
        enable_fan_chart=False,
    )
    # The mean response should change. Check at least one hour moves
    # meaningfully (≥ 2 EUR/MWh) so a future regression that breaks the
    # neighbour-price plumbing is caught.
    diff = out_high["mean_eur_mwh"] - out_no["mean_eur_mwh"]
    assert float(np.max(np.abs(diff))) >= 2.0


def test_compute_forecast_handles_partial_neighbour_data(tmp_path: Path) -> None:
    """Missing zones / NaN entries must not propagate NaN to the mean."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    n = len(ts)
    wind  = np.full(n, 6.0)
    solar = np.zeros(n)
    temp  = np.full(n, 5.0)
    partial = {
        "se3": np.array([50.0] * 12 + [np.nan] * 12),   # 12-hour gap
        # se1 and ee deliberately missing
    }
    out = p.compute_forecast(
        ts, wind, solar, temp,
        recent_neighbour_prices=partial,
        enable_fan_chart=False,
    )
    assert np.isfinite(out["mean_eur_mwh"]).all()


def test_compute_forecast_respects_floor(tmp_path: Path) -> None:
    """All mean predictions must be >= softplus(0) = log(2) above the
    floor (i.e. > floor - epsilon)."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    n = len(ts)
    wind = np.full(n, 20.0)        # very high wind → strongly negative residual
    solar = np.zeros(n)
    temp = np.full(n, 5.0)
    out = p.compute_forecast(ts, wind, solar, temp,
                              enable_fan_chart=False)
    assert (out["mean_eur_mwh"] >= _pf.DEFAULT_FLOOR_EUR_MWH - 1e-3).all()


def test_compute_forecast_fan_bands_are_ordered(tmp_path: Path) -> None:
    """P5 ≤ P25 ≤ P50 ≤ P75 ≤ P95 for every forecast hour."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    n = len(ts)
    wind  = np.full(n, 6.0)
    solar = np.zeros(n)
    temp  = np.full(n, 5.0)
    out = p.compute_forecast(ts, wind, solar, temp,
                              enable_fan_chart=True)
    assert (out["P5_eur_mwh"]  <= out["P25_eur_mwh"]).all()
    assert (out["P25_eur_mwh"] <= out["P50_eur_mwh"]).all()
    assert (out["P50_eur_mwh"] <= out["P75_eur_mwh"]).all()
    assert (out["P75_eur_mwh"] <= out["P95_eur_mwh"]).all()


def test_compute_forecast_disable_fan_chart_omits_bands(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    n = len(ts)
    out = p.compute_forecast(ts, np.full(n, 6.0), np.zeros(n),
                              np.full(n, 5.0), enable_fan_chart=False)
    assert "mean_eur_mwh" in out
    for k in ("P5", "P25", "P50", "P75", "P95"):
        assert f"{k}_eur_mwh" not in out


# ── D(k) duration curves ──────────────────────────────────────────


def test_duration_curves_24_entries_per_direction(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(72)        # 3 full days
    # Synthetic hourly prediction: a clear diurnal cycle
    h = np.arange(len(ts)) % 24
    pred = 50.0 + 30.0 * np.cos(2 * np.pi * (h - 5) / 24)
    dk = p.compute_duration_curves(pred, ts)
    assert len(dk) == 3
    for day in dk:
        assert len(day["dk_cheap_eur_mwh"]) == 24
        assert len(day["dk_peak_eur_mwh"]) == 24


def test_duration_curves_monotone_per_direction(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    pred = np.random.RandomState(0).normal(50, 20, size=24)
    dk = p.compute_duration_curves(pred, ts)
    day = dk[0]
    cheap = np.array(day["dk_cheap_eur_mwh"])
    peak  = np.array(day["dk_peak_eur_mwh"])
    assert (np.diff(cheap) >= -1e-9).all()      # non-decreasing
    assert (np.diff(peak)  <= +1e-9).all()      # non-increasing
    assert cheap[-1] == pytest.approx(peak[-1], abs=1e-9)  # daily mean


# ── update_with_actuals + state persistence ───────────────────────


def test_update_with_actuals_warms_calibrators(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    rng = np.random.RandomState(0)
    # Feed 200 updates (well above warmup_hours=168)
    for _ in range(200):
        pred = 50.0
        actual = 51.0 + rng.normal(0, 5)
        p.update_with_actuals(np.array([pred]), np.array([actual]))
    assert p._bias.warm
    # Bias should be ≈ +1 (we systematically forecast 1 too low)
    assert p._bias.bias_estimate == pytest.approx(1.0, abs=1.5)


def test_state_roundtrips_through_storage_dir(tmp_path: Path) -> None:
    """Save state, instantiate a fresh pipeline pointing at the same
    storage dir, and verify it restores the calibrator state."""
    p1 = _make_pipeline(tmp_path)
    rng = np.random.RandomState(0)
    for _ in range(180):
        p1.update_with_actuals(np.array([50.0]),
                                np.array([52.0 + rng.normal(0, 3)]))
    p1.save_state()
    # New instance — should restore from the persisted JSON files
    p2 = _make_pipeline(tmp_path)
    assert p2._bias.warm
    assert p2._bias.bias_estimate == pytest.approx(
        p1._bias.bias_estimate, abs=1e-9)


def test_update_with_actuals_returns_refit_flag(tmp_path: Path) -> None:
    p = _make_pipeline(tmp_path)
    out = p.update_with_actuals(np.array([50.0]), np.array([55.0]))
    assert "refit_recommended" in out
    assert out["refit_recommended"] is False
    assert "bias_warm" in out
    assert "bias_estimate" in out
    assert "fan_diagnostics" in out


# ── End-to-end smoke test ─────────────────────────────────────────


def test_pipeline_end_to_end_smoke(tmp_path: Path) -> None:
    """Run compute_forecast → compute_duration_curves → update_with_actuals
    in the order the coordinator would, with realistic-shape inputs."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(168)        # 7-day forecast horizon
    n = len(ts)
    rng = np.random.RandomState(0)
    wind  = 6.0 + rng.normal(0, 1.5, size=n).clip(0, None)
    solar = np.maximum(0, 200.0 * np.sin(np.pi * (np.arange(n) % 24) / 24)
                       + rng.normal(0, 30, size=n))
    temp  = 5.0 + 5.0 * np.cos(2 * np.pi * np.arange(n) / (24 * 30)) \
            + rng.normal(0, 2, size=n)

    out = p.compute_forecast(ts, wind, solar, temp,
                              enable_fan_chart=True)
    pred = out["mean_eur_mwh"]
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()

    dk = p.compute_duration_curves(pred, ts)
    assert len(dk) == 7  # 7 days
    for day in dk:
        assert len(day["dk_cheap_eur_mwh"]) == 24

    # Pretend the first 24 hours' actuals came in
    actuals = pred[:24] + rng.normal(0, 5, size=24)
    diag = p.update_with_actuals(pred[:24], actuals)
    assert "refit_recommended" in diag
