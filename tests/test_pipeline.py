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
    # v2.17.0 — intercept + 10 Ridge features: 5 core, 3 LAGGED neighbour
    # zones (same-hour prices leak the target), plus the two demand
    # inputs (lagged net load, public-holiday flag).
    assert p._ridge_coef.shape == (10,)
    assert tuple(p._features) == (
        "intercept", "Y_fi_lag168", "is_workday",
        "Y_sigmoid_wind_rho", "Y_solar_effective", "Y_temp",
        "Y_se1_lag168", "Y_se3_lag168", "Y_ee_lag168",
        "is_holiday",
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
    """v2.10.0 — supplying SE1/SE3/EE neighbour prices shifts the mean
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
        neighbour_prices_lag168=neigh,
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
        neighbour_prices_lag168=partial,
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
    # 480 hourly updates = 20 daily observations per hour-of-day bin,
    # above the per-bin warmup (14 daily observations).
    ts = _hourly_timestamps(480)
    for i in range(480):
        pred = 50.0
        actual = 51.0 + rng.normal(0, 5)
        p.update_with_actuals(np.array([pred]), np.array([actual]),
                              timestamps=ts[i:i + 1])
    assert p._bias.warm
    # Bias should be ≈ +1 (we systematically forecast 1 too low)
    assert p._bias.bias_estimate == pytest.approx(1.0, abs=1.5)


def test_update_with_actuals_without_timestamps_skips_bias(
        tmp_path: Path) -> None:
    """No timestamps -> hour bins unknown -> the bias corrector must not
    update (the fan-chart calibrator still does)."""
    p = _make_pipeline(tmp_path)
    for _ in range(480):
        p.update_with_actuals(np.array([50.0]), np.array([55.0]))
    assert not p._bias.warm
    assert p._bias.bias_estimate == 0.0


def test_per_hour_bias_specialises_by_hour(tmp_path: Path) -> None:
    """Opposite systematic errors at different hours must produce
    opposite per-hour corrections — the property a global EMA lacks
    (see studies/results/phase2_bias_decomposition.md)."""
    p = _make_pipeline(tmp_path)
    # 60 days: the 14-day-halflife EMA converges to ~95 % of the true
    # per-hour offset (1 - 0.5^(60/14)).
    ts = _hourly_timestamps(24 * 60)
    hours = (ts.astype("datetime64[s]").astype("int64") // 3600) % 24
    pred = np.full(len(ts), 50.0)
    actual = np.where(hours == 8, 60.0,
                      np.where(hours == 3, 40.0, 50.0)).astype(float)
    for i in range(len(ts)):
        p.update_with_actuals(pred[i:i + 1], actual[i:i + 1],
                              timestamps=ts[i:i + 1])
    by_hour = p._bias.bias_by_hour
    assert by_hour[8] == pytest.approx(+10.0, abs=2.0)
    assert by_hour[3] == pytest.approx(-10.0, abs=2.0)
    assert by_hour[12] == pytest.approx(0.0, abs=1.0)


def test_per_hour_bias_migrates_legacy_state() -> None:
    """`PerHourBiasCorrector.from_dict` must accept a legacy single-EMA
    `HourlyBiasCorrector` state and seed all 24 bins from it.

    Tested at the class level: the Pipeline no longer routes legacy state
    through this path, because state carrying no model fingerprint was
    learned by an unknown (older) model and is discarded instead — see
    test_pre_existing_state_without_fingerprint_is_discarded. The
    migration is kept for callers that know the model is unchanged.
    """
    legacy = _hc_mod.HourlyBiasCorrector()
    for _ in range(200):    # warm the legacy corrector at +2 bias
        legacy.update(50.0, 52.0)
    migrated = _hc_mod.PerHourBiasCorrector.from_dict(legacy.to_dict())
    assert migrated.warm
    assert migrated.bias_estimate == pytest.approx(legacy.bias_estimate,
                                                    abs=1e-9)
    for h in (0, 8, 23):
        assert migrated.bias_by_hour[h] == pytest.approx(
            legacy.bias_estimate, abs=1e-9)


def test_state_roundtrips_through_storage_dir(tmp_path: Path) -> None:
    """Save state, instantiate a fresh pipeline pointing at the same
    storage dir, and verify it restores the calibrator state."""
    p1 = _make_pipeline(tmp_path)
    rng = np.random.RandomState(0)
    ts = _hourly_timestamps(480)
    for i in range(480):
        p1.update_with_actuals(np.array([50.0]),
                                np.array([52.0 + rng.normal(0, 3)]),
                                timestamps=ts[i:i + 1])
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


# ── Economic sign invariant: zero-marginal-cost generation ────────
#
# Wind and PV have ~zero marginal cost, so more of either can only push
# the price down or leave it unchanged — never up. These tests are a
# permanent guard: they must fail if a retrain, refactor, or hand-edited
# artifact ever reintroduces the inversion that shipped through v2.15.0
# (Y_solar_effective had a POSITIVE coefficient, so a sunnier forecast
# raised the predicted price).


def test_shipped_artifact_obeys_zero_marginal_cost_signs() -> None:
    """The SHIPPED artifact must not price wind/PV with a positive sign."""
    art = json.loads(
        (REPO / "custom_components" / "spot_price_predictor" / "data"
         / "spike_model_default.json").read_text()
    )
    feats = ["intercept"] + list(art["ridge_features"])
    for name in pipeline_mod.NON_POSITIVE_FEATURES:
        assert name in feats, f"{name} missing from shipped ridge_features"
        c = float(art["ridge_coef"][feats.index(name)])
        assert c <= 0.0, (
            f"{name} coefficient {c:+.6f} > 0 — the shipped model would raise "
            f"its price forecast when zero-marginal-cost generation rises. "
            f"Refit with the sign constraint in "
            f"studies/build_fresh_spike_model.py."
        )


def test_runtime_clamps_a_positive_physics_coefficient(tmp_path: Path) -> None:
    """Even a violating artifact must be neutralised at load time."""
    data_dir = REPO / "custom_components" / "spot_price_predictor" / "data"
    art = json.loads((data_dir / "spike_model_default.json").read_text())
    feats = ["intercept"] + list(art["ridge_features"])
    art["ridge_coef"][feats.index("Y_solar_effective")] = +0.5   # inverted
    bad = tmp_path / "data"
    bad.mkdir()
    for f in data_dir.glob("*.json"):
        (bad / f.name).write_text(f.read_text(), encoding="utf-8")
    (bad / "spike_model_default.json").write_text(json.dumps(art),
                                                  encoding="utf-8")
    p = pipeline_mod.Pipeline(data_dir=bad, storage_dir=tmp_path / "state")
    assert p._ridge_coef[p._features.index("Y_solar_effective")] == 0.0


def test_forecast_never_rises_with_more_irradiance(tmp_path: Path) -> None:
    """Behavioural invariant: raising irradiance must never raise the
    forecast at any hour. This is the guard that survives refactors —
    it constrains the model's response, not just a stored number."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(48)
    n = len(ts)
    wind = np.full(n, 6.0)
    temp = np.full(n, 18.0)
    base = p.compute_forecast(ts, wind, np.full(n, 100.0), temp,
                              enable_fan_chart=False)["mean_eur_mwh"]
    for extra in (50.0, 200.0, 500.0):
        sunnier = p.compute_forecast(ts, wind, np.full(n, 100.0 + extra), temp,
                                     enable_fan_chart=False)["mean_eur_mwh"]
        assert np.all(sunnier <= base + 1e-9), (
            f"+{extra:.0f} W/m2 irradiance RAISED the forecast at "
            f"{int(np.sum(sunnier > base + 1e-9))} hour(s); PV is "
            f"zero-marginal-cost and can never increase price."
        )


def test_forecast_never_rises_with_more_wind(tmp_path: Path) -> None:
    """Same invariant for wind (also zero marginal cost)."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(48)
    n = len(ts)
    solar = np.zeros(n)
    temp = np.full(n, 5.0)
    base = p.compute_forecast(ts, np.full(n, 5.0), solar, temp,
                              enable_fan_chart=False)["mean_eur_mwh"]
    for w in (7.0, 10.0, 15.0):
        windier = p.compute_forecast(ts, np.full(n, w), solar, temp,
                                     enable_fan_chart=False)["mean_eur_mwh"]
        assert np.all(windier <= base + 1e-9), (
            f"wind {w:.0f} m/s RAISED the forecast vs 5 m/s; wind is "
            f"zero-marginal-cost and can never increase price."
        )


# ── Leak invariant: no same-hour market data (v2.17.0) ────────────
#
# FI, SE1, SE3 and EE clear in the SAME day-ahead auction, so a same-hour
# neighbour price is never observable before the FI price it is meant to
# predict. Through v2.16 the pipeline consumed same-hour prices, which
# suppressed the physical wind coefficient (-44.6 vs -93.0), inverted the
# solar sign, and left the model mis-specified for every hour production
# must actually forecast. These tests are a permanent guard.


def test_artifact_declares_no_same_hour_neighbour_features() -> None:
    """The shipped artifact must not name an un-lagged neighbour zone."""
    art = json.loads(
        (REPO / "custom_components" / "spot_price_predictor" / "data"
         / "spike_model_default.json").read_text()
    )
    for f in art["ridge_features"]:
        for zone in ("se1", "se3", "ee"):
            if f in (f"Y_{zone}", zone):
                raise AssertionError(
                    f"shipped artifact uses same-hour neighbour feature {f!r}; "
                    f"SE/EE clear in the same auction as FI, so this value is "
                    f"unknowable at forecast time. Use Y_{zone}_lag168."
                )


def test_neighbour_features_read_the_lagged_hour(tmp_path: Path) -> None:
    """A neighbour price supplied for the forecast hours must influence
    the forecast via the LAGGED slot, not the same-hour slot.

    Feeding a spike whose position corresponds to t-168h must move the
    forecast at t; the same spike aligned to t must not.
    """
    p = _make_pipeline(tmp_path)
    n = 24 + pipeline_mod.NEIGHBOUR_LAG_HOURS
    ts = _hourly_timestamps(n)
    wind = np.full(n, 6.0); solar = np.zeros(n); temp = np.full(n, 5.0)
    base_kw = dict(wind=wind, solar=solar, temp=temp, enable_fan_chart=False)
    flat = {z: np.full(n, 40.0) for z in ("se1", "se3", "ee")}
    base = p.compute_forecast(ts, neighbour_prices_lag168=flat, **base_kw)
    # Caller contract: element i holds the price at ts[i] - 168 h. So to
    # perturb the value the model reads for forecast hour i, perturb i.
    bumped = {z: v.copy() for z, v in flat.items()}
    for z in bumped:
        bumped[z][5] = 400.0
    out = p.compute_forecast(ts, neighbour_prices_lag168=bumped, **base_kw)
    d = out["mean_eur_mwh"] - base["mean_eur_mwh"]
    assert abs(d[5]) > 1e-6, "lagged neighbour input had no effect on its hour"
    others = np.delete(np.abs(d), 5)
    assert others.max() < 1e-6, "a single lagged input leaked into other hours"


def test_holiday_flag_suppresses_workday_and_moves_forecast(tmp_path: Path) -> None:
    """A weekday marked as a public holiday must not be priced as a
    normal working day (weekday holidays have weekend-like demand)."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(48)
    n = len(ts)
    kw = dict(wind=np.full(n, 6.0), solar=np.zeros(n), temp=np.full(n, 5.0),
              enable_fan_chart=False)
    normal = p.compute_forecast(ts, is_holiday=np.zeros(n), **kw)["mean_eur_mwh"]
    holiday = p.compute_forecast(ts, is_holiday=np.ones(n), **kw)["mean_eur_mwh"]
    assert not np.allclose(normal, holiday), (
        "is_holiday had no effect — weekday holidays would be priced as "
        "ordinary working days"
    )


def test_shipped_model_does_not_use_lagged_netload() -> None:
    """`Y_netload_lag168` was shipped in v2.17.0 and removed in v2.17.1.

    On the correct (2023-) training window it was worth -0.04 % MAE and
    its fitted coefficient was negative (-1.25) — a collinearity artifact
    against the lagged price (corr +0.587), not a demand relationship.
    Demand enters via the holiday features instead. Re-adding it needs
    fresh evidence, not a silent retrain.
    """
    art = json.loads(
        (REPO / "custom_components" / "spot_price_predictor" / "data"
         / "spike_model_default.json").read_text()
    )
    assert "Y_netload_lag168" not in art["ridge_features"], (
        "shipped model re-introduced Y_netload_lag168; it measured as "
        "noise with an uninterpretable sign. See docs/BACKLOG.md."
    )


def test_netload_plumbing_still_accepted(tmp_path: Path) -> None:
    """The pipeline must keep accepting `netload_lag168` without error so
    a future artifact (e.g. the day-ahead hybrid) can use it, even though
    no shipped model declares it today."""
    p = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24)
    n = len(ts)
    out = p.compute_forecast(
        ts, wind=np.full(n, 6.0), solar=np.zeros(n), temp=np.full(n, 5.0),
        netload_lag168=np.full(n, 9000.0), enable_fan_chart=False)
    assert np.isfinite(out["mean_eur_mwh"]).all()


# ── Calibrator state must not outlive the model that taught it ────
#
# The bias corrector and DtACI fan chart learn THIS model's error. After a
# retrain those corrections describe a model that no longer exists.
# Measured across the v2.16 -> v2.17.1 change, carrying the old state over
# cost ~0.45 % MAE and about +1.5 EUR/MWh of excess bias until the 14-day
# EMA washed it out.


def _warm_bias_state(tmp_path: Path, offset: float = 9.0) -> Path:
    """Create a storage dir holding a warmed corrector + a fingerprint
    claiming it belongs to a different model."""
    storage = tmp_path / "pipeline_state"
    storage.mkdir(parents=True, exist_ok=True)
    bc = _hc_mod.PerHourBiasCorrector()
    for _ in range(60):
        for h in range(24):
            bc.update(50.0, 50.0 + offset, h)
    (storage / "hourly_bias.json").write_text(json.dumps(bc.to_dict()),
                                              encoding="utf-8")
    return storage


def test_calibrator_state_is_discarded_when_the_model_changes(
        tmp_path: Path) -> None:
    storage = _warm_bias_state(tmp_path)
    (storage / pipeline_mod.FINGERPRINT_FILE).write_text(
        json.dumps({"model_fingerprint": "a-different-model"}),
        encoding="utf-8")
    p = _make_pipeline(tmp_path)
    assert p.calibrators_cold_started is True
    assert not p._bias.warm, "stale corrections survived a model change"
    assert p._bias.bias_estimate == 0.0
    assert not (storage / "hourly_bias.json").exists()


def test_calibrator_state_survives_when_the_model_is_unchanged(
        tmp_path: Path) -> None:
    """A restart must NOT throw away hard-won calibration."""
    p1 = _make_pipeline(tmp_path)
    ts = _hourly_timestamps(24 * 30)
    for i in range(len(ts)):
        p1.update_with_actuals(np.array([50.0]), np.array([57.0]),
                               timestamps=ts[i:i + 1])
    p1.save_state()
    assert p1._bias.warm
    p2 = _make_pipeline(tmp_path)
    assert p2.calibrators_cold_started is False
    assert p2._bias.warm, "restart discarded state despite an unchanged model"
    assert p2._bias.bias_estimate == pytest.approx(p1._bias.bias_estimate,
                                                   abs=1e-9)


def test_pre_existing_state_without_fingerprint_is_discarded(
        tmp_path: Path) -> None:
    """Upgrading from a build that never wrote a fingerprint: the state
    was learned by an older model, so it must not be trusted."""
    storage = _warm_bias_state(tmp_path)
    assert not (storage / pipeline_mod.FINGERPRINT_FILE).exists()
    p = _make_pipeline(tmp_path)
    assert p.calibrators_cold_started is True
    assert not p._bias.warm


def test_fresh_install_is_not_reported_as_a_cold_start(tmp_path: Path) -> None:
    """No prior state at all is a normal first run, not an invalidation."""
    p = _make_pipeline(tmp_path)
    assert p.calibrators_cold_started is False


def test_model_fingerprint_is_exposed_and_tracks_the_coefficients(
        tmp_path: Path) -> None:
    """The DtACI D(k) bundles key their own state on this value, so it has
    to be public and it has to move when the model does."""
    p = _make_pipeline(tmp_path)
    assert p.model_fingerprint == p._model_fingerprint
    assert p.model_fingerprint

    before = p.model_fingerprint
    p._ridge_coef[1] += 0.5
    assert p._compute_model_fingerprint() != before


def test_fingerprint_is_recorded_at_init_not_only_at_save(
        tmp_path: Path) -> None:
    """v2.17.2 wrote the fingerprint last of four files in `save_state`.
    A cycle that failed — or never reached — the save left state with no
    fingerprint beside it, so the NEXT start wiped it again, and every
    start after that. The calibrators never got to warm up at all."""
    p = _make_pipeline(tmp_path)          # fresh install, no save_state call
    assert (p._storage_dir / pipeline_mod.FINGERPRINT_FILE).exists()


def test_a_cold_start_is_not_repeated_on_every_restart(
        tmp_path: Path) -> None:
    """The reset must happen once per model change, not once per restart."""
    storage = _warm_bias_state(tmp_path)
    (storage / pipeline_mod.FINGERPRINT_FILE).write_text(
        json.dumps({"model_fingerprint": "a-different-model"}),
        encoding="utf-8")

    p1 = _make_pipeline(tmp_path)
    assert p1.calibrators_cold_started is True

    # Deliberately never call save_state — this is the cycle that failed,
    # or was cut short by a restart, before reaching the persist step.
    p2 = _make_pipeline(tmp_path)

    assert p2.calibrators_cold_started is False, (
        "the model has not changed again; a second wipe on the next start "
        "holds the calibrators permanently cold"
    )


# ── Local-calendar workday flag (v2.18.0) ─────────────────────────


def test_is_workday_uses_the_local_calendar() -> None:
    """The trainer builds `is_workday` from Europe/Helsinki
    (build_fresh_spike_model.py) and the coordinator builds `is_holiday`
    from the local date, so the runtime must agree. Through v2.17.3 this
    was a UTC weekday, which disagreed on 2.93 % of hours — 21:00–23:00
    UTC — at exactly the Fri/Sat and Sun/Mon boundaries.
    """
    ts = np.array([
        "2026-06-07T21:00",   # Sunday 21:00 UTC  = Monday 00:00 EEST
        "2026-06-05T21:00",   # Friday 21:00 UTC  = Saturday 00:00 EEST
        "2026-06-08T09:00",   # Monday, unambiguous
        "2026-06-06T09:00",   # Saturday, unambiguous
    ], dtype="datetime64[ns]")
    got = pipeline_mod.Pipeline._is_workday(ts)
    assert list(got) == [True, False, True, False], (
        "Sunday 21:00 UTC is Monday locally (workday) and Friday 21:00 UTC "
        "is Saturday locally (not) — a UTC weekday gets both backwards"
    )


def test_workday_flag_moves_the_forecast_at_the_local_boundary(
    tmp_path: Path,
) -> None:
    """Behavioural guard: the hour that flips must actually change the
    forecast, by roughly the shipped is_workday coefficient."""
    p = _make_pipeline(tmp_path)
    art = json.loads(
        (REPO / "custom_components" / "spot_price_predictor" / "data"
         / "spike_model_default.json").read_text())
    coef = float(art["ridge_coef"][
        (["intercept"] + list(art["ridge_features"])).index("is_workday")])
    base = dict(wind=np.full(2, 6.0), solar=np.zeros(2), temp=np.full(2, 5.0),
                enable_fan_chart=False)
    sun_local = np.array(["2026-06-07T18:00", "2026-06-07T19:00"],
                         dtype="datetime64[ns]")   # Sunday locally
    mon_local = np.array(["2026-06-07T21:00", "2026-06-07T22:00"],
                         dtype="datetime64[ns]")   # Monday locally
    a = p.compute_forecast(sun_local, **base)["mean_eur_mwh"]
    b = p.compute_forecast(mon_local, **base)["mean_eur_mwh"]
    assert not np.allclose(a, b)
    assert coef > 0.0
