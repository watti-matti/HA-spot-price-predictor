"""Tests for `pv_aware_cvar` (per-day production CVaR computation)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

import pv_aware_cvar  # noqa: E402


# ── Sampler properties ─────────────────────────────────────────────


def test_sampler_mean_preserves_point_forecast() -> None:
    rng = np.random.default_rng(0)
    pv = np.array([0, 0, 0, 0.1, 0.5, 1.2, 2.5, 4.0, 5.0, 5.5, 5.8, 5.5,
                    5.0, 4.0, 2.5, 1.2, 0.5, 0.1, 0, 0, 0, 0, 0, 0])
    paths = pv_aware_cvar._sample_pv_paths(pv, n_paths=5000, rel_std=0.30,
                                             rng=rng)
    # Mean across paths per hour should be within ~1% of the point.
    np.testing.assert_allclose(paths.mean(axis=0), pv, rtol=0.05, atol=0.05)


def test_sampler_zero_hours_stay_zero() -> None:
    rng = np.random.default_rng(1)
    pv = np.zeros(24)
    pv[10] = 4.0
    paths = pv_aware_cvar._sample_pv_paths(pv, n_paths=100, rel_std=0.30,
                                             rng=rng)
    # All paths must be exactly zero at every hour where point = 0.
    for h in range(24):
        if pv[h] == 0:
            assert np.all(paths[:, h] == 0.0)


def test_sampler_std_matches_rel_std_at_pv_hours() -> None:
    rng = np.random.default_rng(2)
    pv = np.full(24, 3.0)
    paths = pv_aware_cvar._sample_pv_paths(pv, n_paths=8000, rel_std=0.30,
                                             rng=rng)
    rel_std_empirical = paths.std(axis=0) / paths.mean(axis=0)
    np.testing.assert_allclose(rel_std_empirical, 0.30, rtol=0.10)


# ── compute_pv_aware_cvar_for_day ──────────────────────────────────


def _dummy_inputs() -> dict[str, np.ndarray]:
    return {
        "buy":  np.full(24, 0.20, dtype=float),
        "sell": np.full(24, 0.04, dtype=float),
        "pv":   np.concatenate([
            np.zeros(7),
            np.array([0.5, 2.0, 3.5, 4.5, 5.0, 5.5, 5.0, 4.5, 3.5, 2.0, 0.8]),
            np.zeros(6),
        ]),
        "cons": np.full(24, 1.5, dtype=float),
    }


def test_compute_returns_expected_keys() -> None:
    d = _dummy_inputs()
    out = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d["buy"], d["sell"], d["pv"], d["cons"],
        rng=np.random.default_rng(0),
    )
    expected = {
        "mean_eur_kwh", "cvar95_eur_kwh",
        "p5_eur_kwh", "p50_eur_kwh", "p95_eur_kwh",
        "mean_eur", "cvar95_eur",
        "pv_self_consumed_kwh", "pv_exported_kwh",
        "n_paths", "rel_std",
    }
    assert set(out.keys()) == expected


def test_compute_quantile_ordering() -> None:
    d = _dummy_inputs()
    out = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d["buy"], d["sell"], d["pv"], d["cons"],
        n_paths=500,
        rng=np.random.default_rng(3),
    )
    assert out["p5_eur_kwh"] <= out["p50_eur_kwh"] <= out["p95_eur_kwh"]
    # CVaR_5% (right-tail) is >= P95 by definition.
    assert out["cvar95_eur_kwh"] >= out["p95_eur_kwh"] - 1e-9


def test_compute_cvar_above_mean_for_realistic_day() -> None:
    """A spread > 0 between CVaR_5% and mean is the whole point of
    publishing this sensor."""
    d = _dummy_inputs()
    out = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d["buy"], d["sell"], d["pv"], d["cons"],
        n_paths=500,
        rng=np.random.default_rng(4),
    )
    assert out["cvar95_eur_kwh"] > out["mean_eur_kwh"]


def test_compute_no_pv_equals_buy_times_consumption() -> None:
    """Sanity: zero PV reduces cost to buy * consumption."""
    d = _dummy_inputs()
    d["pv"] = np.zeros(24)
    out = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d["buy"], d["sell"], d["pv"], d["cons"],
        n_paths=200,
        rng=np.random.default_rng(5),
    )
    # All paths identical when PV=0 — mean = CVaR.
    assert out["mean_eur_kwh"] == pytest.approx(0.20)
    assert out["cvar95_eur_kwh"] == pytest.approx(0.20)


def test_compute_validates_input_lengths() -> None:
    with pytest.raises(ValueError, match="length 24"):
        pv_aware_cvar.compute_pv_aware_cvar_for_day(
            np.zeros(23), np.zeros(24), np.zeros(24), np.zeros(24),
        )
    with pytest.raises(ValueError, match="length 24"):
        pv_aware_cvar.compute_pv_aware_cvar_for_day(
            np.zeros(24), np.zeros(24), np.zeros(24), np.zeros(20),
        )


def test_compute_pv_self_consumption_plus_export_equals_pv() -> None:
    d = _dummy_inputs()
    out = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d["buy"], d["sell"], d["pv"], d["cons"],
        n_paths=300,
        rng=np.random.default_rng(6),
    )
    # Mean total PV across paths
    total_pv_mean = d["pv"].sum()  # sample mean ≈ point sum
    # The kernel reports per-path mean self+exp; sum must approximately
    # equal the mean total PV (sampler is mean-preserving).
    assert (out["pv_self_consumed_kwh"] + out["pv_exported_kwh"]) == \
        pytest.approx(total_pv_mean, rel=0.05)


def test_compute_high_consumption_reduces_export() -> None:
    """When consumption is much higher than PV, surplus collapses."""
    d = _dummy_inputs()
    d_low_load = dict(d)
    d_high_load = dict(d)
    d_high_load["cons"] = np.full(24, 10.0)  # way above PV peak
    out_low = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d_low_load["buy"], d_low_load["sell"], d_low_load["pv"],
        d_low_load["cons"],
        n_paths=200, rng=np.random.default_rng(7),
    )
    out_high = pv_aware_cvar.compute_pv_aware_cvar_for_day(
        d_high_load["buy"], d_high_load["sell"], d_high_load["pv"],
        d_high_load["cons"],
        n_paths=200, rng=np.random.default_rng(8),
    )
    assert out_high["pv_exported_kwh"] < out_low["pv_exported_kwh"]
    assert out_high["pv_self_consumed_kwh"] >= out_low["pv_self_consumed_kwh"]
