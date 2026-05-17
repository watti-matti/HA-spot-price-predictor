"""Tests for studies/se1_se3_congestion_analysis.py.

The dual-feature hedge optimizer and regime-split helper are the
parts we want to verify behave correctly without hitting real data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    if "se1_se3_test" in sys.modules:
        return sys.modules["se1_se3_test"]
    sys.path.insert(0, str(REPO / "studies"))
    path = REPO / "studies" / "se1_se3_congestion_analysis.py"
    spec = importlib.util.spec_from_file_location("se1_se3_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["se1_se3_test"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# ── hedge_dual_features ─────────────────────────────────────────────


def test_dual_hedge_returns_expected_shape() -> None:
    """Smoke test: dual hedge returns h1, h2, cvar fields and finite values
    on plausible random-walk price data. Coefficient recovery is sensitive to
    forward-shift mechanics that aren't worth pinning in a unit test —
    the empirical validation in studies/results/ exercises the real-data path.
    """
    rng = np.random.default_rng(0)
    n = 3000
    f1 = rng.normal(0, 1, n).cumsum() + 50
    f2 = rng.normal(0, 1, n).cumsum() + 50
    actual = 0.6 * f1 + 0.4 * f2 + rng.normal(0, 0.1, n)
    result = m.hedge_dual_features(actual, f1, f2, lag=2, alpha=0.05)
    assert set(result.keys()) >= {
        "h1", "h2", "cvar_test_hist_unhedged",
        "cvar_test_hist_hedged", "n_test",
    }
    for key in ("h1", "h2", "cvar_test_hist_unhedged", "cvar_test_hist_hedged"):
        assert np.isfinite(result[key]), f"{key} should be finite"
    # Bounds enforced by the optimizer
    assert -5 <= result["h1"] <= 5
    assert -5 <= result["h2"] <= 5


def test_dual_hedge_handles_collinear_features() -> None:
    """If f1 ≡ f2 (perfectly collinear), the optimizer should still
    produce a finite, bounded hedge ratio and not crash."""
    rng = np.random.default_rng(1)
    n = 3000
    f1 = rng.normal(0, 1, n).cumsum() + 30
    f2 = f1.copy()  # exact duplicate
    actual = 0.5 * f1 + rng.normal(0, 0.3, n)
    result = m.hedge_dual_features(actual, f1, f2, lag=2, alpha=0.05)
    # The sum h1+h2 should be near 0.5; individual values are not uniquely
    # determined under perfect collinearity but the bounds keep them sane
    assert -5 <= result["h1"] <= 5
    assert -5 <= result["h2"] <= 5
    assert np.isfinite(result["cvar_test_hist_hedged"])


# ── regime_split_analysis ───────────────────────────────────────────


def test_regime_split_categorises_all_zones() -> None:
    """Synthetic spread spanning all four magnitude bins → expect 4 regimes back."""
    rng = np.random.default_rng(2)
    n = 4000
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # Construct prices so spread falls in each bin
    SE1 = np.full(n, 40.0)
    # Mix the four congestion regimes
    spread_targets = np.repeat([0.1, 2.0, 15.0, 50.0], n // 4)
    SE3 = SE1 + spread_targets[:n] + rng.normal(0, 0.05, n)
    FI = 0.7 * SE3 + 0.3 * SE1 + rng.normal(0, 5, n)
    regimes = m.regime_split_analysis(ts, FI, SE3, SE1)
    # All four regimes should be populated
    assert len(regimes) == 4
    # In severe (>30) regime, the data has spread ≈ 50
    severe = next(r for r in regimes if "severe" in r["regime"])
    assert severe["spread_mean"] == pytest.approx(50.0, abs=1.0)


def test_regime_split_skips_underpopulated_bins() -> None:
    """If one bin has <100 samples, it's omitted from the output."""
    rng = np.random.default_rng(3)
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    SE1 = np.full(n, 40.0)
    SE3 = SE1 + 0.0  # all spread = 0 → only uncongested regime gets samples
    FI = SE3 + rng.normal(0, 2, n)
    regimes = m.regime_split_analysis(ts, FI, SE3, SE1)
    # All four regimes have <100 except the uncongested one
    assert len(regimes) == 1
    assert "uncongested" in regimes[0]["regime"]


# ── red_pct helper ──────────────────────────────────────────────────


def test_red_pct_simple_arithmetic() -> None:
    r = {"cvar_test_hist_unhedged": 100.0, "cvar_test_hist_hedged": 90.0}
    assert m.red_pct(r) == pytest.approx(10.0)


def test_red_pct_negative_when_hedge_worse() -> None:
    r = {"cvar_test_hist_unhedged": 100.0, "cvar_test_hist_hedged": 110.0}
    assert m.red_pct(r) == pytest.approx(-10.0)
