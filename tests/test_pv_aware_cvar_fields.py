"""Tests for the PV-aware CVaR fields published since v2.11.9.

Covers the relationships the dashboards rely on:
  * worst-5% (CVaR) >= expected (mean)
  * p5 <= p50 <= p95 and p5 <= mean <= p95
  * no-PV grid baseline >= PV-aware expected cost (PV only lowers the bill)
and a source guard that the coordinator actually publishes the new keys.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "custom_components" / "spot_price_predictor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_load("custom_components.spot_price_predictor.pv_cost_kernel",
      PKG / "pv_cost_kernel.py")
_cvar = _load("custom_components.spot_price_predictor.pv_aware_cvar",
              PKG / "pv_aware_cvar.py")
compute_pv_aware_cvar_for_day = _cvar.compute_pv_aware_cvar_for_day


def _synthetic_day(seed: int = 0):
    rng = np.random.default_rng(seed)
    buy = np.full(24, 0.15) + rng.uniform(-0.03, 0.03, 24)   # EUR/kWh
    sell = buy - 0.10                                        # positive
    cons = np.full(24, 1.0)
    hours = np.arange(24)
    pv = np.maximum(0.0, np.sin((hours - 6) / 12 * np.pi)) * 3.0  # midday bell
    return buy, sell, pv, cons


def test_cvar_field_relationships():
    buy, sell, pv, cons = _synthetic_day(seed=0)
    out = compute_pv_aware_cvar_for_day(
        buy, sell, pv, cons, rng=np.random.default_rng(1))
    # Worst-5% tail mean must be >= the overall mean.
    assert out["cvar95_eur_kwh"] >= out["mean_eur_kwh"] - 1e-9
    # Quantile ordering, and mean within the band.
    assert out["p5_eur_kwh"] <= out["p50_eur_kwh"] <= out["p95_eur_kwh"] + 1e-9
    assert out["p5_eur_kwh"] - 1e-9 <= out["mean_eur_kwh"] <= out["p95_eur_kwh"] + 1e-9


def test_grid_baseline_ge_pv_expected():
    """The no-PV grid cost (consumption-weighted consumer price, exactly how
    the coordinator computes `grid_cost_eur_kwh`) is >= the PV-aware expected
    cost — PV can only reduce the expected bill when the sell price is
    positive."""
    buy, sell, pv, cons = _synthetic_day(seed=3)
    out = compute_pv_aware_cvar_for_day(
        buy, sell, pv, cons, rng=np.random.default_rng(2))
    grid_cost = float((cons * buy).sum() / cons.sum())
    assert grid_cost >= out["mean_eur_kwh"] - 1e-9, (
        f"grid {grid_cost} should be >= PV-aware mean {out['mean_eur_kwh']}")


def test_coordinator_publishes_new_cvar_fields():
    src = (PKG / "coordinator.py").read_text(encoding="utf-8")
    for field in ("pv_aware_mean_eur_kwh", "pv_aware_p5_eur_kwh",
                  "pv_aware_p95_eur_kwh", "grid_cost_eur_kwh"):
        assert f'day_entry["{field}"]' in src, (
            f"coordinator must publish {field} for the parallel CVaR cards")


def test_cvar_widget_binds_to_published_fields():
    """The composite CVaR widget (expected · 90% band · worst-5% · no-PV
    baseline) must read exactly the fields the coordinator publishes —
    guards against a rename drifting the card and sensor apart."""
    dash = (REPO / "docs" / "yaml_examples"
            / "forecast_v2_11_dashboard.yaml").read_text(encoding="utf-8")
    for field in ("d.pv_aware_mean_eur_kwh", "d.pv_aware_p5_eur_kwh",
                  "d.pv_aware_p95_eur_kwh", "d.pv_aware_cvar95_eur_kwh",
                  "d.grid_cost_eur_kwh"):
        assert field in dash, f"CVaR widget must bind {field}"
