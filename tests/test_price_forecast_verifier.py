"""Tests for the learned per-lead-time price-forecast verifier (v2.14.0).

The verifier replaces the static `price_rel_std_for_lead` heuristic
with a site-specific profile learned from forecast-vs-realized error.
Covered here:
  * cold-start returns the static prior byte-for-byte
  * a reconciled cleared day (forecast == realized) drives the learned
    error toward zero — the "0 for cleared days" property from data
  * a persistently-wrong forecast day drives the learned rel_std up
  * shrinkage: the published value moves prior → learned as n grows
  * persistence round-trips through to_dict/from_dict and load/save
  * schema-version / malformed state cold-starts instead of crashing
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "custom_components" / "spot_price_predictor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# pv_aware_cvar supplies the heuristic prior the verifier shrinks toward.
_load("custom_components.spot_price_predictor.pv_cost_kernel",
      PKG / "pv_cost_kernel.py")
_cvar = _load("custom_components.spot_price_predictor.pv_aware_cvar",
              PKG / "pv_aware_cvar.py")
_pfv = _load("custom_components.spot_price_predictor.price_forecast_verifier",
             PKG / "price_forecast_verifier.py")

PriceForecastVerifier = _pfv.PriceForecastVerifier
price_rel_std_for_lead = _cvar.price_rel_std_for_lead


def _flat(v: float) -> list[float]:
    return [v] * 24


# ── cold start ────────────────────────────────────────────────────


def test_cold_start_returns_static_prior():
    v = PriceForecastVerifier()
    for lead in range(7):
        assert v.rel_std_for_lead(lead) == price_rel_std_for_lead(lead)


def test_record_before_reconcile_does_not_change_output():
    v = PriceForecastVerifier()
    v.record_forecast("2026-07-20", 3, _flat(0.15))
    # No realized data yet — still the prior.
    assert v.rel_std_for_lead(3) == price_rel_std_for_lead(3)


# ── reconciliation drives learning ────────────────────────────────


def test_perfect_forecast_learns_near_zero():
    """A cleared day whose forecast equalled the realized price teaches
    the learner that this lead has ~zero error."""
    v = PriceForecastVerifier(min_warmup=1)
    price = [0.10 + 0.01 * h for h in range(24)]
    # Reconcile the same date several times is idempotent; use distinct
    # dates to accumulate samples at lead 2.
    for i, date in enumerate(["2026-07-1%d" % d for d in range(1, 6)]):
        v.record_forecast(date, 2, price)
        n = v.reconcile(date, price)
        assert n == 1
    assert v.n[2] == 5
    # Learned error ~0; blended value well below the 0.10 prior.
    assert v.rel_std_ewma[2] < 1e-6
    assert v.rel_std_for_lead(2) < price_rel_std_for_lead(2)


def test_biased_forecast_learns_large_rel_std():
    """A lead whose forecast is consistently 30% under realized should
    learn a rel_std near 0.30 and publish above the prior."""
    v = PriceForecastVerifier(min_warmup=1)
    fc = _flat(0.10)
    rz = _flat(0.13)  # realized 30% higher every hour
    for d in range(1, 7):
        date = "2026-07-0%d" % d
        v.record_forecast(date, 5, fc)
        v.reconcile(date, rz)
    # RMS relative error = 0.30 exactly (uniform); EWMA converges to it.
    assert abs(v.rel_std_ewma[5] - 0.30) < 0.05
    assert v.rel_std_for_lead(5) > price_rel_std_for_lead(5)


def test_reconcile_is_idempotent_per_date():
    v = PriceForecastVerifier(min_warmup=1)
    v.record_forecast("2026-07-20", 4, _flat(0.10))
    assert v.reconcile("2026-07-20", _flat(0.12)) == 1
    # Second call for the same date is a no-op (already reconciled).
    assert v.reconcile("2026-07-20", _flat(0.12)) == 0
    assert v.n[4] == 1


def test_multiple_leads_for_one_date():
    """The same target date, forecast at several leads, updates each
    lead's accumulator when it clears."""
    v = PriceForecastVerifier(min_warmup=1)
    date = "2026-07-20"
    v.record_forecast(date, 6, _flat(0.10))
    v.record_forecast(date, 3, _flat(0.11))
    v.record_forecast(date, 0, _flat(0.12))
    assert v.reconcile(date, _flat(0.12)) == 3
    assert set(v.n) == {0, 3, 6}
    assert v.n[0] == 1 and v.n[3] == 1 and v.n[6] == 1
    # Lead 0 forecast was exact → ~0 error; lead 6 was most wrong.
    assert v.rel_std_ewma[0] < v.rel_std_ewma[6]


# ── shrinkage behaviour ───────────────────────────────────────────


def test_shrinkage_moves_from_prior_to_learned():
    v = PriceForecastVerifier(min_warmup=4)
    prior = price_rel_std_for_lead(5)
    fc, rz = _flat(0.10), _flat(0.15)  # 50% under → rel_std 0.50
    published = []
    for d in range(1, 9):
        date = "2026-07-0%d" % d if d < 10 else "2026-07-%d" % d
        v.record_forecast(date, 5, fc)
        v.reconcile(date, rz)
        published.append(v.rel_std_for_lead(5))
    # Monotonically increasing toward the learned 0.50 as n grows.
    assert published[0] > prior
    assert published[-1] > published[0]
    assert published[-1] < 0.50  # never overshoots the learned value


# ── persistence ───────────────────────────────────────────────────


def test_to_from_dict_round_trip():
    v = PriceForecastVerifier(min_warmup=2)
    v.record_forecast("2026-07-20", 2, _flat(0.10))
    v.reconcile("2026-07-20", _flat(0.13))
    v.record_forecast("2026-07-21", 3, _flat(0.10))  # still pending
    raw = v.to_dict()
    v2 = PriceForecastVerifier.from_dict(raw)
    assert v2.n == v.n
    assert v2.rel_std_ewma == v.rel_std_ewma
    assert "2026-07-20" in v2.reconciled
    assert "2026-07-21" in v2.pending
    assert v2.rel_std_for_lead(2) == v.rel_std_for_lead(2)


def test_load_save_file_round_trip(tmp_path):
    path = str(tmp_path / "price_rel_std.json")
    v = PriceForecastVerifier(min_warmup=1)
    v.record_forecast("2026-07-20", 4, _flat(0.10))
    v.reconcile("2026-07-20", _flat(0.14))
    _pfv.save(path, v)
    v2 = _pfv.load_or_create(path)
    assert v2.n == v.n
    assert v2.rel_std_for_lead(4) == v.rel_std_for_lead(4)


def test_load_missing_file_cold_starts(tmp_path):
    v = _pfv.load_or_create(str(tmp_path / "does_not_exist.json"))
    assert isinstance(v, PriceForecastVerifier)
    assert v.n == {}


def test_from_dict_bad_schema_cold_starts():
    v = PriceForecastVerifier.from_dict({"schema_version": 999, "n": {"2": 5}})
    assert v.n == {}
    v2 = PriceForecastVerifier.from_dict({"garbage": True})
    assert v2.n == {}
    v3 = PriceForecastVerifier.from_dict(None)
    assert v3.n == {}


# ── robustness ────────────────────────────────────────────────────


def test_short_vectors_are_ignored():
    v = PriceForecastVerifier(min_warmup=1)
    v.record_forecast("2026-07-20", 2, [0.1] * 10)  # too short
    assert "2026-07-20" not in v.pending
    v.record_forecast("2026-07-20", 2, _flat(0.10))
    assert v.reconcile("2026-07-20", [0.1] * 5) == 0  # realized too short


def test_zero_price_hours_excluded():
    """Hours with ~0 forecast price don't blow up the ratio."""
    v = PriceForecastVerifier(min_warmup=1)
    fc = [0.0] * 12 + [0.10] * 12
    rz = [0.0] * 12 + [0.11] * 12
    v.record_forecast("2026-07-20", 3, fc)
    v.reconcile("2026-07-20", rz)
    # Only the 12 non-zero hours count; 10% uniform error → rel_std 0.10.
    assert abs(v.rel_std_ewma[3] - 0.10) < 1e-6


def test_diagnostics_shape():
    v = PriceForecastVerifier(min_warmup=1)
    v.record_forecast("2026-07-20", 2, _flat(0.10))
    v.reconcile("2026-07-20", _flat(0.12))
    diag = v.diagnostics()
    assert diag["enabled"] is True
    assert diag["reconciled_dates"] == 1
    assert "2" in diag["per_lead"]
    row = diag["per_lead"]["2"]
    assert set(row) == {
        "n", "learned_rel_std", "prior_rel_std", "published_rel_std",
    }
    assert row["n"] == 1
