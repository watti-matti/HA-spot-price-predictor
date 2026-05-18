"""Wiring tests for the DtACI UI integration (Phase B v2 deployment).

These tests don't spin up a real Home Assistant — they exercise the
integration helpers and bundle reconciliation flow with synthetic data
to confirm:

* `dtaci_integration.attach_dk_intervals` produces the four band fields
  on a forecast entry once the bundle is warmed up.
* The bundle's `diagnostics()` output has the structure the Lovelace
  card consumes (zones[fi].per_k.cheap[k] / per_k.peak[k] with the
  expected scalar keys).
* Reconciliation flow: feeding (forecast, actual) D(i) pairs increases
  `n_updates` and eventually warms each instance.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import pytest


_PKG = Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_load("custom_components.spot_price_predictor.bias_corrector",
      _PKG / "bias_corrector.py")
_load("custom_components.spot_price_predictor.dtaci",
      _PKG / "dtaci.py")
dk_dtaci = _load("custom_components.spot_price_predictor.dk_dtaci",
                 _PKG / "dk_dtaci.py")
integration = _load(
    "custom_components.spot_price_predictor.dtaci_integration",
    _PKG / "dtaci_integration.py",
)
DkDtACIBundle = dk_dtaci.DkDtACIBundle


def _synth_dk(rng: random.Random, base: float = 0.10):
    """Synthetic (cheap[24], peak[24]) for one day in EUR/kWh range."""
    prices = sorted([rng.gauss(base, 0.04) for _ in range(24)])
    cheap = []; peak = []
    sum_lo = sum_hi = 0.0
    desc = list(reversed(prices))
    for k in range(24):
        sum_lo += prices[k]; sum_hi += desc[k]
        cheap.append(sum_lo / (k + 1))
        peak.append(sum_hi / (k + 1))
    return cheap, peak


# ── attach_dk_intervals shape ─────────────────────────────────────


def test_attach_dk_intervals_writes_four_fields_when_warm():
    rng = random.Random(20260428)
    bundle = DkDtACIBundle(min_warmup=10, bias_warmup_steps=5)
    # Warm up
    for _ in range(40):
        a_c, a_p = _synth_dk(rng)
        f_c = [v + rng.gauss(0, 0.005) for v in a_c]
        f_p = [v + rng.gauss(0, 0.008) for v in a_p]
        bundle.update(f_c, f_p, a_c, a_p)
    f_c, f_p = _synth_dk(rng)
    daily = [{
        "date": "2026-04-29", "source": "forecast",
        "dk_cheap_eur_kwh": f_c, "dk_peak_eur_kwh": f_p,
    }]
    integration.attach_dk_intervals(bundle, daily)
    d = daily[0]
    for key in ("dk_cheap_lower_eur_kwh", "dk_cheap_upper_eur_kwh",
                 "dk_peak_lower_eur_kwh", "dk_peak_upper_eur_kwh"):
        assert key in d, f"missing band field {key}"
        assert len(d[key]) == 24
    # Lower < forecast < upper at every k
    for k in range(24):
        assert d["dk_cheap_lower_eur_kwh"][k] <= f_c[k] + 1e-9
        assert d["dk_cheap_upper_eur_kwh"][k] >= f_c[k] - 1e-9


def test_attach_dk_intervals_skips_entry_with_short_arrays():
    """Defensive: entries missing the new schema shouldn't crash."""
    bundle = DkDtACIBundle()
    daily = [
        {"date": "2026-04-29"},                              # no arrays
        {"date": "2026-04-30", "dk_cheap_eur_kwh": [0.1] * 5},  # too short
    ]
    integration.attach_dk_intervals(bundle, daily)
    for d in daily:
        assert "dk_cheap_lower_eur_kwh" not in d


# ── diagnostics() shape (consumed by Lovelace card) ──────────────


def test_diagnostics_shape_matches_card_consumer():
    """The Lovelace card reads:
        diag.zones.fi.per_k.<direction>.<k>.<param>
    where param ∈ {coverage, alpha_agg, bias_ema, dominant_gamma,
                   weight_entropy_bits, half_width, n_updates, bias_warm}.
    """
    bundle = DkDtACIBundle()
    diag = bundle.diagnostics()
    # Top level keys for the markdown header tiles
    for key in ("target_coverage", "n_warm_instances",
                "n_total_instances", "mean_coverage", "mean_width", "per_k"):
        assert key in diag, f"top-level missing {key}"
    # Per-k keys
    for direction in ("cheap", "peak"):
        for k in range(1, 25):
            d = diag["per_k"][direction][k]
            for key in ("coverage", "alpha_agg", "bias_ema",
                         "dominant_gamma", "weight_entropy_bits",
                         "half_width", "n_updates", "bias_warm"):
                assert key in d, f"per_k.{direction}[{k}] missing {key}"


def test_diagnostics_serialises_to_json():
    """The full diag dict goes through the HA state machine which JSON-
    serialises everything. No tuples, sets, datetimes etc."""
    rng = random.Random(0)
    bundle = DkDtACIBundle()
    for _ in range(20):
        a_c, a_p = _synth_dk(rng)
        bundle.update(a_c, a_p, a_c, a_p)
    diag = bundle.diagnostics()
    s = json.dumps(diag)  # must not raise
    assert "per_k" in s


# ── Persistence helpers ──────────────────────────────────────────


def test_load_or_create_bundle_round_trip(tmp_path: Path):
    state_path = tmp_path / "dtaci_dk_fi.json"
    bundle = integration.load_or_create_bundle(state_path)
    assert (bundle.n_total_instances if hasattr(bundle, "n_total_instances")
            else len(bundle.instances)) == 48
    # Run a few updates, save
    rng = random.Random(2)
    for _ in range(15):
        a_c, a_p = _synth_dk(rng)
        bundle.update(a_c, a_p, a_c, a_p)
    integration.save_bundle(state_path, bundle)
    assert state_path.exists()
    # Load and verify the n_updates persisted
    bundle2 = integration.load_or_create_bundle(state_path)
    sample = bundle2.instances["cheap_4"]
    assert sample.n_updates == 15


def test_atomic_write_does_not_corrupt_on_overwrite(tmp_path: Path):
    """save_bundle uses a temp file + os.replace, so even mid-write
    interruption can't leave a half-written JSON. Smoke test the
    round-trip."""
    p = tmp_path / "dtaci_dk_fi.json"
    bundle1 = DkDtACIBundle()
    integration.save_bundle(p, bundle1)
    # Overwrite with a different bundle
    bundle2 = DkDtACIBundle(target_coverage=0.95)
    integration.save_bundle(p, bundle2)
    # Read back: target_coverage must reflect the second write
    with open(p) as f:
        d = json.load(f)
    assert d["target_coverage"] == 0.95


# ── End-to-end: full reconciliation flow on a synthetic year ─────


def test_full_year_reconciliation_warms_bundle_and_produces_bands():
    """Walk forward 365 days of synthetic forecast/actual pairs.
    All 48 instances must warm up and produce non-trivial bands."""
    rng = random.Random(42)
    bundle = DkDtACIBundle(min_warmup=14, bias_warmup_steps=14)
    for day in range(365):
        a_c, a_p = _synth_dk(rng, base=0.10 + 0.01 * (day // 30))  # slow drift
        f_c = [v + rng.gauss(0, 0.003) for v in a_c]
        f_p = [v + rng.gauss(0, 0.005) for v in a_p]
        bundle.update(f_c, f_p, a_c, a_p)

    diag = bundle.diagnostics()
    assert diag["n_warm_instances"] == diag["n_total_instances"] == 48
    # All instances should have non-zero half-width once warm
    for direction in ("cheap", "peak"):
        for k in range(1, 25):
            r = diag["per_k"][direction][k]
            assert r["half_width"] > 0, (
                f"{direction}[{k}] half_width is zero after warmup"
            )
            assert 0.7 <= r["coverage"] <= 1.0, (
                f"{direction}[{k}] coverage proxy {r['coverage']} "
                "outside plausible range"
            )
