"""Tests for DkDtACIBundle — per-(direction, k) DtACI on D(i) statistics."""
from __future__ import annotations

import importlib.util
import math
import random
import sys
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
DkDtACIBundle = dk_dtaci.DkDtACIBundle


def _synth_day(rng: random.Random, base_mean: float = 50.0):
    """Build one day's actual (cheap[24], peak[24]) from sorted prices."""
    prices = sorted([rng.gauss(base_mean, 15) for _ in range(24)])
    cheap = []
    peak = []
    sum_lo = 0.0
    sum_hi = 0.0
    desc = list(reversed(prices))
    for k in range(24):
        sum_lo += prices[k]
        sum_hi += desc[k]
        cheap.append(sum_lo / (k + 1))
        peak.append(sum_hi / (k + 1))
    return cheap, peak


# ── Construction ──────────────────────────────────────────────────


def test_bundle_has_48_instances():
    b = DkDtACIBundle()
    assert len(b.instances) == 48
    assert all(f"cheap_{k}" in b.instances for k in range(1, 25))
    assert all(f"peak_{k}" in b.instances for k in range(1, 25))


def test_bundle_each_instance_has_bias_corrector():
    b = DkDtACIBundle()
    for inst in b.instances.values():
        assert inst.bias_corrector is not None


# ── Update / predict ──────────────────────────────────────────────


def test_predict_intervals_returns_correct_shape():
    b = DkDtACIBundle()
    cheap = [10.0 + i for i in range(24)]
    peak = [80.0 - 0.5 * i for i in range(24)]
    bands = b.predict_intervals(cheap, peak)
    assert set(bands.keys()) == {"cheap", "peak"}
    for direction in ("cheap", "peak"):
        assert set(bands[direction].keys()) == {"lower", "point", "upper"}
        for series in bands[direction].values():
            assert len(series) == 24


def test_cold_start_intervals_collapse_to_point():
    """Before each instance's min_warmup, intervals collapse to point."""
    b = DkDtACIBundle(min_warmup=10)
    cheap = [10.0 + i for i in range(24)]
    peak = [80.0 - 0.5 * i for i in range(24)]
    bands = b.predict_intervals(cheap, peak)
    for direction, fc in (("cheap", cheap), ("peak", peak)):
        for k_idx in range(24):
            assert bands[direction]["lower"][k_idx] == fc[k_idx]
            assert bands[direction]["upper"][k_idx] == fc[k_idx]


def test_update_rejects_wrong_lengths():
    b = DkDtACIBundle()
    cheap_short = [10.0] * 23
    peak = [80.0] * 24
    with pytest.raises(ValueError, match="cheap arrays must be length 24"):
        b.update(cheap_short, peak, [10.0] * 24, peak)
    peak_short = [80.0] * 23
    with pytest.raises(ValueError, match="peak arrays must be length 24"):
        b.update([10.0] * 24, peak_short, [10.0] * 24, [80.0] * 24)


def test_per_k_coverage_converges_to_target_under_iid_residuals():
    """Walk forward over synthetic D(k) data with i.i.d. forecast noise.
    Per-k realised coverage should land within +/- 5 pp of target after
    enough days."""
    rng = random.Random(20260428)
    target = 0.9
    b = DkDtACIBundle(target_coverage=target, window=200, min_warmup=20)
    n = 1000
    eval_start = 200
    covered = {(d, k): 0 for d in ("cheap", "peak") for k in range(1, 25)}
    predicted = {(d, k): 0 for d in ("cheap", "peak") for k in range(1, 25)}
    for day in range(n):
        a_cheap, a_peak = _synth_day(rng)
        # Forecast = actual + i.i.d. Gaussian noise (no bias)
        f_cheap = [a + rng.gauss(0, 3) for a in a_cheap]
        f_peak = [a + rng.gauss(0, 5) for a in a_peak]
        if day >= eval_start:
            bands = b.predict_intervals(f_cheap, f_peak)
            for direction, ac in (("cheap", a_cheap), ("peak", a_peak)):
                for k in range(1, 25):
                    low = bands[direction]["lower"][k - 1]
                    high = bands[direction]["upper"][k - 1]
                    predicted[(direction, k)] += 1
                    if low <= ac[k - 1] <= high:
                        covered[(direction, k)] += 1
        b.update(f_cheap, f_peak, a_cheap, a_peak)
    for (direction, k), n_pred in predicted.items():
        cov = covered[(direction, k)] / n_pred
        assert 0.85 <= cov <= 0.95, (
            f"{direction}[{k}] coverage {cov:.3f} outside [0.85, 0.95]"
        )


# ── Diagnostics ───────────────────────────────────────────────────


def test_diagnostics_returns_full_per_k_breakdown():
    rng = random.Random(7)
    b = DkDtACIBundle()
    for _ in range(60):
        a_cheap, a_peak = _synth_day(rng)
        f_cheap = [a + rng.gauss(0, 2) for a in a_cheap]
        f_peak = [a + rng.gauss(0, 3) for a in a_peak]
        b.update(f_cheap, f_peak, a_cheap, a_peak)
    diag = b.diagnostics()
    assert diag["n_total_instances"] == 48
    assert diag["n_warm_instances"] == 48
    for direction in ("cheap", "peak"):
        for k in range(1, 25):
            d = diag["per_k"][direction][k]
            for key in ("coverage", "alpha_agg", "bias_ema", "dominant_gamma",
                         "weight_entropy_bits", "half_width", "n_updates",
                         "bias_warm"):
                assert key in d, f"{direction}[{k}] missing {key}"


# ── Persistence ───────────────────────────────────────────────────


def test_to_from_dict_round_trip_preserves_intervals():
    rng = random.Random(99)
    b = DkDtACIBundle(window=100, min_warmup=10)
    for _ in range(150):
        a_cheap, a_peak = _synth_day(rng)
        f_cheap = [a + rng.gauss(0, 4) for a in a_cheap]
        f_peak = [a + rng.gauss(0, 6) for a in a_peak]
        b.update(f_cheap, f_peak, a_cheap, a_peak)
    f_test_c = [12.0 + i for i in range(24)]
    f_test_p = [80.0 - 0.5 * i for i in range(24)]
    bands_a = b.predict_intervals(f_test_c, f_test_p)

    state = b.to_dict()
    b2 = DkDtACIBundle.from_dict(state)
    bands_b = b2.predict_intervals(f_test_c, f_test_p)

    for direction in ("cheap", "peak"):
        for series in ("lower", "point", "upper"):
            for v_a, v_b in zip(bands_a[direction][series],
                                bands_b[direction][series]):
                assert v_a == pytest.approx(v_b, abs=1e-12)


def test_from_dict_missing_instance_cold_starts():
    """A bundle restored from a state dict missing an instance still
    returns a valid (cold-start) DtACI for that key."""
    b = DkDtACIBundle()
    state = b.to_dict()
    del state["instances"]["cheap_5"]
    b2 = DkDtACIBundle.from_dict(state)
    assert "cheap_5" in b2.instances
    # Cold instance returns point-only bands
    bands = b2.predict_intervals([10.0] * 24, [80.0] * 24)
    assert bands["cheap"]["lower"][4] == bands["cheap"]["upper"][4]


def test_from_dict_rejects_wrong_version():
    with pytest.raises(ValueError, match="bundle version"):
        DkDtACIBundle.from_dict({"version": 99, "instances": {}})


# ── Bias correction realised on biased data ───────────────────────


def test_bias_correction_per_k_reduces_mae_on_biased_peak():
    """Inject a +20 EUR/MWh bias on the peak[1] forecast across many days.
    The DtACI bundle's per-instance bias EMA on peak_1 should drift
    toward +20 (positive = forecast higher than actual? NO - EMA tracks
    actual - forecast, which is -20 if we bias the forecast UP).
    Either way, |bias_ema| should grow on peak[1] and stay near zero
    on cheap[1]."""
    rng = random.Random(1)
    b = DkDtACIBundle(bias_warmup_steps=14, cadence_per_day=1)
    for _ in range(300):
        a_cheap, a_peak = _synth_day(rng)
        f_cheap = list(a_cheap)
        # Bias only peak[1]: forecast 20 EUR/MWh too high
        f_peak = list(a_peak)
        f_peak[0] += 20.0
        b.update(f_cheap, f_peak, a_cheap, a_peak)
    diag = b.diagnostics()
    bias_peak_1 = diag["per_k"]["peak"][1]["bias_ema"]
    bias_cheap_1 = diag["per_k"]["cheap"][1]["bias_ema"]
    # bias_estimate tracks (actual - forecast), so when we forecast HIGH,
    # the EMA should be NEGATIVE (~-20)
    assert bias_peak_1 == pytest.approx(-20.0, abs=3.0), (
        f"peak[1] bias_ema {bias_peak_1:+.2f} not near -20"
    )
    assert abs(bias_cheap_1) < 2.0, (
        f"cheap[1] bias_ema {bias_cheap_1:+.2f} drifted from zero"
    )
