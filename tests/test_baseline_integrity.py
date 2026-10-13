"""The recorded baseline must not silently go stale.

`studies/results/BASELINE.json` pins the reference point that future
trainings are compared against. Its fingerprint covers five inputs, but
only three of them live in git:

    inference_code       tracked  -> checkable here
    shipped_artifacts    tracked  -> checkable here
    harness_code         tracked  -> checkable here
    data_store_parquets  NOT tracked (gitignored)
    weather_cache        NOT tracked (gitignored, per-checkout)

So these tests assert only on the tracked three. Asserting on the other
two would fail on any machine that has not synced the data, which would
train everyone to ignore the failure — the opposite of the point.

Use `python studies/verify_baseline.py` for the full check, including
the untracked inputs, before quoting any number.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "studies" / "results" / "BASELINE.json"

_spec = importlib.util.spec_from_file_location(
    "vb_under_test", REPO / "studies" / "verify_baseline.py")
vb = importlib.util.module_from_spec(_spec)
sys.modules["vb_under_test"] = vb
_spec.loader.exec_module(vb)

TRACKED = ("inference_code", "inference_modules",
           "shipped_artifacts", "harness_code")


def _baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip("no BASELINE.json recorded yet")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("component", TRACKED)
def test_tracked_component_matches_baseline(component: str) -> None:
    """A change to the model, the pipeline or the scoring path without
    re-recording the baseline leaves every quoted number wrong. Fail
    loudly instead."""
    base = _baseline()
    want = base["fingerprint"][component]
    got = vb.fingerprint()[component]
    assert got == want, (
        f"{component} no longer matches BASELINE.json.\n"
        f"  baseline {want}\n  current  {got}\n"
        "If this change was intended, re-run the harness and then "
        "`python studies/verify_baseline.py --record`. Do not edit "
        "BASELINE.json by hand — it must come from a measured run."
    )


def test_baseline_records_the_artifacts_it_measured() -> None:
    """The artifact identities stored for humans must agree with the
    artifacts actually on disk, so a reader cannot be misled by a stale
    version string even if the hashes were somehow regenerated."""
    base = _baseline()
    spp = REPO / "custom_components" / "spot_price_predictor" / "data"
    for name, rec in base["artifacts"].items():
        if not rec.get("present"):
            continue
        on_disk = json.loads((spp / name).read_text(encoding="utf-8"))
        assert rec["version"] == on_disk.get("version"), (
            f"{name}: baseline says version {rec['version']}, disk says "
            f"{on_disk.get('version')}")
        if "n_features" in rec:
            assert rec["n_features"] == len(on_disk.get("ridge_features", [])), (
                f"{name}: baseline says {rec['n_features']} features, disk "
                f"has {len(on_disk.get('ridge_features', []))}")


def test_baseline_carries_measured_results_not_assertions() -> None:
    """A baseline is only useful if it came from a real run. Require the
    evidence: a results hash, a scored-row count, and an eval window."""
    base = _baseline()
    h = base.get("harness") or {}
    assert len(h.get("results_sha256", "")) == 64, (
        "baseline has no harness results hash — it was not recorded from "
        "a measured run")
    assert h.get("n_eval", 0) > 1000, "implausible scored-row count"
    assert h.get("eval_window") and len(h["eval_window"]) == 2
    prod = h["segments"]["ALL"].get("PRODUCTION")
    assert prod and prod["mae"] > 0, "no PRODUCTION measurement recorded"


def test_fingerprint_covers_the_weather_cache() -> None:
    """Regression guard for the 2026-10-13 false positive: a
    branch-vs-production comparison showed a 0.0154 EUR/MWh difference
    that was entirely a weather-cache vintage mismatch. If this key ever
    disappears from the fingerprint, that class of error becomes
    invisible again."""
    fp = vb.fingerprint()
    assert "weather_cache" in fp
    assert "data_store_parquets" in fp
    base = _baseline()
    assert "weather_cache" in base["fingerprint"], (
        "BASELINE.json predates weather-cache tracking — re-record it")


def test_inference_closure_is_derived_not_hand_listed() -> None:
    """The reachable set must come from the import graph.

    A hand-maintained list was wrong in both directions when first
    written — it missed ten reachable modules and included `sensor.py`,
    which cannot affect the forecast. An incomplete list means a change
    to real forecast code does not invalidate the baseline, which is the
    one failure this whole mechanism exists to prevent.
    """
    mods = vb.inference_modules()
    assert len(mods) > 15, (
        f"only {len(mods)} forecast-reachable modules found — the import "
        "closure is probably not resolving intra-package imports")
    # Entry points and the core maths must be in any correct closure.
    for required in ("coordinator.py", "pipeline.py", "features.py",
                     "bias_corrector.py", "seasonal_decomposition.py",
                     "hourly_calibration.py", "price_floor.py"):
        assert required in mods, f"{required} missing from the closure"
    # sensor.py reads coordinator output; it cannot move a forecast.
    assert "sensor.py" not in mods
    # retrain.py is lazily imported by a service handler, never while
    # forecasting, and is deliberately excluded.
    assert "retrain.py" not in mods


def test_baseline_records_which_modules_it_hashed() -> None:
    """The hash alone cannot tell a reader whether the reachable set
    changed. Record the list, and require it to still match."""
    base = _baseline()
    recorded = base["fingerprint"].get("inference_modules")
    assert recorded, "BASELINE.json predates module-list recording"
    assert list(vb.inference_modules()) == list(recorded), (
        "the set of forecast-reachable modules changed.\n"
        f"  baseline {sorted(recorded)}\n"
        f"  current  {sorted(vb.inference_modules())}\n"
        "An import was added to or removed from the forecast path. "
        "Re-run the harness and re-record the baseline."
    )


def test_doc_numbers_are_traceable_to_the_baseline() -> None:
    """Every figure in docs/TRAINING_BASELINE.md must appear in
    BASELINE.json.

    The point of the baseline is that nobody quotes a remembered number.
    A prose document is exactly where remembered numbers creep back in,
    so each value is traced mechanically rather than trusted.
    """
    doc_path = REPO / "docs" / "TRAINING_BASELINE.md"
    if not doc_path.exists():
        pytest.skip("TRAINING_BASELINE.md not present")
    doc = doc_path.read_text(encoding="utf-8")
    base = _baseline()
    fp, h, art = base["fingerprint"], base["harness"], base["artifacts"]

    untraceable = []

    def need(label: str, text: str) -> None:
        if text not in doc:
            untraceable.append(f"{label} ({text!r})")

    for k in ("inference_code", "shipped_artifacts", "harness_code",
              "data_store_parquets", "weather_cache"):
        need(k, fp[k][:16])
    need("data_store_snapshot", fp["data_store_snapshot"])
    need("weather_cache_files", f"({fp['weather_cache_files']} files)")
    need("module count", f"({len(fp['inference_modules'])} modules")
    need("n_eval", f"{h['n_eval']:,}")
    need("eval start", h["eval_window"][0][:10])
    need("eval end", h["eval_window"][1][:10])

    sp = art.get("spike_model_default.json", {})
    if sp.get("present"):
        need("spike version", sp["version"])
        need("spike feature count", f"**{sp['n_features']} features**")
        need("spike train window", sp["train_window"][1][:10])

    for seg, v in h["segments"].items():
        p = v.get("PRODUCTION")
        if not p:
            continue
        need(f"{seg} MAE", f"{p['mae']:.4f}")
        need(f"{seg} bias", f"{abs(p['bias']):.4f}")

    assert not untraceable, (
        "docs/TRAINING_BASELINE.md contains values not found in "
        "BASELINE.json:\n  " + "\n  ".join(untraceable) +
        "\nRe-record the baseline and update the document from it, "
        "rather than editing numbers by hand."
    )
