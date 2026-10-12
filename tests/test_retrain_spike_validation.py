"""The retrain pipeline must not install a spike artifact the pipeline
cannot consume.

Regression test for the v2.18.1 defect: `retrain.retrain_spike_model`
refitted via `v2513_layer4_spike_model`, a v2.5.13-era trainer that
writes a five-feature model (no neighbour prices, no `is_holiday`) over
its own short train window. It overwrote the shipped nine-feature
v2.17.1 artifact with no error raised anywhere — the only symptom was a
degraded forecast, which is exactly the failure mode a test should
catch.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "custom_components" / "spot_price_predictor"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load retrain.py as a standalone file. Importing it as a package member
# would execute custom_components/spot_price_predictor/__init__.py, which
# imports Home Assistant — absent on a training PC, which is precisely why
# scripts/retrain_model.* invokes the file directly rather than with -m.
_spec = importlib.util.spec_from_file_location(
    "spp_retrain", REPO / "custom_components" / "spot_price_predictor"
    / "retrain.py")
retrain = importlib.util.module_from_spec(_spec)
sys.modules["spp_retrain"] = retrain
_spec.loader.exec_module(retrain)

_validate_spike_payload = retrain._validate_spike_payload
RIDGE_FEATURES = retrain.pipeline_ridge_features(REPO)
# slot 0 is "intercept"; the artifact lists only the real columns.
ARTIFACT_FEATURES = RIDGE_FEATURES[1:]

SHIPPED = (REPO / "custom_components" / "spot_price_predictor" / "data"
           / "spike_model_default.json")


def _good() -> dict:
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


def test_shipped_artifact_passes_validation() -> None:
    """The artifact in the repo must satisfy the guard, or the guard is
    wrong rather than the artifact."""
    _validate_spike_payload(_good(), SHIPPED, REPO)


def test_legacy_five_feature_payload_is_rejected() -> None:
    """The exact shape the v2.5.13 trainer produced."""
    bad = _good()
    bad["version"] = "2.5.13"
    bad["ridge_features"] = ["Y_fi_lag168", "is_workday",
                             "Y_sigmoid_wind_rho", "Y_solar_effective",
                             "Y_temp"]
    bad["ridge_coef"] = [0.0] * 6
    with pytest.raises(ValueError, match="feature mismatch"):
        _validate_spike_payload(bad, Path("spike_model_fresh.json"), REPO)


def test_coefficient_count_must_include_intercept() -> None:
    bad = _good()
    bad["ridge_coef"] = [0.0] * len(ARTIFACT_FEATURES)   # intercept missing
    with pytest.raises(ValueError, match="coefficient mismatch"):
        _validate_spike_payload(bad, Path("spike_model_fresh.json"), REPO)


def test_missing_physics_seasonal_is_rejected() -> None:
    """Without this block the runtime deseasonalises wind/solar
    differently from training — a silent train/inference mismatch."""
    bad = _good()
    bad.pop("physics_seasonal", None)
    with pytest.raises(ValueError, match="physics_seasonal"):
        _validate_spike_payload(bad, Path("spike_model_fresh.json"), REPO)


def test_retrain_uses_the_nine_feature_trainer() -> None:
    """Guard the import itself: `v2513_layer4_spike_model` must not come
    back, and the trainer that is used must declare the pipeline's
    feature list."""
    src = (REPO / "custom_components" / "spot_price_predictor"
           / "retrain.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*import\s+(\w+)", src, re.MULTILINE)
    assert "v2513_layer4_spike_model" not in imports, (
        "retrain.py is importing the v2.5.13 trainer again")
    assert "build_fresh_spike_model" in imports

    sys.path.insert(0, str(REPO / "studies"))
    import build_fresh_spike_model as bfsm
    assert list(bfsm.FEATS) == list(ARTIFACT_FEATURES), (
        "the trainer's FEATS and pipeline.RIDGE_FEATURES have diverged")


# ── candidate mode: production must be untouchable ─────────────────

def test_candidate_mode_restores_shipped_artifacts(tmp_path) -> None:
    """A candidate run must leave data_dir byte-identical, even though
    the layer trainers write straight to the shipped artifact path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shipped = data_dir / retrain.ARTIFACT_NAMES["spike"]
    original = b'{"version": "shipped", "ridge_features": []}'
    shipped.write_bytes(original)

    with retrain._artifacts_preserved(data_dir, [shipped.name]):
        shipped.write_bytes(b'{"version": "candidate-overwrote-it"}')
        assert shipped.read_bytes() != original      # trainer did write

    assert shipped.read_bytes() == original, (
        "shipped artifact was not restored after a candidate run")


def test_candidate_mode_restores_even_when_a_layer_raises(tmp_path) -> None:
    """The 2026-10-12 run crashed in the solar layer AFTER the spike
    layer had already overwritten its artifact. Restoration must happen
    on the exception path too."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shipped = data_dir / retrain.ARTIFACT_NAMES["seasonal"]
    original = b'{"version": "shipped"}'
    shipped.write_bytes(original)

    with pytest.raises(RuntimeError):
        with retrain._artifacts_preserved(data_dir, [shipped.name]):
            shipped.write_bytes(b'{"version": "half-written"}')
            raise RuntimeError("solar layer blew up")

    assert shipped.read_bytes() == original


def test_candidate_mode_removes_artifacts_that_did_not_exist_before(
        tmp_path) -> None:
    """A newly created artifact must not silently become 'shipped'."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    created = data_dir / retrain.ARTIFACT_NAMES["solar"]
    assert not created.exists()

    with retrain._artifacts_preserved(data_dir, [created.name]):
        created.write_bytes(b'{"version": "brand new"}')

    assert not created.exists(), (
        "an artifact created during a candidate run was left in data_dir")


# ── all-or-nothing rollback ────────────────────────────────────────

def _stub_layers(monkeypatch, data_dir, *, solar_raises: bool):
    """Replace the three layer refitters with stubs that write a marker
    artifact, so rollback can be observed without a real 2-minute fit."""
    def _seasonal(repo_root, dd):
        (dd / retrain.ARTIFACT_NAMES["seasonal"]).write_bytes(b'{"v":"new"}')
        return {"layer": "seasonal"}

    def _spike(repo_root, dd):
        (dd / retrain.ARTIFACT_NAMES["spike"]).write_bytes(b'{"v":"new"}')
        return {"layer": "spike"}

    def _solar(repo_root, dd, key):
        if solar_raises:
            raise RuntimeError("solar sub-model: no overlapping hours")
        (dd / retrain.ARTIFACT_NAMES["solar"]).write_bytes(b'{"v":"new"}')
        return {"layer": "solar"}

    monkeypatch.setattr(retrain, "retrain_seasonal", _seasonal)
    monkeypatch.setattr(retrain, "retrain_spike_model", _spike)
    monkeypatch.setattr(retrain, "retrain_solar_submodel", _solar)


def _seed(data_dir: Path) -> dict[str, bytes]:
    data_dir.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for name in retrain.ARTIFACT_NAMES.values():
        blob = f'{{"v":"shipped-{name}"}}'.encode()
        (data_dir / name).write_bytes(blob)
        blobs[name] = blob
    return blobs


def test_failed_layer_rolls_back_the_whole_run(monkeypatch, tmp_path) -> None:
    """This is the 2026-10-12 failure: seasonal and spike refit, solar
    blew up. Every artifact must be back to its shipped bytes."""
    data_dir = tmp_path / "data"
    shipped = _seed(data_dir)
    _stub_layers(monkeypatch, data_dir, solar_raises=True)

    res = retrain.retrain_all(repo_root=tmp_path, data_dir=data_dir)

    assert res["ok"] is False
    assert res["rolled_back"] is True
    assert res["rolled_back_because"] == ["solar"]
    assert res["artifacts_installed"] == []
    for name, blob in shipped.items():
        assert (data_dir / name).read_bytes() == blob, (
            f"{name} was not rolled back after a failed run")


def test_successful_run_commits_every_artifact(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    shipped = _seed(data_dir)
    _stub_layers(monkeypatch, data_dir, solar_raises=False)

    res = retrain.retrain_all(repo_root=tmp_path, data_dir=data_dir,
                              fingrid_api_key="x")

    assert res["rolled_back"] is False
    for name in shipped:
        assert (data_dir / name).read_bytes() == b'{"v":"new"}', (
            f"{name} should have been installed on a clean run")


def test_allow_partial_keeps_what_succeeded(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "data"
    shipped = _seed(data_dir)
    _stub_layers(monkeypatch, data_dir, solar_raises=True)

    res = retrain.retrain_all(repo_root=tmp_path, data_dir=data_dir,
                              allow_partial=True)

    assert res["rolled_back"] is False
    assert (data_dir / retrain.ARTIFACT_NAMES["spike"]).read_bytes() \
        == b'{"v":"new"}'
    # solar never wrote, so its shipped copy must still be intact
    assert (data_dir / retrain.ARTIFACT_NAMES["solar"]).read_bytes() \
        == shipped[retrain.ARTIFACT_NAMES["solar"]]


def test_skipped_layer_does_not_roll_back_the_others(
        monkeypatch, tmp_path) -> None:
    """Solar skips when no Fingrid key is given. `ok` reports False for a
    skip, but a skip is not a failure and must not discard the refits
    that did happen."""
    data_dir = tmp_path / "data"
    _seed(data_dir)

    def _solar_skip(repo_root, dd, key):
        return {"layer": "solar", "skipped": True, "reason": "no key"}

    _stub_layers(monkeypatch, data_dir, solar_raises=False)
    monkeypatch.setattr(retrain, "retrain_solar_submodel", _solar_skip)

    res = retrain.retrain_all(repo_root=tmp_path, data_dir=data_dir)

    assert res["ok"] is False          # a skip still reports not-ok
    assert res["rolled_back"] is False  # ...but nothing is discarded
    assert (data_dir / retrain.ARTIFACT_NAMES["seasonal"]).read_bytes() \
        == b'{"v":"new"}'
