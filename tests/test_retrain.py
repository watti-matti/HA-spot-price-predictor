"""Tests for custom_components/spot_price_predictor/retrain.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

# Load retrain.py directly so we don't pull in HA-dependent package init
_rt_spec = importlib.util.spec_from_file_location(
    "spot_price_predictor.retrain",
    REPO / "custom_components" / "spot_price_predictor" / "retrain.py",
)
retrain = importlib.util.module_from_spec(_rt_spec)
sys.modules["spot_price_predictor.retrain"] = retrain
_rt_spec.loader.exec_module(retrain)


# ── Atomic write helper ─────────────────────────────────────────────


def test_atomic_write_json_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "out.json"
    payload = {"hello": "world", "n": 7}
    retrain._atomic_write_json(target, payload)
    assert target.exists()
    assert json.loads(target.read_text()) == payload


def test_atomic_write_json_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    target.write_text('{"old": true}')
    retrain._atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}


def test_atomic_write_json_leaves_no_partial_file_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """If serialisation raises mid-flight, the live file must not change."""
    target = tmp_path / "stable.json"
    target.write_text('{"v": 1}')
    # Make json.dump raise inside the helper
    def _boom(*args, **kwargs):
        raise IOError("simulated disk full")
    monkeypatch.setattr(retrain.json, "dump", _boom)
    with pytest.raises(IOError):
        retrain._atomic_write_json(target, {"v": 2})
    # Original file still intact
    assert json.loads(target.read_text()) == {"v": 1}


# ── Orchestrator surface ────────────────────────────────────────────


def test_retrain_all_default_layers_constant() -> None:
    assert retrain.ALL_LAYERS == ("seasonal", "spike", "solar")


def test_retrain_all_unknown_layer_returns_error(tmp_path: Path) -> None:
    """Layers not in ALL_LAYERS get an error block but don't crash the run."""
    result = retrain.retrain_all(
        layers=["bogus"],
        repo_root=tmp_path, data_dir=tmp_path,
    )
    assert "bogus" in result["results"]
    assert "error" in result["results"]["bogus"]
    assert result["ok"] is False


def test_retrain_all_skips_solar_when_no_key(tmp_path: Path) -> None:
    """Solar layer must skip cleanly when no Fingrid key is provided."""
    # Wipe any env var so the skip path is exercised
    import os
    saved = os.environ.pop("FINGRID_API_KEY", None)
    try:
        result = retrain.retrain_solar_submodel(
            repo_root=tmp_path, data_dir=tmp_path, fingrid_api_key=None)
        assert result["skipped"] is True
        assert "no FINGRID_API_KEY" in result["reason"]
    finally:
        if saved is not None:
            os.environ["FINGRID_API_KEY"] = saved


def test_retrain_all_records_started_and_completed(tmp_path: Path) -> None:
    """Even on errors the timing fields must be present."""
    result = retrain.retrain_all(
        layers=["bogus"],
        repo_root=tmp_path, data_dir=tmp_path,
    )
    assert "started_at" in result
    assert "completed_at" in result
    assert result["started_at"] <= result["completed_at"]


def test_retrain_all_solar_skip_does_not_count_as_ok(tmp_path: Path) -> None:
    """A skipped layer marks the run as not-ok (not all layers completed)."""
    import os
    saved = os.environ.pop("FINGRID_API_KEY", None)
    try:
        result = retrain.retrain_all(
            layers=["solar"],
            repo_root=tmp_path, data_dir=tmp_path,
            fingrid_api_key=None,
        )
        assert result["ok"] is False
        assert result["results"]["solar"]["skipped"] is True
    finally:
        if saved is not None:
            os.environ["FINGRID_API_KEY"] = saved


def test_ensure_repo_imports_is_idempotent(tmp_path: Path) -> None:
    """Calling _ensure_repo_imports twice does not duplicate sys.path entries."""
    n_before = len(sys.path)
    retrain._ensure_repo_imports(tmp_path)
    n_after_first = len(sys.path)
    retrain._ensure_repo_imports(tmp_path)
    n_after_second = len(sys.path)
    assert n_after_second == n_after_first
