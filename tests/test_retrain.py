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


def test_retrain_solar_no_longer_requires_a_fingrid_key(tmp_path: Path) -> None:
    """The solar layer used to skip without a Fingrid key, because it
    fetched dataset 248 live. It now reads the national PV series from
    the cached data store (247), so no key is needed and the layer must
    ATTEMPT the refit rather than skip.

    Proven by making the import step raise a sentinel: reaching it at
    all means the key gate is gone. The refit must NOT actually run —
    `solar_clear_sky_submodel.main()` resolves its own paths from the
    real repo and would overwrite the shipped artifact as a test side
    effect (it did, before this was pinned down).
    """
    import os
    saved = os.environ.pop("FINGRID_API_KEY", None)
    sentinel = RuntimeError("reached the import step")

    def _boom(_repo_root):
        raise sentinel

    orig = retrain._ensure_repo_imports
    retrain._ensure_repo_imports = _boom
    try:
        with pytest.raises(RuntimeError) as got:
            retrain.retrain_solar_submodel(
                repo_root=tmp_path, data_dir=tmp_path, fingrid_api_key=None)
        assert got.value is sentinel, "returned early instead of refitting"
    finally:
        retrain._ensure_repo_imports = orig
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


def test_keyless_solar_run_reports_the_error_not_a_skip(tmp_path: Path) -> None:
    """Formerly this asserted a `skipped` marker. Since the layer no
    longer needs a Fingrid key, a keyless run is a real attempt, and a
    failure must be reported as an error so the all-or-nothing rollback
    engages instead of being quietly tolerated.

    The layer is forced to fail at the import step. Letting it reach
    `solar_clear_sky_submodel.main()` is not an option in a test: that
    module resolves paths from the REAL repo, not from `repo_root`, so
    it would refit against the live data store and overwrite the
    shipped artifact — outside this run's `tmp_path` snapshot, so the
    rollback could not undo it. Only visible when the whole suite runs,
    because a sibling test puts studies/ on sys.path first.
    """
    import os
    saved = os.environ.pop("FINGRID_API_KEY", None)

    def _boom(_repo_root):
        raise ModuleNotFoundError("no studies/ under repo_root")

    orig = retrain._ensure_repo_imports
    retrain._ensure_repo_imports = _boom
    try:
        result = retrain.retrain_all(
            layers=["solar"],
            repo_root=tmp_path, data_dir=tmp_path,
            fingrid_api_key=None,
        )
        assert result["ok"] is False
        assert "skipped" not in result["results"]["solar"]
        assert "error" in result["results"]["solar"]
        assert result["rolled_back"] is True
        assert result["rolled_back_because"] == ["solar"]
    finally:
        retrain._ensure_repo_imports = orig
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
