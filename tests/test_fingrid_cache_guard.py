"""A failed Fingrid fetch must never be cached.

On 2026-10-12 a batch retrain hit Fingrid rate limiting and
`fetch_fingrid` cached the empty result, writing 2-byte `[]` files for
datasets 248 and 267. Every later run then read `[]` from disk with no
network call, so a transient 429 became a permanent failure and the
70/30 split raised `IndexError: index 0 is out of bounds` forever.

Deleting the cache files fixes a given machine; this guard stops it
recurring. The dataset itself was never at fault — a run against 248
succeeds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "studies"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_module(tmp_repo: Path):
    """Load the study with REPO pointed at a throwaway directory."""
    spec = importlib.util.spec_from_file_location(
        "scsm_under_test", REPO / "studies" / "solar_clear_sky_submodel.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scsm_under_test"] = mod
    spec.loader.exec_module(mod)
    mod.REPO = tmp_repo
    return mod


def test_empty_fingrid_result_is_not_cached(tmp_path) -> None:
    """The actual 2026-10-12 defect: a failed fetch must raise rather
    than write a 2-byte `[]` cache that poisons every later run."""
    mod = _load_module(tmp_path)
    cache = tmp_path / "studies" / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    mod.CACHE_DIR = cache

    import datetime as _dt
    start = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
    end = _dt.datetime(2024, 1, 2, tzinfo=_dt.timezone.utc)

    class _Resp:
        status_code = 500
        text = "upstream unavailable"

    mod.requests = type("R", (), {"get": staticmethod(lambda *a, **k: _Resp())})

    with pytest.raises(RuntimeError, match="Not caching the empty result"):
        mod.fetch_fingrid(248, start, end, "dummy-key")

    assert not list(cache.glob("*.json")), (
        "an empty Fingrid result was cached, which poisons later runs")
