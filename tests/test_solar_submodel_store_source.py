"""The solar sub-model reads PV from the data store, not Fingrid live.

Context: until v2.18.2 this sub-model fetched Fingrid 248 and 267 over
the full four-year window on every run. On 2026-10-12 those fetches hit
rate limiting during a batch retrain, and `fetch_fingrid` cached the
empty result — turning a transient 429 into a permanent failure, since
later runs read `[]` from disk without any network call. The series
were already in the data store as 247 / 267.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def _store(tmp_repo: Path, **cols) -> Path:
    d = tmp_repo / "data_store"
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
    frame = {"solar_forecast_mw": np.linspace(0, 500, 96),
             "solar_capacity_mw": np.full(96, 1800.0)}
    frame.update(cols)
    pd.DataFrame(frame, index=idx).to_parquet(d / "fi_grid_data.parquet")
    return d / "fi_grid_data.parquet"


def test_missing_store_explains_how_to_build_it(tmp_path) -> None:
    mod = _load_module(tmp_path)
    with pytest.raises(FileNotFoundError, match="src.data_store update"):
        mod.load_pv_from_store()


def test_missing_columns_point_at_the_api_key(tmp_path) -> None:
    """Those columns only populate when the data-store update had a key."""
    mod = _load_module(tmp_path)
    _store(tmp_path)
    g = pd.read_parquet(tmp_path / "data_store" / "fi_grid_data.parquet")
    g.drop(columns=["solar_capacity_mw"]).to_parquet(
        tmp_path / "data_store" / "fi_grid_data.parquet")
    with pytest.raises(KeyError, match="FINGRID_API_KEY"):
        mod.load_pv_from_store()


def test_reads_247_and_forward_fills_capacity(tmp_path) -> None:
    mod = _load_module(tmp_path)
    mod.WINDOW_START = pd.Timestamp("2024-01-01", tz="UTC")
    mod.WINDOW_END = pd.Timestamp("2024-01-05", tz="UTC")
    cap = np.full(96, 1800.0)
    cap[10:20] = np.nan            # a reporting gap in the step series
    _store(tmp_path, solar_capacity_mw=cap)

    prod, capacity = mod.load_pv_from_store()

    assert prod.name == "actual_mw"
    assert capacity.name == "capacity_mw"
    assert len(prod) == 96
    assert capacity.notna().all(), "capacity gap was not carried across"
    assert (capacity == 1800.0).all()
    assert prod.index.tz is not None, "index must stay tz-aware"


def test_window_is_respected(tmp_path) -> None:
    mod = _load_module(tmp_path)
    mod.WINDOW_START = pd.Timestamp("2024-01-02", tz="UTC")
    mod.WINDOW_END = pd.Timestamp("2024-01-03", tz="UTC")
    _store(tmp_path)

    prod, _ = mod.load_pv_from_store()

    assert prod.index.min() >= mod.WINDOW_START
    assert prod.index.max() <= mod.WINDOW_END


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
