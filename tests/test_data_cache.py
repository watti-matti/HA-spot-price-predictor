"""Incremental per-location weather cache (training-time data fetch).

Historical weather never changes, so `_get_location_weather` caches each
location as an append-only parquet keyed by (location, variables) — not by
date range — and on a re-run fetches ONLY the days not already on disk.
This avoids re-downloading years of data after a transient timeout or on a
daily refresh. These tests stub the HTTP layer and assert exactly which
date ranges are fetched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")  # parquet backend for the cache

REPO = Path(__file__).resolve().parent.parent


def _load_data_sources():
    spec = importlib.util.spec_from_file_location(
        "_spp_data_sources", REPO / "src" / "data_sources.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_spp_data_sources"] = mod
    spec.loader.exec_module(mod)
    return mod


_VARS = ("wind_speed_120m", "global_tilted_irradiance_instant", "temperature_2m")
_BASE = {"latitude": 65.0, "longitude": 25.0, "tilt": 45,
         "hourly": ",".join(_VARS), "wind_speed_unit": "ms", "timezone": "UTC"}


def _make_fetch_recorder(mod):
    """Replace _http_get_json with a stub that records fetched ranges and
    returns synthetic hourly data covering exactly the requested range."""
    calls: list[tuple[str, str]] = []

    def fake_get(url, params, timeout, label, **kw):
        calls.append((params["start_date"], params["end_date"]))
        idx = pd.date_range(params["start_date"], params["end_date"] + " 23:00",
                            freq="h", tz="UTC")
        n = len(idx)
        return {"hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx],
            _VARS[0]: [3.0] * n, _VARS[1]: [100.0] * n, _VARS[2]: [10.0] * n,
        }}

    mod._http_get_json = fake_get
    return calls


def _get(mod, cache_dir, start, end):
    return mod._get_location_weather(
        cache_dir, "Loc", "url", dict(_BASE), *_VARS, start, end)


def test_only_new_tail_is_fetched_on_rerun(tmp_path):
    mod = _load_data_sources()
    calls = _make_fetch_recorder(mod)
    d = str(tmp_path)

    df1 = _get(mod, d, "2022-01-01", "2022-01-10")
    assert calls == [("2022-01-01", "2022-01-10")]
    assert len(df1) == 10 * 24

    # Extend the window by 3 days → only the new tail is fetched.
    calls.clear()
    df2 = _get(mod, d, "2022-01-01", "2022-01-13")
    assert calls == [("2022-01-11", "2022-01-13")], "should fetch only the gap"
    assert len(df2) == 13 * 24

    # Same window again → nothing fetched.
    calls.clear()
    df3 = _get(mod, d, "2022-01-01", "2022-01-13")
    assert calls == [], "fully cached window must not re-fetch"
    assert len(df3) == 13 * 24


def test_widening_start_fetches_only_earlier_gap(tmp_path):
    mod = _load_data_sources()
    calls = _make_fetch_recorder(mod)
    d = str(tmp_path)

    _get(mod, d, "2022-01-01", "2022-01-05")
    calls.clear()
    df = _get(mod, d, "2021-12-30", "2022-01-05")
    assert calls == [("2021-12-30", "2021-12-31")], "only the earlier gap"
    assert len(df) == 7 * 24


def test_no_cache_dir_fetches_full_range_each_time(tmp_path):
    mod = _load_data_sources()
    calls = _make_fetch_recorder(mod)

    _get(mod, None, "2022-01-01", "2022-01-03")
    _get(mod, None, "2022-01-01", "2022-01-03")
    assert calls == [("2022-01-01", "2022-01-03"), ("2022-01-01", "2022-01-03")], (
        "with no cache_dir, every call fetches the full range (legacy)")


def test_returned_window_is_sliced_to_request(tmp_path):
    """Cache may hold more than requested; the return is sliced to [start,end]."""
    mod = _load_data_sources()
    _make_fetch_recorder(mod)
    d = str(tmp_path)

    _get(mod, d, "2022-01-01", "2022-01-20")          # cache 20 days
    df = _get(mod, d, "2022-01-05", "2022-01-07")      # ask for 3 days
    assert df.index.min().strftime("%Y-%m-%d") == "2022-01-05"
    assert df.index.max().strftime("%Y-%m-%d") == "2022-01-07"
    assert len(df) == 3 * 24
