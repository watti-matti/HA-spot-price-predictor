"""Tests for the Statnett hydro reservoir client.

Covers parsing, normalisation, error handling, and cache fallback without
requiring network access — uses aiohttp's mock testing utilities and a
synthetic response shaped after the real Statnett API.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_module():
    if "statnett_client_test" in sys.modules:
        return sys.modules["statnett_client_test"]
    path = REPO / "custom_components" / "spot_price_predictor" / "statnett_client.py"
    spec = importlib.util.spec_from_file_location("statnett_client_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["statnett_client_test"] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load_module()


SAMPLE_RESPONSE = {
    "currentYear": [
        {
            "week": 19, "year": 2026, "total": 32.45,
            "no1": 27.42, "no2": 28.19, "no3": 23.06, "no4": 59.78, "no5": 15.10,
        },
        {
            "week": 18, "year": 2026, "total": 32.89,
            "no1": 24.40, "no2": 28.81, "no3": 23.46, "no4": 60.40, "no5": 16.21,
        },
        {
            "week": 17, "year": 2026, "total": 32.34,
            "no1": 19.66, "no2": 28.23, "no3": 22.34, "no4": 60.48, "no5": 16.70,
        },
    ],
    "lastYear": [
        {"week": 19, "year": 2025, "total": 49.17, "no1": 38.97, "no2": 51.32,
         "no3": 53.15, "no4": 61.00, "no5": 32.72},
    ],
}


# ── Normalisation ───────────────────────────────────────────────────


def test_normalize_returns_sorted_newest_first() -> None:
    out = sc.StatnettReservoirClient._normalize(SAMPLE_RESPONSE)
    weeks = out["weeks"]
    assert len(weeks) == 3
    assert weeks[0]["week"] == 19
    assert weeks[1]["week"] == 18
    assert weeks[2]["week"] == 17
    assert out["latest"]["week"] == 19
    assert out["latest"]["total_pct"] == pytest.approx(32.45)


def test_normalize_preserves_all_zones() -> None:
    out = sc.StatnettReservoirClient._normalize(SAMPLE_RESPONSE)
    latest_zones = out["latest"]["zones"]
    assert set(latest_zones) == {"no1", "no2", "no3", "no4", "no5"}
    assert latest_zones["no4"] == pytest.approx(59.78)


def test_normalize_skips_malformed_entries() -> None:
    bad = {
        "currentYear": [
            {"week": "garbage", "year": 2026, "total": 50.0},  # bad week
            {"week": 5, "year": 2026, "total": 45.0,            # good
             "no1": 30.0, "no2": 30.0, "no3": 30.0, "no4": 30.0, "no5": 30.0},
        ],
    }
    out = sc.StatnettReservoirClient._normalize(bad)
    assert len(out["weeks"]) == 1
    assert out["weeks"][0]["week"] == 5


def test_normalize_raises_when_no_data() -> None:
    with pytest.raises(sc.StatnettClientError):
        sc.StatnettReservoirClient._normalize({"currentYear": []})


def test_normalize_raises_when_all_entries_malformed() -> None:
    with pytest.raises(sc.StatnettClientError):
        sc.StatnettReservoirClient._normalize(
            {"currentYear": [{"week": "x"}, {"year": "y"}]}
        )


def test_normalize_raises_on_non_dict() -> None:
    with pytest.raises(sc.StatnettClientError):
        sc.StatnettReservoirClient._normalize([1, 2, 3])


# ── Cache fallback ──────────────────────────────────────────────────


class _FakeClientError(OSError):
    """Stand-in for aiohttp.ClientError that doesn't require aiohttp installed.

    The coordinator catches by class-name match including 'ClientError' or
    inherits from OSError, both of which apply here.
    """


def test_cache_fallback_when_fetch_fails(tmp_path: Path) -> None:
    """If fetch fails but cache exists, return cache."""
    cache_path = tmp_path / sc.CACHE_FILENAME
    cached = sc.StatnettReservoirClient._normalize(SAMPLE_RESPONSE)
    cache_path.write_text(json.dumps(cached), encoding="utf-8")

    session = MagicMock()

    async def _raise(*_a, **_k):
        raise _FakeClientError("simulated failure")

    client = sc.StatnettReservoirClient(session, cache_dir=tmp_path)
    client._fetch_raw = _raise  # type: ignore[assignment]
    out = asyncio.run(client.async_get_latest())
    assert out is not None
    assert out["latest"]["week"] == 19


def test_returns_none_when_fetch_fails_and_no_cache(tmp_path: Path) -> None:
    session = MagicMock()

    async def _raise(*_a, **_k):
        raise _FakeClientError("simulated failure")

    client = sc.StatnettReservoirClient(session, cache_dir=tmp_path)
    client._fetch_raw = _raise  # type: ignore[assignment]
    out = asyncio.run(client.async_get_latest())
    assert out is None


# ── Cache freshness ─────────────────────────────────────────────────


def test_cache_freshness_initially_false() -> None:
    client = sc.StatnettReservoirClient(MagicMock(), cache_dir=None)
    assert client._is_cache_fresh() is False
