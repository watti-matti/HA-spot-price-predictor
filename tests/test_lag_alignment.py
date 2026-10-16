"""Lags must be measured in HOURS, not in row positions.

The analysis frame is built by an inner join followed by `dropna()`, so
it is not contiguous. `pandas.Series.shift(168)` moves 168 ROWS, which
on a gappy index reads the wrong hour: measured 2026-10-16, 1,417 of
32,970 rows (4.3 %) were lagged between 1 h and 21 h off target.

That coupled lag alignment to NaN coverage. Fixing an unrelated
capacity gap-fill added 159 rows, which re-positioned every later row,
which moved predictions by up to 18.1 EUR/MWh on rows both frames
shared — indistinguishable from a model change unless you look.

`tests/test_harness_production_parity.py` could not catch it because
its synthetic frame is contiguous. These tests use a frame WITH HOLES,
which is the only configuration where the two definitions differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "studies"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp_extra_features import lag_by_hours      # noqa: E402

LAG = 168


def _gappy(n: int = 600, drop=(100, 101, 102, 300, 455)) -> pd.DatetimeIndex:
    full = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return full.delete(list(drop))


def test_contiguous_index_matches_shift() -> None:
    """With no holes the two definitions must agree, or the helper has
    changed semantics rather than fixed alignment."""
    idx = pd.date_range("2024-01-01", periods=600, freq="1h", tz="UTC")
    v = np.arange(len(idx), dtype=float)
    got = lag_by_hours(v, idx, LAG)
    want = pd.Series(v, index=idx).shift(LAG).values
    np.testing.assert_allclose(got[LAG:], want[LAG:])


def test_gappy_index_diverges_from_shift() -> None:
    """The regression itself: on a holed index `.shift()` is wrong, so
    the helper MUST differ from it. If this ever passes by agreeing with
    shift, the fix has been reverted."""
    idx = _gappy()
    v = np.arange(len(idx), dtype=float)
    got = lag_by_hours(v, idx, LAG)
    shifted = pd.Series(v, index=idx).shift(LAG).values
    m = np.isfinite(got) & np.isfinite(shifted)
    assert (got[m] != shifted[m]).any(), (
        "lag_by_hours agrees with .shift() on a gappy index — the "
        "position-based bug is back")


def test_lag_reads_the_hour_exactly_168h_earlier() -> None:
    """The actual invariant, checked value by value."""
    idx = _gappy()
    v = np.arange(len(idx), dtype=float)
    lookup = dict(zip(idx, v))
    got = lag_by_hours(v, idx, LAG)
    checked = 0
    for i, t in enumerate(idx):
        want_t = t - pd.Timedelta(hours=LAG)
        if want_t in lookup:
            assert got[i] == lookup[want_t], (
                f"row {t} read {got[i]} but hour {want_t} holds "
                f"{lookup[want_t]}")
            checked += 1
        else:
            assert np.isnan(got[i]), (
                f"row {t} has no observation at {want_t} yet returned "
                f"{got[i]} instead of NaN")
    assert checked > 300, "test frame did not exercise enough rows"


def test_missing_source_hour_yields_nan_not_a_neighbour() -> None:
    """A hole 168 h back must produce NaN, never the nearest available
    hour — silently substituting a neighbour is how a leak hides."""
    idx = _gappy(drop=(5, 6, 7))
    v = np.arange(len(idx), dtype=float)
    got = lag_by_hours(v, idx, LAG)
    holes = [t for t in (pd.Timestamp("2024-01-01 05:00", tz="UTC"),
                         pd.Timestamp("2024-01-01 06:00", tz="UTC"),
                         pd.Timestamp("2024-01-01 07:00", tz="UTC"))]
    for h in holes:
        target = h + pd.Timedelta(hours=LAG)
        if target in idx:
            assert np.isnan(got[idx.get_loc(target)]), (
                f"row {target} should be NaN because {h} is missing")


def test_analysis_frame_lag_is_time_aligned() -> None:
    """End-to-end on the real frame, if the data store is present."""
    store = REPO / "data_store" / "fi_prices.parquet"
    if not store.exists():
        pytest.skip("data store not present (gitignored)")
    from exp_extra_features import build_dataframe
    df = build_dataframe()
    idx = df.index
    lookup = dict(zip(idx, df["Y_fi"].values))
    got = df["Y_fi_lag168"].values
    bad = 0
    for i in range(0, len(idx), 37):          # sample, the frame is large
        want_t = idx[i] - pd.Timedelta(hours=LAG)
        if want_t in lookup and np.isfinite(got[i]):
            if not np.isclose(got[i], lookup[want_t], equal_nan=True):
                bad += 1
    assert bad == 0, f"{bad} sampled rows have a mis-aligned Y_fi_lag168"
