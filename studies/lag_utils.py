"""Time-based lag, isolated so it has no study dependencies.

Lives apart from exp_extra_features because
tests/test_harness_production_parity.py execs the harness source into
a hand-built namespace and must supply this helper without importing
the whole study data pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lag_by_hours(values, index: pd.DatetimeIndex, hours: int) -> np.ndarray:
    """Lag a series by WALL-CLOCK HOURS, aligned to `index`.

    `pandas.Series.shift(n)` shifts by POSITION. The analysis frame is
    built by an inner join followed by `dropna()`, so it is NOT
    contiguous — measured: 9 irregular steps and 56 missing hours over
    33,026 — and for every row after a gap `.shift(168)` reads the wrong
    hour. Measured on 2026-10-16: 1,417 of 32,970 rows (4.3 %) were
    lagged by the wrong amount, between 1 h and 21 h off, mean -6.5 h.

    That silently coupled lag alignment to NaN coverage: adding 159 rows
    by fixing the capacity gap-fill moved predictions by up to 18.1
    EUR/MWh on rows both frames contained, which looked like a model
    change and was not.

    Production is unaffected — `coordinator.py` aligns neighbours by
    timestamp (`t - 168 h`), not by position. This is an offline-analysis
    defect, and the parity test never caught it because its synthetic
    frame is contiguous.

    Returns a positionally-aligned ndarray, so it is a drop-in
    replacement for `.shift(h).values`.
    """
    s = pd.Series(np.asarray(values, dtype=float), index=index)
    return s.reindex(index - pd.Timedelta(hours=hours)).values
