"""v2.5.5 — Per-input seasonal decomposition: training + runtime helpers.

Implements the Moazeni-Powell sequential additive decomposition

    X(t) = P_hour(h) + P_day(d) + P_week(w) + Y(t)

at two levels:

* :func:`fit_components` — offline training: takes a series and the list
  of components to fit (per the v2.5.4 audit recommendations), returns
  the seasonal vectors as a JSON-serialisable dict.

* :func:`compute_residual` — runtime: takes raw X(t), timestamps, and
  a loaded artifact, returns the residual `Y(t) = X − Σ components`.
  Pure-numpy; no fit; no Fingrid; safe to call inside the coordinator.

* :func:`build_artifact` / :func:`load_components` — JSON persistence
  matching the v2.5.3 solar pattern. The artifact ships in
  ``data/seasonal_components_default.json`` and is refreshed quarterly
  by the operator via ``studies/build_seasonal_components.py``.

The artifact maps each input name (e.g. ``"fi"``, ``"wind"``,
``"ghi_cs"``) to the seasonal vectors actually fit for that input —
absent components (e.g. ``P_day`` for wind) are not subtracted at
runtime. This matches the per-input decomposition depths set in v2.5.4.

Refer to ``studies/results/V2_5_4_RELEASE_NOTES.md`` for the audit
result that drives the per-input depth choices.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


# Per-input depth specification — matches v2.5.4 audit verdict.
DEFAULT_DEPTHS: dict[str, tuple[str, ...]] = {
    "fi":     ("P_hour", "P_day", "P_week"),
    "se3":    ("P_hour", "P_day", "P_week"),
    "se1":    ("P_day",  "P_week"),
    "ee":     ("P_hour", "P_day", "P_week"),
    "wind":   ("P_hour", "P_week"),
    "solar":  ("P_hour", "P_week"),
    "ghi_cs": ("P_hour", "P_week"),
    "temp":   ("P_hour", "P_week"),
    "cloud":  ("P_week",),
}


def _hour_index(ts: np.ndarray) -> np.ndarray:
    """Hour-of-day index 0..23 from numpy datetime64[ns]/[us]/[s] array."""
    # numpy datetime64 in any unit → seconds since epoch → modulo 86400
    secs = ts.astype("datetime64[s]").astype("int64")
    return (secs // 3600) % 24


def _weekday_index(ts: np.ndarray) -> np.ndarray:
    """Weekday index 0=Mon..6=Sun for numpy datetime64 array.

    1970-01-01 is a Thursday (=3) so the offset accounts for that.
    """
    secs = ts.astype("datetime64[s]").astype("int64")
    days = secs // 86400
    return (days + 3) % 7


def _week_of_year_index(ts: np.ndarray) -> np.ndarray:
    """ISO week of year minus 1 (0..52) for numpy datetime64 array.

    Implemented via stdlib `datetime.isocalendar()` rather than numpy
    because ISO week boundaries are tricky. Acceptable cost since this
    runs once at fit time and once per coordinator cycle at inference.
    """
    secs = ts.astype("datetime64[s]").astype("int64")
    out = np.empty(len(secs), dtype=np.int64)
    for i, s in enumerate(secs):
        out[i] = datetime.fromtimestamp(int(s), tz=timezone.utc).isocalendar().week - 1
    return out


# ── Fitting (offline) ──────────────────────────────────────────────


def circular_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Wrap-around centred moving average for periodic seasonal vectors.

    `arr` is treated as a circular sequence (P_hour: bin 23 is adjacent
    to bin 0; P_week: bin 52 is adjacent to bin 0). The output has the
    same length as the input; preserves the mean.

    Used after :func:`fit_components` for inputs where short averaging
    history makes per-bin estimates noisy (typically weather inputs
    with only 3-8 years of history). For inputs with strong physical
    structure (P_hour for solar, P_week for temperature) smoothing is
    unnecessary — the fit is already dominated by signal not noise.

    Args:
        arr: 1-D periodic vector (length 24 / 7 / 53).
        window: smoothing window length in bins. Must be odd; `window=1`
            returns the input unchanged.

    Returns:
        Smoothed copy of `arr`, same length, same mean.
    """
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        raise ValueError(f"window must be odd; got {window}")
    n = len(arr)
    if window >= n:
        return np.full(n, float(arr.mean()))
    half = window // 2
    # Tile + roll for true circular boundary handling
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    assert len(smoothed) == n
    # Restore exact mean to preserve sequential-subtraction guarantees
    smoothed = smoothed - smoothed.mean() + float(arr.mean())
    return smoothed


def fit_components(
    x: np.ndarray,
    timestamps: np.ndarray,
    depth: Iterable[str] = ("P_hour", "P_day", "P_week"),
    smooth: Mapping[str, int] | None = None,
) -> dict[str, list[float]]:
    """Fit the Moazeni-Powell sequential additive decomposition.

    Args:
        x: 1-D series of observations (any units).
        timestamps: matching 1-D numpy datetime64 array (UTC).
        depth: which components to fit, in the canonical order
            ``("P_hour", "P_day", "P_week")``. Components NOT listed are
            skipped — both at fit and at runtime.

    Returns:
        Dict mapping each fitted component name to a list of floats:
            P_hour → length 24
            P_day  → length 7
            P_week → length 53
        Components in `depth` that aren't computable on the given data
        (e.g. P_hour with no daylight variation) still appear with
        zeros — calling code can safely subtract them.
    """
    x = np.asarray(x, dtype=float)
    timestamps = np.asarray(timestamps)
    if x.shape != timestamps.shape:
        raise ValueError(
            f"x and timestamps must have the same shape; "
            f"got {x.shape} vs {timestamps.shape}"
        )
    depth_set = set(depth)
    smooth = smooth or {}
    out: dict[str, list[float]] = {}

    residual = x.copy()
    if "P_hour" in depth_set:
        h = _hour_index(timestamps)
        p_hour = np.zeros(24, dtype=float)
        for k in range(24):
            mask = h == k
            if mask.any():
                p_hour[k] = float(residual[mask].mean())
        if smooth.get("P_hour", 1) > 1:
            p_hour = circular_smooth(p_hour, int(smooth["P_hour"]))
        residual = residual - p_hour[h]
        out["P_hour"] = p_hour.tolist()

    if "P_day" in depth_set:
        d = _weekday_index(timestamps)
        p_day = np.zeros(7, dtype=float)
        for k in range(7):
            mask = d == k
            if mask.any():
                p_day[k] = float(residual[mask].mean())
        # P_day has only 7 bins; smoothing is rarely useful and is left
        # off by default. Caller can still pass smooth={"P_day": 3}.
        if smooth.get("P_day", 1) > 1:
            p_day = circular_smooth(p_day, int(smooth["P_day"]))
        residual = residual - p_day[d]
        out["P_day"] = p_day.tolist()

    if "P_week" in depth_set:
        w = _week_of_year_index(timestamps)
        p_week = np.zeros(53, dtype=float)
        for k in range(53):
            mask = w == k
            if mask.any():
                p_week[k] = float(residual[mask].mean())
        if smooth.get("P_week", 1) > 1:
            p_week = circular_smooth(p_week, int(smooth["P_week"]))
        residual = residual - p_week[w]
        out["P_week"] = p_week.tolist()

    return out


# ── Runtime: compute the residual Y_X = X − seasonal ────────────────


def compute_seasonal_part(
    timestamps: np.ndarray,
    components: Mapping[str, list[float]],
) -> np.ndarray:
    """Reconstruct the seasonal sum at the given timestamps.

    Components that are absent from the dict are treated as zero (i.e.
    not subtracted) — this is how per-input decomposition depths are
    honoured at runtime without needing a separate spec.
    """
    timestamps = np.asarray(timestamps)
    out = np.zeros(len(timestamps), dtype=float)
    if "P_hour" in components:
        p = np.asarray(components["P_hour"], dtype=float)
        if p.size != 24:
            raise ValueError(f"P_hour must have length 24, got {p.size}")
        out += p[_hour_index(timestamps)]
    if "P_day" in components:
        p = np.asarray(components["P_day"], dtype=float)
        if p.size != 7:
            raise ValueError(f"P_day must have length 7, got {p.size}")
        out += p[_weekday_index(timestamps)]
    if "P_week" in components:
        p = np.asarray(components["P_week"], dtype=float)
        if p.size != 53:
            raise ValueError(f"P_week must have length 53, got {p.size}")
        out += p[_week_of_year_index(timestamps)]
    return out


def compute_residual(
    x: np.ndarray,
    timestamps: np.ndarray,
    components: Mapping[str, list[float]],
) -> np.ndarray:
    """Y_X(t) = X(t) − seasonal_components(t).

    Components absent from `components` are not subtracted — so the same
    function transparently handles per-input depth choices.
    """
    x = np.asarray(x, dtype=float)
    seasonal = compute_seasonal_part(timestamps, components)
    return x - seasonal


# ── Artifact persistence ────────────────────────────────────────────


def build_artifact(
    inputs: Mapping[str, dict[str, list[float]]],
    train_window: tuple[str, str] | None = None,
    stats: Mapping[str, dict[str, float]] | None = None,
    depths: Mapping[str, Iterable[str]] | None = None,
    notes: str = "",
) -> dict:
    """Assemble the JSON-serialisable artifact for persistence.

    Args:
        inputs: mapping of input-name → component dict (output of
            :func:`fit_components`).
        train_window: optional ``("YYYY-MM-DD HH:MM", "...")`` window
            used to fit the components; recorded for traceability.
        stats: optional mapping input-name → ``{"raw_std", "residual_std",
            "var_reduction"}`` for the auto-generated markdown.
        depths: optional mapping input-name → tuple of component names;
            defaults to :data:`DEFAULT_DEPTHS` if absent. Recorded in
            the artifact so the operator can audit what was fit.
        notes: free-form text appended to the artifact.

    Returns:
        Dict ready for ``json.dumps``.
    """
    depths_resolved = {
        k: list(depths[k] if (depths and k in depths) else DEFAULT_DEPTHS.get(k, ()))
        for k in inputs
    }
    return {
        "version": "2.5.7",
        "train_window": list(train_window) if train_window else None,
        "depths": depths_resolved,
        "components": dict(inputs),
        "stats": dict(stats or {}),
        "notes": notes,
    }


def load_components(path: str | Path) -> dict | None:
    """Load the persisted artifact from JSON; return ``None`` if missing.

    Returns the full artifact dict (use ``["components"][input_name]``
    to get the per-input component dict suitable for
    :func:`compute_residual`).
    """
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
