"""Intraday PV nowcast correction (clear-sky-index persistence).

The day-ahead PV forecast (external entity or internal irradiance model)
is fixed for ~6 hours at a time and cannot tell that *today* is running
sunnier or cloudier than predicted. This module corrects TODAY's
remaining PV forecast using a real-time measurement of actual
production, so the PV-aware effective-price curves react to the real
sky — the input a downstream planner needs to exploit a surplus while
it is happening rather than one cycle later.

**Method — clear-sky-index persistence.** The standard, robust solar
nowcasting primitive (Perez / Marquez persistence family): form the
ratio of what is being produced *now* to what was forecast for *now*,

    k = measured_now / forecast_now,

smooth it across cycles to reject passing-cloud transients, and scale
the forecast's remaining daylight hours by ``k``. Past hours are
already realized and left untouched. When today is genuinely clearer
than forecast, ``k > 1`` lifts the afternoon; when overcast, ``k < 1``
suppresses it. Because the coordinator re-evaluates every nowcast
cycle, any over-correction self-heals within one interval.

**Confidence.** The module also reports the *realized PV-energy
fraction* of today — how much of the day's forecast solar energy is
already in the past. This is a PV-weighted "how much of today is
known" signal (0 before sunrise even at 06:00; ~1 after sunset even
though the clock says 20:00), strictly better than a wall-clock
fraction for telling a downstream risk model how certain today's
effective prices now are.

Pure module: no Home Assistant imports. The coordinator supplies the
measured value, the forecast arrays, and the current hour offset; this
module does the arithmetic and is unit-tested in isolation.
"""

from __future__ import annotations

# Clamp the raw index to reject sensor glitches, divide-by-near-zero
# blowups at dawn/dusk, and physically implausible ratios. 0.15..3.0
# spans "much cloudier than forecast" to "much clearer than forecast"
# without letting a single bad sample rewrite the day.
DEFAULT_CLAMP_LO = 0.15
DEFAULT_CLAMP_HI = 3.0

# Forecast production (kWh/h) below this is treated as "sun effectively
# down" — no meaningful ratio can be formed, so no correction.
MIN_FORECAST_KWH = 0.05

# EMA weight for the smoothed index. 0.3 → a passing cloud moves the
# applied k by ~30% per cycle, so with a ~15-min nowcast interval the
# index tracks a real sky change within ~30-45 min while averaging out
# single-sample noise.
DEFAULT_SMOOTHING_ALPHA = 0.3


def clear_sky_index(
    measured_now_kwh_h: float | None,
    forecast_now_kwh_h: float | None,
    *,
    clamp_lo: float = DEFAULT_CLAMP_LO,
    clamp_hi: float = DEFAULT_CLAMP_HI,
    min_forecast_kwh: float = MIN_FORECAST_KWH,
) -> float | None:
    """Instantaneous clear-sky index ``k = measured / forecast``.

    Both inputs are same-unit production *rates* at the current hour
    (kWh per hour; equivalently average kW). A power sensor in watts
    should be divided by 1000 by the caller before it gets here.

    Returns the clamped ratio, or ``None`` when no meaningful ratio
    can be formed (sun down, missing/invalid measurement) — the caller
    then keeps the previous smoothed index rather than reverting.
    """
    if forecast_now_kwh_h is None or forecast_now_kwh_h < min_forecast_kwh:
        return None
    if measured_now_kwh_h is None or measured_now_kwh_h < 0.0:
        return None
    k = float(measured_now_kwh_h) / float(forecast_now_kwh_h)
    return max(clamp_lo, min(clamp_hi, k))


def smooth_index(
    prev_k: float | None,
    new_k: float | None,
    *,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> float | None:
    """EMA-blend a fresh index into the running smoothed index.

    ``new_k = None`` (no ratio this cycle) keeps the previous value —
    the smoothed index persists across sun-down gaps rather than
    resetting. First valid sample seeds the EMA directly.
    """
    if new_k is None:
        return prev_k
    if prev_k is None:
        return float(new_k)
    return (1.0 - alpha) * float(prev_k) + alpha * float(new_k)


def apply_nowcast(
    pv_forecast: list[float],
    k: float | None,
    hours_from_now: list[float],
) -> list[float]:
    """Scale the forecast's *future* hours by the smoothed index ``k``.

    Parameters
    ----------
    pv_forecast
        Per-hour PV production forecast (kWh/h).
    k
        Smoothed clear-sky index. ``None`` or ``1.0`` → forecast
        returned unchanged (copy).
    hours_from_now
        For each forecast element, its offset in hours from the current
        moment. Elements with offset ``<= 0`` are past/current and are
        left untouched (already realized); elements ``> 0`` are the
        remaining forecast and get scaled by ``k``.

    Returns a new list; the input is not mutated. Scaled values are
    floored at 0.

    Note the correction is applied uniformly to all remaining hours,
    not decayed with lead time: the smoothed index represents a
    persistent same-day sky bias (the clear/overcast day the forecast
    missed), and the frequent re-evaluation self-corrects transients.
    Lead-time decay is a possible future refinement.
    """
    if k is None or abs(k - 1.0) < 1e-9:
        return list(pv_forecast)
    out: list[float] = []
    for pv, h in zip(pv_forecast, hours_from_now):
        if h > 0.0:
            out.append(max(0.0, float(pv) * float(k)))
        else:
            out.append(float(pv))
    return out


def realized_pv_fraction(
    pv_today_by_hour: list[float],
    hours_from_now: list[float],
) -> float:
    """Fraction of today's forecast PV energy already in the past.

    A PV-energy-weighted "how much of today is known" measure: 0.0
    before any production has occurred (pre-dawn), ~1.0 once the
    day's solar energy is spent (post-sunset), rising steeply only
    across daylight hours — unlike a wall-clock fraction which rises
    linearly through the dark morning and evening.

    ``pv_today_by_hour`` must cover exactly today's 24 hours;
    ``hours_from_now[i]`` is that hour's offset from now (negative =
    already elapsed). Returns 0.0 for a zero-energy day (polar night)
    so the caller can treat it as "nothing known yet".
    """
    total = sum(max(0.0, v) for v in pv_today_by_hour)
    if total <= 1e-9:
        return 0.0
    elapsed = sum(
        max(0.0, v)
        for v, h in zip(pv_today_by_hour, hours_from_now)
        if h <= 0.0
    )
    return max(0.0, min(1.0, elapsed / total))


def nowcast_confidence(
    realized_fraction: float,
    *,
    measurement_live: bool,
) -> str:
    """Coarse confidence label for today's corrected effective prices.

    Consumed downstream (e.g. ENP) to size the residual uncertainty
    band on day 0. ``low`` when no live measurement is available (we
    are back to the raw forecast); otherwise scales with how much of
    the day is realized: the more of today's solar energy is already
    banked and measured, the tighter tomorrow-planning can trust
    today's remaining curve.
    """
    if not measurement_live:
        return "low"
    if realized_fraction >= 0.6:
        return "high"
    if realized_fraction >= 0.2:
        return "medium"
    return "low"


def should_trigger_refresh(
    k: float | None,
    k_applied: float | None,
    seconds_since_last_refresh: float | None,
    *,
    k_delta: float,
    min_gap_seconds: float,
) -> bool:
    """Whether the fast tick should request a full forecast refresh.

    Fire only when the live smoothed index ``k`` has drifted from the
    one currently baked into the published forecast (``k_applied``, or
    an implicit 1.0 when nothing has been applied yet) by at least
    ``k_delta``, AND at least ``min_gap_seconds`` have elapsed since the
    last triggered refresh (rate limit protecting the weather API).
    """
    if k is None:
        return False
    if (
        seconds_since_last_refresh is not None
        and seconds_since_last_refresh < min_gap_seconds
    ):
        return False
    baseline = 1.0 if k_applied is None else k_applied
    return abs(k - baseline) >= k_delta


__all__ = [
    "DEFAULT_CLAMP_HI",
    "DEFAULT_CLAMP_LO",
    "DEFAULT_SMOOTHING_ALPHA",
    "MIN_FORECAST_KWH",
    "apply_nowcast",
    "clear_sky_index",
    "nowcast_confidence",
    "realized_pv_fraction",
    "should_trigger_refresh",
    "smooth_index",
]
