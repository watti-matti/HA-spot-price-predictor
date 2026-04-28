"""D(k) cheap/peak duration curve utilities — HA inference side.

Mirror of `src/dk_utils.py` for the custom component (which must be self-
contained without importing from `src/`). Keep these in lockstep.

  dk_cheap[k-1] = average price of the cheapest k hours      (k=1..12, monotone non-decreasing)
  dk_peak[k-1]  = average price of the most expensive k hours (k=1..12, monotone non-increasing)

Both carry decision-relevant signal:
  - dk_cheap: best achievable scheduling cost for k hours of deferrable load
  - dk_peak:  worst-case price if forced to run during k peak hours
"""
from __future__ import annotations


def compute_dk_cheap_peak(
    hourly_prices: list[float],
    *,
    half_horizon: int = 12,
) -> tuple[list[float], list[float]]:
    """Compute cheap-end and peak-end D(k) curves from 24 hourly prices.

    Returns (dk_cheap, dk_peak), each length `half_horizon`:
      dk_cheap[k-1] = mean of cheapest k hours (k=1..half_horizon)
      dk_peak[k-1]  = mean of most expensive k hours (k=1..half_horizon)
    """
    if len(hourly_prices) != 24:
        raise ValueError(
            f"compute_dk_cheap_peak requires exactly 24 hourly prices, "
            f"got {len(hourly_prices)}"
        )
    if half_horizon < 1 or half_horizon > 12:
        raise ValueError(
            f"half_horizon must be in [1, 12], got {half_horizon}"
        )

    asc = sorted(float(p) for p in hourly_prices)
    desc = list(reversed(asc))

    dk_cheap: list[float] = []
    dk_peak: list[float] = []
    cum_lo = 0.0
    cum_hi = 0.0
    for k in range(1, half_horizon + 1):
        cum_lo += asc[k - 1]
        cum_hi += desc[k - 1]
        dk_cheap.append(cum_lo / k)
        dk_peak.append(cum_hi / k)

    return dk_cheap, dk_peak


def reconstruct_sorted_prices(
    dk_cheap: list[float],
    dk_peak: list[float],
) -> tuple[list[float], list[float]]:
    """Recover individual sorted prices from cumulative-mean curves.

    Inverse of `compute_dk_cheap_peak`:
      cheap_sorted_asc[k-1] = price of the k-th cheapest hour
      peak_sorted_desc[k-1] = price of the k-th most expensive hour
    """
    cheap: list[float] = []
    for k, dk in enumerate(dk_cheap, start=1):
        if k == 1:
            cheap.append(dk)
        else:
            cheap.append(k * dk - (k - 1) * dk_cheap[k - 2])

    peak: list[float] = []
    for k, dk in enumerate(dk_peak, start=1):
        if k == 1:
            peak.append(dk)
        else:
            peak.append(k * dk - (k - 1) * dk_peak[k - 2])

    return cheap, peak


def is_monotone_cheap(dk_cheap: list[float], *, atol: float = 1e-9) -> bool:
    """True iff dk_cheap is non-decreasing (any valid cheap-end D(k))."""
    return all(
        dk_cheap[i] <= dk_cheap[i + 1] + atol
        for i in range(len(dk_cheap) - 1)
    )


def is_monotone_peak(dk_peak: list[float], *, atol: float = 1e-9) -> bool:
    """True iff dk_peak is non-increasing (any valid peak-end D(k))."""
    return all(
        dk_peak[i] >= dk_peak[i + 1] - atol
        for i in range(len(dk_peak) - 1)
    )
