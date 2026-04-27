"""
D(k) cheap/peak duration curve utilities.

The D(k) duration curve historically encoded "average price of cheapest k hours"
for k=1..24, but values D(13)..D(24) lose decision-relevance — they smoothly
approach the daily average. This module provides the symmetric interpretation:

  dk_cheap[k-1] = average price of the cheapest k hours,    k=1..12
  dk_peak[k-1]  = average price of the most expensive k hours, k=1..12

Both arrays carry decision-relevant signal:
  - dk_cheap: best achievable scheduling cost for k hours of deferrable load
  - dk_peak:  worst-case price if forced to run during k peak hours
              (storage planning, peak-avoidance, grid backup margin)

Each array is monotone:
  - dk_cheap is non-decreasing  (averaging in higher prices)
  - dk_peak  is non-increasing  (averaging in lower prices)

The total information content is the same as the legacy 24-element D(k) array,
but each value is meaningful.
"""
from __future__ import annotations


def compute_dk_cheap_peak(
    hourly_prices: list[float],
    *,
    half_horizon: int = 12,
) -> tuple[list[float], list[float]]:
    """Compute cheap-end and peak-end D(k) curves from hourly prices for one day.

    Parameters
    ----------
    hourly_prices : list[float]
        24 hourly prices for a single day (any consistent unit: EUR/MWh, c/kWh, ...).
    half_horizon : int, default 12
        Length of each output array. Must satisfy 2 * half_horizon <= 24.

    Returns
    -------
    (dk_cheap, dk_peak) : tuple[list[float], list[float]]
        Each list has length `half_horizon`.
        dk_cheap[k-1] = mean of cheapest k hours (k=1..half_horizon)
        dk_peak[k-1]  = mean of most expensive k hours (k=1..half_horizon)

    Raises
    ------
    ValueError
        If hourly_prices does not contain exactly 24 finite values, or if
        half_horizon is out of range.
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

    # Sort ascending and descending; cheapest end and peak end of the same data
    asc = sorted(float(p) for p in hourly_prices)
    desc = list(reversed(asc))  # priciest first

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

    Inverse of compute_dk_cheap_peak (in the sense of recovering the underlying
    sorted price vector when the input is from real data, modulo numerical noise).

    Returns
    -------
    (cheap_sorted_asc, peak_sorted_desc)
        cheap_sorted_asc[k-1] = the k-th cheapest hour's price
        peak_sorted_desc[k-1] = the k-th most expensive hour's price
    """
    cheap = []
    for k, dk in enumerate(dk_cheap, start=1):
        if k == 1:
            cheap.append(dk)
        else:
            # k * dk_cheap[k-1] - (k-1) * dk_cheap[k-2]
            cheap.append(k * dk - (k - 1) * dk_cheap[k - 2])

    peak = []
    for k, dk in enumerate(dk_peak, start=1):
        if k == 1:
            peak.append(dk)
        else:
            peak.append(k * dk - (k - 1) * dk_peak[k - 2])

    return cheap, peak


def is_monotone_cheap(dk_cheap: list[float], *, atol: float = 1e-9) -> bool:
    """Check that dk_cheap is non-decreasing (true for any valid cheap-end D(k))."""
    return all(dk_cheap[i] <= dk_cheap[i + 1] + atol for i in range(len(dk_cheap) - 1))


def is_monotone_peak(dk_peak: list[float], *, atol: float = 1e-9) -> bool:
    """Check that dk_peak is non-increasing (true for any valid peak-end D(k))."""
    return all(dk_peak[i] >= dk_peak[i + 1] - atol for i in range(len(dk_peak) - 1))
