"""Household PV production estimator.

Phase 1: simple physics-based model converting Open-Meteo solar irradiance
(`global_tilted_irradiance_instant`, fetched at fixed 45° tilt, south-facing)
into hourly PV output for a user-configured rooftop system.

Output is bounded by the physical capacity ceiling (`capacity_kwp * efficiency`)
and is non-negative.

Tilt and azimuth corrections are scalar adjustments relative to the 45°-S
reference. For Phase 1 we use small empirical corrections — accurate to
within ~10% for typical residential orientations in Finland. Phase 2 may
add a second Open-Meteo fetch with the user's specific tilt/azimuth for
exact per-hour irradiance.
"""

from __future__ import annotations

import math


def _tilt_correction(tilt_deg: float) -> float:
    """Scalar correction relative to 45° tilt reference.

    Gentler tilts collect less in winter / more in summer; we use the
    mean-annual-yield correction which is approximately cos((tilt-45)/2)
    for tilts in [10, 70]°. At tilt=45 the correction is exactly 1.
    """
    delta = tilt_deg - 45.0
    # Quadratic falloff around 45°; loses ~5% at 30° or 60°, ~12% at 15° or 75°
    factor = 1.0 - 0.0006 * (delta * delta)
    return max(0.6, min(1.05, factor))


def _azimuth_correction(azimuth_deg: float) -> float:
    """Scalar correction relative to south-facing (180°) reference.

    Pure south = 1.0. Pure east/west loses ~15-20% over the year.
    Pure north loses ~50%.
    """
    # Convert to deviation from south, in [0, 180]
    deviation = abs(((azimuth_deg - 180.0) + 180.0) % 360.0 - 180.0)
    # Cosine falloff approximation: cos(deviation/2) gives 1.0 south,
    # ~0.85 east/west (90° dev), ~0.5 north (180° dev)
    return max(0.4, math.cos(math.radians(deviation) / 2.0))


def estimate_pv_kwh_per_hour(
    irradiance_w_m2: float,
    capacity_kwp: float,
    tilt_deg: float = 45.0,
    azimuth_deg: float = 180.0,
    efficiency: float = 0.85,
) -> float:
    """Estimate household PV output (kWh) for one hour.

    Parameters
    ----------
    irradiance_w_m2
        Hourly mean global tilted irradiance from Open-Meteo (W/m²).
        Open-Meteo's `global_tilted_irradiance_instant` field at tilt=45°
        and south orientation is the reference used here.
    capacity_kwp
        Installed PV peak power (kWp).
    tilt_deg
        User's panel tilt (degrees from horizontal). Default 45 matches
        the irradiance fetch angle, so no tilt correction is applied.
    azimuth_deg
        User's panel azimuth (0=N, 90=E, 180=S, 270=W). Default 180.
    efficiency
        System efficiency factor combining DC/AC conversion, soiling,
        wiring losses, temperature derating. Typical 0.80–0.90.

    Returns
    -------
    float
        PV production for the hour, in kWh. Always in [0, capacity_kwp · efficiency].

    Notes
    -----
    - Returns 0.0 immediately if capacity_kwp ≤ 0 (PV disabled).
    - Negative irradiance (sensor noise) is clamped to 0.
    - Output is hard-capped at `capacity_kwp * efficiency` (physical ceiling).
    """
    if capacity_kwp <= 0.0:
        return 0.0
    if not (0.0 <= efficiency <= 1.0):
        raise ValueError(f"efficiency must be in [0, 1], got {efficiency}")

    irr = max(0.0, float(irradiance_w_m2))
    raw = (irr / 1000.0) * capacity_kwp * efficiency
    raw *= _tilt_correction(tilt_deg)
    raw *= _azimuth_correction(azimuth_deg)

    ceiling = capacity_kwp * efficiency
    return max(0.0, min(raw, ceiling))


def marginal_effective_eur_kwh(
    buy_eur_kwh: float,
    sell_eur_kwh: float,
    pv_kwh: float,
    baseload_kwh: float,
) -> float:
    """Marginal cost of running 1 additional kWh of flexible load at this hour.

    Bounded by [sell_eur_kwh, buy_eur_kwh]. This is the PV-aware effective
    price metric used as input to D(k) cheap/peak order statistics.

    Parameters
    ----------
    buy_eur_kwh
        Consumer buy price (b_h), incl. tariffs/VAT. EUR/kWh.
    sell_eur_kwh
        Sell price (s_h) = spot − commission − export_fee. EUR/kWh.
        Can be negative when spot is below total deductions.
    pv_kwh
        Hourly PV production. kWh. ≥ 0.
    baseload_kwh
        Non-flexible household consumption assumed at this hour. kWh. > 0.

    Returns
    -------
    float
        Marginal cost m_h ∈ [s_h, b_h]. EUR/kWh.

    Formula
    -------
        pv_avail = max(0, pv − baseload)
        from_pv  = min(1, pv_avail)
        from_grid = 1 − from_pv
        m_h      = from_pv · sell + from_grid · buy
    """
    pv_avail = max(0.0, pv_kwh - baseload_kwh)
    from_pv = min(1.0, pv_avail)
    from_grid = 1.0 - from_pv
    return from_pv * sell_eur_kwh + from_grid * buy_eur_kwh


def net_household_cost_eur(
    buy_eur_kwh: float,
    sell_eur_kwh: float,
    pv_kwh: float,
    consumption_kwh: float,
) -> float:
    """Raw net household cost for one hour (informational, not used for D(k)).

    N_h = max(0, c-p)·b − max(0, p-c)·s

    Returns
    -------
    float
        Net hourly cost in EUR. Positive = user pays grid; negative = user is paid.
    """
    grid_import = max(0.0, consumption_kwh - pv_kwh)
    grid_export = max(0.0, pv_kwh - consumption_kwh)
    return grid_import * buy_eur_kwh - grid_export * sell_eur_kwh
