"""Consumption profile loader for PV-aware CVaR computation.

Two sources of profile data:

1. **External EMA module** (e.g. HA-consumption-profiler running as a
   sibling HA integration) publishes a sensor whose attributes follow
   `docs/household_profile_schema.md`. The user configures
   ``CONF_CONSUMPTION_PROFILE_ENTITY`` to point at that sensor.

2. **Synthetic fallback** for fresh installs that have no profiler.
   A generic Finnish-climate household shape calibrated to
   ``CONF_ANNUAL_CONSUMPTION_KWH``. Data provenance is reported as
   ``"synthetic_cold_start"`` so dashboards can show low-confidence.

This module deliberately does NOT learn the profile itself — that's
the EMA module's job (separate repo). It only consumes whatever the
profiler publishes or falls back to a generic baseline.

Privacy contract per `docs/household_profile_schema.md`: the synthetic
fallback is NOT derived from any individual user's data. The reference
household's empirically-extracted profile lives in
``studies/_private/`` (gitignored) and is never the public default.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np


# ── Synthetic Finnish household profile ──────────────────────────────
#
# Generic shape calibrated to public Finnish-residential statistics
# (Energiateollisuus typical-household curves, non-optimised). Hour
# normalised to mean = 1.0. Monthly factor reflects heating-dominated
# seasonal swing typical for the FI bidding zone.

_SYNTHETIC_HOUR_SHAPE: tuple[float, ...] = (
    # 00-05: low (sleeping)
    0.65, 0.55, 0.50, 0.48, 0.50, 0.65,
    # 06-11: morning rise + cooking + occupancy
    0.85, 1.20, 1.30, 1.10, 0.95, 1.00,
    # 12-17: midday + early evening
    1.05, 1.00, 1.00, 1.05, 1.15, 1.40,
    # 18-23: evening peak + wind-down
    1.55, 1.50, 1.35, 1.15, 0.95, 0.80,
)

_SYNTHETIC_WEEKEND_MULT: tuple[float, ...] = (
    1.00, 1.00, 1.00, 1.00, 1.00,   # Mon..Fri
    1.05, 1.05,                      # Sat, Sun
)

_SYNTHETIC_MONTHLY_FACTOR: tuple[float, ...] = (
    # Jan-Apr: high (heating)
    1.50, 1.40, 1.20, 1.00,
    # May-Aug: low (no heating, longer daylight)
    0.85, 0.65, 0.55, 0.60,
    # Sep-Dec: rising back into heating
    0.80, 1.00, 1.20, 1.45,
)


# ── Profile object ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsumptionProfile:
    """Per-hour consumption derived from a shape + monthly factor."""

    mean_kwh_per_hour:    float
    shape_hour_weekday:   np.ndarray   # ``[7, 24]``, mean 1.0
    monthly_factor:       np.ndarray   # ``[12]``, mean 1.0
    data_provenance:      str

    def __post_init__(self) -> None:
        if self.shape_hour_weekday.shape != (7, 24):
            raise ValueError(
                f"shape_hour_weekday must be [7, 24], got "
                f"{self.shape_hour_weekday.shape}"
            )
        if self.monthly_factor.shape != (12,):
            raise ValueError(
                f"monthly_factor must be [12], got "
                f"{self.monthly_factor.shape}"
            )
        if self.mean_kwh_per_hour < 0:
            raise ValueError(
                f"mean_kwh_per_hour must be non-negative, "
                f"got {self.mean_kwh_per_hour}"
            )

    def consumption_for_timestamps(
        self, timestamps_local: Sequence[datetime],
    ) -> np.ndarray:
        """Per-hour kWh for a sequence of local-time timestamps."""
        out = np.empty(len(timestamps_local), dtype=float)
        for i, ts in enumerate(timestamps_local):
            wd = ts.weekday()
            h = ts.hour
            m = ts.month - 1
            out[i] = (
                float(self.shape_hour_weekday[wd, h])
                * self.mean_kwh_per_hour
                * float(self.monthly_factor[m])
            )
        return out


# ── Builders ─────────────────────────────────────────────────────────


def synthetic_profile(annual_kwh: float) -> ConsumptionProfile:
    """Generic non-optimised Finnish-climate household.

    Mean hourly kWh = ``annual_kwh / 8760``. Hour-of-day × weekday
    shape and monthly factor are taken from public-statistic constants
    in this module. Provenance: ``"synthetic_cold_start"``.
    """
    if annual_kwh < 0:
        raise ValueError(f"annual_kwh must be non-negative, got {annual_kwh}")

    hour = np.array(_SYNTHETIC_HOUR_SHAPE, dtype=float)
    weekend = np.array(_SYNTHETIC_WEEKEND_MULT, dtype=float)
    # 7×24 shape: weekday multiplier outer-product hour shape.
    shape = np.outer(weekend, hour)
    # Renormalise so overall mean across all (weekday, hour) cells = 1.
    shape = shape / shape.mean()

    monthly = np.array(_SYNTHETIC_MONTHLY_FACTOR, dtype=float)
    monthly = monthly / monthly.mean()

    return ConsumptionProfile(
        mean_kwh_per_hour=float(annual_kwh) / 8760.0,
        shape_hour_weekday=shape,
        monthly_factor=monthly,
        data_provenance="synthetic_cold_start",
    )


def load_profile_from_entity_attrs(
    entity_attrs: dict[str, Any] | None,
    *,
    fallback_annual_kwh: float,
) -> ConsumptionProfile:
    """Build a profile from a HA sensor's attributes, with fallback.

    Expected attributes per ``docs/household_profile_schema.md``:

    - ``mean_kwh_per_hour`` : float
    - ``shape_hour_weekday`` : list of 7 lists of 24 floats
    - ``monthly_factor`` : list of 12 floats
    - ``data_provenance`` : str (optional, defaults to ``"ema_unknown"``)

    If attributes are missing, malformed, or absent entirely, falls
    back to :func:`synthetic_profile` calibrated to ``fallback_annual_kwh``.
    """
    if not entity_attrs:
        return synthetic_profile(fallback_annual_kwh)

    try:
        mean_raw = entity_attrs.get("mean_kwh_per_hour")
        shape_raw = entity_attrs.get("shape_hour_weekday")
        monthly_raw = entity_attrs.get("monthly_factor")
        provenance = str(entity_attrs.get("data_provenance", "ema_unknown"))

        if mean_raw is None or shape_raw is None or monthly_raw is None:
            return synthetic_profile(fallback_annual_kwh)

        # Parse shape ─ must be [7][24] of floats; replace None with 1.0
        # so a sparse profile still works without crashing.
        shape = np.array(
            [
                [
                    float(v) if v is not None else 1.0
                    for v in row
                ]
                for row in shape_raw
            ],
            dtype=float,
        )
        if shape.shape != (7, 24):
            return synthetic_profile(fallback_annual_kwh)

        monthly = np.array(
            [float(v) if v is not None else 1.0 for v in monthly_raw],
            dtype=float,
        )
        if monthly.shape != (12,):
            return synthetic_profile(fallback_annual_kwh)

        return ConsumptionProfile(
            mean_kwh_per_hour=float(mean_raw),
            shape_hour_weekday=shape,
            monthly_factor=monthly,
            data_provenance=provenance,
        )

    except (KeyError, ValueError, TypeError):
        return synthetic_profile(fallback_annual_kwh)
