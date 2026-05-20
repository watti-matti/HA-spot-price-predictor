"""PV scenario generator — block-bootstrap from cached irradiance history.

Phase A of `pv_adjusted_cvar_plan.md`. For a target forecast window
of hourly timestamps, generates N alternative PV-production paths
by sampling historical irradiance trajectories from the same
calendar position across years. Each path preserves intra-day
temporal structure (diurnal cycle, synoptic cloud sequences) by
sampling whole-day blocks rather than independent hours.

The generator is generic — given a weather DataFrame and target
timestamps, it returns ``[N_paths, n_hours]`` PV kWh tensors.
Households parameterise via ``(kWp, tilt, azimuth, efficiency)``.

Validation logic is in `studies/exp_pv_scenarios_validation.py`.
This module just produces the paths.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from pv_estimate import estimate_pv_kwh_per_hour  # noqa: E402


@dataclass(frozen=True)
class PVConfig:
    capacity_kwp:  float
    tilt_deg:      float = 45.0
    azimuth_deg:   float = 180.0
    efficiency:    float = 0.85


def _day_of_year_distance(d1: int, d2: int) -> int:
    """Smallest distance between two day-of-year values on a 365-day ring."""
    diff = abs(d1 - d2)
    return min(diff, 365 - diff)


def _candidate_pool(
    target_doy: int,
    history_doys: np.ndarray,
    history_dates: np.ndarray,
    window_days: int,
) -> np.ndarray:
    """Indices of historical dates whose day-of-year is within
    ``window_days`` of ``target_doy``."""
    distances = np.array([
        _day_of_year_distance(target_doy, int(d)) for d in history_doys
    ])
    return np.where(distances <= window_days)[0]


def generate_pv_scenarios(
    target_timestamps: pd.DatetimeIndex,
    weather_df: pd.DataFrame,
    pv_config: PVConfig,
    *,
    n_paths: int = 500,
    block_size_hours: int = 24,
    candidate_window_days: int = 7,
    rng_seed: int = 42,
    irradiance_col: str = "solar_irradiance_weighted",
) -> np.ndarray:
    """Generate PV-production scenarios.

    Parameters
    ----------
    target_timestamps : DatetimeIndex
        UTC hourly timestamps the forecast covers. Length must be a
        whole number of ``block_size_hours``.
    weather_df : DataFrame indexed by UTC hourly timestamps
        Must contain ``irradiance_col``. Provides the historical
        pool.
    pv_config : PVConfig
        Household PV parameters.
    n_paths : int
        How many alternative paths to generate.
    block_size_hours : int
        Length of each sampled block. 24 (whole days) preserves the
        diurnal cycle; 168 (whole weeks) preserves synoptic cloud
        sequences.
    candidate_window_days : int
        Pool radius around the target day-of-year (in days). 7
        means "any historical date within ±1 week of the target
        date's day-of-year, in any year."
    rng_seed : int
        Bootstrap RNG seed.

    Returns
    -------
    pv_paths : ndarray of shape ``[n_paths, len(target_timestamps)]``
        PV kWh per hour per path.
    """
    n_hours = len(target_timestamps)
    if n_hours % block_size_hours != 0:
        raise ValueError(
            f"target window length {n_hours} not divisible by "
            f"block_size_hours {block_size_hours}"
        )
    n_blocks = n_hours // block_size_hours

    # Drop rows where irradiance isn't observed.
    weather = weather_df.dropna(subset=[irradiance_col]).sort_index()
    if len(weather) < block_size_hours * 30:
        raise ValueError(
            f"weather history too short to bootstrap (need >= "
            f"{block_size_hours * 30} hours, have {len(weather)})"
        )

    # Pre-compute candidate block starts. A block start is any
    # historical UTC hour from which `block_size_hours` of contiguous
    # hourly observations are available. We allow starts at any UTC
    # hour-of-day (not just midnight) so the block's diurnal phase
    # can be matched to the target's diurnal phase.
    # Build a sorted index of available hours and find which can
    # start a valid block via fast lookahead.
    weather_idx = weather.index
    # Hours since epoch — unit-agnostic via Timedelta arithmetic
    # (pandas internal int64 is ns for some series, µs for others).
    ref = pd.Timestamp("1970-01-01", tz="UTC")
    hours = ((weather_idx - ref) // pd.Timedelta(hours=1)).to_numpy()
    valid_start_mask = np.zeros(len(weather_idx), dtype=bool)
    # An entry i is a valid start iff hours[i:i+block_size] is a
    # consecutive integer sequence.
    if len(hours) >= block_size_hours:
        seq_check = (
            hours[block_size_hours - 1:]
            - hours[:len(hours) - block_size_hours + 1]
        )
        valid_start_mask[:len(seq_check)] = seq_check == (block_size_hours - 1)
    valid_starts_arr = weather_idx[valid_start_mask]
    valid_doys = valid_starts_arr.dayofyear.values
    valid_hours_of_day = valid_starts_arr.hour.values

    rng = np.random.default_rng(rng_seed)
    pv_paths = np.zeros((n_paths, n_hours), dtype=float)

    for block_idx in range(n_blocks):
        target_block_start = target_timestamps[block_idx * block_size_hours]
        target_doy = int(target_block_start.dayofyear)
        target_hod = int(target_block_start.hour)
        # Restrict to candidates that start at the same UTC hour-of-day
        # so the block's diurnal phase aligns with the target.
        hod_mask = valid_hours_of_day == target_hod
        pool_idx = _candidate_pool(
            target_doy, valid_doys[hod_mask],
            valid_starts_arr[hod_mask],
            candidate_window_days,
        )
        # Translate pool_idx (relative to hod-filtered array) back to
        # absolute indices into valid_starts_arr.
        if pool_idx.size > 0:
            abs_idx = np.where(hod_mask)[0][pool_idx]
            pool_idx = abs_idx
        if pool_idx.size == 0:
            # No historical match for that hour-of-day — fall back
            # to expanding the window and ignoring hour alignment.
            pool_idx = _candidate_pool(
                target_doy, valid_doys, valid_starts_arr,
                candidate_window_days * 3,
            )
        if pool_idx.size == 0:
            pool_idx = np.arange(len(valid_starts_arr))

        sampled = rng.choice(pool_idx, size=n_paths, replace=True)

        for path_idx, sample_idx in enumerate(sampled):
            block_start = valid_starts_arr[sample_idx]
            block_end = block_start + pd.Timedelta(hours=block_size_hours)
            irr_block = weather.loc[
                block_start:block_end - pd.Timedelta(seconds=1),
                irradiance_col,
            ].values
            if len(irr_block) != block_size_hours:
                irr_block = irr_block[:block_size_hours]
                if len(irr_block) < block_size_hours:
                    # Pad with zeros if a row was missing — robust fallback.
                    pad = np.zeros(block_size_hours - len(irr_block))
                    irr_block = np.concatenate([irr_block, pad])
            pv_block = np.array([
                estimate_pv_kwh_per_hour(
                    float(g) if np.isfinite(g) else 0.0,
                    capacity_kwp=pv_config.capacity_kwp,
                    tilt_deg=pv_config.tilt_deg,
                    azimuth_deg=pv_config.azimuth_deg,
                    efficiency=pv_config.efficiency,
                )
                for g in irr_block
            ])
            start = block_idx * block_size_hours
            pv_paths[path_idx, start:start + block_size_hours] = pv_block

    return pv_paths


def summarise_paths(
    pv_paths: np.ndarray, alphas: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> dict[str, np.ndarray]:
    """Convenience: marginal quantiles per hour across paths."""
    return {
        f"q{int(a * 100):02d}": np.quantile(pv_paths, a, axis=0)
        for a in alphas
    }
