"""Coordinator-side glue for the DtACI per-D(i) calibration bundle.

Encapsulates load/save of the `DkDtACIBundle` state and produces
calibrated per-(direction, k) bands for the duration-forecast sensor.

State files live under `<data_dir>/dtaci_dk_<zone>.json` per zone.
Atomic writes (temp file + os.replace) prevent corruption on
power-cut. Schema is forward-compatible: missing instances within a
bundle cold-start when load fails.

Activation is opt-in via a coordinator-level config flag
`enable_dtaci_dk` (default `False` until production-side validation
against thermal-cost outcomes confirms net benefit).

The `DTACI_ANALYSIS_V2.md` walk-forward backtest (FI, SE1, SE3, EE,
700–914 days each) shows DtACI closes the coverage gap by 1.3–3.7 pp
on the three under-covering zones and reduces point-forecast MAE by
4–13% on every zone. That evidence supports turning the flag on; the
gate stays in place to de-risk the production rollout.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .bias_corrector import OnlineBiasCorrector
from .dk_dtaci import DkDtACIBundle
from .dtaci import DtACI

_LOGGER = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically — temp file + os.replace.

    Prevents corruption if the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".dtaci.", suffix=".json",
                                dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_or_create_bundle(
    state_path: Path, target_coverage: float = 0.9,
) -> DkDtACIBundle:
    """Load `DkDtACIBundle` state from `state_path`, or create a fresh one."""
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return DkDtACIBundle.from_dict(json.load(f))
        except Exception as exc:
            _LOGGER.warning(
                "DkDtACIBundle state %s unreadable (%s); starting fresh",
                state_path, exc,
            )
    return DkDtACIBundle(target_coverage=target_coverage)


def save_bundle(state_path: Path, bundle: DkDtACIBundle) -> None:
    """Persist a `DkDtACIBundle` to disk atomically."""
    _atomic_write_json(state_path, bundle.to_dict())


def attach_dk_intervals(
    bundle: DkDtACIBundle,
    daily_forecast: list[dict[str, Any]],
) -> None:
    """Mutate `daily_forecast` in place to add per-(direction, k) bands.

    For each day entry that has both `dk_cheap_eur_kwh[12]` and
    `dk_peak_eur_kwh[12]`, we read the forecast arrays and write back:

        dk_cheap_lower_eur_kwh[12]
        dk_cheap_upper_eur_kwh[12]
        dk_peak_lower_eur_kwh[12]
        dk_peak_upper_eur_kwh[12]

    During the bundle's warmup the bands collapse to the point — by
    design (no spurious confidence before enough observations).

    Note: the bundle's internal scoring is in whatever unit the caller
    feeds it on `update()`. By convention we keep the unit consistent
    with the forecast arrays here (EUR/kWh), so the bands are also in
    EUR/kWh.
    """
    for day in daily_forecast:
        cheap = day.get("dk_cheap_eur_kwh") or []
        peak = day.get("dk_peak_eur_kwh") or []
        if len(cheap) < 12 or len(peak) < 12:
            continue
        bands = bundle.predict_intervals(list(cheap[:12]), list(peak[:12]))
        day["dk_cheap_lower_eur_kwh"] = [round(v, 4) for v in bands["cheap"]["lower"]]
        day["dk_cheap_upper_eur_kwh"] = [round(v, 4) for v in bands["cheap"]["upper"]]
        day["dk_peak_lower_eur_kwh"] = [round(v, 4) for v in bands["peak"]["lower"]]
        day["dk_peak_upper_eur_kwh"] = [round(v, 4) for v in bands["peak"]["upper"]]


def update_from_actuals(
    bundle: DkDtACIBundle,
    forecast_history: dict[str, dict[str, list[float]]],
    actual_dk_history: dict[str, dict[str, list[float]]],
) -> int:
    """Feed reconciled (forecast, actual) D(k) pairs to the bundle.

    Args:
        forecast_history: {date_str: {"cheap": [12], "peak": [12]}}
            forecast issued for that day, captured at the time the
            forecast cycle ran.
        actual_dk_history: {date_str: {"cheap": [12], "peak": [12]}}
            computed once that day's 24 hourly actuals reconcile.

    Returns the number of (date, direction, k) updates performed.
    """
    n = 0
    for date_str, actual in actual_dk_history.items():
        if date_str not in forecast_history:
            continue
        f = forecast_history[date_str]
        try:
            bundle.update(
                forecast_dk_cheap=f["cheap"],
                forecast_dk_peak=f["peak"],
                actual_dk_cheap=actual["cheap"],
                actual_dk_peak=actual["peak"],
            )
            n += 1
        except (KeyError, ValueError) as exc:
            _LOGGER.warning(
                "DkDtACIBundle update failed for %s: %s", date_str, exc,
            )
    return n


def diagnostics_summary(bundle: DkDtACIBundle) -> dict[str, Any]:
    """Compact dict for the duration-forecast sensor's diagnostic block.

    Mirrors the parameter set the reference UI card expects:
      * mean_coverage, mean_width, n_warm_instances
      * per-(direction, k) breakdown of coverage / bias_ema /
        alpha_agg / dominant_gamma / weight_entropy_bits / half_width
    """
    return bundle.diagnostics()


# Legacy hourly-DtACI path (Phase B v1) -- kept for one transition
# release in case downstream consumers wire up the v1 hourly intervals.
# New code should use the per-D(i) bundle above.

def load_or_create(state_path: Path,
                   target_coverage: float = 0.9,
                   halflife_days: float = 20.0) -> DtACI:
    """Legacy: hourly DtACI (single instance). Prefer `load_or_create_bundle`."""
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return DtACI.from_dict(json.load(f))
        except Exception as exc:
            _LOGGER.warning(
                "DtACI state %s unreadable (%s); starting fresh",
                state_path, exc,
            )
    bc = OnlineBiasCorrector(halflife_days=halflife_days,
                             warmup_steps=168, cadence_per_day=24)
    return DtACI(target_coverage=target_coverage,
                 window=720, min_warmup=24, bias_corrector=bc)


def save(state_path: Path, instance: DtACI) -> None:
    """Legacy hourly DtACI persistence."""
    _atomic_write_json(state_path, instance.to_dict())
