"""Coordinator-side glue for DtACI online calibration (Phase B.4).

Encapsulates load/save of DtACI state and produces interval bands for
the hourly price forecast. Kept in a separate module so the coordinator
itself stays minimal and the integration is opt-in.

State files live under `<data_dir>/dtaci_state_<key>.json` per zone /
forecaster. Currently only the FI hourly forecaster is wired up (the
zone where the production AR(2) is consumed downstream); SE1, SE3, EE
neighbour states are scaffolded but not active until the validation
matrix is extended to multi-step horizons (see DTACI_ANALYSIS.md
"Future work prioritization").

Activation
----------
The integration is **opt-in** via a coordinator-level config flag
`enable_dtaci` (read from the config entry; default `False` until
production-side validation against thermal-cost outcomes confirms net
benefit). When disabled this module is not imported; when enabled it
is initialised once per coordinator and called from
`_async_update_data` after the hourly forecast list is finalised.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .bias_corrector import OnlineBiasCorrector
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


def load_or_create(state_path: Path,
                   target_coverage: float = 0.9,
                   halflife_days: float = 20.0) -> DtACI:
    """Load DtACI state from `state_path`, or create a fresh instance."""
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
    """Persist a DtACI instance to disk atomically."""
    _atomic_write_json(state_path, instance.to_dict())


def consume_observations(
    instance: DtACI,
    pairs: Iterable[tuple[float, float]],
) -> int:
    """Feed (forecast, actual) pairs to the DtACI instance.

    Returns the number of pairs ingested.
    """
    n = 0
    for forecast, actual in pairs:
        instance.update(forecast, actual)
        n += 1
    return n


def attach_intervals(
    instance: DtACI,
    forecast_entries: list[dict[str, Any]],
    point_field: str = "spot_eur_mwh",
    consumer_field: str = "consumer_eur_kwh",
    spot_to_consumer_eur_kwh=None,
) -> None:
    """Mutate `forecast_entries` in place to add interval band fields.

    Adds these keys per entry:
      * `forecast_lower_eur_mwh` / `forecast_upper_eur_mwh`
      * `forecast_lower_eur_kwh` / `forecast_upper_eur_kwh`
        (only when `spot_to_consumer_eur_kwh` callable is provided —
         takes (spot_eur_mwh, is_night) and returns EUR/kWh)
    """
    for entry in forecast_entries:
        spot = entry.get(point_field)
        if spot is None:
            continue
        low, _point, high = instance.predict_interval(float(spot))
        entry["forecast_lower_eur_mwh"] = round(low, 2)
        entry["forecast_upper_eur_mwh"] = round(high, 2)
        if spot_to_consumer_eur_kwh is not None:
            ts = entry.get("timestamp", "")
            try:
                hour = int(ts[11:13])
                is_night = hour < 7 or hour >= 22
                entry["forecast_lower_eur_kwh"] = round(
                    spot_to_consumer_eur_kwh(low, is_night), 4)
                entry["forecast_upper_eur_kwh"] = round(
                    spot_to_consumer_eur_kwh(high, is_night), 4)
            except (ValueError, IndexError):
                pass


def state_summary(instance: DtACI) -> dict[str, Any]:
    """Compact dict for the sensor's diagnostic attributes."""
    bc = instance.bias_corrector
    return {
        "n_updates": instance.n_updates,
        "effective_coverage": round(instance.effective_coverage, 4),
        "current_half_width": round(instance.current_half_width, 2),
        "dominant_expert_gamma": (
            instance.gammas[instance.dominant_expert]
            if instance.weights else None
        ),
        "bias_estimate": (
            round(bc.bias_estimate, 4) if bc is not None else None
        ),
        "bias_warm": (bc.warm if bc is not None else None),
    }
