"""v2.5.15 — Hourly-level DtACI + bias-corrector wrappers.

The existing `dtaci.DtACI` and `bias_corrector.OnlineBiasCorrector` are
generic enough to wrap an hourly point forecast and its fan-chart
quantile bands directly. This module provides three small helpers that
compose them into per-hour pipeline use:

  * :class:`HourlyBiasCorrector` — wraps a single :class:`OnlineBiasCorrector`
    tuned for hourly cadence (cadence=24/day, halflife=14 days, warmup
    7 days). Apply after L1+L2+L3 mean prediction and BEFORE the
    softplus floor. Closes the systematic-bias loop that v2.5.14
    documented as currently open.

  * :class:`HourlyFanChartCalibrator` — maintains a small set of DtACI
    instances, one per (left, right) symmetric coverage band the user
    wants to publish. For example, target_coverages=(0.5, 0.9) yields
    P25–P75 and P5–P95 bands whose ACTUAL realised coverage tracks the
    nominal target over time. Replaces / complements the model-based
    GPD POT fan chart shipped in v2.5.14 — when both are present,
    DtACI's realised-coverage guarantee wins out as a calibration
    cross-check.

  * :class:`RefitMonitor` — accumulates realised coverage error across
    bands. When the deviation from target exceeds a threshold for
    N consecutive observations, emits a refit-recommended event. Does
    NOT auto-trigger anything in production — the operator validates
    and runs the offline refit script.

All three are pure-numpy / pure-Python, serialisable via to_dict /
from_dict, no external state outside the integration's `.storage/`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import dtaci as _dtaci_mod
from . import bias_corrector as _bias_mod


# ── 1. Hourly bias corrector ───────────────────────────────────────


@dataclass
class HourlyBiasCorrector:
    """Thin wrapper over `OnlineBiasCorrector` with hourly defaults.

    Wraps the L1+L2+L3 mean prediction so that systematic offset (e.g.
    "predictions consistently 2 EUR/MWh too high in spring") is
    eliminated over time. EMA halflife 14 days (≈ 336 hours), 7-day
    warmup. Winsorisation at 5× absolute-residual EMA so single price
    spikes don't poison the bias estimate.
    """
    halflife_days: float = 14.0
    warmup_hours: int = 7 * 24
    winsor_limit: float | None = 5.0
    _inner: _bias_mod.OnlineBiasCorrector | None = None

    def __post_init__(self) -> None:
        if self._inner is None:
            self._inner = _bias_mod.OnlineBiasCorrector(
                halflife_days=self.halflife_days,
                warmup_steps=self.warmup_hours,
                winsor_limit=self.winsor_limit,
                cadence_per_day=24,
            )

    def correct(self, forecast: float) -> float:
        return float(self._inner.correct(float(forecast)))

    def update(self, forecast: float, actual: float) -> None:
        self._inner.update(float(forecast), float(actual))

    @property
    def warm(self) -> bool:
        return bool(self._inner.warm)

    @property
    def bias_estimate(self) -> float:
        return float(self._inner.bias_estimate)

    def to_dict(self) -> dict:
        return {
            "kind": "HourlyBiasCorrector",
            "halflife_days": self.halflife_days,
            "warmup_hours": self.warmup_hours,
            "winsor_limit": self.winsor_limit,
            "inner": self._inner.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HourlyBiasCorrector":
        obj = cls(
            halflife_days=float(d.get("halflife_days", 14.0)),
            warmup_hours=int(d.get("warmup_hours", 168)),
            winsor_limit=d.get("winsor_limit", 5.0),
        )
        obj._inner = _bias_mod.OnlineBiasCorrector.from_dict(d["inner"])
        return obj


# ── 1b. Per-hour-of-day bias corrector ─────────────────────────────


@dataclass
class PerHourBiasCorrector:
    """24 independent `OnlineBiasCorrector` instances, one per UTC
    hour-of-day.

    Rationale (studies/results/phase2_bias_decomposition.md): after a
    single global bias EMA, the residual bias of the v2.12 pipeline still
    carries a strong hour-of-day shape (morning +4…+7, night −3…−4,
    afternoon −2…−4 EUR/MWh) that a global scalar cannot express because
    it averages to ~0. Binning the EMA by hour-of-day removed it
    (harness walk-forward: 19.40 → 18.72 MAE vs the global corrector,
    midday bias +4.6 → +0.4).

    Each bin sees ~1 observation/day, so the inner correctors run at
    ``cadence_per_day=1`` with the same 14-day halflife and warm up
    after `warmup_updates` daily observations (default 14). Same 5×
    winsorisation so single spikes don't poison a bin.

    Migration: `from_dict` accepts a legacy `HourlyBiasCorrector` state
    dict and seeds every bin with the legacy global bias estimate (and
    its warm status), so behaviour at switchover matches the old
    corrector and then specialises per hour.
    """
    halflife_days: float = 14.0
    warmup_updates: int = 14
    winsor_limit: float | None = 5.0
    _inner: dict | None = None

    def __post_init__(self) -> None:
        if self._inner is None:
            self._inner = {h: self._new_bin() for h in range(24)}

    def _new_bin(self) -> _bias_mod.OnlineBiasCorrector:
        return _bias_mod.OnlineBiasCorrector(
            halflife_days=self.halflife_days,
            warmup_steps=self.warmup_updates,
            winsor_limit=self.winsor_limit,
            cadence_per_day=1,
        )

    def correct(self, forecast: float, hour: int) -> float:
        return float(self._inner[int(hour) % 24].correct(float(forecast)))

    def update(self, forecast: float, actual: float, hour: int) -> None:
        self._inner[int(hour) % 24].update(float(forecast), float(actual))

    @property
    def warm(self) -> bool:
        """True once ANY hour bin is actively correcting."""
        return any(b.warm for b in self._inner.values())

    @property
    def bias_estimate(self) -> float:
        """Mean bias estimate across warm bins (diagnostic scalar)."""
        warm = [b.bias_estimate for b in self._inner.values() if b.warm]
        return float(np.mean(warm)) if warm else 0.0

    @property
    def bias_by_hour(self) -> dict[int, float]:
        """Per-hour applied correction (0.0 for bins still warming)."""
        return {h: (float(b.bias_estimate) if b.warm else 0.0)
                for h, b in self._inner.items()}

    def to_dict(self) -> dict:
        return {
            "kind": "PerHourBiasCorrector",
            "halflife_days": self.halflife_days,
            "warmup_updates": self.warmup_updates,
            "winsor_limit": self.winsor_limit,
            "inner": {str(h): b.to_dict() for h, b in self._inner.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerHourBiasCorrector":
        if d.get("kind") == "HourlyBiasCorrector":
            # Legacy single-EMA state — seed all bins with the global
            # estimate so switchover is behaviour-preserving.
            legacy = _bias_mod.OnlineBiasCorrector.from_dict(d["inner"])
            obj = cls()
            for b in obj._inner.values():
                b.bias_estimate = float(legacy.bias_estimate)
                b.abs_bias_estimate = float(legacy.abs_bias_estimate)
                b.n_updates = obj.warmup_updates if legacy.warm else 0
            return obj
        obj = cls(
            halflife_days=float(d.get("halflife_days", 14.0)),
            warmup_updates=int(d.get("warmup_updates", 14)),
            winsor_limit=d.get("winsor_limit", 5.0),
            _inner={},
        )
        obj._inner = {
            int(h): _bias_mod.OnlineBiasCorrector.from_dict(ser)
            for h, ser in d.get("inner", {}).items()
        }
        for h in range(24):   # guard against partial state
            obj._inner.setdefault(h, obj._new_bin())
        return obj


# ── 2. Per-hour fan-chart calibrator ──────────────────────────────


@dataclass
class HourlyFanChartCalibrator:
    """Maintain one DtACI instance per (symmetric) coverage band.

    Target coverages ``(0.5, 0.9)`` produce, after sufficient warmup:
      band_0.5 = (P25, P75)         (50 % central interval)
      band_0.9 = (P5,  P95)         (90 % central interval)

    The realised coverage of each band tracks its nominal target via
    DtACI's discounted-loss expert reweighting. When the underlying
    forecast distribution shifts, band widths automatically adapt.

    Composes with v2.5.14's GPD POT fan chart — the GPD bands are the
    PRIOR (model-based, calibrated on training tail); DtACI bands are
    the POSTERIOR calibration based on realised coverage.
    """
    target_coverages: tuple[float, ...] = (0.5, 0.9)
    window: int = 720           # ~30 days of hourly conformity scores
    min_warmup: int = 24
    eta: float = 5.0
    rho: float = 0.99
    instances: dict[float, _dtaci_mod.DtACI] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instances:
            for tc in self.target_coverages:
                self.instances[tc] = _dtaci_mod.DtACI(
                    target_coverage=tc,
                    window=self.window,
                    min_warmup=self.min_warmup,
                    eta=self.eta,
                    rho=self.rho,
                )

    def predict_bands(self, forecast: float) -> dict[float, tuple[float, float]]:
        """For each target coverage, return (lower, upper) band edges."""
        out: dict[float, tuple[float, float]] = {}
        for tc, inst in self.instances.items():
            lo, _, hi = inst.predict_interval(float(forecast))
            out[tc] = (float(lo), float(hi))
        return out

    def update(self, forecast: float, actual: float) -> None:
        for inst in self.instances.values():
            inst.update(float(forecast), float(actual))

    def diagnostics(self) -> dict[float, dict[str, float]]:
        return {
            tc: {
                "effective_alpha":    float(inst.effective_alpha),
                "effective_coverage": float(inst.effective_coverage),
                "current_half_width": float(inst.current_half_width),
                "dominant_gamma":     float(inst.dominant_gamma),
                "weight_entropy_bits": float(inst.weight_entropy_bits),
            }
            for tc, inst in self.instances.items()
        }

    def to_dict(self) -> dict:
        return {
            "kind": "HourlyFanChartCalibrator",
            "target_coverages": list(self.target_coverages),
            "window":     self.window,
            "min_warmup": self.min_warmup,
            "eta":        self.eta,
            "rho":        self.rho,
            "instances":  {str(tc): inst.to_dict()
                           for tc, inst in self.instances.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HourlyFanChartCalibrator":
        obj = cls(
            target_coverages=tuple(d.get("target_coverages", (0.5, 0.9))),
            window=int(d.get("window", 720)),
            min_warmup=int(d.get("min_warmup", 24)),
            eta=float(d.get("eta", 5.0)),
            rho=float(d.get("rho", 0.99)),
            instances={},
        )
        for k, ser in d.get("instances", {}).items():
            obj.instances[float(k)] = _dtaci_mod.DtACI.from_dict(ser)
        return obj


# ── 3. Refit-recommendation monitor ───────────────────────────────


@dataclass
class RefitMonitor:
    """Watch the realised coverage of a DtACI / calibrator. When the
    deviation from target exceeds ``drift_pp`` for ``persistence``
    consecutive observations, set a ``refit_recommended`` flag.

    Does NOT trigger any retraining itself. The integration polls the
    flag at the end of each coordinator cycle; the operator decides
    whether to run the offline refit script.

    Default: drift > 5 pp from target for 14 consecutive days at hourly
    cadence (336 hours). Anything below that is normal sampling noise.
    """
    target_coverage: float = 0.9
    drift_pp: float = 0.05
    persistence_steps: int = 14 * 24
    _consecutive: int = 0
    _refit_recommended: bool = False
    _trigger_history: list = field(default_factory=list)

    def observe(self, realised_coverage: float, timestamp_iso: str | None = None
                ) -> None:
        deviation = abs(float(realised_coverage) - self.target_coverage)
        if deviation > self.drift_pp:
            self._consecutive += 1
            if self._consecutive >= self.persistence_steps:
                if not self._refit_recommended:
                    self._refit_recommended = True
                    self._trigger_history.append({
                        "timestamp": timestamp_iso,
                        "realised_coverage": float(realised_coverage),
                        "deviation_pp": float(deviation),
                    })
        else:
            self._consecutive = 0
            self._refit_recommended = False

    @property
    def refit_recommended(self) -> bool:
        return bool(self._refit_recommended)

    @property
    def consecutive_drift_hours(self) -> int:
        return int(self._consecutive)

    @property
    def trigger_history(self) -> list:
        return list(self._trigger_history)

    def reset(self) -> None:
        self._consecutive = 0
        self._refit_recommended = False

    def to_dict(self) -> dict:
        return {
            "kind": "RefitMonitor",
            "target_coverage": self.target_coverage,
            "drift_pp":        self.drift_pp,
            "persistence_steps": self.persistence_steps,
            "_consecutive":   self._consecutive,
            "_refit_recommended": self._refit_recommended,
            "_trigger_history": list(self._trigger_history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RefitMonitor":
        obj = cls(
            target_coverage=float(d.get("target_coverage", 0.9)),
            drift_pp=float(d.get("drift_pp", 0.05)),
            persistence_steps=int(d.get("persistence_steps", 336)),
        )
        obj._consecutive = int(d.get("_consecutive", 0))
        obj._refit_recommended = bool(d.get("_refit_recommended", False))
        obj._trigger_history = list(d.get("_trigger_history", []))
        return obj
