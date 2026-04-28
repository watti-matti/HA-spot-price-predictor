"""Online bias correction via exponentially-weighted moving average.

Tracks the running mean of *signed* forecast residuals  e_t = actual - forecast
with a half-life parameterised in days. Subtracts the estimate from new
forecasts so slow regime shifts (the 2022 European spike, the 2025-26
recovery, multi-week weather anomalies) are debiased automatically
without retraining the underlying model.

Why an EMA, why signed
----------------------
Two failure modes commonly observed in AR/Ridge price forecasters:

* **Level drift**  — the forecast systematically under- or over-estimates
  the actual for a sustained period (e.g. a heat-wave week).  A signed
  residual mean tracks this and corrects it directly.
* **Heavy-tail noise** — single 500 EUR/MWh spikes can blow up a sample
  mean over a short window. The EMA naturally down-weights old data
  geometrically, but a single huge outlier can still dominate. We
  therefore expose `winsor_limit` to softly clip per-step residuals to
  a multiple of the running absolute mean.

The EMA's half-life corresponds to the time after which a single
observation's weight in the estimate has decayed to half. With
`halflife_days = 20`:

    lambda_per_step = log(2) / (20 * 24)  ~= 1.44e-3

Each step the new estimate is `(1 - lambda) * prev + lambda * err_t`.

A minimum-update gate prevents the corrector from acting on too little
data: until `n_updates >= warmup_steps` (default 168 = 1 week of hourly
observations) `correct(forecast)` returns the input unchanged.

State is JSON-serialisable for persistence across HA restarts.
"""

from __future__ import annotations

import math
from typing import Any


class OnlineBiasCorrector:
    """EMA tracker of signed forecast residuals with cold-start gating.

    Parameters
    ----------
    halflife_days
        Decay half-life of the EMA in days. Larger = slower adaptation,
        less variance, more lag.
    warmup_steps
        Number of `update()` calls required before `correct()` starts
        applying any correction. Until then `correct(f) == f`.
    winsor_limit
        Soft cap on per-step absolute residual, expressed as a multiple
        of the running absolute residual mean. None disables. Default 5.0
        means single observations are clipped at 5x the typical-error
        magnitude — protects against single-step extremes corrupting the
        slow-drift estimator.
    cadence_per_day
        Number of update steps per day. Default 24 = hourly cadence.
        Used to derive the per-step EMA factor from `halflife_days`.
    """

    def __init__(
        self,
        halflife_days: float = 20.0,
        warmup_steps: int = 168,
        winsor_limit: float | None = 5.0,
        cadence_per_day: int = 24,
    ) -> None:
        if halflife_days <= 0:
            raise ValueError(f"halflife_days must be > 0, got {halflife_days}")
        if cadence_per_day <= 0:
            raise ValueError(
                f"cadence_per_day must be > 0, got {cadence_per_day}"
            )
        self.halflife_days = float(halflife_days)
        self.warmup_steps = int(warmup_steps)
        self.winsor_limit = (
            float(winsor_limit) if winsor_limit is not None else None
        )
        self.cadence_per_day = int(cadence_per_day)
        # Per-step EMA decay factor: weight on new observation each step.
        # `(1 - lambda)^n_steps_per_halflife = 0.5`  =>  lambda = 1 - 2^(-1/n)
        n_per_half = self.halflife_days * self.cadence_per_day
        self.lambda_: float = 1.0 - math.pow(0.5, 1.0 / max(1.0, n_per_half))

        self.bias_estimate: float = 0.0
        self.abs_bias_estimate: float = 0.0  # for winsorisation reference
        self.n_updates: int = 0

    # ── Public API ──────────────────────────────────────────────────

    def correct(self, forecast: float) -> float:
        """Return the bias-corrected forecast.

        During warm-up returns `forecast` unchanged. Once warm, returns
        `forecast + bias_estimate` (so a positive `bias_estimate` means
        the model has been under-forecasting on recent data and we shift
        the forecast up).
        """
        if self.n_updates < self.warmup_steps:
            return float(forecast)
        return float(forecast) + self.bias_estimate

    def update(self, forecast: float, actual: float) -> None:
        """Incorporate a new (forecast, actual) observation.

        Computes the signed residual `actual - forecast`, optionally
        winsorises it against the running absolute-residual mean, then
        EMA-updates `bias_estimate`. Also EMA-tracks the absolute
        residual so the winsorisation reference stays current.
        """
        err = float(actual) - float(forecast)
        abs_err = abs(err)

        # Maintain abs-error EMA with the same lambda. Pre-warmup we
        # can't winsorise yet (no reference), so fall through.
        if self.n_updates >= self.warmup_steps and self.winsor_limit:
            cap = self.winsor_limit * max(self.abs_bias_estimate, 1e-9)
            if err > cap:
                err = cap
            elif err < -cap:
                err = -cap
            abs_err = abs(err)

        # EMA updates
        l = self.lambda_
        self.bias_estimate = (1.0 - l) * self.bias_estimate + l * err
        self.abs_bias_estimate = (
            (1.0 - l) * self.abs_bias_estimate + l * abs_err
        )
        self.n_updates += 1

    # ── Diagnostics ─────────────────────────────────────────────────

    @property
    def warm(self) -> bool:
        """Whether `correct()` is actively applying a correction."""
        return self.n_updates >= self.warmup_steps

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "halflife_days": self.halflife_days,
            "warmup_steps": self.warmup_steps,
            "winsor_limit": self.winsor_limit,
            "cadence_per_day": self.cadence_per_day,
            "bias_estimate": self.bias_estimate,
            "abs_bias_estimate": self.abs_bias_estimate,
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OnlineBiasCorrector":
        if d.get("version", 1) != 1:
            raise ValueError(
                f"Unknown OnlineBiasCorrector state version: {d.get('version')}"
            )
        inst = cls(
            halflife_days=d.get("halflife_days", 20.0),
            warmup_steps=d.get("warmup_steps", 168),
            winsor_limit=d.get("winsor_limit", 5.0),
            cadence_per_day=d.get("cadence_per_day", 24),
        )
        inst.bias_estimate = float(d.get("bias_estimate", 0.0))
        inst.abs_bias_estimate = float(d.get("abs_bias_estimate", 0.0))
        inst.n_updates = int(d.get("n_updates", 0))
        return inst
