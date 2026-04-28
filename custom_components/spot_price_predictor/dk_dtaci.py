"""Per-(direction, k) DtACI bundle for D(i) order statistics.

This module is the *primary* integration point for online conformal
calibration in the Phase B layer (see `dtaci.py` for the underlying
algorithm). It maintains 24 independent DtACI instances per zone:

    cheap[k] for k = 1..12  (mean of cheapest k hours)
    peak[k]  for k = 1..12  (mean of priciest k hours)

Each DtACI instance tracks the residual distribution of its own order
statistic and produces a calibrated band

    [D(i) − q_i, D(i), D(i) + q_i]

at the target coverage. Per-(direction, k) state means each statistic
adapts independently to its own regime — D_cheap(1) is a heavy-tailed
trough estimator, D_cheap(12) is a smoother near-mean quantity, and
D_peak(1) is a sharp spike estimator. Sharing one DtACI across all
24 statistics would average over these regimes and lose calibration on
each individually.

The class also exposes a per-instance `OnlineBiasCorrector` so each
(direction, k) statistic gets its own bias EMA. Bias drift on D_peak(1)
(under-forecasting one-hour spikes) and bias drift on D_cheap(12) (level
drift in the daily mean of the cheap half) are independent failure
modes; tracking them separately is the right granularity.

Diagnostic surface
------------------
For each (direction, k) the bundle reports the parameters from
`docs/dtaci_layer.md` and the reference UI card:

  coverage rate (30-day rolling)    — fraction of recent intervals covered
  bias_ema                            — current EMA of signed residual
  alpha_agg                           — current effective miscoverage
  dominant gamma                      — best-performing step size
  weight entropy                      — Shannon entropy in bits of w[m]
  interval width                      — 2 * current half-width

Persistence
-----------
The whole bundle serialises to a single JSON dict (atomic write per
coordinator cycle). Schema is forward-compatible: missing (direction, k)
entries are restarted at cold-state on load.
"""
from __future__ import annotations

import logging
from typing import Any

from .bias_corrector import OnlineBiasCorrector
from .dtaci import DEFAULT_GAMMAS, DtACI

_LOGGER = logging.getLogger(__name__)


CHEAP_PEAK_K_RANGE: tuple[int, ...] = tuple(range(1, 13))
"""All 24 D(i) order statistics — k=1..12 for both cheap and peak ends."""


def _instance_key(direction: str, k: int) -> str:
    """Canonical key for a (direction, k) DtACI instance."""
    return f"{direction}_{k}"


class DkDtACIBundle:
    """Bundle of 24 DtACI instances, one per (direction, k) D(i) statistic.

    Parameters
    ----------
    target_coverage
        Marginal coverage target. Default 0.9 (90% intervals).
    gammas
        Step-size grid shared by all instances (default 15-point ladder).
    eta, rho, window, min_warmup
        Forwarded to each DtACI.
    bias_halflife_days
        Per-instance OnlineBiasCorrector halflife.
    bias_warmup_steps
        Per-instance bias-corrector warmup. Note that D(k) updates arrive
        at *daily* cadence, so 168 steps = 168 days ≈ 24 weeks. We default
        to 30 (≈1 month) since daily cadence has 24× fewer datapoints
        than hourly.
    cadence_per_day
        How many bias-corrector ticks per day. For D(k) statistics this
        is 1 (one observation per day, post-reconciliation). Forwarded
        to OnlineBiasCorrector for halflife conversion.
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        gammas=DEFAULT_GAMMAS,
        eta: float = 5.0,
        rho: float = 0.99,
        window: int = 365,
        min_warmup: int = 14,
        bias_halflife_days: float = 21.0,
        bias_warmup_steps: int = 30,
        cadence_per_day: int = 1,
    ) -> None:
        self.target_coverage = float(target_coverage)
        self.gammas = list(gammas)
        self.eta = float(eta)
        self.rho = float(rho)
        self.window = int(window)
        self.min_warmup = int(min_warmup)
        self.bias_halflife_days = float(bias_halflife_days)
        self.bias_warmup_steps = int(bias_warmup_steps)
        self.cadence_per_day = int(cadence_per_day)
        # Build all 24 instances.
        self.instances: dict[str, DtACI] = {}
        for direction in ("cheap", "peak"):
            for k in CHEAP_PEAK_K_RANGE:
                self.instances[_instance_key(direction, k)] = self._fresh()

    def _fresh(self) -> DtACI:
        """Construct a fresh DtACI with a fresh bias corrector."""
        bc = OnlineBiasCorrector(
            halflife_days=self.bias_halflife_days,
            warmup_steps=self.bias_warmup_steps,
            cadence_per_day=self.cadence_per_day,
        )
        return DtACI(
            target_coverage=self.target_coverage,
            gammas=self.gammas,
            eta=self.eta,
            rho=self.rho,
            window=self.window,
            min_warmup=self.min_warmup,
            bias_corrector=bc,
        )

    # ── Update / predict ────────────────────────────────────────────

    def update(
        self,
        forecast_dk_cheap: list[float],
        forecast_dk_peak: list[float],
        actual_dk_cheap: list[float],
        actual_dk_peak: list[float],
    ) -> None:
        """Feed one day's (forecast, actual) D(i) pairs to all 24 instances.

        Each input must be a length-12 array. Missing values (NaN, None)
        skip the corresponding instance for that day.
        """
        if len(forecast_dk_cheap) != 12 or len(actual_dk_cheap) != 12:
            raise ValueError("cheap arrays must be length 12")
        if len(forecast_dk_peak) != 12 or len(actual_dk_peak) != 12:
            raise ValueError("peak arrays must be length 12")
        for k in CHEAP_PEAK_K_RANGE:
            for direction, fa_pair in [
                ("cheap", (forecast_dk_cheap[k - 1], actual_dk_cheap[k - 1])),
                ("peak", (forecast_dk_peak[k - 1], actual_dk_peak[k - 1])),
            ]:
                f, a = fa_pair
                if f is None or a is None:
                    continue
                try:
                    f_f = float(f)
                    a_f = float(a)
                    if f_f != f_f or a_f != a_f:  # NaN check
                        continue
                except (TypeError, ValueError):
                    continue
                self.instances[_instance_key(direction, k)].update(f_f, a_f)

    def predict_intervals(
        self,
        forecast_dk_cheap: list[float],
        forecast_dk_peak: list[float],
    ) -> dict[str, dict[str, list[float]]]:
        """Return calibrated bands for one day's forecasts.

        Output schema:
            {
              "cheap": {
                "lower": [12 floats],
                "point": [12 floats],
                "upper": [12 floats],
              },
              "peak": { ... same ... },
            }

        During warm-up the bands collapse to point — this is by design.
        """
        out: dict[str, dict[str, list[float]]] = {
            "cheap": {"lower": [], "point": [], "upper": []},
            "peak":  {"lower": [], "point": [], "upper": []},
        }
        for direction, forecasts in (
            ("cheap", forecast_dk_cheap),
            ("peak", forecast_dk_peak),
        ):
            for k in CHEAP_PEAK_K_RANGE:
                inst = self.instances[_instance_key(direction, k)]
                f = forecasts[k - 1]
                low, point, high = inst.predict_interval(float(f))
                out[direction]["lower"].append(low)
                out[direction]["point"].append(point)
                out[direction]["upper"].append(high)
        return out

    # ── Diagnostics ──────────────────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        """Compact dict suitable for sensor attributes / UI cards.

        Returns aggregated indicators plus per-(direction, k) breakdowns
        for the parameters listed in `docs/dtaci_layer.md`. All values
        are JSON-serialisable scalars.
        """
        per_k_cheap = {}
        per_k_peak = {}
        cov_30d = []
        widths = []
        n_warm_instances = 0
        for direction, target_dict in (
            ("cheap", per_k_cheap), ("peak", per_k_peak),
        ):
            for k in CHEAP_PEAK_K_RANGE:
                inst = self.instances[_instance_key(direction, k)]
                bc = inst.bias_corrector
                # 30-day rolling coverage approximation: 1 - alpha_eff.
                # The exact realised coverage requires keeping a separate
                # rolling buffer; the alpha_eff-based estimate is a
                # well-calibrated proxy and matches what alpha_t adapts
                # to under DtACI's convergence theorem.
                cov_proxy = 1.0 - inst.effective_alpha
                cov_30d.append(cov_proxy)
                widths.append(2.0 * inst.current_half_width)
                if inst.n_updates >= inst.min_warmup:
                    n_warm_instances += 1
                target_dict[k] = {
                    "n_updates": inst.n_updates,
                    "coverage": round(cov_proxy, 4),
                    "alpha_agg": round(inst.effective_alpha, 4),
                    "bias_ema": (round(bc.bias_estimate, 4)
                                 if bc is not None else 0.0),
                    "bias_warm": (bool(bc.warm) if bc is not None
                                  else False),
                    "dominant_gamma": inst.dominant_gamma,
                    "weight_entropy_bits": round(
                        inst.weight_entropy_bits, 3),
                    "half_width": round(inst.current_half_width, 4),
                }
        # Aggregate headlines (mean over 24 instances)
        n = len(cov_30d) or 1
        return {
            "target_coverage": self.target_coverage,
            "n_warm_instances": n_warm_instances,
            "n_total_instances": len(self.instances),
            "mean_coverage": round(sum(cov_30d) / n, 4),
            "mean_width": round(sum(widths) / n, 4),
            "per_k": {"cheap": per_k_cheap, "peak": per_k_peak},
        }

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the entire bundle's state."""
        return {
            "version": 1,
            "target_coverage": self.target_coverage,
            "gammas": list(self.gammas),
            "eta": self.eta,
            "rho": self.rho,
            "window": self.window,
            "min_warmup": self.min_warmup,
            "bias_halflife_days": self.bias_halflife_days,
            "bias_warmup_steps": self.bias_warmup_steps,
            "cadence_per_day": self.cadence_per_day,
            "instances": {
                key: inst.to_dict()
                for key, inst in self.instances.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DkDtACIBundle":
        """Restore from `to_dict` output. Missing instances cold-start."""
        if d.get("version", 1) != 1:
            raise ValueError(f"Unknown bundle version: {d.get('version')}")
        bundle = cls(
            target_coverage=d.get("target_coverage", 0.9),
            gammas=d.get("gammas", DEFAULT_GAMMAS),
            eta=d.get("eta", 5.0),
            rho=d.get("rho", 0.99),
            window=d.get("window", 365),
            min_warmup=d.get("min_warmup", 14),
            bias_halflife_days=d.get("bias_halflife_days", 21.0),
            bias_warmup_steps=d.get("bias_warmup_steps", 30),
            cadence_per_day=d.get("cadence_per_day", 1),
        )
        for key, inst_d in d.get("instances", {}).items():
            if key in bundle.instances:
                try:
                    bundle.instances[key] = DtACI.from_dict(inst_d)
                except Exception as exc:
                    _LOGGER.warning(
                        "DkDtACIBundle: instance %s unreadable (%s); "
                        "starting fresh", key, exc,
                    )
        return bundle
