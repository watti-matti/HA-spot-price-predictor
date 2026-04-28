"""Dynamically-tuned Adaptive Conformal Inference (DtACI).

Reference: Gibbs & Candes, "Conformal Inference for Online Prediction
with Arbitrary Distribution Shifts", JMLR 25 (2024), paper 22-1218.

DtACI wraps an arbitrary point forecaster and produces:

  1. **Calibrated prediction intervals**  [low, point, high] that achieve
     the target marginal coverage (e.g. 90%) under arbitrary distribution
     shift, without requiring prior knowledge of the shift magnitude.

  2. **An online adaptive miscoverage rate**  alpha_eff[t] which is the
     dynamically-weighted combination of K expert miscoverage rates,
     each with a fixed step size gamma_k.

The algorithm maintains, at every step t:

  * `alpha_k[t]`   — expert k's current miscoverage threshold, updated by
                     ACI rule:
                         alpha_k[t+1] = alpha_k[t] + gamma_k * (alpha - err_k[t])
                     where err_k[t] = 1 if interval at level (1-alpha_k[t])
                     does NOT cover the actual; 0 otherwise.
                     alpha = 1 - target_coverage (target miscoverage).

  * `weight_k[t]` — exponentially-weighted-majority weight, updated on
                     the pinball loss of expert k's quantile prediction:
                         L_k(t) = alpha * (s_t - q_k)^+ + (1-alpha) * (q_k - s_t)^+
                     where s_t is the conformity score and q_k is the
                     quantile of recent scores at expert k's level.
                     Weights are renormalised to sum to 1.

  * `score_window` — rolling buffer of the last `window` conformity scores;
                     used to compute the empirical quantile that turns
                     alpha into an interval half-width.

Coverage guarantees: under bounded scores, DtACI's long-run realised
coverage converges to the target — see Theorem 1 of the paper.

Implementation notes
--------------------
* Pure Python, no numpy / scipy. Suitable for the HA custom component
  runtime constraint (no extra deps beyond stdlib).
* Symmetric scoring: s = |y - y_hat|. The interval is therefore
  symmetric around the point forecast: [y_hat - q, y_hat + q]. For
  asymmetric distributions this loses some efficiency but is robust
  and well-defined.
* The bias_corrector (if attached) is applied *before* scoring on the
  way in (so the conformity score is computed against the debiased
  forecast) and *before* the interval is centred on the way out.
* State is JSON-serialisable via to_dict / from_dict so the coordinator
  can persist DtACI between restarts.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable


DEFAULT_GAMMAS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1)
"""Default DtACI step sizes — span 2 orders of magnitude.

Smaller gammas = slower adaptation, suited to stationary regimes.
Larger gammas = faster adaptation, suited to abrupt shifts.
The expert-weighting layer dynamically picks whichever value is
performing best on the recent pinball loss.
"""


def _empirical_quantile(values: list[float], level: float) -> float:
    """Linear-interpolation empirical quantile.

    `level` is in [0, 1].  level=0.9 = 90th percentile.
    For an empty list returns 0.0 (cold-start: no interval).
    For singletons returns the single value.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    n = len(s)
    # NumPy "linear" interpolation (default):
    #   pos = level * (n - 1)
    pos = level * (n - 1)
    if pos <= 0:
        return float(s[0])
    if pos >= n - 1:
        return float(s[-1])
    lo = int(pos)
    frac = pos - lo
    return float(s[lo]) * (1.0 - frac) + float(s[lo + 1]) * frac


def _pinball_loss(score: float, q: float, alpha: float) -> float:
    """Pinball loss at quantile level (1-alpha) for a single observation.

    L(s, q) = alpha * max(s - q, 0) + (1 - alpha) * max(q - s, 0)

    Minimised when q is the (1-alpha) quantile of the score distribution.
    Used by DtACI to score expert quality.
    """
    if score >= q:
        return alpha * (score - q)
    return (1.0 - alpha) * (q - score)


class DtACI:
    """Dynamically-tuned Adaptive Conformal Inference wrapper.

    Parameters
    ----------
    target_coverage
        Desired marginal coverage, e.g. 0.9 for 90% intervals.
        Internally, target miscoverage `alpha = 1 - target_coverage`.
    gammas
        Per-expert step sizes. Defaults to DEFAULT_GAMMAS.
    eta
        Learning rate for expert weights' exponential update.
    window
        Number of recent conformity scores to retain for empirical
        quantile estimation. 720 = 30 days at hourly cadence.
    min_warmup
        Number of updates required before `predict_interval` returns a
        non-trivial half-width (before this, the interval collapses to
        the point forecast). Prevents over-confident bands during cold
        start.
    bias_corrector
        Optional object exposing `correct(forecast)` and `update(forecast,
        actual)` methods. If provided, every forecast is debiased before
        scoring / centering.
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        gammas: Iterable[float] = DEFAULT_GAMMAS,
        eta: float = 0.1,
        window: int = 720,
        min_warmup: int = 24,
        bias_corrector: Any = None,
    ) -> None:
        if not (0.0 < target_coverage < 1.0):
            raise ValueError(
                f"target_coverage must be in (0, 1), got {target_coverage}"
            )
        self.target_coverage = float(target_coverage)
        self.alpha_target = 1.0 - self.target_coverage
        self.gammas: list[float] = [float(g) for g in gammas]
        if not self.gammas:
            raise ValueError("gammas must not be empty")
        self.eta = float(eta)
        self.window = int(window)
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        self.min_warmup = int(min_warmup)
        self.bias_corrector = bias_corrector

        K = len(self.gammas)
        # Each expert starts at the target miscoverage rate.
        self.alphas: list[float] = [self.alpha_target] * K
        # Uniform weights at start.
        self.weights: list[float] = [1.0 / K] * K
        # Rolling conformity score buffer.
        self.score_window: deque[float] = deque(maxlen=self.window)
        # Cumulative count of updates (used for warmup gating).
        self.n_updates: int = 0

    # ── Internal helpers ─────────────────────────────────────────────

    def _combined_alpha(self) -> float:
        """Weighted average of expert alphas. Clamped to (eps, 1-eps)."""
        a = sum(w * a for w, a in zip(self.weights, self.alphas))
        eps = 1e-6
        return max(eps, min(1.0 - eps, a))

    def _quantile_for_alpha(self, alpha: float) -> float:
        """Empirical quantile of recent scores at level (1 - alpha)."""
        return _empirical_quantile(list(self.score_window), 1.0 - alpha)

    # ── Public API ───────────────────────────────────────────────────

    def predict_interval(self, forecast: float) -> tuple[float, float, float]:
        """Return (lower, point, upper) interval for a new forecast.

        The point is the (optionally debiased) forecast. The half-width
        is the empirical (1-alpha_eff) quantile of the rolling score
        window. During cold-start (n_updates < min_warmup) the interval
        collapses to the point.
        """
        if self.bias_corrector is not None:
            point = float(self.bias_corrector.correct(forecast))
        else:
            point = float(forecast)

        if self.n_updates < self.min_warmup or not self.score_window:
            return point, point, point

        alpha_eff = self._combined_alpha()
        half_width = self._quantile_for_alpha(alpha_eff)
        return point - half_width, point, point + half_width

    def update(self, forecast: float, actual: float) -> None:
        """Update DtACI with one observed (forecast, actual) pair.

        Steps:
          1. Apply bias correction (and update its tracker).
          2. Compute conformity score s = |actual - debiased_forecast|.
          3. For each expert k:
             a. Determine whether expert k's interval would have covered:
                err_k = 1 if s > q_k where q_k = quantile at level (1-alpha_k)
             b. ACI update of alpha_k.
             c. Compute pinball loss L_k = pinball(s, q_k, alpha_target).
          4. Multiplicative-weights update on expert weights using L_k.
          5. Append s to the score window.
          6. Increment update counter.
        """
        forecast_f = float(forecast)
        actual_f = float(actual)

        # 1. Bias correction & tracker update
        if self.bias_corrector is not None:
            point = float(self.bias_corrector.correct(forecast_f))
            self.bias_corrector.update(forecast_f, actual_f)
        else:
            point = forecast_f

        # 2. Conformity score
        score = abs(actual_f - point)

        # Compute per-expert quantile q_k from the *current* window (pre-append),
        # so the score we just observed does not leak into the threshold that
        # was effectively used at prediction time.
        scores_now = list(self.score_window)
        for k, gamma in enumerate(self.gammas):
            alpha_k = self.alphas[k]
            q_k = _empirical_quantile(scores_now, 1.0 - alpha_k)
            err_k = 1.0 if score > q_k else 0.0
            # 3b. ACI update: alpha_{t+1} = alpha_t + gamma * (alpha - err)
            new_alpha = alpha_k + gamma * (self.alpha_target - err_k)
            # Clamp to (0, 1) — strictly inside, never at the boundary
            eps = 1e-6
            self.alphas[k] = max(eps, min(1.0 - eps, new_alpha))

        # 4. Multiplicative-weights update on pinball loss
        #    Loss is scaled to be of order ~score; eta=0.1 is a safe default
        #    for hourly EUR/MWh-scale residuals (10..100). Compute loss then
        #    subtract the min so we don't underflow exp() for large losses.
        losses = []
        for k, gamma in enumerate(self.gammas):
            alpha_k = self.alphas[k]  # post-update
            q_k = _empirical_quantile(scores_now, 1.0 - alpha_k)
            losses.append(_pinball_loss(score, q_k, self.alpha_target))
        if losses:
            min_loss = min(losses)
            new_weights = [
                w * math.exp(-self.eta * (l - min_loss))
                for w, l in zip(self.weights, losses)
            ]
            total = sum(new_weights)
            if total > 0:
                self.weights = [w / total for w in new_weights]

        # 5. Update rolling window with the new score
        self.score_window.append(score)
        # 6. Bump counter
        self.n_updates += 1

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def effective_alpha(self) -> float:
        """Current combined miscoverage rate (lower = wider interval)."""
        return self._combined_alpha()

    @property
    def effective_coverage(self) -> float:
        """Current target coverage = 1 - effective_alpha."""
        return 1.0 - self._combined_alpha()

    @property
    def current_half_width(self) -> float:
        """Half-width that would be used for a forecast right now."""
        if not self.score_window:
            return 0.0
        return self._quantile_for_alpha(self._combined_alpha())

    @property
    def dominant_expert(self) -> int:
        """Index of the expert with the highest current weight."""
        return max(range(len(self.weights)), key=lambda i: self.weights[i])

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the full state."""
        d = {
            "version": 1,
            "target_coverage": self.target_coverage,
            "gammas": list(self.gammas),
            "eta": self.eta,
            "window": self.window,
            "min_warmup": self.min_warmup,
            "alphas": list(self.alphas),
            "weights": list(self.weights),
            "score_window": list(self.score_window),
            "n_updates": self.n_updates,
        }
        if self.bias_corrector is not None and hasattr(
                self.bias_corrector, "to_dict"):
            d["bias_corrector"] = self.bias_corrector.to_dict()
        return d

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        bias_corrector: Any = None,
    ) -> "DtACI":
        """Restore a DtACI instance from `to_dict` output.

        If a bias_corrector was serialised under "bias_corrector" and the
        caller did not pass one explicitly, attempts to reconstruct via
        `OnlineBiasCorrector.from_dict` (lazy import to avoid circular).
        """
        if d.get("version", 1) != 1:
            raise ValueError(f"Unknown DtACI state version: {d.get('version')}")
        bc = bias_corrector
        if bc is None and "bias_corrector" in d:
            from .bias_corrector import OnlineBiasCorrector
            bc = OnlineBiasCorrector.from_dict(d["bias_corrector"])
        inst = cls(
            target_coverage=d["target_coverage"],
            gammas=d["gammas"],
            eta=d.get("eta", 0.1),
            window=d.get("window", 720),
            min_warmup=d.get("min_warmup", 24),
            bias_corrector=bc,
        )
        # Restore mutable state. Length checks guard against schema drift.
        K = len(inst.gammas)
        alphas = d.get("alphas") or [inst.alpha_target] * K
        weights = d.get("weights") or [1.0 / K] * K
        if len(alphas) == K:
            inst.alphas = [float(a) for a in alphas]
        if len(weights) == K:
            total = sum(weights) or 1.0
            inst.weights = [float(w) / total for w in weights]
        for s in d.get("score_window", []):
            inst.score_window.append(float(s))
        inst.n_updates = int(d.get("n_updates", 0))
        return inst
