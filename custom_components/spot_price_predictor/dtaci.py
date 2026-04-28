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

  * `loss_k[t]`   — discounted miscoverage loss per expert (Gibbs &
                     Candes 2024 §4):
                         loss_k[t+1] = rho * loss_k[t] + err_k[t]
                     so a miss contributes 1 and decays geometrically.

  * `weight_k[t]` — softmax of the negative discounted loss:
                         weight_k = exp(-eta * loss_k) / sum_j(...)
                     Weights are renormalised to sum to 1. Experts with
                     fewer recent misses get more weight.

  * `score_window` — rolling buffer of the last `window` conformity scores;
                     used to compute the empirical quantile that turns
                     alpha into an interval half-width.

Coverage guarantees: under bounded scores, DtACI's long-run realised
coverage converges to the target — see Theorem 2 of the paper.

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


DEFAULT_GAMMAS: tuple[float, ...] = (
    0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015,
    0.02,   0.03,  0.05,  0.08,  0.1,   0.15,  0.2,
)
"""Default DtACI step sizes — log-spaced 15-point ladder over
[0.0005, 0.2], spanning 2.5 orders of magnitude.

Smaller gammas = slower adaptation, suited to stationary regimes.
Larger gammas = faster adaptation, suited to abrupt shifts.
The expert-weighting layer dynamically picks whichever value is
currently performing best on the discounted miscoverage loss.

A 15-expert grid gives the weight-entropy diagnostic up to
log2(15) ≈ 3.91 bits of resolution — useful for surfacing
"algorithm uncertainty" to a user-facing UI.
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

    Retained as a helper for diagnostics / tests. The DtACI expert weights
    use the discounted miscoverage loss instead (see Gibbs & Candes 2024
    Section 4) — it converges faster and is what the reference UI cards
    expect.
    """
    if score >= q:
        return alpha * (score - q)
    return (1.0 - alpha) * (q - score)


def _shannon_entropy_bits(weights: list[float]) -> float:
    """Shannon entropy of a probability vector, in bits.

    Returns 0.0 when one expert dominates, log2(K) when uniform. Used
    by the diagnostic layer to surface "algorithm uncertainty" — high
    entropy means the algorithm is unsure which gamma is best, low
    entropy means it has converged on one.
    """
    h = 0.0
    for w in weights:
        if w > 1e-12:
            h -= w * math.log2(w)
    return h


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
        eta: float = 5.0,
        rho: float = 0.99,
        window: int = 720,
        min_warmup: int = 24,
        bias_corrector: Any = None,
    ) -> None:
        if not (0.0 < target_coverage < 1.0):
            raise ValueError(
                f"target_coverage must be in (0, 1), got {target_coverage}"
            )
        if not (0.0 < rho < 1.0):
            raise ValueError(
                f"rho must be in (0, 1), got {rho}"
            )
        self.target_coverage = float(target_coverage)
        self.alpha_target = 1.0 - self.target_coverage
        self.gammas: list[float] = [float(g) for g in gammas]
        if not self.gammas:
            raise ValueError("gammas must not be empty")
        self.eta = float(eta)
        self.rho = float(rho)
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
        # Discounted miscoverage loss per expert (Gibbs & Candes 2024 §4).
        # Each step:  loss_k <- rho * loss_k + err_k        where err_k in {0,1}
        # Weights:    w_k    = softmax(-eta * loss_k)
        # Steady state for an expert covering at rate (1-alpha_target):
        #   loss_k* = alpha_target / (1 - rho)   (independent of k).
        # Differentiation comes from transient excess misses during regime shifts.
        self.cumulative_losses: list[float] = [0.0] * K
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

        Implements the Gibbs & Candes 2024 §4 update:

          For each expert k:
            err_k = 1 if score > q_k (interval misses)  else 0
            alpha_k <- alpha_k + gamma_k * (alpha_target - err_k)
            loss_k  <- rho * loss_k + err_k                 (discounted)
          Then renormalise the weights via softmax(-eta * loss_k).

        Steps:
          1. Apply bias correction (and update its tracker).
          2. Compute conformity score s = |actual - debiased_forecast|.
          3. Per expert: compute err_k vs the pre-append window quantile,
             update alpha_k and the discounted miscoverage loss.
          4. Recompute weights via numerically-stable softmax.
          5. Append s to the score window. Increment counter.
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
            # 3a. ACI update: alpha_{t+1} = alpha_t + gamma * (alpha_target - err)
            new_alpha = alpha_k + gamma * (self.alpha_target - err_k)
            # Clamp to (0, 1) strictly
            eps = 1e-6
            self.alphas[k] = max(eps, min(1.0 - eps, new_alpha))
            # 3b. Discounted miscoverage loss (Gibbs & Candes 2024 §4)
            self.cumulative_losses[k] = (
                self.rho * self.cumulative_losses[k] + err_k
            )

        # 4. Numerically-stable softmax over -eta * loss_k.
        #    Subtracting the minimum loss before exponentiating prevents
        #    overflow when one expert lags far behind, and renormalising
        #    by the sum makes weights invariant to the additive shift.
        min_loss = min(self.cumulative_losses)
        exps = [math.exp(-self.eta * (l - min_loss))
                for l in self.cumulative_losses]
        total = sum(exps)
        if total > 0:
            self.weights = [e / total for e in exps]

        # 5. Update rolling window with the new score, bump counter
        self.score_window.append(score)
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

    @property
    def dominant_gamma(self) -> float:
        """Step size of the highest-weighted expert."""
        return self.gammas[self.dominant_expert]

    @property
    def weight_entropy_bits(self) -> float:
        """Shannon entropy of the expert weight distribution, in bits.

        Range [0, log2(K)]:
          0     = one expert dominates  (algorithm is confident)
          high  = uniform weights        (algorithm is unsure)

        Surfaced to UI as a "calibrator confidence" indicator.
        """
        return _shannon_entropy_bits(self.weights)

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the full state."""
        d = {
            "version": 2,
            "target_coverage": self.target_coverage,
            "gammas": list(self.gammas),
            "eta": self.eta,
            "rho": self.rho,
            "window": self.window,
            "min_warmup": self.min_warmup,
            "alphas": list(self.alphas),
            "weights": list(self.weights),
            "cumulative_losses": list(self.cumulative_losses),
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

        Versions accepted:
          1 = pinball-loss weights, no `cumulative_losses` field. Treated
              as cold-start for the loss vector; alphas / scores preserved.
          2 = current — discounted-loss weights with `cumulative_losses`.

        If a bias_corrector was serialised under "bias_corrector" and the
        caller did not pass one explicitly, attempts to reconstruct via
        `OnlineBiasCorrector.from_dict` (lazy import to avoid circular).
        """
        version = d.get("version", 1)
        if version not in (1, 2):
            raise ValueError(f"Unknown DtACI state version: {version}")
        bc = bias_corrector
        if bc is None and "bias_corrector" in d:
            from .bias_corrector import OnlineBiasCorrector
            bc = OnlineBiasCorrector.from_dict(d["bias_corrector"])
        inst = cls(
            target_coverage=d["target_coverage"],
            gammas=d["gammas"],
            eta=d.get("eta", 5.0),
            rho=d.get("rho", 0.99),
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
        cum = d.get("cumulative_losses")
        if cum is not None and len(cum) == K:
            inst.cumulative_losses = [float(x) for x in cum]
        # else: cold start at zeros, weights will reset to uniform on next update
        for s in d.get("score_window", []):
            inst.score_window.append(float(s))
        inst.n_updates = int(d.get("n_updates", 0))
        return inst
