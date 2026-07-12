"""Learned per-lead-time price-forecast uncertainty (replaces the
`pv_aware_cvar.price_rel_std_for_lead` cold-start heuristic).

v2.13.0 shipped a *static* profile `DEFAULT_PRICE_REL_STD_BY_LEAD`
(0 for cleared days 0-1, growing 0.10→0.30 for forecast days 2-6)
that feeds the CVaR kernel's multiplicative buy-price perturbation.
That was an explicit cold-start prior. This module closes the loop:
it keeps a rolling buffer of each day's *forecast* buy-price curve
tagged by the lead time at which it was made, and once the day's
cleared (realized) price is known it measures the actual relative
forecast error per lead time and learns a site-specific
`rel_std_for_lead(days_ahead)`.

**Why per-lead-time and not DtACI.** The DtACI layer (`dk_dtaci.py`)
calibrates prediction intervals stratified by (direction, D(k) order
statistic) — it does *not* stratify by how far ahead the forecast
was made. The CVaR discontinuity the v2.13.0 fix addressed is a
pure lead-time effect (cleared vs. forecast horizon), so this is a
genuinely new, orthogonal learner rather than a second copy of DtACI.

**Estimator.** For a reconciled (target_date, lead) pair, the sample
relative error is the RMS of `(realized_h − forecast_h) / forecast_h`
over the hours where the forecast price is non-trivial. RMS relative
error is exactly the dispersion magnitude the lognormal buy-price
sampler in `pv_aware_cvar._sample_price_paths` consumes, so the
learned value drops straight into `price_rel_std` with no rescaling.
Samples accumulate per lead via EWMA; the published value shrinks
from the heuristic prior toward the learned mean as evidence grows,
so behaviour is identical to v2.13.0 until real data arrives.

For cleared days (lead 0-1) the forecast *is* the published
day-ahead price, so realized ≈ forecast and the learned error is
≈ 0 — the "0 for cleared days" property of the old static profile
emerges from the data rather than being hard-coded.

Pure module — no Home Assistant imports. The coordinator supplies
forecast/realized curves and a state-file path; this does the
arithmetic, persistence round-trip, and is unit-tested in isolation
via the importlib direct-load pattern.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

try:  # package import (HA runtime)
    from .pv_aware_cvar import price_rel_std_for_lead as _prior_for_lead
except ImportError:  # bare import (test / studies path)
    from pv_aware_cvar import price_rel_std_for_lead as _prior_for_lead  # type: ignore[no-redef]

SCHEMA_VERSION = 1

# Reconciled (date, lead) samples needed before the learned value is
# weighted equal to the prior. Shrinkage weight w = n / (n + warmup),
# so at n == warmup the published rel_std is the midpoint of prior and
# learned; it never jumps from the prior in a single step.
DEFAULT_MIN_WARMUP = 3

# EWMA smoothing for successive per-lead samples. 0.3 keeps ~10 days
# of effective memory while still tracking a genuine regime shift.
DEFAULT_EWMA_ALPHA = 0.3

# Hours whose forecast price is below this (EUR/kWh) are excluded from
# the relative-error RMS — a near-zero denominator makes the ratio
# explode and says nothing useful about tail risk.
_MIN_PRICE_EUR_KWH = 1e-3

# Bound the rolling buffers so a long-running install stays small.
_MAX_PENDING_DATES = 12
_MAX_RECONCILED_DATES = 40


def _sample_rel_std(
    forecast_24: list[float], realized_24: list[float],
) -> float | None:
    """RMS relative buy-price error for one reconciled day, or None.

    Returns None when no hour has a usable (non-trivial) forecast
    price, so a degenerate day contributes nothing to the learner.
    """
    sq = 0.0
    n = 0
    for f, r in zip(forecast_24, realized_24):
        try:
            fv = float(f)
            rv = float(r)
        except (TypeError, ValueError):
            continue
        if abs(fv) <= _MIN_PRICE_EUR_KWH:
            continue
        e = (rv - fv) / fv
        sq += e * e
        n += 1
    if n == 0:
        return None
    return math.sqrt(sq / n)


class PriceForecastVerifier:
    """Rolling per-lead-time price-forecast-error learner."""

    def __init__(
        self,
        *,
        min_warmup: int = DEFAULT_MIN_WARMUP,
        alpha: float = DEFAULT_EWMA_ALPHA,
    ) -> None:
        self.min_warmup = int(min_warmup)
        self.alpha = float(alpha)
        # lead (int) → learned EWMA rel_std
        self.rel_std_ewma: dict[int, float] = {}
        # lead (int) → number of reconciled samples
        self.n: dict[int, int] = {}
        # target_date (YYYY-MM-DD) → {lead: forecast_buy_24}
        self.pending: dict[str, dict[int, list[float]]] = {}
        # target_dates already reconciled (never re-count)
        self.reconciled: set[str] = set()

    # ── recording / reconciliation ────────────────────────────────

    def record_forecast(
        self, target_date: str, days_ahead: int, forecast_buy_24: list[float],
    ) -> None:
        """Store the forecast buy curve for a target date at its lead.

        Overwrites any prior forecast at the same (date, lead) — the
        latest forecast for that lead wins. No-op once the date has
        been reconciled (its cleared price is already known).
        """
        if not target_date or target_date in self.reconciled:
            return
        if forecast_buy_24 is None or len(forecast_buy_24) < 24:
            return
        lead = max(0, int(days_ahead))
        slot = self.pending.setdefault(target_date, {})
        slot[lead] = [float(x) for x in forecast_buy_24[:24]]
        self._prune_pending()

    def reconcile(self, target_date: str, realized_buy_24: list[float]) -> int:
        """Match a now-cleared day's realized curve to stored forecasts.

        For every lead we stored a forecast at for this date, measure
        the relative error and fold it into that lead's EWMA. Each
        date is reconciled at most once. Returns the number of
        (lead) samples ingested.
        """
        if not target_date or target_date in self.reconciled:
            return 0
        if realized_buy_24 is None or len(realized_buy_24) < 24:
            return 0
        slot = self.pending.get(target_date)
        if not slot:
            return 0
        realized = [float(x) for x in realized_buy_24[:24]]
        ingested = 0
        for lead, fc in slot.items():
            s = _sample_rel_std(fc, realized)
            if s is None:
                continue
            prev = self.rel_std_ewma.get(lead)
            self.rel_std_ewma[lead] = (
                s if prev is None else (1.0 - self.alpha) * prev + self.alpha * s
            )
            self.n[lead] = self.n.get(lead, 0) + 1
            ingested += 1
        self.reconciled.add(target_date)
        self.pending.pop(target_date, None)
        self._prune_reconciled()
        return ingested

    # ── query ─────────────────────────────────────────────────────

    def rel_std_for_lead(self, days_ahead: int) -> float:
        """Published price rel_std for a horizon day (0 = today).

        Shrinkage blend of the heuristic prior and the learned EWMA:
        ``w·learned + (1−w)·prior`` with ``w = n / (n + min_warmup)``.
        With no samples this equals the v2.13.0 static profile.
        """
        lead = max(0, int(days_ahead))
        prior = _prior_for_lead(lead)
        n = self.n.get(lead, 0)
        if n <= 0:
            return prior
        learned = self.rel_std_ewma.get(lead, prior)
        w = n / (n + self.min_warmup)
        return (1.0 - w) * prior + w * learned

    def diagnostics(self) -> dict[str, Any]:
        """Compact per-lead learner state for the sensor attribute."""
        leads = sorted(set(self.n) | set(self.rel_std_ewma))
        return {
            "enabled": True,
            "schema_version": SCHEMA_VERSION,
            "pending_dates": len(self.pending),
            "reconciled_dates": len(self.reconciled),
            "per_lead": {
                str(lead): {
                    "n": self.n.get(lead, 0),
                    "learned_rel_std": round(self.rel_std_ewma.get(lead, 0.0), 4),
                    "prior_rel_std": round(_prior_for_lead(lead), 4),
                    "published_rel_std": round(self.rel_std_for_lead(lead), 4),
                }
                for lead in leads
            },
        }

    # ── internal buffer bounds ────────────────────────────────────

    def _prune_pending(self) -> None:
        if len(self.pending) <= _MAX_PENDING_DATES:
            return
        for old in sorted(self.pending)[:-_MAX_PENDING_DATES]:
            del self.pending[old]

    def _prune_reconciled(self) -> None:
        if len(self.reconciled) <= _MAX_RECONCILED_DATES:
            return
        for old in sorted(self.reconciled)[:-_MAX_RECONCILED_DATES]:
            self.reconciled.discard(old)

    # ── persistence ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "min_warmup": self.min_warmup,
            "alpha": self.alpha,
            "rel_std_ewma": {str(k): v for k, v in self.rel_std_ewma.items()},
            "n": {str(k): v for k, v in self.n.items()},
            "pending": {
                date: {str(lead): fc for lead, fc in slot.items()}
                for date, slot in self.pending.items()
            },
            "reconciled": sorted(self.reconciled),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PriceForecastVerifier":
        """Rebuild from persisted state; cold-start on any mismatch.

        A schema-version bump or a malformed file yields a fresh
        verifier rather than a crash — the learner simply re-warms.
        """
        inst = cls()
        if not isinstance(raw, dict):
            return inst
        if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
            return inst
        try:
            inst.min_warmup = int(raw.get("min_warmup", DEFAULT_MIN_WARMUP))
            inst.alpha = float(raw.get("alpha", DEFAULT_EWMA_ALPHA))
            inst.rel_std_ewma = {
                int(k): float(v)
                for k, v in (raw.get("rel_std_ewma") or {}).items()
            }
            inst.n = {int(k): int(v) for k, v in (raw.get("n") or {}).items()}
            inst.pending = {
                str(date): {
                    int(lead): [float(x) for x in fc]
                    for lead, fc in (slot or {}).items()
                }
                for date, slot in (raw.get("pending") or {}).items()
            }
            inst.reconciled = {str(d) for d in (raw.get("reconciled") or [])}
        except (TypeError, ValueError):
            return cls()
        return inst


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON to a temp file then os.replace — never a torn file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def load_or_create(path: str) -> PriceForecastVerifier:
    """Load the verifier from ``path`` or cold-start if absent/bad."""
    try:
        with open(path, encoding="utf-8") as fh:
            return PriceForecastVerifier.from_dict(json.load(fh))
    except (OSError, ValueError):
        return PriceForecastVerifier()


def save(path: str, verifier: PriceForecastVerifier) -> None:
    """Persist the verifier atomically. Best-effort (raises only on
    truly unexpected errors; the caller wraps in try/except)."""
    _atomic_write_json(path, verifier.to_dict())


__all__ = [
    "DEFAULT_EWMA_ALPHA",
    "DEFAULT_MIN_WARMUP",
    "SCHEMA_VERSION",
    "PriceForecastVerifier",
    "load_or_create",
    "save",
]
