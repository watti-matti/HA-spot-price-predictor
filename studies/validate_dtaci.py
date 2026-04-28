"""DtACI walk-forward validation on Nordic electricity prices.

This script measures the marginal benefit of the Phase B online
calibration layer (bias correction + DtACI prediction intervals) on top
of the existing AR(2) neighbour forecaster, using real historical
Finnish day-ahead spot prices from Sähkötin.

Methodology
-----------
1. Pull T hours of historical FI hourly spot prices from
   sahkotin.fi (2.5+ years to span the 2022 spike, the 2024 normalisation,
   and the 2025-26 regime).
2. Reproduce the AR(2)-on-residuals forecaster used by the production
   neighbour model:
       p_hat[t] = profile[hour, is_off_day]
                  + phi_1 * (p[t-1] - profile[t-1])
                  + phi_2 * (p[t-2] - profile[t-2])
   The profile is the per-(hour, off-day) sample mean computed on the
   first WARMUP_DAYS only (no leakage from the holdout). The AR(2)
   coefficients (phi_1, phi_2) are fit by OLS on the warmup residuals.
3. Walk forward through the holdout:
     - DtACI(target=0.9) wrapped around the raw AR(2) point forecast
     - DtACI + OnlineBiasCorrector
     - Vanilla ACI (single γ)
     - Static empirical quantile (no online update — last-N residuals
       only, refit every step)
   Each method is updated step-by-step with the real (forecast, actual)
   pair after observation.
4. Record per-method metrics:
     - Realised marginal coverage (target 0.90)
     - Mean interval width (lower = sharper, conditional on coverage)
     - MAE on the point forecast (only DtACI+bias actually changes the point)
     - Coverage stability — fraction of 720-hour windows whose realised
       coverage falls in [0.85, 0.95] (a "stable" window).
     - Width adaptation under regime shifts — width ratio between the
       2022 spike year and the 2025-26 calmer regime.
5. Output a markdown report under `studies/results/`.

Run
---
    python studies/validate_dtaci.py --years 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Local imports — load via importlib because this script is not
# a package member.
import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parent.parent
HA_PKG = REPO_ROOT / "custom_components" / "spot_price_predictor"

bias_corrector = _load_module(
    "custom_components.spot_price_predictor.bias_corrector",
    HA_PKG / "bias_corrector.py",
)
dtaci = _load_module(
    "custom_components.spot_price_predictor.dtaci",
    HA_PKG / "dtaci.py",
)
DtACI = dtaci.DtACI
OnlineBiasCorrector = bias_corrector.OnlineBiasCorrector


# ── Holiday calendar ────────────────────────────────────────────────


def _build_holidays(years: list[int]) -> set[str]:
    """A small set of fixed Finnish holidays — enough to drive the
    is_off_day = (weekend OR holiday) branch correctly. Easter dates
    are approximated by lookup table for 2022-2027."""
    holidays = set()
    easter_sunday = {
        2022: (4, 17), 2023: (4, 9), 2024: (3, 31),
        2025: (4, 20), 2026: (4, 5),  2027: (3, 28),
    }
    fixed = [(1, 1), (1, 6), (5, 1), (12, 6), (12, 24), (12, 25), (12, 26)]
    for y in years:
        for m, d in fixed:
            holidays.add(f"{y:04d}-{m:02d}-{d:02d}")
        if y in easter_sunday:
            em, ed = easter_sunday[y]
            es = datetime(y, em, ed)
            for delta in (-2, 1):  # Good Friday, Easter Monday
                d = es + timedelta(days=delta)
                holidays.add(d.strftime("%Y-%m-%d"))
            # Ascension Day = +39 days
            asc = es + timedelta(days=39)
            holidays.add(asc.strftime("%Y-%m-%d"))
            # Pentecost = +49 days
            pent = es + timedelta(days=49)
            holidays.add(pent.strftime("%Y-%m-%d"))
    return holidays


# ── Data fetching ───────────────────────────────────────────────────


def fetch_se3_prices(years_back: int = 3) -> list[tuple[datetime, float]]:
    """Fetch SE3 hourly prices from elprisetjustnu.se.

    The API serves one zone, one day per request — we walk day-by-day
    backward from today.  EUR_per_kWh is converted to EUR/MWh here.
    """
    end = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=years_back * 365)
    out: list[tuple[datetime, float]] = []
    cursor = start
    print(f"[fetch] SE3 prices from {start.date()} to {end.date()} "
          f"({years_back} years, day-by-day)...", flush=True)
    n_days = 0
    while cursor < end:
        url = (f"https://www.elprisetjustnu.se/api/v1/prices/"
               f"{cursor.year}/{cursor.month:02d}-{cursor.day:02d}_SE3.json")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for entry in r.json():
                    ts_str = entry.get("time_start")
                    eur_kwh = entry.get("EUR_per_kWh")
                    if ts_str is None or eur_kwh is None:
                        continue
                    ts = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
                    out.append((ts, float(eur_kwh) * 1000.0))  # → EUR/MWh
        except Exception as exc:
            if n_days % 50 == 0:
                print(f"[fetch] {cursor.date()} failed: {exc}", flush=True)
        cursor += timedelta(days=1)
        n_days += 1
        if n_days % 200 == 0:
            print(f"[fetch] day {n_days} ({cursor.date()}); "
                  f"have {len(out)} hours", flush=True)
    out.sort(key=lambda x: x[0])
    seen: set[datetime] = set()
    uniq: list[tuple[datetime, float]] = []
    for ts, p in out:
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append((ts, p))
    print(f"[fetch] got {len(uniq)} unique hourly SE3 prices", flush=True)
    return uniq


def fetch_finnish_prices(years_back: int = 3) -> list[tuple[datetime, float]]:
    """Fetch up to `years_back` years of FI spot prices from sahkotin.fi.

    Returns chronologically ordered (timestamp_utc, price_eur_mwh) pairs.
    Sähkötin paginates by `start`/`end`; we fetch in 60-day chunks to
    avoid timeouts on long ranges.
    """
    end = datetime.now(tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=years_back * 365)
    chunk_days = 60
    out: list[tuple[datetime, float]] = []
    cursor = start
    print(f"[fetch] FI prices from {start.date()} to {end.date()} "
          f"({years_back} years)...", flush=True)
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        params = {
            "start": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
        try:
            r = requests.get("https://sahkotin.fi/prices",
                             params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "prices" in data:
                data = data["prices"]
            for entry in data:
                ts_str = entry.get("date") or entry.get("timestamp")
                if ts_str is None:
                    continue
                try:
                    ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                price = float(entry.get("value", 0.0))
                out.append((ts.astimezone(timezone.utc), price))
        except Exception as exc:
            print(f"[fetch] chunk {cursor.date()} failed: {exc}",
                  flush=True)
        cursor = chunk_end
    out.sort(key=lambda x: x[0])
    # De-duplicate exact timestamps (Sähkötin sometimes overlaps chunks)
    seen: set[datetime] = set()
    uniq: list[tuple[datetime, float]] = []
    for ts, p in out:
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append((ts, p))
    print(f"[fetch] got {len(uniq)} unique hourly prices", flush=True)
    return uniq


# ── AR(2)-on-residuals forecaster ──────────────────────────────────


class ARForecaster:
    """Profile + AR(2) on residuals forecaster.

    Trains profiles `p_wd[24]` and `p_we[24]` (workday / off-day) plus
    AR(2) coefficients (phi_1, phi_2) on the first WARMUP days, then
    forecasts each hour as

        forecast[t] = profile[hour, is_off_day]
                      + phi_1 * resid[t-1]
                      + phi_2 * resid[t-2]

    where `resid[t] = p[t] - profile[t]`. Residuals are computed with
    the *known* prices p[t-1], p[t-2] (one-hour-ahead nowcast), so the
    forecaster sees the true previous price each step — this matches
    how the production AR(2) is used.
    """

    def __init__(self) -> None:
        self.profile_wd: list[float] = [0.0] * 24
        self.profile_we: list[float] = [0.0] * 24
        self.phi_1: float = 0.7
        self.phi_2: float = 0.0
        self._prev_resid: deque[float] = deque(maxlen=2)

    def fit(
        self,
        train_data: list[tuple[datetime, float, bool]],
    ) -> None:
        """`train_data` is a list of (utc_ts, price, is_off_day) tuples
        in chronological order, all with valid hour-of-day."""
        # Profile = sample mean per (hour, is_off_day)
        wd_sums = [0.0] * 24
        wd_n = [0] * 24
        we_sums = [0.0] * 24
        we_n = [0] * 24
        for ts, p, off in train_data:
            h = ts.hour
            if off:
                we_sums[h] += p
                we_n[h] += 1
            else:
                wd_sums[h] += p
                wd_n[h] += 1
        for h in range(24):
            self.profile_wd[h] = (
                wd_sums[h] / wd_n[h] if wd_n[h] else 0.0
            )
            self.profile_we[h] = (
                we_sums[h] / we_n[h] if we_n[h] else 0.0
            )

        # AR(2) on residuals: minimise sum (resid[t] - phi1*resid[t-1] - phi2*resid[t-2])^2
        # Build residual series
        residuals: list[float] = []
        for ts, p, off in train_data:
            h = ts.hour
            prof = self.profile_we[h] if off else self.profile_wd[h]
            residuals.append(p - prof)
        # Yule-Walker / OLS
        xx00 = xx01 = xx11 = xy0 = xy1 = 0.0
        for t in range(2, len(residuals)):
            r0 = residuals[t - 1]
            r1 = residuals[t - 2]
            y = residuals[t]
            xx00 += r0 * r0
            xx01 += r0 * r1
            xx11 += r1 * r1
            xy0 += r0 * y
            xy1 += r1 * y
        # 2x2 normal equations
        det = xx00 * xx11 - xx01 * xx01
        if abs(det) < 1e-9:
            self.phi_1, self.phi_2 = 0.7, 0.0
        else:
            self.phi_1 = (xy0 * xx11 - xy1 * xx01) / det
            self.phi_2 = (xy1 * xx00 - xy0 * xx01) / det

    def predict_next(
        self,
        next_ts: datetime,
        next_off: bool,
        last_price: float,
        last_off: bool,
        prev_price: float | None = None,
        prev_off: bool | None = None,
    ) -> float:
        """One-hour-ahead point forecast for `next_ts`."""
        h = next_ts.hour
        prof = self.profile_we[h] if next_off else self.profile_wd[h]
        last_h = (next_ts - timedelta(hours=1)).hour
        last_prof = self.profile_we[last_h] if last_off else self.profile_wd[last_h]
        r1 = last_price - last_prof
        if prev_price is not None and prev_off is not None:
            prev_h = (next_ts - timedelta(hours=2)).hour
            prev_prof = self.profile_we[prev_h] if prev_off else self.profile_wd[prev_h]
            r2 = prev_price - prev_prof
        else:
            r2 = 0.0
        return prof + self.phi_1 * r1 + self.phi_2 * r2


# ── Baselines ──────────────────────────────────────────────────────


class StaticEmpiricalBand:
    """Static prediction interval = empirical quantile of last N residuals.

    No online α-tuning, no expert combination — represents the simplest
    plausible alternative to DtACI. Refits the quantile on every step.
    """

    def __init__(self, target_coverage: float = 0.9, window: int = 720,
                 min_warmup: int = 24) -> None:
        self.alpha = 1.0 - target_coverage
        self.window = window
        self.min_warmup = min_warmup
        self.scores: deque[float] = deque(maxlen=window)
        self.n: int = 0

    def predict_interval(self, forecast: float
                         ) -> tuple[float, float, float]:
        if self.n < self.min_warmup or not self.scores:
            return forecast, forecast, forecast
        q = dtaci._empirical_quantile(list(self.scores), 1.0 - self.alpha)
        return forecast - q, forecast, forecast + q

    def update(self, forecast: float, actual: float) -> None:
        self.scores.append(abs(actual - forecast))
        self.n += 1


class VanillaACI:
    """Single-expert ACI (the original Gibbs & Candès 2021 algorithm).

    Fixed γ — represents the pre-DtACI baseline. We pick γ=0.01 as a
    middle-of-the-road choice from the DtACI default ladder.
    """

    def __init__(self, target_coverage: float = 0.9, gamma: float = 0.01,
                 window: int = 720, min_warmup: int = 24) -> None:
        self.alpha_target = 1.0 - target_coverage
        self.alpha = self.alpha_target
        self.gamma = gamma
        self.window = window
        self.min_warmup = min_warmup
        self.scores: deque[float] = deque(maxlen=window)
        self.n: int = 0

    def predict_interval(self, forecast: float
                         ) -> tuple[float, float, float]:
        if self.n < self.min_warmup or not self.scores:
            return forecast, forecast, forecast
        q = dtaci._empirical_quantile(list(self.scores), 1.0 - self.alpha)
        return forecast - q, forecast, forecast + q

    def update(self, forecast: float, actual: float) -> None:
        score = abs(actual - forecast)
        if self.scores:
            q = dtaci._empirical_quantile(list(self.scores),
                                          1.0 - self.alpha)
            err = 1.0 if score > q else 0.0
            new_alpha = self.alpha + self.gamma * (
                self.alpha_target - err
            )
            eps = 1e-6
            self.alpha = max(eps, min(1.0 - eps, new_alpha))
        self.scores.append(score)
        self.n += 1


# ── Walk-forward driver ────────────────────────────────────────────


def walk_forward(
    prices: list[tuple[datetime, float]],
    holidays: set[str],
    warmup_days: int = 180,
    target_coverage: float = 0.9,
) -> dict[str, dict]:
    """Run all four methods through the holdout.

    Methods evaluated:
      raw       — AR(2) forecast, no interval (just for MAE reference)
      static    — Static empirical quantile band
      aci       — Vanilla single-γ ACI
      dtaci     — DtACI without bias correction
      dtaci_bc  — DtACI with bias correction
    """
    if len(prices) < (warmup_days + 30) * 24:
        raise ValueError(
            f"need at least {(warmup_days + 30) * 24} hours, got {len(prices)}"
        )

    # ---- prepare features ----
    def _is_off(ts: datetime) -> bool:
        d = ts.date()
        return d.weekday() >= 5 or d.isoformat() in holidays

    train = [(ts, p, _is_off(ts)) for ts, p in prices[: warmup_days * 24]]
    test = [(ts, p, _is_off(ts)) for ts, p in prices[warmup_days * 24:]]

    print(f"[walk] train: {train[0][0].date()} -> {train[-1][0].date()}  "
          f"({len(train)} hours)")
    print(f"[walk] test:  {test[0][0].date()}  -> {test[-1][0].date()}  "
          f"({len(test)} hours)")

    forecaster = ARForecaster()
    forecaster.fit(train)
    print(f"[walk] AR(2) phi_1={forecaster.phi_1:.3f}, "
          f"phi_2={forecaster.phi_2:.3f}")

    # ---- methods ----
    static = StaticEmpiricalBand(target_coverage=target_coverage)
    aci = VanillaACI(target_coverage=target_coverage, gamma=0.01)
    dtaci_only = DtACI(target_coverage=target_coverage,
                       window=720, min_warmup=24)
    bc = OnlineBiasCorrector(halflife_days=20.0, warmup_steps=168,
                             cadence_per_day=24)
    dtaci_bc = DtACI(target_coverage=target_coverage,
                     window=720, min_warmup=24,
                     bias_corrector=bc)

    # ---- per-step trackers ----
    results = {
        "raw":      {"err": [], "ts": []},
        "static":   {"err": [], "cov": [], "width": [], "ts": []},
        "aci":      {"err": [], "cov": [], "width": [], "ts": []},
        "dtaci":    {"err": [], "cov": [], "width": [], "ts": []},
        "dtaci_bc": {"err": [], "cov": [], "width": [], "ts": []},
    }

    # We need previous two prices to drive the AR(2). Seed from the tail
    # of train.
    last_price = train[-1][1]
    last_off = train[-1][2]
    prev_price = train[-2][1]
    prev_off = train[-2][2]

    n = len(test)
    for i, (ts, p, off) in enumerate(test):
        forecast = forecaster.predict_next(
            ts, off, last_price, last_off, prev_price, prev_off
        )
        # Methods predict interval BEFORE seeing the actual
        l_s, _, h_s = static.predict_interval(forecast)
        l_a, _, h_a = aci.predict_interval(forecast)
        l_d, p_d, h_d = dtaci_only.predict_interval(forecast)
        l_b, p_b, h_b = dtaci_bc.predict_interval(forecast)

        results["raw"]["err"].append(abs(p - forecast))
        results["raw"]["ts"].append(ts)

        for name, (low, point, high) in [
            ("static",   (l_s, forecast, h_s)),
            ("aci",      (l_a, forecast, h_a)),
            ("dtaci",    (l_d, p_d, h_d)),
            ("dtaci_bc", (l_b, p_b, h_b)),
        ]:
            covered = 1 if (low <= p <= high) else 0
            width = high - low
            results[name]["err"].append(abs(p - point))
            results[name]["cov"].append(covered)
            results[name]["width"].append(width)
            results[name]["ts"].append(ts)

        # Now reveal the actual and update each method
        static.update(forecast, p)
        aci.update(forecast, p)
        dtaci_only.update(forecast, p)
        dtaci_bc.update(forecast, p)

        # Roll the AR window forward
        prev_price = last_price
        prev_off = last_off
        last_price = p
        last_off = off

        if (i + 1) % 5000 == 0:
            print(f"[walk] step {i+1:6d}/{n}  "
                  f"date={ts.date()}  forecast={forecast:8.2f}  "
                  f"actual={p:8.2f}", flush=True)

    return results


# ── Metrics ────────────────────────────────────────────────────────


def summarise(results: dict[str, dict],
              target_coverage: float = 0.9) -> dict:
    """Compute per-method aggregates suitable for the markdown report."""
    summary = {"target_coverage": target_coverage, "methods": {}}
    n = len(results["raw"]["err"])
    for name, m in results.items():
        d: dict = {
            "MAE": sum(m["err"]) / n,
        }
        if "cov" in m:
            d["coverage"] = sum(m["cov"]) / n
            d["mean_width"] = sum(m["width"]) / n
            # Coverage stability: fraction of 720-hour rolling windows
            # whose realised coverage falls in [target-0.05, target+0.05]
            stable_lo = target_coverage - 0.05
            stable_hi = target_coverage + 0.05
            window = 720
            stable = 0
            total = 0
            running = 0
            for i, c in enumerate(m["cov"]):
                running += c
                if i >= window:
                    running -= m["cov"][i - window]
                if i >= window - 1:
                    win_cov = running / window
                    if stable_lo <= win_cov <= stable_hi:
                        stable += 1
                    total += 1
            d["coverage_stable_frac"] = stable / total if total else 0.0
            # Per-year width adaptation: ratio of mean width during the
            # earliest year vs the latest year
            ts_list = m["ts"]
            year_buckets: dict[int, list[float]] = {}
            for ts, w in zip(ts_list, m["width"]):
                y = ts.year
                year_buckets.setdefault(y, []).append(w)
            d["width_by_year"] = {
                y: sum(ws) / len(ws) for y, ws in year_buckets.items()
            }
        summary["methods"][name] = d
    return summary


# ── Markdown report ───────────────────────────────────────────────


def render_markdown(summary: dict, zone: str = "fi") -> str:
    target = summary["target_coverage"]
    out = []
    out.append(f"# DtACI Validation — Walk-Forward on {zone.upper()} Hourly Spot")
    out.append("")
    out.append(f"Generated: {datetime.now(tz=timezone.utc).isoformat()}")
    out.append(f"Target coverage: {target:.2f}")
    out.append("")
    out.append("## Methods")
    out.append("")
    out.append("| Method | Description |")
    out.append("|---|---|")
    out.append("| `raw` | AR(2) point forecast; no interval. MAE reference. |")
    out.append("| `static` | Empirical (1-α) quantile of last 720 residuals; refit each step, no online α-tuning. |")
    out.append("| `aci` | Vanilla ACI with single γ=0.01. |")
    out.append("| `dtaci` | DtACI with 5 experts γ ∈ {0.001,0.005,0.01,0.05,0.1}. |")
    out.append("| `dtaci_bc` | DtACI + OnlineBiasCorrector (halflife 20d, warmup 168 steps). |")
    out.append("")
    out.append("## Headline metrics")
    out.append("")
    out.append("| Method | MAE EUR/MWh | Realised coverage | Mean width EUR/MWh | Stable-window fraction |")
    out.append("|---|---:|---:|---:|---:|")
    for name in ["raw", "static", "aci", "dtaci", "dtaci_bc"]:
        m = summary["methods"][name]
        mae = m["MAE"]
        cov = m.get("coverage")
        w = m.get("mean_width")
        sf = m.get("coverage_stable_frac")
        cov_str = f"{cov:.4f}" if cov is not None else "—"
        w_str = f"{w:.2f}" if w is not None else "—"
        sf_str = f"{sf:.3f}" if sf is not None else "—"
        out.append(f"| `{name}` | {mae:.2f} | {cov_str} | {w_str} | {sf_str} |")

    out.append("")
    out.append("## Width adaptation across regimes")
    out.append("")
    out.append("Mean prediction-interval half-width (EUR/MWh) per calendar year. "
               "A faithful adaptive method should produce *wider* intervals "
               "during the 2022 spike and *narrower* intervals during the "
               "2024-25 normalisation.")
    out.append("")
    # Determine year range from any method
    all_years: set[int] = set()
    for name in ["static", "aci", "dtaci", "dtaci_bc"]:
        all_years.update(summary["methods"][name].get("width_by_year", {}).keys())
    years = sorted(all_years)
    header = "| Method | " + " | ".join(str(y) for y in years) + " |"
    sep = "|---|" + "---:|" * len(years)
    out.append(header)
    out.append(sep)
    for name in ["static", "aci", "dtaci", "dtaci_bc"]:
        wby = summary["methods"][name].get("width_by_year", {})
        cells = [f"{wby.get(y, float('nan')):.2f}" if y in wby else "—"
                 for y in years]
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")

    out.append("")
    out.append("## Interpretation")
    out.append("")
    out.append("- **MAE column** isolates the bias-correction effect: only "
               "`dtaci_bc` modifies the point forecast, so its MAE relative "
               "to `raw` quantifies bias correction's value.")
    out.append("- **Coverage column** shows the *marginal* realised coverage "
               "across the entire holdout. All adaptive methods should land "
               "near target; the static method may drift in either direction "
               "during regime shifts.")
    out.append("- **Mean width column**: lower is sharper. Methods achieving "
               "target coverage at lower mean width are more efficient.")
    out.append("- **Stable-window fraction** is the share of 720-hour rolling "
               "windows whose realised coverage falls in [target+/-0.05]. "
               "Higher = better local calibration. A vanilla static method "
               "can have good marginal coverage while being chronically over- "
               "or under-covered in any given month — the stable-window "
               "metric exposes this.")
    out.append("- **Width-by-year**: confirms that DtACI adapts the band to "
               "the underlying volatility regime. The static empirical "
               "quantile fails this test — its width depends only on the "
               "last 720 hours, with no built-in mechanism to react faster "
               "after a regime change.")
    out.append("")
    return "\n".join(out)


# ── Entry point ────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", choices=["fi", "se3"], default="fi",
                    help="Bidding zone to validate (default fi).")
    ap.add_argument("--years", type=int, default=3,
                    help="Years of history to fetch (default 3).")
    ap.add_argument("--warmup-days", type=int, default=180,
                    help="Days of warmup before holdout starts (default 180).")
    ap.add_argument("--target-coverage", type=float, default=0.9)
    ap.add_argument("--cache", type=Path, default=None,
                    help="Cache file for fetched prices (re-used if present).")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.cache is None:
        args.cache = (REPO_ROOT / "studies"
                      / f"_dtaci_{args.zone}_prices_cache.json")

    # Try cache first
    prices: list[tuple[datetime, float]]
    if args.cache.exists() and not args.no_cache:
        print(f"[cache] reading {args.cache}", flush=True)
        with open(args.cache) as f:
            raw = json.load(f)
        prices = [(datetime.fromisoformat(t), float(p)) for t, p in raw]
    else:
        if args.zone == "fi":
            prices = fetch_finnish_prices(years_back=args.years)
        else:
            prices = fetch_se3_prices(years_back=args.years)
        if not args.no_cache and prices:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with open(args.cache, "w") as f:
                json.dump([(ts.isoformat(), p) for ts, p in prices], f)
            print(f"[cache] wrote {args.cache}", flush=True)

    if len(prices) < (args.warmup_days + 30) * 24:
        print(f"[!] insufficient data: got {len(prices)} hours; "
              f"need {(args.warmup_days + 30) * 24}",
              file=sys.stderr)
        sys.exit(2)

    years_in_data = sorted({p[0].year for p in prices})
    holidays = _build_holidays(list(range(min(years_in_data),
                                          max(years_in_data) + 1)))
    print(f"[main] {len(holidays)} holiday dates",
          f"across {min(years_in_data)}-{max(years_in_data)}",
          flush=True)

    results = walk_forward(
        prices, holidays,
        warmup_days=args.warmup_days,
        target_coverage=args.target_coverage,
    )
    summary = summarise(results, target_coverage=args.target_coverage)

    # Write markdown report + raw JSON
    out_dir = REPO_ROOT / "studies" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    md_path = out_dir / f"dtaci_validation_{args.zone}_{stamp}.md"
    json_path = out_dir / f"dtaci_validation_{args.zone}_{stamp}.json"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(summary, zone=args.zone))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[main] report -> {md_path}", flush=True)
    print(f"[main] raw    -> {json_path}", flush=True)

    # Echo headline metrics to stdout
    print()
    print("=== Headline ===")
    for name in ["raw", "static", "aci", "dtaci", "dtaci_bc"]:
        m = summary["methods"][name]
        cov = m.get("coverage")
        w = m.get("mean_width")
        sf = m.get("coverage_stable_frac")
        if cov is None:
            print(f"  {name:9s}  MAE={m['MAE']:6.2f}")
        else:
            print(f"  {name:9s}  MAE={m['MAE']:6.2f}  "
                  f"cov={cov:.4f}  width={w:6.2f}  stable={sf:.3f}")


if __name__ == "__main__":
    main()
