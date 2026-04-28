"""Per-(direction, k) DtACI walk-forward validation on D(i) statistics.

Replaces the v1 hourly-DtACI validation harness. The architecture
under test now matches the design specified by the user's reference UI
card: one DtACI instance per (direction, k) for k = 1..12 in both the
cheap and peak directions = 24 instances per zone.

Methodology
-----------
1. Pull T years of historical hourly prices for each zone:
     FI  via sahkotin.fi
     SE1, SE3 via elprisetjustnu.se
     EE  via elering.ee
2. For each zone, fit an AR(2)-on-residuals day-ahead forecaster on
   the first WARMUP_DAYS, then walk forward through the holdout one
   day at a time.
3. On each day d:
   * Forecast 24 hourly prices for day d using the trained profile
     plus AR(2) iterated 24 hours from the last two known prices.
   * Compute forecast_dk_cheap[12] / forecast_dk_peak[12] from the 24
     predictions via `compute_dk_cheap_peak`.
   * Observe actual 24 hourly prices for day d.
   * Compute actual_dk_cheap[12] / actual_dk_peak[12].
   * Predict intervals from each method's CURRENT state (pre-update).
   * Score per-(direction, k) coverage, width, MAE.
   * Update each method with the (forecast, actual) D(i) pair.
4. Methods compared:
   * `raw`    — point forecast only (MAE reference; no interval)
   * `static` — empirical (1−α) quantile of last `window` residuals
                per (direction, k); refit each step.
   * `dtaci`  — DkDtACIBundle (this is the system under test).
5. Per-zone outputs:
   * Per-k coverage table (target 0.9)
   * Per-k MAE table
   * Per-k bias_ema, alpha_agg, dominant_gamma at end of holdout
   * Mean interval width per-k

Run
---
    python studies/validate_dtaci_dk.py --zones fi,se1,se3,ee --years 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parent.parent
HA_PKG = REPO_ROOT / "custom_components" / "spot_price_predictor"
SRC = REPO_ROOT / "src"

dk_utils = _load("src.dk_utils", SRC / "dk_utils.py")
_load("custom_components.spot_price_predictor.bias_corrector",
      HA_PKG / "bias_corrector.py")
_load("custom_components.spot_price_predictor.dtaci",
      HA_PKG / "dtaci.py")
dk_dtaci = _load("custom_components.spot_price_predictor.dk_dtaci",
                 HA_PKG / "dk_dtaci.py")
DkDtACIBundle = dk_dtaci.DkDtACIBundle
compute_dk_cheap_peak = dk_utils.compute_dk_cheap_peak


# ── Holiday calendar ───────────────────────────────────────────────


def _build_holidays(years: list[int]) -> set[str]:
    """Holiday set common to FI/SE/EE — all observe the major Christian
    holidays plus midsummer plus national days. Crude but adequate for
    is_off_day classification in the AR(2) forecaster."""
    holidays = set()
    easter_sunday = {
        2022: (4, 17), 2023: (4, 9), 2024: (3, 31),
        2025: (4, 20), 2026: (4, 5), 2027: (3, 28),
    }
    fixed = [(1, 1), (1, 6), (5, 1), (12, 24), (12, 25), (12, 26)]
    for y in years:
        for m, d in fixed:
            holidays.add(f"{y:04d}-{m:02d}-{d:02d}")
        if y in easter_sunday:
            em, ed = easter_sunday[y]
            es = datetime(y, em, ed)
            for delta in (-2, 1, 39, 49):
                d = es + timedelta(days=delta)
                holidays.add(d.strftime("%Y-%m-%d"))
    return holidays


# ── Data fetchers ─────────────────────────────────────────────────


def fetch_finnish_prices(years: int) -> list[tuple[datetime, float]]:
    """FI hourly spot via Sähkötin."""
    end = datetime.now(tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0)
    start = end - timedelta(days=years * 365)
    out: list[tuple[datetime, float]] = []
    cursor = start
    print(f"[fetch] FI {start.date()} -> {end.date()}", flush=True)
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=60), end)
        try:
            r = requests.get(
                "https://sahkotin.fi/prices",
                params={
                    "start": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                }, timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "prices" in data:
                data = data["prices"]
            for entry in data:
                ts_str = entry.get("date") or entry.get("timestamp")
                if ts_str is None:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except ValueError:
                    continue
                price = float(entry.get("value", 0.0))
                out.append((ts.astimezone(timezone.utc), price))
        except Exception as exc:
            print(f"  fail {cursor.date()}: {exc}", flush=True)
        cursor = chunk_end
    return _dedup_sorted(out)


def fetch_se_prices(zone: str, years: int) -> list[tuple[datetime, float]]:
    """SE1, SE3, ... via elprisetjustnu.se."""
    end = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=years * 365)
    out: list[tuple[datetime, float]] = []
    cursor = start
    print(f"[fetch] {zone.upper()} {start.date()} -> {end.date()} day-by-day",
          flush=True)
    n_days = 0
    while cursor < end:
        url = (f"https://www.elprisetjustnu.se/api/v1/prices/"
               f"{cursor.year}/{cursor.month:02d}-{cursor.day:02d}_"
               f"{zone.upper()}.json")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for entry in r.json():
                    ts_str = entry.get("time_start")
                    eur_kwh = entry.get("EUR_per_kWh")
                    if ts_str is None or eur_kwh is None:
                        continue
                    ts = datetime.fromisoformat(ts_str).astimezone(
                        timezone.utc)
                    out.append((ts, float(eur_kwh) * 1000.0))
        except Exception:
            pass
        cursor += timedelta(days=1)
        n_days += 1
        if n_days % 200 == 0:
            print(f"  day {n_days} -> {cursor.date()} ({len(out)} hrs)",
                  flush=True)
    return _dedup_sorted(out)


def fetch_ee_prices(years: int) -> list[tuple[datetime, float]]:
    """EE hourly spot via Elering."""
    end = datetime.now(tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0)
    start = end - timedelta(days=years * 365)
    out: list[tuple[datetime, float]] = []
    cursor = start
    print(f"[fetch] EE {start.date()} -> {end.date()}", flush=True)
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        try:
            r = requests.get(
                "https://dashboard.elering.ee/api/nps/price",
                params={
                    "start": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                }, timeout=30,
            )
            r.raise_for_status()
            data = r.json().get("data", {}).get("ee", [])
            for e in data:
                ts = datetime.fromtimestamp(
                    e["timestamp"], tz=timezone.utc)
                out.append((ts, float(e["price"])))
        except Exception as exc:
            print(f"  fail {cursor.date()}: {exc}", flush=True)
        cursor = chunk_end
    return _dedup_sorted(out)


def _dedup_sorted(entries: list[tuple[datetime, float]]
                  ) -> list[tuple[datetime, float]]:
    entries.sort(key=lambda x: x[0])
    seen: set[datetime] = set()
    out: list[tuple[datetime, float]] = []
    for ts, p in entries:
        if ts in seen:
            continue
        seen.add(ts)
        out.append((ts, p))
    print(f"  -> {len(out)} unique hourly entries", flush=True)
    return out


def fetch_zone(zone: str, years: int) -> list[tuple[datetime, float]]:
    if zone == "fi":
        return fetch_finnish_prices(years)
    if zone in ("se1", "se3"):
        return fetch_se_prices(zone, years)
    if zone == "ee":
        return fetch_ee_prices(years)
    raise ValueError(f"unknown zone {zone}")


# ── AR(2) day-ahead forecaster ─────────────────────────────────────


class DayAheadAR2:
    """Profile + AR(2) day-ahead 24-hour forecaster.

    Trained on warmup data, each forecast day produces 24 hourly prices
    by iterating the AR(2) recursion forward from the last two known
    prices. Residuals decay geometrically toward 0 over the 24 hours,
    so the forecast smoothly converges to the day's profile.

    Returns 24 hourly prices for the forecast day (in chronological
    order, hour 0..23) which are then sorted to compute D(k).
    """

    def __init__(self) -> None:
        self.profile_wd: list[float] = [0.0] * 24
        self.profile_we: list[float] = [0.0] * 24
        self.phi_1: float = 0.7
        self.phi_2: float = 0.0

    def fit(self, train_data: list[tuple[datetime, float, bool]]) -> None:
        wd_s = [0.0] * 24
        wd_n = [0] * 24
        we_s = [0.0] * 24
        we_n = [0] * 24
        for ts, p, off in train_data:
            h = ts.hour
            if off:
                we_s[h] += p
                we_n[h] += 1
            else:
                wd_s[h] += p
                wd_n[h] += 1
        for h in range(24):
            self.profile_wd[h] = wd_s[h] / wd_n[h] if wd_n[h] else 0.0
            self.profile_we[h] = we_s[h] / we_n[h] if we_n[h] else 0.0

        residuals: list[float] = []
        for ts, p, off in train_data:
            prof = self.profile_we[ts.hour] if off else self.profile_wd[ts.hour]
            residuals.append(p - prof)
        # OLS for AR(2) coefficients
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
        det = xx00 * xx11 - xx01 * xx01
        if abs(det) < 1e-9:
            self.phi_1, self.phi_2 = 0.7, 0.0
        else:
            self.phi_1 = (xy0 * xx11 - xy1 * xx01) / det
            self.phi_2 = (xy1 * xx00 - xy0 * xx01) / det

    def forecast_day(
        self,
        target_date: datetime,
        target_off: bool,
        last_price: float,
        last_off: bool,
        prev_price: float,
        prev_off: bool,
    ) -> list[float]:
        """Forecast 24 hourly prices for the day starting at `target_date`."""
        last_h = (target_date - timedelta(hours=1)).hour
        prev_h = (target_date - timedelta(hours=2)).hour
        last_prof = (self.profile_we[last_h]
                     if last_off else self.profile_wd[last_h])
        prev_prof = (self.profile_we[prev_h]
                     if prev_off else self.profile_wd[prev_h])
        r1 = last_price - last_prof   # residual at t-1
        r2 = prev_price - prev_prof   # residual at t-2
        out = []
        for h in range(24):
            prof = (self.profile_we[h] if target_off
                    else self.profile_wd[h])
            r_hat = self.phi_1 * r1 + self.phi_2 * r2
            out.append(prof + r_hat)
            r2 = r1
            r1 = r_hat
        return out


# ── Static empirical baseline (per (direction, k)) ─────────────────


class StaticDkBaseline:
    """For each of 24 (direction, k) statistics, keep a rolling buffer
    of recent residuals and serve the empirical (1-α) quantile."""

    def __init__(self, target_coverage: float = 0.9, window: int = 365,
                 min_warmup: int = 14) -> None:
        self.alpha = 1.0 - target_coverage
        self.window = window
        self.min_warmup = min_warmup
        self.residuals: dict[tuple[str, int], deque[float]] = {
            (d, k): deque(maxlen=window)
            for d in ("cheap", "peak") for k in range(1, 13)
        }

    def predict_intervals(
        self, forecast_dk_cheap: list[float], forecast_dk_peak: list[float],
    ) -> dict[str, dict[str, list[float]]]:
        out = {"cheap": {"lower": [], "point": [], "upper": []},
               "peak":  {"lower": [], "point": [], "upper": []}}
        for direction, fc in (("cheap", forecast_dk_cheap),
                              ("peak", forecast_dk_peak)):
            for k in range(1, 13):
                buf = self.residuals[(direction, k)]
                f = float(fc[k - 1])
                if len(buf) < self.min_warmup:
                    out[direction]["lower"].append(f)
                    out[direction]["point"].append(f)
                    out[direction]["upper"].append(f)
                    continue
                vals = sorted(abs(r) for r in buf)
                pos = (1.0 - self.alpha) * (len(vals) - 1)
                lo = int(pos)
                frac = pos - lo
                if lo >= len(vals) - 1:
                    q = vals[-1]
                else:
                    q = vals[lo] * (1.0 - frac) + vals[lo + 1] * frac
                out[direction]["lower"].append(f - q)
                out[direction]["point"].append(f)
                out[direction]["upper"].append(f + q)
        return out

    def update(
        self, forecast_dk_cheap: list[float], forecast_dk_peak: list[float],
        actual_dk_cheap: list[float], actual_dk_peak: list[float],
    ) -> None:
        for direction, fc, ac in (
            ("cheap", forecast_dk_cheap, actual_dk_cheap),
            ("peak", forecast_dk_peak, actual_dk_peak),
        ):
            for k in range(1, 13):
                self.residuals[(direction, k)].append(
                    float(ac[k - 1]) - float(fc[k - 1])
                )


# ── Walk-forward driver ────────────────────────────────────────────


def walk_forward(
    prices: list[tuple[datetime, float]],
    holidays: set[str],
    warmup_days: int = 180,
    target_coverage: float = 0.9,
) -> dict:
    """Walk forward through one zone's prices, returning per-(direction,k)
    metrics for static and DtACI methods."""
    if len(prices) < (warmup_days + 30) * 24:
        raise ValueError(
            f"need at least {(warmup_days + 30) * 24} hours, "
            f"got {len(prices)}"
        )

    def _is_off(ts: datetime) -> bool:
        return ts.weekday() >= 5 or ts.date().isoformat() in holidays

    train = [(ts, p, _is_off(ts))
             for ts, p in prices[: warmup_days * 24]]
    test_hours = [(ts, p, _is_off(ts))
                  for ts, p in prices[warmup_days * 24:]]

    fc = DayAheadAR2()
    fc.fit(train)
    print(f"  AR(2) phi_1={fc.phi_1:.3f}, phi_2={fc.phi_2:.3f}")

    # Group test hours into days
    test_days: dict[str, list[tuple[datetime, float, bool]]] = {}
    for ts, p, off in test_hours:
        day_key = ts.date().isoformat()
        test_days.setdefault(day_key, []).append((ts, p, off))
    # Only keep complete-24h days
    complete_days = sorted(d for d, hrs in test_days.items()
                           if len(hrs) == 24)
    print(f"  test days: {len(complete_days)} complete days")

    # Methods
    static = StaticDkBaseline(target_coverage=target_coverage,
                              window=365, min_warmup=14)
    bundle = DkDtACIBundle(target_coverage=target_coverage,
                           window=365, min_warmup=14,
                           bias_warmup_steps=14, cadence_per_day=1)

    # Per-(direction,k) accumulators
    per_k = {
        method: {
            (direction, k): {"covered": 0, "predicted": 0, "abs_err": 0.0,
                             "width_sum": 0.0, "raw_abs_err": 0.0}
            for direction in ("cheap", "peak") for k in range(1, 13)
        }
        for method in ("static", "dtaci")
    }

    # Seed AR(2) state from end of train
    last_price = train[-1][1]
    last_off = train[-1][2]
    prev_price = train[-2][1]
    prev_off = train[-2][2]

    for d_idx, day_key in enumerate(complete_days):
        day_hours = test_days[day_key]
        # Sort within day by hour
        day_hours.sort(key=lambda x: x[0])
        target_off = day_hours[0][2]
        target_date = day_hours[0][0]
        # Forecast 24 hourly prices
        f_24 = fc.forecast_day(
            target_date, target_off,
            last_price, last_off, prev_price, prev_off,
        )
        # Actual D(k)
        a_24 = [p for _, p, _ in day_hours]
        a_cheap, a_peak = compute_dk_cheap_peak(a_24)
        f_cheap, f_peak = compute_dk_cheap_peak(f_24)

        # Predict bands BEFORE update
        bands_static = static.predict_intervals(f_cheap, f_peak)
        bands_dtaci = bundle.predict_intervals(f_cheap, f_peak)

        # Score
        for direction, ac in (("cheap", a_cheap), ("peak", a_peak)):
            for k in range(1, 13):
                f_k = f_cheap[k - 1] if direction == "cheap" else f_peak[k - 1]
                a_k = ac[k - 1]
                raw_err = abs(a_k - f_k)
                for method, bands in (("static", bands_static),
                                      ("dtaci", bands_dtaci)):
                    low = bands[direction]["lower"][k - 1]
                    point = bands[direction]["point"][k - 1]
                    high = bands[direction]["upper"][k - 1]
                    rec = per_k[method][(direction, k)]
                    rec["raw_abs_err"] += raw_err
                    rec["abs_err"] += abs(a_k - point)
                    rec["width_sum"] += (high - low)
                    rec["predicted"] += 1
                    if low <= a_k <= high:
                        rec["covered"] += 1

        # Now update
        static.update(f_cheap, f_peak, a_cheap, a_peak)
        bundle.update(f_cheap, f_peak, a_cheap, a_peak)

        # Roll AR(2) state
        prev_price = day_hours[-2][1]
        prev_off = day_hours[-2][2]
        last_price = day_hours[-1][1]
        last_off = day_hours[-1][2]

        if (d_idx + 1) % 100 == 0:
            print(f"  day {d_idx+1}/{len(complete_days)} {day_key}",
                  flush=True)

    # Aggregate per-method, per-k
    summary = {"target_coverage": target_coverage,
               "n_days_evaluated": len(complete_days),
               "methods": {}}
    for method, per_dk in per_k.items():
        m_per_k = {"cheap": {}, "peak": {}}
        all_cov = []
        all_w = []
        all_mae = []
        all_raw = []
        for (direction, k), rec in per_dk.items():
            n = rec["predicted"] or 1
            cov = rec["covered"] / n
            mae = rec["abs_err"] / n
            raw_mae = rec["raw_abs_err"] / n
            w = rec["width_sum"] / n
            m_per_k[direction][k] = {
                "coverage": round(cov, 4),
                "MAE": round(mae, 3),
                "raw_MAE": round(raw_mae, 3),
                "mean_width": round(w, 3),
                "n": rec["predicted"],
            }
            all_cov.append(cov)
            all_w.append(w)
            all_mae.append(mae)
            all_raw.append(raw_mae)
        summary["methods"][method] = {
            "mean_coverage": round(sum(all_cov) / len(all_cov), 4),
            "mean_width": round(sum(all_w) / len(all_w), 3),
            "mean_MAE": round(sum(all_mae) / len(all_mae), 3),
            "mean_raw_MAE": round(sum(all_raw) / len(all_raw), 3),
            "per_k": m_per_k,
        }
    summary["dtaci_diagnostics"] = bundle.diagnostics()
    return summary


# ── Markdown rendering ────────────────────────────────────────────


def render_markdown(zone: str, summary: dict) -> str:
    target = summary["target_coverage"]
    out = []
    out.append(f"# DtACI Per-D(i) Validation — {zone.upper()}")
    out.append("")
    out.append(f"Generated: {datetime.now(tz=timezone.utc).isoformat()}")
    out.append(f"Target coverage: {target:.2f}")
    out.append(f"Days evaluated: {summary['n_days_evaluated']}")
    out.append("")

    out.append("## Headline (mean over 24 instances)")
    out.append("")
    out.append("| Method | mean coverage | mean MAE | mean width |")
    out.append("|---|---:|---:|---:|")
    for m, label in (("static", "static empirical"),
                     ("dtaci", "DtACI bundle")):
        s = summary["methods"][m]
        out.append(f"| {label} | {s['mean_coverage']:.4f} | "
                   f"{s['mean_MAE']:.2f} | {s['mean_width']:.2f} |")
    out.append("")

    out.append("## Per-k coverage (cheap end, target 0.9)")
    out.append("")
    out.append("| k | static | DtACI |")
    out.append("|---:|---:|---:|")
    for k in range(1, 13):
        sc = summary["methods"]["static"]["per_k"]["cheap"][k]["coverage"]
        dc = summary["methods"]["dtaci"]["per_k"]["cheap"][k]["coverage"]
        out.append(f"| {k} | {sc:.3f} | {dc:.3f} |")
    out.append("")

    out.append("## Per-k coverage (peak end, target 0.9)")
    out.append("")
    out.append("| k | static | DtACI |")
    out.append("|---:|---:|---:|")
    for k in range(1, 13):
        sc = summary["methods"]["static"]["per_k"]["peak"][k]["coverage"]
        dc = summary["methods"]["dtaci"]["per_k"]["peak"][k]["coverage"]
        out.append(f"| {k} | {sc:.3f} | {dc:.3f} |")
    out.append("")

    out.append("## Per-k DtACI diagnostics (end of holdout)")
    out.append("")
    out.append("| direction | k | coverage | bias_ema | alpha_agg | "
               "dom γ | width | weight entropy |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    diag = summary["dtaci_diagnostics"]["per_k"]
    for direction in ("cheap", "peak"):
        for k in range(1, 13):
            r = diag[direction][k]
            out.append(
                f"| {direction} | {k} | {r['coverage']:.3f} | "
                f"{r['bias_ema']:+.2f} | {r['alpha_agg']:.3f} | "
                f"{r['dominant_gamma']:.4f} | "
                f"{2 * r['half_width']:.2f} | "
                f"{r['weight_entropy_bits']:.2f} bits |"
            )
    out.append("")
    return "\n".join(out)


# ── Entry point ────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", default="fi,se1,se3,ee",
                    help="Comma-separated zones (default fi,se1,se3,ee)")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--warmup-days", type=int, default=180)
    ap.add_argument("--target-coverage", type=float, default=0.9)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    zones = [z.strip().lower() for z in args.zones.split(",") if z.strip()]
    out_dir = REPO_ROOT / "studies" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / "studies"
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")

    # Holiday calendar (computed once)
    years_seen: set[int] = set()
    for z in zones:
        cache_path = cache_dir / f"_dtaci_dk_{z}_prices_cache.json"
        if cache_path.exists() and not args.no_cache:
            with open(cache_path) as f:
                raw = json.load(f)
            years_seen.update(int(t[:4]) for t, _ in raw[:: 1000])
    if not years_seen:
        years_seen = set(range(datetime.utcnow().year - 4,
                               datetime.utcnow().year + 1))
    holidays = _build_holidays(sorted(years_seen))

    combined: dict = {"target_coverage": args.target_coverage,
                       "zones": {}}

    for zone in zones:
        cache_path = cache_dir / f"_dtaci_dk_{zone}_prices_cache.json"
        if cache_path.exists() and not args.no_cache:
            print(f"[cache] {cache_path.name}")
            with open(cache_path) as f:
                raw = json.load(f)
            prices = [(datetime.fromisoformat(t), float(p)) for t, p in raw]
        else:
            prices = fetch_zone(zone, args.years)
            if not args.no_cache and prices:
                with open(cache_path, "w") as f:
                    json.dump([(ts.isoformat(), p) for ts, p in prices], f)
                print(f"[cache] wrote {cache_path.name}")
        if len(prices) < (args.warmup_days + 30) * 24:
            print(f"[!] {zone}: insufficient data, skipping")
            continue
        # Add any missing holiday years
        ys = sorted({p[0].year for p in prices})
        for y in ys:
            if y not in years_seen:
                holidays.update(_build_holidays([y]))
                years_seen.add(y)
        print(f"\n=== {zone.upper()} ===")
        summary = walk_forward(
            prices, holidays,
            warmup_days=args.warmup_days,
            target_coverage=args.target_coverage,
        )
        md = render_markdown(zone, summary)
        md_path = out_dir / f"dtaci_dk_{zone}_{stamp}.md"
        json_path = out_dir / f"dtaci_dk_{zone}_{stamp}.json"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        combined["zones"][zone] = summary
        print()
        print(f"  static : cov {summary['methods']['static']['mean_coverage']:.4f}  "
              f"width {summary['methods']['static']['mean_width']:.2f}  "
              f"MAE {summary['methods']['static']['mean_MAE']:.2f}")
        print(f"  dtaci  : cov {summary['methods']['dtaci']['mean_coverage']:.4f}  "
              f"width {summary['methods']['dtaci']['mean_width']:.2f}  "
              f"MAE {summary['methods']['dtaci']['mean_MAE']:.2f}")
        print(f"  raw MAE: {summary['methods']['dtaci']['mean_raw_MAE']:.2f}")

    # Combined JSON
    combined_path = out_dir / f"dtaci_dk_combined_{stamp}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n[done] combined -> {combined_path.name}")


if __name__ == "__main__":
    main()
