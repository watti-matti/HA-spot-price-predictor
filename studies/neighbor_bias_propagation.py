"""Does neighbor-zone bias correction improve FI accuracy?

The FI hourly Ridge model takes AR(2) day-ahead forecasts of SE1, SE3,
EE as features (coefs 0.287, 0.252, 0.146 in the latest retrain). If
neighbor residuals are correlated with FI residuals, bias-correcting
the neighbor features should propagate into FI accuracy improvement.
If neighbor residuals are uncorrelated with FI residuals, bias
correction on neighbors does nothing for FI (and the 4-zone DtACI
deployment is overscoped — only the FI bundle matters).

This script measures the empirical correlation directly, with no
training-time mismatch confound:

  1. Reuse the cached hourly prices for FI / SE1 / SE3 / EE.
  2. Walk forward day-ahead AR(2) forecasts for each zone in parallel.
  3. Collect day-level forecast residuals: r_z[d] = actual_z[d] -
     forecast_z[d], evaluated as the daily mean for simplicity.
  4. Compute Pearson correlations cor(r_FI, r_SEi), cor(r_FI, r_SE3),
     cor(r_FI, r_EE).
  5. Report.

Interpretation:
  * |r| < 0.2  → no leverage; neighbor bias correction would not
                  improve FI MAE through the linear Ridge feed.
  * 0.2 ≤ |r| < 0.5 → moderate leverage; partial improvement plausible.
  * |r| ≥ 0.5  → strong leverage; neighbor bias correction is
                  expected to materially improve FI accuracy.
"""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cache(zone: str) -> list[tuple[datetime, float]]:
    p = REPO_ROOT / "studies" / f"_dtaci_dk_{zone}_prices_cache.json"
    with open(p) as f:
        raw = json.load(f)
    return [(datetime.fromisoformat(t), float(p)) for t, p in raw]


def is_off(ts: datetime, holidays: set[str]) -> bool:
    return ts.weekday() >= 5 or ts.date().isoformat() in holidays


def fit_profile_ar2(train_data: list[tuple[datetime, float, bool]]
                    ) -> tuple[list[float], list[float], float, float]:
    """Same as validate_dtaci_dk.DayAheadAR2.fit, returned as a tuple."""
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
    profile_wd = [wd_s[h] / wd_n[h] if wd_n[h] else 0.0 for h in range(24)]
    profile_we = [we_s[h] / we_n[h] if we_n[h] else 0.0 for h in range(24)]
    residuals = [p - (profile_we[ts.hour] if off else profile_wd[ts.hour])
                 for ts, p, off in train_data]
    xx00 = xx01 = xx11 = xy0 = xy1 = 0.0
    for t in range(2, len(residuals)):
        r0 = residuals[t - 1]; r1 = residuals[t - 2]; y = residuals[t]
        xx00 += r0 * r0; xx01 += r0 * r1; xx11 += r1 * r1
        xy0 += r0 * y; xy1 += r1 * y
    det = xx00 * xx11 - xx01 * xx01
    if abs(det) < 1e-9:
        return profile_wd, profile_we, 0.7, 0.0
    phi_1 = (xy0 * xx11 - xy1 * xx01) / det
    phi_2 = (xy1 * xx00 - xy0 * xx01) / det
    return profile_wd, profile_we, phi_1, phi_2


def forecast_day(profile_wd, profile_we, phi_1, phi_2,
                 target_off, last_h, last_price, last_off,
                 prev_h, prev_price, prev_off) -> list[float]:
    last_prof = profile_we[last_h] if last_off else profile_wd[last_h]
    prev_prof = profile_we[prev_h] if prev_off else profile_wd[prev_h]
    r1 = last_price - last_prof
    r2 = prev_price - prev_prof
    out = []
    for h in range(24):
        prof = profile_we[h] if target_off else profile_wd[h]
        r_hat = phi_1 * r1 + phi_2 * r2
        out.append(prof + r_hat)
        r2 = r1; r1 = r_hat
    return out


def build_holidays(years: list[int]) -> set[str]:
    holidays: set[str] = set()
    easter_sunday = {
        2022: (4, 17), 2023: (4, 9), 2024: (3, 31),
        2025: (4, 20), 2026: (4, 5), 2027: (3, 28),
    }
    fixed = [(1, 1), (1, 6), (5, 1), (12, 24), (12, 25), (12, 26)]
    for y in years:
        for m, d in fixed:
            holidays.add(f"{y:04d}-{m:02d}-{d:02d}")
        if y in easter_sunday:
            from datetime import timedelta as _td
            em, ed = easter_sunday[y]
            es = datetime(y, em, ed)
            for delta in (-2, 1, 39, 49):
                holidays.add((es + _td(days=delta)).strftime("%Y-%m-%d"))
    return holidays


def walk_zone(prices, holidays, warmup_days: int = 180):
    """Walk forward day-ahead forecasts for one zone. Return per-day
    (date_str, forecast_24, actual_24)."""
    n_warmup = warmup_days * 24
    train = [(ts, p, is_off(ts, holidays)) for ts, p in prices[:n_warmup]]
    profile_wd, profile_we, phi_1, phi_2 = fit_profile_ar2(train)

    # Group test hours by date
    by_day: dict[str, list[tuple[datetime, float, bool]]] = {}
    for ts, p in prices[n_warmup:]:
        d = ts.date().isoformat()
        by_day.setdefault(d, []).append((ts, p, is_off(ts, holidays)))
    days = sorted(d for d, hrs in by_day.items() if len(hrs) == 24)

    last_price = train[-1][1]; last_off = train[-1][2]
    prev_price = train[-2][1]; prev_off = train[-2][2]
    last_h = train[-1][0].hour; prev_h = train[-2][0].hour

    out: dict[str, dict] = {}
    for day_key in days:
        hrs = sorted(by_day[day_key], key=lambda x: x[0])
        target_off = hrs[0][2]
        f_24 = forecast_day(
            profile_wd, profile_we, phi_1, phi_2,
            target_off, last_h, last_price, last_off,
            prev_h, prev_price, prev_off,
        )
        a_24 = [p for _, p, _ in hrs]
        out[day_key] = {"forecast": f_24, "actual": a_24}
        # Roll
        prev_price = hrs[-2][1]; prev_off = hrs[-2][2]; prev_h = hrs[-2][0].hour
        last_price = hrs[-1][1]; last_off = hrs[-1][2]; last_h = hrs[-1][0].hour
    return out


def daily_residual(rec: dict) -> float:
    """Day-level residual = mean(actual) - mean(forecast)."""
    f = rec["forecast"]; a = rec["actual"]
    return sum(a) / 24 - sum(f) / 24


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(sx2 * sy2)
    return sxy / den if den > 0 else 0.0


def main():
    print("[load] caches...", flush=True)
    zones_data = {z: load_cache(z) for z in ("fi", "se1", "se3", "ee")}
    years = set()
    for prices in zones_data.values():
        years.update(p[0].year for p in prices[::5000])
    holidays = build_holidays(sorted(years))

    print("[walk] day-ahead AR(2) per zone (180-day warmup)...", flush=True)
    zone_runs = {z: walk_zone(prices, holidays) for z, prices in zones_data.items()}

    # Find common days across all four zones
    common_days = sorted(set.intersection(
        *(set(run.keys()) for run in zone_runs.values())))
    print(f"  {len(common_days)} days common to all 4 zones")

    # Daily residuals per zone
    daily_r = {
        z: [daily_residual(zone_runs[z][d]) for d in common_days]
        for z in zone_runs
    }

    # Pearson correlations of FI vs neighbour residuals
    print()
    print("=== Pearson correlation of daily residuals ===")
    for nb in ("se1", "se3", "ee"):
        r = pearson(daily_r["fi"], daily_r[nb])
        print(f"  cor(r_FI, r_{nb.upper():3s}) = {r:+.4f}")

    # Cross-correlations among neighbours (sanity)
    print()
    print("=== Among neighbours ===")
    for a, b in (("se1", "se3"), ("se1", "ee"), ("se3", "ee")):
        r = pearson(daily_r[a], daily_r[b])
        print(f"  cor(r_{a.upper()}, r_{b.upper()}) = {r:+.4f}")

    # Joint variance explained by linear combination using Ridge weights
    # from the latest retrain (model_coefs.json):
    #   coefs in standardized space: ar_se1 +0.287, ar_se3 +0.252, ar_ee +0.146
    # We use these as relative weights for a synthetic predictor of FI residual.
    coefs = {"se1": 0.287, "se3": 0.252, "ee": 0.146}
    # Simple OLS R^2 of r_FI ~ a*r_SE1 + b*r_SE3 + c*r_EE  (3-feature regression)
    # Reuse Pearson algebra: for orthogonalised features we'd need a proper
    # multi-OLS, but we can also fit it directly.
    n = len(common_days)
    X = [[daily_r["se1"][i], daily_r["se3"][i], daily_r["ee"][i]]
         for i in range(n)]
    y = list(daily_r["fi"])
    # Multiple linear regression (least squares)
    # Solve normal equations  X^T X b = X^T y
    p = 3
    # Build XtX and Xty
    XtX = [[0.0] * p for _ in range(p)]
    Xty = [0.0] * p
    for i in range(n):
        for r in range(p):
            for c in range(p):
                XtX[r][c] += X[i][r] * X[i][c]
            Xty[r] += X[i][r] * y[i]

    # Gauss elimination
    def solve(A, b):
        n_ = len(A)
        # Augment
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for i in range(n_):
            # Pivot
            piv = i + max(range(n_ - i),
                          key=lambda j: abs(M[i + j][i]))
            M[i], M[piv] = M[piv], M[i]
            if abs(M[i][i]) < 1e-12:
                return None
            inv = 1.0 / M[i][i]
            M[i] = [v * inv for v in M[i]]
            for j in range(n_):
                if j == i:
                    continue
                fac = M[j][i]
                M[j] = [v - fac * w for v, w in zip(M[j], M[i])]
        return [row[-1] for row in M]
    beta = solve(XtX, Xty)
    print()
    print("=== Multi-OLS  r_FI ~ a*r_SE1 + b*r_SE3 + c*r_EE ===")
    if beta is not None:
        print(f"  fitted: {beta[0]:+.3f}*r_SE1 {beta[1]:+.3f}*r_SE3 "
              f"{beta[2]:+.3f}*r_EE")
        # R^2
        y_mean = sum(y) / n
        sst = sum((yi - y_mean) ** 2 for yi in y)
        ssr = 0.0
        for i in range(n):
            yhat = sum(beta[k] * X[i][k] for k in range(p))
            ssr += (y[i] - yhat) ** 2
        r2 = 1.0 - ssr / sst if sst > 0 else 0.0
        print(f"  R^2 = {r2:.4f}  "
              f"(fraction of FI residual variance explained "
              f"by neighbour residuals)")
    else:
        print("  multi-OLS singular")

    # Variance of daily residual per zone
    print()
    print("=== Daily residual statistics ===")
    for z, vals in daily_r.items():
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        print(f"  {z.upper()}: mean residual {m:+.2f}, "
              f"std {s:.2f} EUR/MWh, |max|={max(abs(v) for v in vals):.1f}")

    # Implication: if r_FI variance explained by neighbours > 10%, then
    # bias-correcting AR features would propagate into a meaningful FI
    # MAE improvement (~ sqrt(R^2) * Ridge feature contribution).
    # If R^2 < 5%, neighbour bias correction is irrelevant for FI.
    if beta is not None:
        if r2 > 0.10:
            print()
            print(f"VERDICT: R^2 = {r2:.3f} > 0.10 — multi-zone neighbour "
                  f"bias correction is expected to improve FI accuracy. "
                  f"4-zone DtACI deployment is supported.")
        elif r2 > 0.05:
            print()
            print(f"VERDICT: R^2 = {r2:.3f} marginal (5%-10%) — small "
                  f"FI improvement plausible. Decision-grade evidence "
                  f"would require running an actual FI Ridge inference "
                  f"experiment with bias-corrected AR features.")
        else:
            print()
            print(f"VERDICT: R^2 = {r2:.3f} < 0.05 — neighbour residuals "
                  f"explain almost no FI variance. Multi-zone DtACI does "
                  f"NOT improve FI accuracy. Deploy FI bundle only.")


if __name__ == "__main__":
    main()
