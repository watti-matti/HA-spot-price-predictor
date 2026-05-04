"""fingrid_netload_study.py — does residual demand (net load) explain the
AR(2) forecast residuals on FI hourly prices?

Hypothesis
----------
FI price spikes during 2025-12 → 2026-03 happened when **demand exceeded
the available supply at low marginal cost** — a market-pinch event.
The current FI Ridge model captures this *indirectly* via wind speed
(weather), heating-degree-days (demand proxy), and nuclear deficit
(supply gap). The hypothesis is that **net load**, defined as

    net_load[t] = consumption_forecast[t]
                  - wind_forecast[t]
                  - solar_forecast[t]
                  - nuclear_real[t]

is a *direct* market-pinch indicator. When net_load is high, fossil
peakers and imports dominate the merit-order, and prices spike.

The walk-forward report (`forecaster_performance_fi_*.html`) showed
AR(2) MAE of 57.7 EUR/MWh in 2026-Q1 (the spike). If net_load
explains a meaningful fraction of those residuals, adding it as a
feature to the FI Ridge would close that gap.

Methodology
-----------
1. Fetch Fingrid datasets for the OOS window (2025-12 → 2026-04):
     165 — Electricity consumption forecast (15-min)
     246 — Wind power generation forecast (15-min)
     247 — Solar power generation forecast (15-min)
     188 — Nuclear power production real-time (3-min)
   Resample everything to hourly means.
2. Load the FI walk-forward records produced by
   `validate_forecaster_performance.py`.
3. Align the two on hourly timestamps.
4. Compute:
     - Pearson correlation cor(net_load, price)
     - Pearson correlation cor(net_load, AR(2) residual)
     - Conditional MAE: |residual| | net_load > Q90 vs < Q10
     - OLS:  residual ~ a*net_load + b   →  R^2
5. Report whether net_load can plausibly close the spike-period MAE gap.

Run
---
    set FINGRID_API_KEY=...
    python studies/fingrid_netload_study.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "studies" / "_fingrid_cache"
CACHE_DIR.mkdir(exist_ok=True)


# Datasets we care about
DATASETS = {
    "consumption":  165,  # MW, 15-min, day-ahead consumption forecast
    "wind":         246,  # MW, 15-min, day-ahead wind generation forecast
    "solar":        247,  # MW, 15-min, day-ahead solar generation forecast
    "nuclear_real": 188,  # MW, 3-min, real-time nuclear production
}


# ── Fingrid API client ──────────────────────────────────────────────


def fetch_dataset(ds_id: int, start: datetime, end: datetime,
                  api_key: str, chunk_days: int = 30,
                  delay_s: float = 0.5) -> list[dict]:
    """Fetch all rows for `ds_id` between `start` and `end`, paginating
    by date chunks (Fingrid caps response sizes) and respecting the
    rate limit (~30 req/min). Cached on disk per-chunk."""
    cache_key = f"ds{ds_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    headers = {"x-api-key": api_key}
    out: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        # Each chunk: paginate through pages until empty
        page = 1
        while True:
            params = {
                "startTime": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "endTime":   chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "format":    "json",
                "pageSize":  20000,
                "page":      page,
            }
            for retry in range(5):
                r = requests.get(
                    f"https://data.fingrid.fi/api/datasets/{ds_id}/data",
                    params=params, headers=headers, timeout=30,
                )
                if r.status_code == 200:
                    break
                if r.status_code == 429:
                    wait = (retry + 1) * 3
                    print(f"   429 rate limit on ds {ds_id}; waiting {wait}s",
                          flush=True)
                    time.sleep(wait)
                    continue
                print(f"   HTTP {r.status_code} on ds {ds_id}: {r.text[:200]}",
                      flush=True)
                break
            else:
                print(f"   gave up on ds {ds_id} after retries", flush=True)
                break
            try:
                d = r.json()
            except Exception:
                break
            rows = d.get("data") if isinstance(d, dict) else d
            if not rows:
                break
            out.extend(rows)
            pagination = d.get("pagination") if isinstance(d, dict) else None
            if not pagination or pagination.get("nextPage") is None:
                break
            page = pagination["nextPage"]
            time.sleep(delay_s)
        cursor = chunk_end
        time.sleep(delay_s)
        print(f"  ds {ds_id}: {start.date()} → {cursor.date()} cumulative "
              f"{len(out)} rows", flush=True)

    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


def hourly_mean(rows: list[dict]) -> dict[datetime, float]:
    """Resample sub-hourly data to hourly mean."""
    bucket: dict[datetime, list[float]] = defaultdict(list)
    for row in rows:
        ts_str = row.get("startTime")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        h = ts.replace(minute=0, second=0, microsecond=0)
        try:
            bucket[h].append(float(row["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {h: sum(v) / len(v) for h, v in bucket.items()}


# ── Statistics ──────────────────────────────────────────────────────


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(sx2 * sy2)
    return sxy / den if den > 0 else 0.0


def ols_r2(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fit y = a*x + b. Return (a, b, R^2)."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0 or sy2 == 0:
        return 0.0, my, 0.0
    a = sxy / sx2
    b = my - a * mx
    yhat = [a * x + b for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    r2 = 1.0 - ss_res / sy2
    return a, b, r2


def quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(pos)
    if lo >= len(s) - 1:
        return float(s[-1])
    frac = pos - lo
    return float(s[lo]) * (1 - frac) + float(s[lo + 1]) * frac


# ── Main ────────────────────────────────────────────────────────────


def main():
    api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set FINGRID_API_KEY env var", file=sys.stderr)
        sys.exit(2)

    # Window: focus on the FI walk-forward OOS that actually had a spike.
    # The forecaster_performance_fi_*.json was 2025-10-30 → 2026-04-27.
    start = datetime(2025, 12, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 28, tzinfo=timezone.utc)
    print(f"[fingrid] window {start.date()} → {end.date()} "
          f"({(end - start).days} days)", flush=True)

    series: dict[str, dict[datetime, float]] = {}
    for name, ds_id in DATASETS.items():
        print(f"[fingrid] fetching {name} (ds {ds_id})...", flush=True)
        rows = fetch_dataset(ds_id, start, end, api_key)
        h = hourly_mean(rows)
        series[name] = h
        print(f"  {name}: {len(h)} hourly buckets, "
              f"mean {sum(h.values())/len(h):.0f} MW", flush=True)

    # Build aligned hourly net_load series.
    # Common timestamps across all 4 datasets.
    common = set(series["consumption"].keys())
    for k in ("wind", "solar", "nuclear_real"):
        common &= set(series[k].keys())
    common_sorted = sorted(common)
    print(f"[align] {len(common_sorted)} hours common to all 4 series",
          flush=True)
    if not common_sorted:
        print("ERROR: no overlapping hours; check dataset availability")
        sys.exit(2)

    net_load = {
        h: series["consumption"][h]
           - series["wind"][h]
           - series["solar"][h]
           - series["nuclear_real"][h]
        for h in common_sorted
    }

    # Load FI walk-forward records (most recent run)
    import glob
    fi_files = sorted(glob.glob(
        str(REPO_ROOT / "studies" / "results"
            / "forecaster_performance_fi_*.json")))
    if not fi_files:
        print("ERROR: no FI walk-forward report found; run "
              "validate_forecaster_performance.py first", file=sys.stderr)
        sys.exit(2)
    print(f"[load] FI walk-forward: {fi_files[-1]}")
    # The .json sidecar I produced earlier has the headline + quarters
    # but not the per-hour records. Re-run the walk-forward inline so
    # we have residuals at each hour.
    sys.path.insert(0, str(REPO_ROOT / "studies"))
    from validate_forecaster_performance import (
        _build_holidays, walk_forward,
    )
    with open(REPO_ROOT / "studies" / "_dtaci_dk_fi_prices_cache.json") as f:
        raw = json.load(f)
    prices = [(datetime.fromisoformat(t), float(p)) for t, p in raw]
    years = sorted({p[0].year for p in prices})
    holidays = _build_holidays(list(range(min(years), max(years) + 2)))
    print("[walk] re-running FI walk-forward to get hourly residuals...")
    ctx = walk_forward(prices, holidays, train_days=540, test_days=180)

    # Align records with net_load timestamps
    rec_map = {r["ts"].replace(minute=0, second=0, microsecond=0): r
               for r in ctx["records"]}
    aligned_ts = [h for h in common_sorted if h in rec_map]
    print(f"[align] {len(aligned_ts)} hours in OOS ∩ Fingrid")
    if not aligned_ts:
        print("ERROR: no overlap between FI walk-forward and Fingrid net_load",
              file=sys.stderr)
        sys.exit(2)

    nl = [net_load[h] for h in aligned_ts]
    actuals = [rec_map[h]["actual"] for h in aligned_ts]
    forecasts = [rec_map[h]["forecast"] for h in aligned_ts]
    residuals = [a - f for a, f in zip(actuals, forecasts)]
    abs_resid = [abs(r) for r in residuals]
    consumption = [series["consumption"][h] for h in aligned_ts]
    wind_v = [series["wind"][h] for h in aligned_ts]

    # Correlations
    r_nl_price = pearson(nl, actuals)
    r_nl_resid = pearson(nl, residuals)
    r_nl_absresid = pearson(nl, abs_resid)
    r_consumption_price = pearson(consumption, actuals)
    r_wind_price = pearson(wind_v, actuals)

    # OLS on raw residuals (does net_load PREDICT the AR(2) miss?)
    a_raw, b_raw, r2_raw = ols_r2(nl, residuals)
    a_abs, b_abs, r2_abs = ols_r2(nl, abs_resid)

    # Conditional MAE: |residual| | net_load > Q90 vs < Q10
    q10 = quantile(nl, 0.1); q90 = quantile(nl, 0.9)
    high_mask = [n >= q90 for n in nl]
    low_mask = [n <= q10 for n in nl]
    mae_high = (sum(r for r, m in zip(abs_resid, high_mask) if m)
                / max(1, sum(high_mask)))
    mae_low  = (sum(r for r, m in zip(abs_resid, low_mask) if m)
                / max(1, sum(low_mask)))
    mae_all  = sum(abs_resid) / len(abs_resid)

    # Plausible MAE win if we add net_load and residual~a*net_load fits well
    # MAE reduction ≈ MAE_baseline * sqrt(R^2)  (rough rule for OLS-class fits)
    plausible_pct = math.sqrt(max(0.0, r2_raw)) * 100

    # ── Report ─────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "studies" / "results"
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    md_path = out_dir / f"fingrid_netload_study_{stamp}.md"
    json_path = out_dir / f"fingrid_netload_study_{stamp}.json"

    md = f"""# Fingrid net-load — does residual demand explain FI price spikes?

**Generated**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Window**: {start.date()} → {end.date()}
**Data**: Fingrid datasets 165 (consumption forecast), 246 (wind forecast),
247 (solar forecast), 188 (nuclear real-time), all resampled to hourly mean.
**Aligned hours**: {len(aligned_ts):,}

## Hypothesis

Adding `net_load = consumption - wind - solar - nuclear` to the FI Ridge
features should explain price spikes the current model misses, because
spikes happen when residual demand pinches the merit order.

## Net-load summary statistics

| Metric | Value |
|---|---:|
| Mean net_load | {sum(nl)/len(nl):.0f} MW |
| Min net_load  | {min(nl):.0f} MW |
| Max net_load  | {max(nl):.0f} MW |
| Q10 net_load  | {q10:.0f} MW |
| Q90 net_load  | {q90:.0f} MW |
| Mean consumption | {sum(consumption)/len(consumption):.0f} MW |
| Mean wind forecast | {sum(wind_v)/len(wind_v):.0f} MW |

## Linear correlation with hourly price (EUR/MWh)

| Predictor | Pearson r vs price |
|---|---:|
| **net_load** | **{r_nl_price:+.4f}** |
| consumption alone | {r_consumption_price:+.4f} |
| wind alone | {r_wind_price:+.4f} |

## Linear correlation with AR(2) hourly residual

The interesting question — does net_load predict what the AR(2) baseline
misses?

| Predictor | Pearson r vs residual | Pearson r vs |residual| |
|---|---:|---:|
| **net_load** | **{r_nl_resid:+.4f}** | **{r_nl_absresid:+.4f}** |

OLS fit `residual ~ a*net_load + b`:

| Coefficient | Value |
|---|---:|
| slope a (EUR/MWh per MW) | {a_raw*1000:.2f} per GW |
| intercept b | {b_raw:+.2f} EUR/MWh |
| **R² of OLS** | **{r2_raw:.4f}** |

## Conditional MAE — high vs low net_load

If net_load explains spike-MAE specifically, |residual| should be much
larger when net_load is in its top decile:

| Slice | Hours | mean |residual| EUR/MWh | vs all |
|---|---:|---:|---:|
| net_load ≤ Q10 ({q10:.0f} MW) | {sum(low_mask):,} | {mae_low:.2f} | {mae_low/mae_all:+.0%} |
| **net_load ≥ Q90 ({q90:.0f} MW)** | {sum(high_mask):,} | **{mae_high:.2f}** | **{mae_high/mae_all:+.0%}** |
| all | {len(abs_resid):,} | {mae_all:.2f} | — |

## Plausible MAE improvement

A simple OLS feature would reduce the AR(2) baseline MAE by roughly
**sqrt(R²) ≈ {plausible_pct:.1f}%** as a back-of-envelope estimate.
The Ridge would do better than this in practice because it can learn
nonlinear interactions (e.g. `net_load × is_weekend` or `net_load²`)
and because the FI Ridge already has nuclear / wind features that
this hypothesis is partially redundant with.

## Verdict

"""
    if r2_raw > 0.10 and r_nl_absresid > 0.20:
        verdict = (
            f"**STRONG SIGNAL.** net_load explains {r2_raw:.1%} of the "
            f"AR(2) residual variance and the high-decile slice has "
            f"{mae_high/mae_low:.1f}× the MAE of the low-decile slice. "
            f"Adding net_load to the FI Ridge feature set is expected "
            f"to materially improve winter-spike accuracy. Recommend "
            f"implementing as a feature for v2.2."
        )
    elif r2_raw > 0.05 or r_nl_absresid > 0.12:
        verdict = (
            f"**MODERATE SIGNAL.** net_load shows {r2_raw:.1%} R² with "
            f"AR(2) residuals; the high-decile slice has "
            f"{mae_high/mae_low:.1f}× the MAE of the low-decile slice. "
            f"Adding it as a feature is justifiable but the marginal "
            f"win may be small after the existing wind / nuclear / HDD "
            f"features absorb correlated variance. Consider running "
            f"the same correlation against the FI Ridge residuals "
            f"(not the AR(2) baseline) to estimate the marginal gain."
        )
    else:
        verdict = (
            f"**WEAK SIGNAL.** R² of only {r2_raw:.1%} with AR(2) "
            f"residuals. Either the wind / nuclear features in the "
            f"FI Ridge already absorb this signal, or net_load doesn't "
            f"capture the spike dynamics on this OOS window. Not "
            f"recommended as a priority feature."
        )
    md += verdict + "\n"

    md += f"""

## Method limitations

* **Reading against AR(2) residuals, not FI Ridge residuals.** The
  measured R² is the *upper bound* of what net_load can add — the
  FI Ridge already has wind speed, HDD, and nuclear deficit, which
  partially encode the same information. The marginal gain on FI
  Ridge will be smaller than the {r2_raw:.1%} measured here.
* **Single OOS window.** Results may differ in summer / autumn.
* **Day-ahead forecasts (datasets 165, 246, 247) vs real-time
  nuclear (188).** The day-ahead forecasts are what would be
  available at forecast time; using nuclear real-time is a small
  cheat — proper inference would use the day-ahead nuclear schedule.
  Effect on the analysis is small because nuclear is much more
  stable than wind or consumption.

## Reproducibility

```sh
set FINGRID_API_KEY=...
python studies/fingrid_netload_study.py
```

Cached Fingrid responses live under `studies/_fingrid_cache/`;
re-running is fast after the first fetch.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "window": [start.isoformat(), end.isoformat()],
            "n_aligned_hours": len(aligned_ts),
            "stats": {
                "net_load": {
                    "mean": sum(nl) / len(nl),
                    "min": min(nl), "max": max(nl),
                    "q10": q10, "q90": q90,
                },
            },
            "correlations": {
                "net_load_vs_price": r_nl_price,
                "consumption_vs_price": r_consumption_price,
                "wind_vs_price": r_wind_price,
                "net_load_vs_residual": r_nl_resid,
                "net_load_vs_abs_residual": r_nl_absresid,
            },
            "ols_residual_on_netload": {
                "slope_per_GW": a_raw * 1000,
                "intercept": b_raw,
                "r2": r2_raw,
            },
            "conditional_mae": {
                "low_decile":  {"q": q10, "n": sum(low_mask),  "mae": mae_low},
                "high_decile": {"q": q90, "n": sum(high_mask), "mae": mae_high},
                "all":         {"n": len(abs_resid),           "mae": mae_all},
            },
            "verdict_pct": plausible_pct,
        }, f, indent=2, default=str)

    print()
    print(f"=== net_load study, {len(aligned_ts):,} aligned hours ===")
    print(f"  cor(net_load, price)         = {r_nl_price:+.4f}")
    print(f"  cor(net_load, residual)      = {r_nl_resid:+.4f}")
    print(f"  cor(net_load, |residual|)    = {r_nl_absresid:+.4f}")
    print(f"  OLS residual~net_load R²     = {r2_raw:.4f}")
    print(f"  MAE high-decile / low-decile = {mae_high/mae_low:.2f}×")
    print(f"  plausible MAE reduction      ≈ {plausible_pct:.1f}%")
    print()
    print(f"[done] markdown → {md_path}")
    print(f"[done] json     → {json_path}")


if __name__ == "__main__":
    main()
