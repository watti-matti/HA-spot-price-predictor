"""validate_forecaster_performance.py — walk-forward MAE / R² report for the
spot-price day-ahead forecaster.

Purpose
-------
Confidence-build a quantitative answer to "is the forecaster actually fitting
recent FI price statistics?". We:

  1. Load 3 years of cached FI hourly spot prices (Sähkötin).
  2. Reserve the most recent ``test_days`` (default 180) as the OOS window.
  3. For every day in that window, fit the AR(2)-on-residuals forecaster on
     a rolling ``train_days`` window (default 540) of *prior* days only.
  4. Generate a 24-hour day-ahead forecast for the test day, using only the
     two most-recent actual prices as AR(2) seed (no peeking).
  5. Collect (timestamp, actual, forecast) tuples; compute headline metrics,
     hour-of-day breakdown, weekly trend, residual distribution, regime-
     adaptation diagnostics.
  6. Render a self-contained HTML report under
     ``studies/results/forecaster_performance_<stamp>.html`` with embedded
     SVG charts (no external CSS/JS, opens in any browser).

What this validates
-------------------
The AR(2)-on-residuals forecaster is the *production neighbour-zone forecaster*
(SE1, SE3, EE feed into the FI Ridge model as features) and the day-ahead
backbone the FI hourly Ridge layers nuclear / weather / wind features on top
of. AR(2) MAE is an *upper bound* on FI Ridge MAE — extra features can only
reduce error — so this report gives a conservative read of model performance.

Headline metrics produced
-------------------------
* Hourly MAE / RMSE / R² across the OOS window
* MAE by hour-of-day (24 bars)
* Rolling 14-day MAE (does the model adapt?)
* Weekly bias trace (signed mean residual)
* Per-quarter MAE comparison (regime adaptation)
* Worst-25 days residual table

Run
---
    python studies/validate_forecaster_performance.py [--test-days 180] \\
        [--train-days 540] [--zone fi]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Holiday set (small) ─────────────────────────────────────────────


def _build_holidays(years: list[int]) -> set[str]:
    out: set[str] = set()
    easter_sunday = {
        2022: (4, 17), 2023: (4, 9), 2024: (3, 31),
        2025: (4, 20), 2026: (4, 5), 2027: (3, 28),
    }
    fixed = [(1, 1), (1, 6), (5, 1), (12, 24), (12, 25), (12, 26)]
    for y in years:
        for m, d in fixed:
            out.add(f"{y:04d}-{m:02d}-{d:02d}")
        if y in easter_sunday:
            em, ed = easter_sunday[y]
            es = datetime(y, em, ed)
            for delta in (-2, 1, 39, 49):
                out.add((es + timedelta(days=delta)).strftime("%Y-%m-%d"))
    return out


def is_off(ts: datetime, holidays: set[str]) -> bool:
    return ts.weekday() >= 5 or ts.date().isoformat() in holidays


# ── AR(2)-on-residuals forecaster ──────────────────────────────────


def fit_ar2(train: list[tuple[datetime, float, bool]]
            ) -> tuple[list[float], list[float], float, float]:
    """Profile (workday + off-day) plus AR(2) coefficients on residuals."""
    wd_s = [0.0] * 24; wd_n = [0] * 24
    we_s = [0.0] * 24; we_n = [0] * 24
    for ts, p, off in train:
        h = ts.hour
        if off: we_s[h] += p; we_n[h] += 1
        else:   wd_s[h] += p; wd_n[h] += 1
    profile_wd = [wd_s[h] / wd_n[h] if wd_n[h] else 0.0 for h in range(24)]
    profile_we = [we_s[h] / we_n[h] if we_n[h] else 0.0 for h in range(24)]

    residuals = [p - (profile_we[ts.hour] if off else profile_wd[ts.hour])
                 for ts, p, off in train]
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
                 target_off, last_ts: datetime, last_price, last_off,
                 prev_ts: datetime, prev_price, prev_off) -> list[float]:
    last_h = last_ts.hour; prev_h = prev_ts.hour
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


# ── Walk-forward driver ────────────────────────────────────────────


def walk_forward(prices: list[tuple[datetime, float]],
                 holidays: set[str],
                 train_days: int = 540,
                 test_days: int = 180,
                 ) -> dict:
    """For each day in the most recent `test_days`, fit on prior
    `train_days` and forecast 24h. Return per-hour records."""
    # Group hours by date
    by_date: dict[str, list[tuple[datetime, float, bool]]] = defaultdict(list)
    for ts, p in prices:
        by_date[ts.date().isoformat()].append(
            (ts, p, is_off(ts, holidays)))
    complete = sorted(d for d, h in by_date.items() if len(h) == 24)
    print(f"[walk] {len(complete)} complete days available", flush=True)

    if len(complete) < train_days + test_days:
        raise ValueError(
            f"need ≥ {train_days + test_days} days, "
            f"have {len(complete)}")

    test_start_idx = len(complete) - test_days
    records: list[dict] = []
    print(f"[walk] testing days {complete[test_start_idx]} → "
          f"{complete[-1]}", flush=True)
    print(f"[walk] training window: rolling {train_days} days behind "
          f"each test day", flush=True)

    for d_idx in range(test_start_idx, len(complete)):
        train_slice_dates = complete[d_idx - train_days: d_idx]
        target_date = complete[d_idx]
        train: list[tuple[datetime, float, bool]] = []
        for d in train_slice_dates:
            train.extend(sorted(by_date[d], key=lambda x: x[0]))
        # Sort by ts to make sure AR(2) residuals are in order
        train.sort(key=lambda x: x[0])

        profile_wd, profile_we, phi_1, phi_2 = fit_ar2(train)

        # Last two known prices = the last two hours of the day BEFORE the
        # target. AR(2) seed.
        target_hours = sorted(by_date[target_date], key=lambda x: x[0])
        target_off = target_hours[0][2]
        prev_day_hours = sorted(by_date[complete[d_idx - 1]],
                                key=lambda x: x[0])
        last_ts, last_price, last_off = prev_day_hours[-1]
        prev_ts, prev_price, prev_off = prev_day_hours[-2]

        forecast_24 = forecast_day(
            profile_wd, profile_we, phi_1, phi_2,
            target_off, last_ts, last_price, last_off,
            prev_ts, prev_price, prev_off,
        )

        for hr_idx, (ts, actual, _) in enumerate(target_hours):
            records.append({
                "ts": ts,
                "date": target_date,
                "hour": ts.hour,
                "dow": ts.weekday(),
                "is_off": is_off(ts, holidays),
                "actual": actual,
                "forecast": forecast_24[hr_idx],
            })

        if (d_idx - test_start_idx + 1) % 30 == 0:
            print(f"[walk] day {d_idx - test_start_idx + 1}/{test_days} "
                  f"{target_date}", flush=True)

    return {
        "records": records,
        "train_days": train_days,
        "test_days": test_days,
        "test_start": complete[test_start_idx],
        "test_end": complete[-1],
    }


# ── Metrics ────────────────────────────────────────────────────────


def headline(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {}
    err = [r["actual"] - r["forecast"] for r in records]
    abs_err = [abs(e) for e in err]
    sq_err = [e * e for e in err]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum(sq_err) / n)
    bias = sum(err) / n
    mu = sum(r["actual"] for r in records) / n
    sst = sum((r["actual"] - mu) ** 2 for r in records)
    ssr = sum(sq_err)
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    return {
        "n_hours": n,
        "MAE": mae,
        "RMSE": rmse,
        "bias": bias,
        "R2": r2,
        "mean_actual": mu,
        "std_actual": math.sqrt(sst / n),
        "min_actual": min(r["actual"] for r in records),
        "max_actual": max(r["actual"] for r in records),
    }


def hour_of_day_mae(records: list[dict]) -> list[float]:
    sums = [0.0] * 24; counts = [0] * 24
    for r in records:
        sums[r["hour"]] += abs(r["actual"] - r["forecast"])
        counts[r["hour"]] += 1
    return [sums[h] / counts[h] if counts[h] else 0.0 for h in range(24)]


def rolling_daily_mae(records: list[dict], window_days: int = 14) -> list[dict]:
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_day[r["date"]].append(abs(r["actual"] - r["forecast"]))
    days = sorted(by_day.keys())
    daily = [(d, sum(by_day[d]) / len(by_day[d])) for d in days]
    out = []
    for i in range(len(daily)):
        lo = max(0, i - window_days + 1)
        sub = [v for _, v in daily[lo: i + 1]]
        out.append({"date": daily[i][0],
                    "daily_mae": daily[i][1],
                    "rolling_mae": sum(sub) / len(sub)})
    return out


def weekly_bias(records: list[dict]) -> list[dict]:
    by_week: dict[str, list[float]] = defaultdict(list)
    for r in records:
        ts = r["ts"]
        # ISO week label
        iso_year, iso_week, _ = ts.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        by_week[key].append(r["actual"] - r["forecast"])
    out = []
    for k in sorted(by_week.keys()):
        vals = by_week[k]
        m = sum(vals) / len(vals)
        out.append({"week": k, "bias": m, "n": len(vals)})
    return out


def quarterly_mae(records: list[dict]) -> list[dict]:
    """Split OOS into 4 equal-time quarters; report MAE per quarter."""
    if not records:
        return []
    records = sorted(records, key=lambda r: r["ts"])
    n = len(records)
    step = n // 4
    out = []
    for i in range(4):
        lo = i * step
        hi = (i + 1) * step if i < 3 else n
        sub = records[lo:hi]
        if not sub: continue
        err = [abs(r["actual"] - r["forecast"]) for r in sub]
        out.append({
            "quarter": f"Q{i+1}",
            "from": sub[0]["ts"].date().isoformat(),
            "to": sub[-1]["ts"].date().isoformat(),
            "MAE": sum(err) / len(err),
            "mean_actual": sum(r["actual"] for r in sub) / len(sub),
            "n_hours": len(sub),
        })
    return out


def worst_days(records: list[dict], k: int = 25) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_day[r["date"]].append(r)
    daily = []
    for d, rs in by_day.items():
        err = [abs(r["actual"] - r["forecast"]) for r in rs]
        actuals = [r["actual"] for r in rs]
        forecasts = [r["forecast"] for r in rs]
        daily.append({
            "date": d,
            "MAE": sum(err) / len(err),
            "actual_mean": sum(actuals) / 24,
            "forecast_mean": sum(forecasts) / 24,
            "actual_max": max(actuals),
            "actual_min": min(actuals),
        })
    daily.sort(key=lambda x: -x["MAE"])
    return daily[:k]


# ── SVG chart generators (no external libs) ────────────────────────


def _svg_open(w: int, h: int, title: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" '
            f'height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{title}" font-family="-apple-system,'
            'Segoe UI,Helvetica,Arial,sans-serif" font-size="11">')


def line_chart_svg(daily: list[dict], width: int = 920, height: int = 280,
                   title: str = "Rolling 14-day MAE") -> str:
    """Two-series line chart: daily MAE + rolling MAE."""
    if not daily:
        return ""
    pad_l, pad_r, pad_t, pad_b = 50, 12, 28, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    ymax = max(max(d["daily_mae"] for d in daily),
               max(d["rolling_mae"] for d in daily)) * 1.1
    ymin = 0
    n = len(daily)
    def x(i): return pad_l + (i / max(1, n - 1)) * plot_w
    def y(v): return pad_t + plot_h * (1 - (v - ymin) / max(1e-9, ymax - ymin))

    # Axis ticks (5 horizontal lines)
    grid = []
    for i in range(6):
        v = ymax * i / 5
        gy = y(v)
        grid.append(
            f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l + plot_w}" '
            f'y2="{gy}" stroke="#e5e5e5" stroke-width="1"/>')
        grid.append(
            f'<text x="{pad_l - 6}" y="{gy + 3}" text-anchor="end" '
            f'fill="#666">{v:.0f}</text>')
    # X ticks (5 evenly spaced dates)
    xticks = []
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        if i < 0 or i >= n: continue
        xticks.append(
            f'<text x="{x(i)}" y="{height - pad_b + 14}" '
            f'text-anchor="middle" fill="#666">{daily[i]["date"][5:]}</text>')

    # Daily MAE: thin grey
    pts_daily = " ".join(f"{x(i)},{y(d['daily_mae'])}"
                         for i, d in enumerate(daily))
    # Rolling MAE: thicker red-orange
    pts_roll = " ".join(f"{x(i)},{y(d['rolling_mae'])}"
                        for i, d in enumerate(daily))

    legend = (
        f'<g transform="translate({pad_l + 12},{pad_t + 4})">'
        f'<line x1="0" y1="6" x2="20" y2="6" stroke="#bbb" stroke-width="1.5"/>'
        f'<text x="26" y="9" fill="#444">daily MAE</text>'
        f'<line x1="100" y1="6" x2="120" y2="6" stroke="#d97706" stroke-width="2.5"/>'
        f'<text x="126" y="9" fill="#444">14-day rolling</text>'
        f'</g>'
    )

    return (
        _svg_open(width, height, title)
        + f'<text x="{width // 2}" y="16" text-anchor="middle" '
          f'fill="#222" font-weight="600">{title}</text>'
        + "".join(grid)
        + f'<polyline fill="none" stroke="#bbb" stroke-width="1" '
          f'points="{pts_daily}"/>'
        + f'<polyline fill="none" stroke="#d97706" stroke-width="2.5" '
          f'points="{pts_roll}"/>'
        + legend
        + "".join(xticks)
        + "</svg>"
    )


def bar_chart_svg(values: list[float], labels: list[str],
                  title: str, ylabel: str = "EUR/MWh",
                  width: int = 920, height: int = 240,
                  highlight_indices: list[int] = None,
                  signed: bool = False) -> str:
    if not values:
        return ""
    pad_l, pad_r, pad_t, pad_b = 50, 12, 28, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(values)
    if signed:
        ymax = max(abs(min(values)), abs(max(values))) * 1.1 or 1
        ymin = -ymax
    else:
        ymax = max(values) * 1.15 or 1
        ymin = 0
    span = ymax - ymin
    bar_w = plot_w / n * 0.8
    gap = plot_w / n * 0.2
    def y(v): return pad_t + plot_h * (1 - (v - ymin) / span)
    zero_y = y(0) if signed else (pad_t + plot_h)

    grid = []
    for i in range(5):
        v = ymin + span * i / 4
        gy = y(v)
        grid.append(
            f'<line x1="{pad_l}" y1="{gy}" x2="{pad_l + plot_w}" y2="{gy}" '
            f'stroke="#e5e5e5" stroke-width="1"/>')
        grid.append(
            f'<text x="{pad_l - 6}" y="{gy + 3}" text-anchor="end" '
            f'fill="#666">{v:+.0f}</text>')

    bars = []
    for i, v in enumerate(values):
        bx = pad_l + i * (bar_w + gap) + gap / 2
        if signed:
            by = y(max(v, 0))
            bh = abs(y(v) - zero_y)
            color = "#dc2626" if v > 0 else "#2563eb"
        else:
            by = y(v); bh = (pad_t + plot_h) - by
            color = "#0ea5e9"
        if highlight_indices and i in highlight_indices:
            color = "#ea580c"
        bars.append(
            f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" '
            f'fill="{color}" rx="2"/>')
    label_skip = max(1, n // 20)
    text_labels = [
        f'<text x="{pad_l + i * (bar_w + gap) + gap / 2 + bar_w / 2}" '
        f'y="{height - pad_b + 14}" text-anchor="middle" fill="#666" '
        f'font-size="10">{labels[i]}</text>'
        for i in range(n) if i % label_skip == 0
    ]
    return (
        _svg_open(width, height, title)
        + f'<text x="{width // 2}" y="16" text-anchor="middle" '
          f'fill="#222" font-weight="600">{title}</text>'
        + "".join(grid)
        + "".join(bars)
        + "".join(text_labels)
        + f'<text x="14" y="{pad_t + plot_h / 2}" '
          f'transform="rotate(-90,14,{pad_t + plot_h / 2})" '
          f'text-anchor="middle" fill="#666">{ylabel}</text>'
        + "</svg>"
    )


def histogram_svg(values: list[float], title: str = "Residual histogram",
                  bins: int = 50, width: int = 460, height: int = 220) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        return ""
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
        counts[idx] += 1
    return bar_chart_svg(
        counts,
        [f"{edges[i]:.0f}" for i in range(bins)],
        title=title, ylabel="count",
        width=width, height=height,
    )


def scatter_svg(records: list[dict], width: int = 460, height: int = 460,
                title: str = "Forecast vs actual (hourly)") -> str:
    if not records:
        return ""
    pad = 40
    pw = width - 2 * pad; ph = height - 2 * pad
    actuals = [r["actual"] for r in records]
    forecasts = [r["forecast"] for r in records]
    lo = min(min(actuals), min(forecasts))
    hi = max(max(actuals), max(forecasts))
    pad_v = (hi - lo) * 0.05
    lo -= pad_v; hi += pad_v
    span = hi - lo or 1
    def x(v): return pad + (v - lo) / span * pw
    def y(v): return pad + ph * (1 - (v - lo) / span)
    points = "".join(
        f'<circle cx="{x(r["actual"])}" cy="{y(r["forecast"])}" r="1.4" '
        f'fill="#2563eb" fill-opacity="0.25"/>'
        for r in records
    )
    diag = (
        f'<line x1="{x(lo)}" y1="{y(lo)}" x2="{x(hi)}" y2="{y(hi)}" '
        f'stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3"/>'
    )
    ticks = []
    for i in range(5):
        v = lo + span * i / 4
        gx = x(v); gy = y(v)
        ticks.append(f'<text x="{gx}" y="{height - 18}" '
                     f'text-anchor="middle" fill="#666">{v:.0f}</text>')
        ticks.append(f'<text x="{pad - 6}" y="{gy + 3}" '
                     f'text-anchor="end" fill="#666">{v:.0f}</text>')

    return (
        _svg_open(width, height, title)
        + f'<text x="{width // 2}" y="14" text-anchor="middle" '
          f'fill="#222" font-weight="600">{title}</text>'
        + f'<rect x="{pad}" y="{pad}" width="{pw}" height="{ph}" '
          f'fill="none" stroke="#ddd"/>'
        + diag + points + "".join(ticks)
        + f'<text x="{width // 2}" y="{height - 4}" text-anchor="middle" '
          f'fill="#444">actual EUR/MWh</text>'
        + f'<text x="14" y="{height // 2}" '
          f'transform="rotate(-90,14,{height // 2})" '
          f'text-anchor="middle" fill="#444">forecast EUR/MWh</text>'
        + "</svg>"
    )


# ── HTML report ────────────────────────────────────────────────────


def render_html(zone: str, ctx: dict, hl: dict, hod: list[float],
                rolling: list[dict], wbias: list[dict],
                quarters: list[dict], worst: list[dict],
                records: list[dict]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
         max-width:1000px;margin:24px auto;padding:0 16px;color:#222}
    h1{font-size:20px;font-weight:600;margin:0 0 4px}
    h2{font-size:15px;font-weight:600;margin:24px 0 8px;color:#333}
    .sub{color:#666;font-size:13px;margin-bottom:18px}
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
              gap:10px;margin:14px 0}
    .kpi{background:#f8f8f6;border:1px solid #eee;border-radius:8px;padding:10px 12px}
    .kpi .v{font-size:20px;font-weight:600;color:#0f172a;line-height:1.1}
    .kpi .l{font-size:11px;color:#666;text-transform:uppercase;
            letter-spacing:0.04em;margin-top:2px}
    .chart{background:#fff;border:1px solid #eee;border-radius:8px;
           padding:6px;margin:8px 0}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
    th{text-align:left;padding:6px 8px;border-bottom:1px solid #ddd;
       font-weight:600;color:#444}
    td{padding:5px 8px;border-bottom:1px solid #f0f0f0}
    tr:hover td{background:#fafafa}
    .num{text-align:right;font-variant-numeric:tabular-nums}
    .bad{color:#dc2626}
    .good{color:#16a34a}
    .footer{color:#888;font-size:11px;margin-top:32px;
            padding-top:12px;border-top:1px solid #eee}
    code{background:#f0f0ee;padding:1px 4px;border-radius:3px;font-size:11px}
    """
    def fmt(v, d=2):
        if v is None or (isinstance(v, float) and (v != v)): return "—"
        return f"{v:.{d}f}"

    # KPI grid
    kpis = [
        ("Hours evaluated", f"{hl['n_hours']:,}", "OOS window"),
        ("MAE", f"{hl['MAE']:.2f}", "EUR/MWh"),
        ("RMSE", f"{hl['RMSE']:.2f}", "EUR/MWh"),
        ("Bias", f"{hl['bias']:+.2f}", "EUR/MWh"),
        ("R²", f"{hl['R2']:.3f}", "vs daily mean"),
        ("Mean actual", f"{hl['mean_actual']:.1f}", "EUR/MWh"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div>'
        f'<div class="l">{lab}<br><span style="color:#aaa">{sub}</span></div></div>'
        for lab, v, sub in kpis
    )

    # Hour-of-day bars
    hod_svg = bar_chart_svg(
        hod, [f"{h:02d}" for h in range(24)],
        title="MAE by hour-of-day", ylabel="EUR/MWh",
        width=920, height=220,
        highlight_indices=[i for i, v in enumerate(hod)
                           if v >= max(hod) * 0.92],
    )
    # Rolling MAE
    roll_svg = line_chart_svg(rolling, title="Rolling 14-day MAE")
    # Weekly bias
    wbias_vals = [w["bias"] for w in wbias]
    wbias_lbl = [w["week"][2:] for w in wbias]
    wb_svg = bar_chart_svg(
        wbias_vals, wbias_lbl,
        title="Weekly mean signed residual (positive = under-forecast)",
        ylabel="EUR/MWh", width=920, height=220, signed=True,
    )
    # Residual histogram
    resids = [r["actual"] - r["forecast"] for r in records]
    hist_svg = histogram_svg(resids, title="Residual distribution",
                             width=460, height=220)
    # Scatter (downsample to 4000 points to keep SVG small)
    scatter_recs = records if len(records) <= 4000 else records[::len(records) // 4000]
    sct_svg = scatter_svg(scatter_recs, width=460, height=440,
                          title="Forecast vs actual (hourly)")

    # Quarters table
    q_rows = "".join(
        f'<tr><td>{q["quarter"]}</td><td>{q["from"]} → {q["to"]}</td>'
        f'<td class="num">{q["n_hours"]:,}</td>'
        f'<td class="num">{q["mean_actual"]:.1f}</td>'
        f'<td class="num">{q["MAE"]:.2f}</td></tr>'
        for q in quarters
    )
    # Worst days table
    w_rows = "".join(
        f'<tr><td>{d["date"]}</td>'
        f'<td class="num bad">{d["MAE"]:.2f}</td>'
        f'<td class="num">{d["actual_mean"]:.1f}</td>'
        f'<td class="num">{d["forecast_mean"]:.1f}</td>'
        f'<td class="num">{d["actual_max"]:.1f}</td></tr>'
        for d in worst[:15]
    )

    # Adaptation interpretation
    early_mae = quarters[0]["MAE"] if quarters else 0
    late_mae = quarters[-1]["MAE"] if quarters else 0
    delta = late_mae - early_mae
    delta_pct = (delta / early_mae * 100) if early_mae else 0
    if abs(delta_pct) < 5:
        adapt_msg = (
            f'<span class="good">Stable</span>: MAE moved from '
            f'{early_mae:.2f} → {late_mae:.2f} ({delta_pct:+.1f}%) '
            f'across the OOS window. The forecaster is tracking the '
            f'price regime as it evolves.'
        )
    elif delta_pct > 0:
        adapt_msg = (
            f'<span class="bad">MAE growing</span>: {early_mae:.2f} → '
            f'{late_mae:.2f} ({delta_pct:+.1f}%). Recent regime is '
            f'harder than what the model was seeing earlier in the '
            f'OOS window. Consider shorter training window or '
            f'enabling DtACI bias correction.'
        )
    else:
        adapt_msg = (
            f'<span class="good">MAE improving</span>: {early_mae:.2f} '
            f'→ {late_mae:.2f} ({delta_pct:+.1f}%). The forecaster is '
            f'adapting well as the OOS window progresses.'
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Forecaster performance — {zone.upper()}</title>
<style>{css}</style></head><body>
<h1>Spot-price forecaster — walk-forward performance ({zone.upper()})</h1>
<div class="sub">
  Generated {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
  · OOS window <b>{ctx['test_start']} → {ctx['test_end']}</b>
  ({ctx['test_days']} days)
  · Rolling fit on prior <b>{ctx['train_days']} days</b>
  · Forecaster: AR(2)-on-residuals (production neighbour-zone model;
    upper bound on FI Ridge MAE since FI Ridge adds wind/nuclear/HDD
    features that can only reduce error)
</div>

<h2>Headline metrics</h2>
<div class="kpi-grid">{kpi_html}</div>

<h2>Adaptation across the OOS window</h2>
<div class="sub">{adapt_msg}</div>
<table>
<tr><th>Quarter</th><th>Range</th><th class="num">Hours</th>
<th class="num">Mean actual EUR/MWh</th><th class="num">MAE EUR/MWh</th></tr>
{q_rows}
</table>

<h2>MAE by hour-of-day</h2>
<div class="chart">{hod_svg}</div>

<h2>Rolling 14-day MAE</h2>
<div class="chart">{roll_svg}</div>

<h2>Weekly bias (signed residual)</h2>
<div class="chart">{wb_svg}</div>

<h2>Calibration scatter + residual distribution</h2>
<div class="grid-2">
  <div class="chart">{sct_svg}</div>
  <div class="chart">{hist_svg}</div>
</div>

<h2>Worst-15 forecast days</h2>
<table>
<tr><th>Date</th><th class="num">MAE</th>
<th class="num">Actual mean</th><th class="num">Forecast mean</th>
<th class="num">Actual peak</th></tr>
{w_rows}
</table>

<div class="footer">
  Methodology: walk-forward day-ahead forecasts. For each test day, the
  forecaster is refit from scratch on the immediately-preceding
  {ctx['train_days']} days, then asked to forecast all 24 hours of the
  test day using only the last two known prices as AR(2) seed. No future
  information leaks. The headline R² is hourly; in EPF literature this
  metric typically lands in 0.3–0.6 range and varies strongly with how
  spike-heavy the OOS window is. Conservative read: this is the lower
  performance bound of the production FI forecaster, since the production
  Ridge layer adds wind, nuclear, HDD, and cross-border features that can
  only further reduce error. See <code>studies/results/VERSION_COMPARISON.md</code>
  for a comparison of bundled-model metrics across versions.
</div>

</body></html>"""


# ── Entry point ────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="fi", choices=["fi", "se1", "se3", "ee"])
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--test-days", type=int, default=180)
    args = ap.parse_args()

    cache_path = REPO_ROOT / "studies" / f"_dtaci_dk_{args.zone}_prices_cache.json"
    if not cache_path.exists():
        # Fall back to legacy fi cache
        cache_path = REPO_ROOT / "studies" / f"_dtaci_{args.zone}_prices_cache.json"
    print(f"[cache] {cache_path}")
    with open(cache_path) as f:
        raw = json.load(f)
    prices = [(datetime.fromisoformat(t), float(p)) for t, p in raw]
    years = sorted({p[0].year for p in prices})
    holidays = _build_holidays(list(range(min(years), max(years) + 2)))

    ctx = walk_forward(
        prices, holidays,
        train_days=args.train_days,
        test_days=args.test_days,
    )
    records = ctx["records"]
    hl = headline(records)
    hod = hour_of_day_mae(records)
    roll = rolling_daily_mae(records, window_days=14)
    wbias = weekly_bias(records)
    quarters = quarterly_mae(records)
    worst = worst_days(records, k=25)

    print()
    print(f"=== {args.zone.upper()} headline ===")
    print(f"  hours: {hl['n_hours']:,}")
    print(f"  MAE:   {hl['MAE']:.2f} EUR/MWh")
    print(f"  RMSE:  {hl['RMSE']:.2f} EUR/MWh")
    print(f"  bias:  {hl['bias']:+.2f} EUR/MWh")
    print(f"  R^2:   {hl['R2']:.3f}")
    print()
    print("=== quarters ===")
    for q in quarters:
        print(f"  {q['quarter']} {q['from']} → {q['to']:>10}: "
              f"MAE {q['MAE']:.2f}  mean_actual {q['mean_actual']:.1f}")

    out_dir = REPO_ROOT / "studies" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    html_path = out_dir / f"forecaster_performance_{args.zone}_{stamp}.html"
    json_path = out_dir / f"forecaster_performance_{args.zone}_{stamp}.json"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(args.zone, ctx, hl, hod, roll, wbias,
                            quarters, worst, records))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "zone": args.zone, "ctx": {k: v for k, v in ctx.items()
                                        if k != "records"},
            "headline": hl, "hour_of_day_mae": hod,
            "weekly_bias": wbias, "quarters": quarters,
            "worst_days": worst[:25],
        }, f, indent=2, default=str)

    print()
    print(f"[done] HTML → {html_path}")
    print(f"[done] JSON → {json_path}")


if __name__ == "__main__":
    main()
