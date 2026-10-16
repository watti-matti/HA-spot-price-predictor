"""Is the Aurora Line visible in Fingrid's cross-border data?

A direct data question with no price model involved. If the Aurora Line
(SE1-FI) raised transfer capacity, the capacity series should step. If
it does not, the model cannot be given a capacity signal from this
source and the realised flows are the fallback regime indicator.

WHY THE MONTHLY MAXIMUM, NOT THE MEAN
-------------------------------------
Fingrid defines the active intraday series (38, 44, 39, 45) as "free
capacity after day-ahead market". That is PHYSICAL capacity minus what
the auction already used, so its mean is confounded: a month of heavy
trade looks like a month of low capacity. The monthly MAXIMUM is the
better proxy for physical capacity — across a month there will be hours
when the day-ahead market left the link largely unused, so the maximum
approaches the physical limit from below.

So read the `max` column for capacity, and treat `mean` as a usage
statistic rather than a capacity one. For flows, the maximum is the
peak realised transfer, which is also bounded by physical capacity and
is therefore a second, independent proxy.

The day-ahead capacity series that would have answered this directly
(24/25/26/27) were discontinued after 2024-10-29 and do not cover the
period at all.

Run:  python studies/exp_ntc_step_check.py
      python studies/exp_ntc_step_check.py --date 2026-05-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "studies" / "results"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default="2025-11-01",
                    help="candidate step date (default 2025-11-01, where "
                         "corr(FI,SE1) jumps 0.343 -> 0.885)")
    ap.add_argument("--from", dest="start", default="2025-01-01",
                    help="first month to tabulate")
    a = ap.parse_args()
    step = pd.Timestamp(a.date, tz="UTC")

    g = pd.read_parquet(REPO / "data_store" / "fi_grid_data.parquet")
    cols = [c for c in g.columns if c.startswith(("ntc_", "flow_"))]
    if not cols:
        print("ERROR: no ntc_/flow_ columns in fi_grid_data.parquet.\n"
              "       Re-run `python -m src.data_store update` with "
              "FINGRID_API_KEY set, from a checkout whose "
              "config/regions/finland.yaml declares them.")
        raise SystemExit(2)

    print(f"  {len(cols)} cross-border columns: {', '.join(cols)}")
    out: dict = {"step_date": a.date, "columns": {}}

    for c in cols:
        s = g[c].dropna()
        if s.empty:
            print(f"\n  {c}: EMPTY — nothing returned by the API")
            out["columns"][c] = {"empty": True}
            continue

        print(f"\n  {c}   {len(s):,} non-null, "
              f"{s.index.min().date()} .. {s.index.max().date()}")
        m = s[s.index >= pd.Timestamp(a.start, tz="UTC")]
        if not m.empty:
            t = m.groupby(m.index.strftime("%Y-%m")).agg(
                ["count", "mean", "median", "max", "min"])
            print(f"    {'month':8s} {'n':>5s} {'mean':>9s} {'median':>9s} "
                  f"{'max':>9s} {'min':>9s}")
            for mo, r in t.iterrows():
                mark = " <-- step" if mo == a.date[:7] else ""
                print(f"    {mo:8s} {int(r['count']):5d} {r['mean']:9.1f} "
                      f"{r['median']:9.1f} {r['max']:9.1f} "
                      f"{r['min']:9.1f}{mark}")

        pre, post = s[s.index < step], s[s.index >= step]
        rec: dict = {"n": int(len(s)),
                     "start": str(s.index.min()), "end": str(s.index.max())}
        if len(pre) > 500 and len(post) > 500:
            # Capacity is a step function, so the maximum is the
            # informative statistic (see module docstring).
            rec.update({
                "pre_max": float(pre.max()), "post_max": float(post.max()),
                "pre_p95": float(np.percentile(pre, 95)),
                "post_p95": float(np.percentile(post, 95)),
                "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
            })
            d_max = rec["post_max"] - rec["pre_max"]
            d_p95 = rec["post_p95"] - rec["pre_p95"]
            rec["delta_max"] = float(d_max)
            rec["delta_p95"] = float(d_p95)
            print(f"    across {a.date}:  max {rec['pre_max']:.0f} -> "
                  f"{rec['post_max']:.0f} ({d_max:+.0f} MW)   "
                  f"p95 {rec['pre_p95']:.0f} -> {rec['post_p95']:.0f} "
                  f"({d_p95:+.0f})   mean {rec['pre_mean']:.0f} -> "
                  f"{rec['post_mean']:.0f}")
            verdict = ("STEP UP" if d_p95 > 50 else
                       "STEP DOWN" if d_p95 < -50 else "no step")
            rec["verdict_p95"] = verdict
            print(f"    verdict (p95, >50 MW): {verdict}")
        else:
            print(f"    insufficient history either side of {a.date} "
                  f"(pre={len(pre)}, post={len(post)})")
        out["columns"][c] = rec

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "exp_ntc_step_check.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
