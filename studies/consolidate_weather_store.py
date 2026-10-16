"""Move the per-site weather inputs out of the cache and into the data store.

The problem
-----------
The model's weather inputs live in two places with very different
guarantees:

  data_store/fi_weather.parquet     3 WEIGHTED AGGREGATES, tracked
  studies/.cache/wx_*.parquet       7 sites x 4 variables, GITIGNORED

The per-site series are the actual raw inputs — and they include
`cloud_cover`, which the data store does not hold in any form even
though `build_seasonal_components` fits a `cloud` seasonal component
from it. 28 series, 77,064 hourly rows each, 2018-01-01 onward: the data
is local already, just not durable.

That split has already caused a real error. `studies/.cache` is
per-checkout and incrementally appended, so two clones silently hold
different vintages of recent hours. On 2026-10-13 a branch-vs-production
comparison appeared to show the branch moving the forecast by 0.0154
EUR/MWh; it was entirely a weather-cache mismatch, and the results were
byte-identical once the caches were synced.

What this does
--------------
Consolidates the cache into `data_store/fi_weather_sites.parquet`, one
column per (variable, site), on a single hourly UTC index. No refetch —
it reads what is already on disk, so it is cheap and repeatable.

The cache stays as the incremental fetch mechanism
(`build_seasonal_components._fetch_openmeteo_var_weighted` appends to
it). Run this after any weather fetch to refresh the durable copy.

Clear-sky GHI is deliberately NOT stored: it is computed
deterministically from site coordinates and timestamps by
`solar_clear_sky.clear_sky_series`, so storing it would duplicate a pure
function of data already here.

Run:  python studies/consolidate_weather_store.py
      python studies/consolidate_weather_store.py --years 4
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "studies" / ".cache"
STORE = REPO / "data_store"
OUT = STORE / "fi_weather_sites.parquet"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=int, default=0,
                    help="trim to the last N years (0 = keep everything; "
                         "the cache reaches back to 2018)")
    a = ap.parse_args()

    files = sorted(CACHE.glob("wx_*.parquet"))
    if not files:
        print(f"ERROR: no wx_*.parquet in {CACHE.relative_to(REPO)}.\n"
              "       The cache is populated by any study that calls\n"
              "       build_seasonal_components.load_weather_extended "
              "(e.g. backtest_harness).")
        return 2

    cols: dict[str, pd.Series] = {}
    for f in files:
        # wx_<variable>_<Site Name>.parquet — the variable is lower-case
        # with underscores, the site starts at the first capital.
        m = re.match(r"wx_(.+?)_(?=[A-Z])(.+)$", f.stem)
        if not m:
            print(f"  skipping unparseable name: {f.name}")
            continue
        var, site = m.group(1), m.group(2)
        d = pd.read_parquet(f)
        s = d.iloc[:, 0] if d.shape[1] == 1 else d.squeeze()
        cols[f"{var}__{site}"] = pd.Series(s.values, index=pd.DatetimeIndex(d.index))

    df = pd.DataFrame(cols).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if a.years:
        cutoff = df.index.max() - pd.Timedelta(days=365 * a.years)
        df = df[df.index >= cutoff]

    STORE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    nvar = len({c.split("__")[0] for c in df.columns})
    nsite = len({c.split("__")[1] for c in df.columns})
    print(f"  wrote {OUT.relative_to(REPO)}")
    print(f"    {len(df):,} hourly rows  {df.index.min()} .. {df.index.max()}")
    print(f"    {len(df.columns)} columns = {nvar} variables x {nsite} sites")
    print(f"    {OUT.stat().st_size / 1e6:.1f} MB")
    worst = df.isna().mean().sort_values(ascending=False)
    bad = worst[worst > 0.001]
    if len(bad):
        print("    columns >0.1% NaN:")
        for c, v in bad.items():
            print(f"      {c:52s} {100*v:5.2f}%")
    else:
        print("    no column above 0.1% NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
