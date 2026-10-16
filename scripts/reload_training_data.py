"""Reload every input and rebuild the modelling dataset, end to end.

One command, four steps, each verified before the next runs:

  1. data store update      prices, neighbours, weather, 18 Fingrid series
  2. weather consolidation  rescue the per-site series out of the cache
  3. modelling dataset      one aligned hourly table + metadata
  4. verification           coverage, contiguity, causality tags

Why a script rather than four commands
--------------------------------------
Each step has a prerequisite the next one silently tolerates, and the
failures are quiet rather than loud:

* `src.data_store`'s `_missing_ranges` compares TIME COVERAGE only, so a
  newly configured Fingrid series is never fetched if the window is
  already covered — the run reports success with an unchanged hash. The
  column check added in v2.18.2 fixes that, and this script asserts the
  expected column count so a regression is caught here too.
* The per-site weather lives in `studies/.cache/wx_*.parquet`, which is
  gitignored and per-checkout. It is the ONLY copy of 28 series going
  back to 2018. A cleared cache loses them silently, and two checkouts
  holding different vintages already produced one false-positive
  model comparison (2026-10-13).
* `build_dataframe` inner-joins then `dropna()`s, so one gappy column
  shortens the frame for every study. 207 hours were being lost this way.

Usage
-----
    python scripts/reload_training_data.py                 # full reload
    python scripts/reload_training_data.py --skip-fetch     # rebuild only
    python scripts/reload_training_data.py --years 4

Needs `FINGRID_API_KEY` in the environment or in `.env` at the repo root
(gitignored). Without it the Fingrid series are skipped and the run
fails at verification rather than producing a quietly incomplete store.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / "data_store"

EXPECTED_GRID_COLUMNS = 18
EXPECTED_SITE_COLUMNS = 28
MIN_FEATURE_COVERAGE = 0.99

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _key() -> str:
    """Environment first, then .env. Read with utf-8-sig: PowerShell's
    `Set-Content -Encoding utf8` writes a BOM, which defeats a regex
    anchored at the start of the line."""
    k = os.environ.get("FINGRID_API_KEY", "").strip()
    if k:
        return k
    f = REPO / ".env"
    if f.exists():
        m = re.search(r"FINGRID_API_KEY\s*=\s*(\S+)",
                      f.read_text(encoding="utf-8-sig", errors="replace"))
        if m:
            return m.group(1).strip()
    return ""


def _run(label: str, args: list[str], env: dict | None = None) -> bool:
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=REPO,
                       env={**os.environ, **(env or {})})
    if r.returncode != 0:
        print(f"  FAILED ({label}), exit {r.returncode}")
        return False
    return True


def verify(years: float) -> int:
    import pandas as pd

    print("\n=== 4. verification ===")
    problems: list[str] = []

    grid = STORE / "fi_grid_data.parquet"
    if not grid.exists():
        print("  fi_grid_data.parquet missing")
        return 1
    g = pd.read_parquet(grid)
    print(f"  grid           {len(g.columns):2d} columns "
          f"(expected {EXPECTED_GRID_COLUMNS})")
    if len(g.columns) < EXPECTED_GRID_COLUMNS:
        problems.append(
            f"grid has {len(g.columns)} of {EXPECTED_GRID_COLUMNS} columns — "
            "a configured Fingrid series was not fetched. Check "
            "FINGRID_API_KEY, and that src/data_store.py still compares "
            "configured columns against the store (not just time coverage).")

    sites = STORE / "fi_weather_sites.parquet"
    if sites.exists():
        w = pd.read_parquet(sites)
        print(f"  weather sites  {len(w.columns):2d} columns, "
              f"{len(w):,} rows from {w.index.min().date()}")
        if len(w.columns) < EXPECTED_SITE_COLUMNS:
            problems.append(
                f"weather sites has {len(w.columns)} of "
                f"{EXPECTED_SITE_COLUMNS} columns — the wx_*.parquet cache "
                "may have been cleared; it is the only copy.")
    else:
        problems.append("fi_weather_sites.parquet missing — run "
                        "studies/consolidate_weather_store.py")

    meta_f = STORE / "modelling_dataset_4y.meta.json"
    ds_f = STORE / "modelling_dataset_4y.parquet"
    if not (meta_f.exists() and ds_f.exists()):
        problems.append("modelling dataset missing")
    else:
        m = json.loads(meta_f.read_text(encoding="utf-8"))
        d = pd.read_parquet(ds_f)
        feats = m["feature_columns"]
        tgt = m["target_column"]
        ok = int(d[feats + [tgt]].notna().all(axis=1).sum())
        frac = ok / len(d)
        print(f"  dataset        {len(d):,} rows x {len(d.columns)} columns, "
              f"{ds_f.stat().st_size / 1e6:.1f} MB")
        print(f"  causality      {m['causality_counts']}")
        print(f"  modelling set  {len(feats)} features + target, "
              f"{ok:,} complete rows ({100 * frac:.2f}%)")

        step = pd.Series(d.index).diff().dropna()
        gaps = int((step != pd.Timedelta("1h")).sum())
        print(f"  index          hourly, {gaps} irregular steps, "
              f"{int(d.index.duplicated().sum())} duplicates")
        if gaps:
            problems.append(f"{gaps} irregular steps in the hourly index")
        if d.index.duplicated().any():
            problems.append("duplicate timestamps in the index")
        if frac < MIN_FEATURE_COVERAGE:
            problems.append(
                f"only {100 * frac:.2f}% of rows are complete on the "
                f"feature set (want >= {100 * MIN_FEATURE_COVERAGE:.0f}%)")

        ended = [c for c, v in m["columns_meta"].items()
                 if v["causality"] == "ex-ante-ended"]
        if ended:
            print(f"  discontinued    {len(ended)} series excluded from "
                  f"features: {', '.join(ended)}")

        span_years = (d.index[-1] - d.index[0]).days / 365.25
        print(f"  span           {span_years:.2f} years "
              f"({d.index[0].date()} .. {d.index[-1].date()})")
        if span_years < years - 0.05:
            problems.append(
                f"span {span_years:.2f} y is short of the requested {years} y")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  All checks passed. Dataset is ready for modelling:")
    print(f"    {ds_f.relative_to(REPO)}")
    print(f"    feature list and causality tags in "
          f"{meta_f.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=float, default=4.0)
    ap.add_argument("--region", default="finland")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="rebuild from the existing store without fetching")
    a = ap.parse_args()

    if not a.skip_fetch:
        key = _key()
        if not key:
            print("ERROR: FINGRID_API_KEY not found in the environment or "
                  ".env.\n"
                  "       Without it the 18 Fingrid series are skipped and "
                  "the dataset\n"
                  "       would be quietly incomplete. Set it, or pass "
                  "--skip-fetch to\n"
                  "       rebuild from what is already stored.")
            return 2
        print(f"  Fingrid key found (len {len(key)}), not printed")
        if not _run("1. data store update",
                    ["-m", "src.data_store", "update",
                     "--years", str(int(a.years)), "--region", a.region],
                    env={"FINGRID_API_KEY": key,
                         "PYTHONPATH": str(REPO)}):
            return 1
    else:
        print("  skipping fetch (--skip-fetch)")

    if not _run("2. consolidate per-site weather",
                ["studies/consolidate_weather_store.py"]):
        return 1
    if not _run("3. build modelling dataset",
                ["studies/build_modelling_dataset.py",
                 "--years", str(a.years)]):
        return 1
    return verify(a.years)


if __name__ == "__main__":
    raise SystemExit(main())
