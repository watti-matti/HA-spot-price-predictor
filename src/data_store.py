"""Canonical incremental training-data store.

One place all training data lives, fetched from the original sources,
grown by APPENDING new data (historical rows are immutable and never
re-fetched), and tracked by a manifest that records per-source coverage
and a content hash. The manifest's ``snapshot_id`` is the fingerprint a
trained model records in its provenance, so any model can be linked back
to the exact data it was trained on.

Layout (under ``data_store/``):
    fi_prices.parquet          spot price (Sahkotin)
    fi_weather.parquet         capacity-weighted wind/solar/temp (Open-Meteo)
    fi_neighbor_prices.parquet SE1/SE3/EE (elpriset / Elering)
    fi_grid_data.parquet       Fingrid: nuclear, consumption, wind, solar,
                               solar_capacity (dataset 267)
    manifest.json              per-source coverage + hashes + snapshot_id
    history.jsonl              append-only log of every update

CLI:
    python -m src.data_store update   [--years N] [--region R]
    python -m src.data_store status
    python -m src.data_store validate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.data_sources import (
    fetch_prices, fetch_weather, fetch_neighbor_prices, fetch_grid_data,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data_store"
MANIFEST = STORE / "manifest.json"
HISTORY = STORE / "history.jsonl"
LEGACY_SEED = REPO / "output"     # seed the store from prior parquets on first run

# name -> (parquet filename, original source description)
SOURCES: dict[str, tuple[str, str]] = {
    "prices":   ("fi_prices.parquet",          "Sahkotin FI spot price"),
    "weather":  ("fi_weather.parquet",         "Open-Meteo historical-forecast (7 FI sites, weighted)"),
    "neighbor": ("fi_neighbor_prices.parquet", "elprisetjustnu SE1/SE3 + Elering EE"),
    "grid":     ("fi_grid_data.parquet",       "Fingrid 188/165/246/247/267"),
}


# ── helpers ─────────────────────────────────────────────────────────

def _path(name: str) -> Path:
    return STORE / SOURCES[name][0]


def _content_hash(df: pd.DataFrame) -> str:
    try:
        h = pd.util.hash_pandas_object(df.sort_index(), index=True).values
        return hashlib.sha256(h.tobytes()).hexdigest()[:16]
    except Exception:
        return "?"


def _load(name: str) -> pd.DataFrame | None:
    p = _path(name)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _as_frame(obj) -> pd.DataFrame | None:
    """Normalise a fetcher return (Series / DataFrame / dict-of-Series)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        obj = pd.DataFrame(obj)
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name or "value")
    return obj if len(obj) else None


def _coverage(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0, "start": None, "end": None}
    idx = df.index
    return {"rows": int(len(df)),
            "start": str(idx.min()), "end": str(idx.max())}


def _missing_ranges(df: pd.DataFrame | None, start: datetime, end: datetime):
    """Date ranges (inclusive, as datetimes) not covered by ``df``."""
    if df is None or df.empty:
        return [(start, end)]
    c0, c1 = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
    gaps = []
    if start < c0:
        gaps.append((start, c0 - timedelta(hours=1)))
    if end > c1:
        gaps.append((c1 + timedelta(hours=1), end))
    return gaps


def _fetch(name: str, config: dict, s: datetime, e: datetime) -> pd.DataFrame | None:
    if name == "prices":
        return _as_frame(fetch_prices(config, s, e))
    if name == "weather":
        return _as_frame(fetch_weather(
            config, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"),
            cache_dir=STORE / ".weather_cache"))
    if name == "neighbor":
        return _as_frame(fetch_neighbor_prices(config, s, e))
    if name == "grid":
        return _as_frame(fetch_grid_data(config, s, e))
    return None


def _merge(existing: pd.DataFrame | None, parts: list[pd.DataFrame]) -> pd.DataFrame:
    frames = ([existing] if existing is not None else []) + parts
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


# ── public API ──────────────────────────────────────────────────────

def update(config: dict, years: int = 4, seed: bool = True) -> dict:
    """Incrementally bring every source up to `now`, appending only new rows."""
    STORE.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * years)

    manifest: dict[str, Any] = {"sources": {}}
    for name, (fname, desc) in SOURCES.items():
        existing = _load(name)
        # First run: seed from the legacy output/ parquet so we don't
        # re-fetch years of history that was already downloaded.
        if existing is None and seed and (LEGACY_SEED / fname).exists():
            try:
                existing = pd.read_parquet(LEGACY_SEED / fname)
                logger.info("  %s: seeded %d rows from output/%s",
                            name, len(existing), fname)
            except Exception:
                existing = None

        ranges = _missing_ranges(existing, start, end)
        parts: list[pd.DataFrame] = []
        for gs, ge in ranges:
            if gs >= ge:
                continue
            logger.info("  %s: fetching %s .. %s", name, gs.date(), ge.date())
            got = _fetch(name, config, gs, ge)
            if got is not None:
                parts.append(got)

        if existing is None and not parts:
            logger.warning("  %s: no data available, skipped", name)
            continue
        merged = _merge(existing, parts)
        # Keep only the requested window (drop rows older than `start`).
        merged = merged[merged.index >= pd.Timestamp(start)]
        merged.to_parquet(_path(name))
        manifest["sources"][name] = {
            "file": fname, "source": desc,
            **_coverage(merged),
            "content_sha256": _content_hash(merged),
            "columns": [str(c) for c in merged.columns],
        }
        logger.info("  %s: %d rows [%s .. %s] hash=%s",
                    name, len(merged), merged.index.min(), merged.index.max(),
                    manifest["sources"][name]["content_sha256"])

    # snapshot_id = hash of all per-source content hashes → the data version
    # a trained model records to prove its lineage.
    joined = "|".join(f"{n}:{m['content_sha256']}"
                      for n, m in sorted(manifest["sources"].items()))
    manifest["snapshot_id"] = hashlib.sha256(joined.encode()).hexdigest()[:16]
    manifest["updated_at"] = end.isoformat()
    manifest["window"] = {"start": start.isoformat(), "end": end.isoformat(),
                          "years": years}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "updated_at": manifest["updated_at"],
            "snapshot_id": manifest["snapshot_id"],
            "coverage": {n: {"rows": m["rows"], "end": m["end"]}
                         for n, m in manifest["sources"].items()},
        }) + "\n")
    return manifest


def load_manifest() -> dict | None:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return None


def validate() -> list[str]:
    """Return a list of problems (empty = clean). Checks per source:
    presence, hourly-gap coverage, NaN fraction, and recency."""
    problems: list[str] = []
    now = datetime.now(timezone.utc)
    for name in SOURCES:
        df = _load(name)
        if df is None or df.empty:
            problems.append(f"{name}: MISSING")
            continue
        end = df.index.max().to_pydatetime()
        stale_days = (now - end).days
        if stale_days > 3:
            problems.append(f"{name}: stale — newest row is {stale_days}d old ({end.date()})")
        # gap check on an hourly grid (skip 15-min neighbour resolution)
        nan_frac = float(df.isna().mean().mean())
        if nan_frac > 0.02:
            problems.append(f"{name}: {nan_frac:.1%} NaN cells")
    return problems


def _print_status() -> None:
    m = load_manifest()
    if not m:
        print("No manifest — run `python -m src.data_store update` first.")
        return
    print(f"snapshot_id: {m['snapshot_id']}   updated: {m['updated_at']}")
    print(f"{'source':10} {'rows':>7} {'start':>12} {'end':>12} {'hash':>17}  columns")
    for n, s in m["sources"].items():
        st = (s['start'] or '')[:10]; en = (s['end'] or '')[:10]
        print(f"{n:10} {s['rows']:>7} {st:>12} {en:>12} {s['content_sha256']:>17}  "
              f"{','.join(s['columns'])}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Training-data store")
    ap.add_argument("cmd", choices=["update", "status", "validate"])
    ap.add_argument("--years", type=int, default=4)
    ap.add_argument("--region", default="finland")
    a = ap.parse_args()
    if a.cmd == "status":
        _print_status(); return 0
    if a.cmd == "validate":
        probs = validate()
        if probs:
            print("PROBLEMS:"); [print("  -", p) for p in probs]; return 1
        print("data store OK"); return 0
    config = yaml.safe_load(
        (REPO / "config" / "regions" / f"{a.region}.yaml").read_text())
    m = update(config, years=a.years)
    print(f"\nsnapshot_id: {m['snapshot_id']}")
    _print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
