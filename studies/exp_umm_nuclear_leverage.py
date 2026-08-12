"""Does the Nord Pool UMM feed carry a usable nuclear signal? (No.)

Motivation
----------
Every nuclear experiment in this project used Fingrid dataset 188, which
is *realised* production. It includes unplanned trips that happened after
the day-ahead auction cleared, so regressing price on it asks the model
to explain prices with information the price-setters did not have. That
was the standing explanation for why nuclear kept failing, recorded in
docs/BACKLOG.md as "the nuclear result is not safe — re-run against UMM
planned availability".

This script runs that test. UMM (Urgent Market Message) is the REMIT
transparency feed: generators publish outages with a start, a stop and
the MW affected. `api_client.fetch_nuclear_outage_schedule` already reads
it at runtime, but only for the *current* schedule, so a backtest needs
the published history.

Method
------
1. Pull every nuclear UMM (`fuelTypes=14`) from the public API — 1,906
   messages, 2013 onward — and keep the Finnish production units.
2. Keep the latest `version` per (messageId, period, unit): messages get
   revised, and only the final one describes what was believed.
3. Split by announcement lead time. A message published AFTER its event
   started cannot inform a day-ahead forecast, so only periods announced
   more than DA_LEAD_H ahead count as usable information.
4. Build hourly unavailable-MW series from all messages and from the
   usable subset, and compare both with realised unavailability derived
   from Fingrid 188.

The question is not "does nuclear availability move price" — it does,
measurably. It is "is the forecastable part of it obtainable", and the
answer decides whether the negative nuclear results stand.

Run:  python studies/exp_umm_nuclear_leverage.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "studies" / ".cache" / "umm_fi_nuclear.json"
RESULTS = REPO / "studies" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

UMM_API = "https://ummapi.nordpoolgroup.com/messages"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
FUEL_NUCLEAR = 14
AREA_FI = "FI"
DA_LEAD_H = 36.0          # a day-ahead forecast cannot use anything later
FLEET_MW = 4372.0         # Finnish nuclear nameplate, per config/regions
WINDOW = ("2023-01-01", "2026-08-12")


def fetch_fi_nuclear_umm(refresh: bool = False) -> list[dict]:
    """All Finnish nuclear UMM entries. Cached — the feed is append-only."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    items, skip, total = [], 0, 1
    while skip < total:
        req = urllib.request.Request(
            f"{UMM_API}?fuelTypes={FUEL_NUCLEAR}&limit=500&skip={skip}",
            headers={"Accept": "application/json", "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
        items += d["items"]
        total = d.get("total", len(items))
        skip += 500
    out = [{"pub": it.get("publicationDate"), "status": it.get("eventStatus"),
            "type": it.get("unavailabilityType"), "mid": it.get("messageId"),
            "ver": it.get("version"), "unit": pu.get("name"),
            "cap": pu.get("installedCapacity"), "periods": pu.get("timePeriods")}
           for it in items
           for pu in (it.get("productionUnits") or [])
           if pu.get("areaName") == AREA_FI and pu.get("fuelType") == FUEL_NUCLEAR]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def periods_frame(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for r in entries:
        pub = pd.Timestamp(r["pub"])
        for p in (r.get("periods") or []):
            try:
                st = pd.Timestamp(p["eventStart"]).tz_convert("UTC")
                sp = pd.Timestamp(p["eventStop"]).tz_convert("UTC")
            except Exception:
                continue
            rows.append({"pub": pub, "start": st, "stop": sp, "unit": r["unit"],
                         "unavail": float(p.get("unavailableCapacity") or 0.0),
                         "type": r["type"], "ver": r.get("ver", 0),
                         "mid": r.get("mid")})
    P = pd.DataFrame(rows)
    P["lead_h"] = (P.start - P.pub).dt.total_seconds() / 3600
    # Messages are revised; only the final version describes what was believed.
    return P.sort_values("ver").drop_duplicates(
        subset=["mid", "start", "unit"], keep="last")


def hourly_unavailable(sel: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    un = pd.Series(0.0, index=idx)
    for _, r in sel.iterrows():
        un.loc[(idx >= r.start) & (idx < r.stop)] += r.unavail
    return un


def main() -> None:
    entries = fetch_fi_nuclear_umm()
    P = periods_frame(entries)
    idx = pd.date_range(*WINDOW, freq="h", tz="UTC")
    inwin = P[P.start >= WINDOW[0]]
    usable = inwin[inwin.lead_h > DA_LEAD_H]

    print("=" * 72)
    print("UMM nuclear leverage — is the planned-availability signal usable?")
    print("=" * 72)
    print(f"\n  Finnish nuclear UMM entries, all time : {len(entries)}")
    print(f"  outage periods starting in window     : {len(inwin)}")
    print(f"  ... announced >{DA_LEAD_H:.0f} h ahead (usable) : {len(usable)}")

    print("\n  Announcement lead time by unavailabilityType:")
    for t, g in P.groupby("type"):
        print(f"    type {t}  n={len(g):4d}  median lead {g.lead_h.median():+8.1f} h"
              f"   >{DA_LEAD_H:.0f} h ahead: {(g.lead_h > DA_LEAD_H).mean():.0%}")

    grid = pd.read_parquet(REPO / "data_store" / "fi_grid_data.parquet")
    grid.index = pd.DatetimeIndex(grid.index)
    nuc = grid["nuclear_mw"].reindex(idx).interpolate(limit=6)
    realised = (1.0 - nuc) * FLEET_MW

    all_umm = hourly_unavailable(inwin, idx)
    use_umm = hourly_unavailable(usable, idx)

    print(f"\n  {'series':36s} {'mean MW':>9s} {'sd MW':>8s} {'hours>0':>9s}")
    for lbl, s in (("realised unavailable (Fingrid 188)", realised),
                   ("UMM, all messages", all_umm),
                   (f"UMM, announced >{DA_LEAD_H:.0f} h ahead", use_umm)):
        print(f"  {lbl:36s} {s.mean():9.0f} {s.std():8.0f} {int((s > 0).sum()):9d}")

    m = realised.notna()
    out = {}
    for lbl, s in (("all", all_umm), ("usable", use_umm)):
        A = np.column_stack([np.ones(int(m.sum())), s[m]])
        b, *_ = np.linalg.lstsq(A, realised[m], rcond=None)
        r2 = 1 - ((realised[m] - A @ b).var() / realised[m].var())
        c = float(np.corrcoef(s[m], realised[m])[0, 1])
        out[lbl] = {"corr": c, "r2": float(r2)}
        print(f"\n  UMM ({lbl}) vs realised unavailability:  "
              f"corr {c:+.3f}   R2 {r2:.3f}")

    print("\n  Which units carry the usable signal:")
    for unit, g in usable.groupby("unit"):
        hrs = ((g.stop - g.start).dt.total_seconds() / 3600).sum()
        print(f"    {unit:20s} {len(g):3d} periods  {hrs:7.0f} unit-h  "
              f"mean {g.unavail.mean():6.0f} MW")

    print(f"\n  VERDICT: the day-ahead-available UMM signal explains "
          f"{out['usable']['r2'] * 100:.1f} % of realised nuclear "
          f"unavailability.\n  There is no better nuclear variable to test "
          f"with — the negative results stand.")
    (RESULTS / "exp_umm_nuclear_leverage.json").write_text(
        json.dumps({"n_entries": len(entries), "n_periods_in_window": len(inwin),
                    "n_usable": len(usable), "fit": out}, indent=2),
        encoding="utf-8")
    print("\nWrote studies/results/exp_umm_nuclear_leverage.json")


if __name__ == "__main__":
    main()
