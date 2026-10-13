"""Pin and verify the training baseline, so comparisons cannot drift.

Why this exists
---------------
On 2026-10-13 a branch-vs-production comparison appeared to show the
branch changing the forecast by up to 0.0154 EUR/MWh. It did not. The
two checkouts held different vintages of the Open-Meteo weather cache
(`studies/.cache/wx_{var}_{site}.parquet`, which is per-checkout,
incremental and append-only), so the two runs were scored on different
input data. After syncing it the results were byte-identical.

Nothing in the output said so. `backtest_harness.json` records
`snapshot_id` for the data store but nothing for the weather cache, and
there is no record of which artifacts or which code produced a given
set of numbers. Any two numbers from different moments were therefore
only loosely comparable, and the difference between "the model changed"
and "the inputs changed" was invisible.

What a baseline has to pin
--------------------------
A forecast number is reproducible only if ALL of these are fixed:

  1. inference code     the 12 modules the pipeline actually executes
  2. shipped artifacts  the 4 JSON files it loads
  3. price/grid data    data_store parquets (+ its own snapshot_id)
  4. weather data       studies/.cache/wx_*.parquet  <- the one that bit us
  5. harness code       the scoring path itself

This module hashes all five into a provenance fingerprint, records it
next to the measured results, and can re-verify it later.

HOW TO COMPARE FAIRLY — read this before quoting a baseline number
------------------------------------------------------------------
The recorded MAE/bias figures are valid ONLY for the exact fingerprint
they were recorded under. They are a reproducibility reference, not a
yardstick to hold a future model against.

The reason: the harness eval window ends wherever the data store ends,
so refreshing data changes which hours are scored. A new model measured
on 2026-11 data and compared against a baseline number from 2026-10
data differs for two reasons at once, and you cannot separate them.

So:
  * To check a code change is a no-op  -> `verify` must report an
    IDENTICAL results hash. That is exact and unambiguous.
  * To check a new model is better     -> run the harness ONCE with
    both configs in it (PRODUCTION vs FRESH/candidate) so both are
    scored on the same rows. Never compare across runs.
  * Numbers recorded here are quotable only alongside their
    fingerprint, and are void the moment any of the five inputs move.

Usage
-----
  python studies/verify_baseline.py                 # verify current state
  python studies/verify_baseline.py --record        # write BASELINE.json
  python studies/verify_baseline.py --print         # show the fingerprint

`--record` refuses to run unless a harness result exists and its
fingerprint matches the current tree, so a baseline can never be
recorded against results produced by different inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPP = REPO / "custom_components" / "spot_price_predictor"
RESULTS = REPO / "studies" / "results"
BASELINE = RESULTS / "BASELINE.json"
HARNESS_JSON = RESULTS / "backtest_harness.json"

# Entry points of the forecast. The coordinator drives the update; the
# pipeline computes the numbers. Everything reachable from these by
# intra-package import can move a predicted price.
FORECAST_ENTRY_POINTS = ("coordinator", "pipeline")


def inference_modules() -> tuple[str, ...]:
    """The forecast-reachable modules, DERIVED not hand-listed.

    A hand-maintained list was tried first and was wrong in both
    directions: it missed ten modules that the forecast really does
    reach (`features`, `dtaci_integration`, `pv_estimate`, `pv_nowcast`,
    `pv_aware_cvar`, `pv_cost_kernel`, `consumption_profile_loader`,
    `dk_dtaci`, `dk_utils`, `price_forecast_verifier`, `holidays`) and
    included `sensor.py`, which is an HA platform that reads coordinator
    output and cannot affect it. A fingerprint over an incomplete list
    silently fails to invalidate itself when real forecast code changes,
    which is the one thing it must never do.

    So the closure is computed from the import graph every run. Adding
    an import to the forecast path therefore changes the fingerprint on
    its own, with no list to remember to update.
    """
    import ast

    local = {p.stem for p in SPP.glob("*.py")}

    def imports_of(stem: str) -> set[str]:
        src = (SPP / f"{stem}.py").read_text(encoding="utf-8",
                                             errors="replace")
        out: set[str] = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.ImportFrom):
                if n.level and n.module is None:          # from . import a, b
                    out |= {a.name for a in n.names if a.name in local}
                elif n.level and n.module:                # from .mod import x
                    root = n.module.split(".")[0]
                    if root in local:
                        out.add(root)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    root = a.name.split(".")[0]
                    if root in local:
                        out.add(root)
        return out

    seen: set[str] = set()
    stack = list(FORECAST_ENTRY_POINTS)
    while stack:
        m = stack.pop()
        if m in seen or not (SPP / f"{m}.py").exists():
            continue
        seen.add(m)
        stack.extend(imports_of(m) - seen)
    return tuple(f"{m}.py" for m in sorted(seen))
SHIPPED_ARTIFACTS = (
    "spike_model_default.json", "seasonal_components_default.json",
    "solar_submodel_default.json", "model_coefs_default.json",
)
# The scoring path. A change here changes the numbers without changing
# the model, which is exactly the confusion this file exists to prevent.
HARNESS_MODULES = (
    "backtest_harness.py", "exp_extra_features.py",
    "build_seasonal_components.py", "honest_horizon_study.py",
)


def _sha_files(paths: list[Path]) -> str:
    """Order-independent-by-name digest over file contents."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda q: q.name):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes() if p.exists() else b"<MISSING>")
        h.update(b"\0")
    return h.hexdigest()


def fingerprint() -> dict:
    """Hash everything that can move a forecast number."""
    store = REPO / "data_store"
    wx = sorted((REPO / "studies" / ".cache").glob("wx_*.parquet"))
    snapshot = "no-data-store"
    mf = store / "manifest.json"
    if mf.exists():
        snapshot = json.loads(mf.read_text()).get("snapshot_id", "unknown")

    return {
        "inference_code": _sha_files(
            [SPP / m for m in inference_modules()]),
        # Recorded so a reader can see WHAT was hashed, and so a change
        # in the reachable set is visible and not just a hash delta.
        "inference_modules": list(inference_modules()),
        "shipped_artifacts": _sha_files(
            [SPP / "data" / a for a in SHIPPED_ARTIFACTS]),
        "harness_code": _sha_files(
            [REPO / "studies" / m for m in HARNESS_MODULES]),
        "data_store_snapshot": snapshot,
        "data_store_parquets": _sha_files(sorted(store.glob("*.parquet"))),
        # The variable that caused the 2026-10-13 false positive. It is
        # per-checkout and NOT in git, so two clones differ silently.
        "weather_cache": _sha_files(wx),
        "weather_cache_files": len(wx),
    }


def _artifact_facts() -> dict:
    """Human-readable identity of the artifacts, so a stale baseline is
    obvious to a reader and not only to a hash comparison."""
    out = {}
    for a in SHIPPED_ARTIFACTS:
        p = SPP / "data" / a
        if not p.exists():
            out[a] = {"present": False}
            continue
        d = json.loads(p.read_text())
        rec = {"present": True, "version": d.get("version")}
        if "ridge_features" in d:
            rec["n_features"] = len(d["ridge_features"])
            rec["ridge_features"] = d["ridge_features"]
        if "train_window" in d:
            rec["train_window"] = d["train_window"]
        if "data_store_snapshot" in d:
            rec["data_store_snapshot"] = d["data_store_snapshot"]
        out[a] = rec
    return out


def _harness_facts() -> dict:
    if not HARNESS_JSON.exists():
        return {}
    d = json.loads(HARNESS_JSON.read_text())
    segs = {}
    for name, seg in d.get("segments", {}).items():
        segs[name] = {
            cfg: {"mae": seg[cfg]["mae"], "bias": seg[cfg]["bias"]}
            for cfg in seg if isinstance(seg.get(cfg), dict)
        }
        segs[name]["n"] = seg.get("n")
        segs[name]["mean_price"] = seg.get("mean_price")
    return {
        "results_sha256": hashlib.sha256(
            HARNESS_JSON.read_bytes()).hexdigest(),
        "snapshot_id": d.get("snapshot_id"),
        "n_eval": d.get("n_eval"),
        "eval_window": d.get("eval_window"),
        "price_p95": d.get("price_p95"),
        "segments": segs,
    }


def record() -> int:
    fp = fingerprint()
    hf = _harness_facts()
    if not hf:
        print(f"ERROR: {HARNESS_JSON.relative_to(REPO)} not found.\n"
              "       Run `python studies/backtest_harness.py` first — a "
              "baseline must be recorded from real measured results, never "
              "from remembered numbers.")
        return 2
    if hf.get("snapshot_id") != fp["data_store_snapshot"]:
        print("ERROR: the harness results were produced against data store "
              f"snapshot {hf.get('snapshot_id')}, but the tree now holds "
              f"{fp['data_store_snapshot']}.\n"
              "       Re-run the harness so results and inputs agree; "
              "recording them together would pin a baseline to data that "
              "did not produce it.")
        return 2

    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps({
        "_what": "Reference point for the training baseline. The numbers "
                 "below are valid ONLY for this exact fingerprint. To "
                 "judge whether a new model is better, score both configs "
                 "in a SINGLE harness run; never compare across runs.",
        "fingerprint": fp,
        "artifacts": _artifact_facts(),
        "harness": hf,
    }, indent=2), encoding="utf-8")
    print(f"recorded -> {BASELINE.relative_to(REPO)}")
    _print(fp, hf)
    return 0


def _print(fp: dict, hf: dict) -> None:
    print("\n  fingerprint")
    for k, v in fp.items():
        v = str(v)
        print(f"    {k:22s} {v[:16] + '…' if len(v) > 17 else v}")
    if hf:
        print(f"\n  measured: n_eval={hf['n_eval']:,}  "
              f"window {hf['eval_window'][0][:10]} .. "
              f"{hf['eval_window'][1][:10]}")
        print(f"    {'segment':20s} {'PRODUCTION MAE':>15s} {'bias':>9s}")
        for name, seg in hf["segments"].items():
            p = seg.get("PRODUCTION")
            if p:
                print(f"    {name:20s} {p['mae']:15.4f} {p['bias']:+9.4f}")


def verify() -> int:
    if not BASELINE.exists():
        print(f"ERROR: no {BASELINE.relative_to(REPO)}. Record one with "
              "`--record` after a harness run.")
        return 2
    base = json.loads(BASELINE.read_text())
    want, got = base["fingerprint"], fingerprint()

    print("  component                current vs baseline")
    bad = []
    for k in want:
        ok = want[k] == got[k]
        if not ok:
            bad.append(k)
        print(f"    {k:22s} {'MATCH' if ok else 'DIFFERS'}")
        if not ok:
            print(f"      baseline {str(want[k])[:24]}")
            print(f"      current  {str(got[k])[:24]}")

    if not bad:
        print("\n  Inputs and code match the baseline exactly.")
        if HARNESS_JSON.exists():
            cur = hashlib.sha256(HARNESS_JSON.read_bytes()).hexdigest()
            ref = base["harness"]["results_sha256"]
            same = cur == ref
            print(f"  Harness results hash: "
                  f"{'IDENTICAL — reproduced exactly' if same else 'DIFFERS'}")
            if not same:
                print("    The inputs match but the results do not. That is "
                      "a genuine non-determinism or code-path bug, not a "
                      "data difference — investigate before trusting any "
                      "number from this tree.")
                return 1
        return 0

    print(f"\n  {len(bad)} component(s) differ, so the baseline's recorded "
          "numbers do NOT apply to this tree.")
    if "weather_cache" in bad:
        print("    weather_cache: studies/.cache/wx_*.parquet is "
              "per-checkout, incremental and untracked, so two clones hold "
              "different vintages. Copy it from the reference checkout "
              "before comparing anything.")
    if "shipped_artifacts" in bad:
        print("    shipped_artifacts: the model changed. Re-record the "
              "baseline deliberately, and remember installing a new "
              "artifact wipes every per-hour bias calibrator.")
    if "data_store_parquets" in bad or "data_store_snapshot" in bad:
        print("    data store moved: the eval window follows the data, so "
              "scored hours changed. Re-run both configs in one harness "
              "run instead of comparing to the old numbers.")
    if "harness_code" in bad:
        print("    harness_code: the scoring path changed, which moves the "
              "numbers without moving the model.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", action="store_true",
                    help="write BASELINE.json from the current tree")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the current fingerprint and exit")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if a.show:
        _print(fingerprint(), _harness_facts())
        return 0
    return record() if a.record else verify()


if __name__ == "__main__":
    raise SystemExit(main())
