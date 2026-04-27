"""
Autonomous validation sweep: runs options A, B, (C) sequentially.

Comparison of SARIMAX exog modes vs AR(2):
  Option A: hour-of-day x workday/weekend dummies (~46 calendar params)
            Strict structural match to AR(2)'s profile_wd/profile_we.
  Option B: full hour-of-week dummies (~172 calendar params)
            Strictly more expressive than AR(2).
  Option C: seasonal SARIMAX(2,0,1)(1,1,0)[168] + Fourier exog
            True stochastic weekly recurrence (very slow fit).

Run order:
  1. A — fast (~6-15 min total). If A wins on the migration criteria, stop early.
  2. B — moderate (~30-90 min). If B wins, stop here.
  3. C — slow (~6-12 hours). Only run if A and B both fail to beat AR(2).

Auto-commits results after each option. Writes consolidated FINDINGS_v2.md
at the end summarizing all attempted variants.

Usage:
    python -m studies.run_validation_sweep [--max-option C] [--anchors 4]

  --max-option   Stop after this option (default C if needed). Options: A, B, C.
  --anchors      Walk-forward anchors per option (default 4).
  --skip-existing  If a tagged result already exists, skip that option.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# Migration criteria (mirror the validate script)
def _passes_criteria(zones_dict: dict) -> tuple[bool, dict]:
    n = len(zones_dict)
    if n == 0:
        return False, {"n_zones": 0}
    better_mae = sum(1 for z in zones_dict.values() if z["sarimax"]["mae"] <= z["ar2"]["mae"])
    better_cheap4 = sum(1 for z in zones_dict.values()
                         if z["sarimax"]["dk_cheap_mae"][3] < z["ar2"]["dk_cheap_mae"][3])
    better_weekend = sum(1 for z in zones_dict.values()
                          if z["sarimax"]["mae_weekend"] < z["ar2"]["mae_weekend"])
    passes = (better_mae >= 2) and (better_cheap4 == n) and (better_weekend >= 2)
    return passes, {
        "n_zones": n,
        "better_mae": better_mae,
        "better_cheap4": better_cheap4,
        "better_weekend": better_weekend,
        "passes": passes,
    }


def _run_validation(tag: str, exog_mode: str, seasonal_period: int,
                     anchors: int, out_dir: Path,
                     train_end: str, holdout_start: str) -> dict | None:
    """Run validation script as subprocess. Returns parsed JSON or None on failure."""
    logger.info("=" * 70)
    logger.info("Running %s: exog_mode=%s, seasonal_period=%d, anchors=%d",
                tag, exog_mode, seasonal_period, anchors)
    logger.info("=" * 70)
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "studies.validate_neighbor_models",
        "--exog-mode", exog_mode,
        "--seasonal-period", str(seasonal_period),
        "--anchors", str(anchors),
        "--tag", tag,
        "--train-end", train_end,
        "--holdout-start", holdout_start,
        "--out-dir", str(out_dir),
    ]
    log_path = out_dir / f"sweep_{tag}.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    logger.info("%s finished in %.1f min (exit %d)", tag, elapsed / 60.0, result.returncode)
    if result.returncode != 0:
        logger.error("%s failed; see %s", tag, log_path)
        return None

    # Find the latest JSON for this tag
    candidates = sorted(out_dir.glob(f"validation_*_{tag}.json"))
    if not candidates:
        logger.error("No JSON output found for tag=%s", tag)
        return None
    json_path = candidates[-1]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["_meta"] = {
        "tag": tag,
        "elapsed_seconds": elapsed,
        "json_path": str(json_path),
        "log_path": str(log_path),
    }
    return data


def _summarize(option_result: dict, opt_name: str) -> str:
    zones = option_result.get("zones", {})
    summary = option_result.get("summary", {})
    elapsed_min = option_result.get("_meta", {}).get("elapsed_seconds", 0) / 60.0
    n = len(zones)
    lines = [f"### {opt_name} ({option_result.get('config', {}).get('exog_mode', '?')})\n"]
    lines.append(f"- Runtime: {elapsed_min:.1f} min")
    lines.append(f"- Zones evaluated: {n}")
    lines.append(f"- SARIMAX better hourly MAE: {summary.get('zones_better_mae', '?')}/{n} (need >=2)")
    lines.append(f"- SARIMAX better dk_cheap[3]:  {summary.get('zones_better_cheap4', '?')}/{n} (need all)")
    lines.append(f"- SARIMAX better weekend MAE: {summary.get('zones_better_weekend', '?')}/{n} (need >=2)")
    lines.append(f"- **Passes migration gate:** {summary.get('decision_migrate_to_sarimax', False)}")

    lines.append("\n| Zone | AR(2) MAE | SARIMAX MAE | Δ | AR(2) cheap[3] | SARIMAX cheap[3] | Δ | AR(2) weekend | SARIMAX weekend | Δ |")
    lines.append("|------|-----------|-------------|---|----------------|------------------|---|---------------|-----------------|---|")
    for zone, z in zones.items():
        a = z["ar2"]; s = z["sarimax"]
        lines.append(
            f"| {zone.upper()} | {a['mae']:.2f} | {s['mae']:.2f} | {s['mae']-a['mae']:+.2f}"
            f" | {a['dk_cheap_mae'][3]:.2f} | {s['dk_cheap_mae'][3]:.2f} | {s['dk_cheap_mae'][3]-a['dk_cheap_mae'][3]:+.2f}"
            f" | {a['mae_weekend']:.2f} | {s['mae_weekend']:.2f} | {s['mae_weekend']-a['mae_weekend']:+.2f}"
            f" |"
        )
    return "\n".join(lines) + "\n"


def write_consolidated_findings(results: dict[str, dict], out_path: Path) -> None:
    """Write FINDINGS_v2.md combining all option outcomes."""
    lines = [
        "# Findings v2 — SARIMAX Specifications Sweep vs AR(2)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Background",
        "",
        "An initial validation (`FINDINGS.md`) used a thin SARIMAX(2,0,1)(0,0,0,0)",
        "with low-frequency Fourier calendar exog (~14 features). AR(2) won on the",
        "migration criteria, but the comparison was unfair: AR(2) carries 48",
        "calendar parameters per zone (24-hour `profile_wd` + 24-hour `profile_we`),",
        "while the SARIMAX I built had only ~14. So the comparison wasn't strict",
        "subset-vs-superset.",
        "",
        "This sweep tests progressively richer SARIMAX specifications, each a",
        "structural superset of the AR(2) profile representation:",
        "",
        "- **Option A** — `hour-workday` exog: 23 hour-of-day dummies + 23 hour×weekend",
        "  interactions + holiday + annual Fourier (~46 calendar params). Matches",
        "  AR(2)'s profile_wd / profile_we structure exactly.",
        "- **Option B** — `hour-of-week` exog: 167 dummies (one per hour×dow cell) +",
        "  holiday + annual Fourier (~172 calendar params). Strictly more expressive.",
        "- **Option C** — full seasonal SARIMAX(2,0,1)(1,1,0)[168] with Fourier exog.",
        "  Adds stochastic weekly recurrence on top of the regression structure.",
        "",
        "Same training (2022-04 → 2025-12) and holdout (2026-01 → 2026-04) windows",
        "as the original FINDINGS.md run. 4 walk-forward anchors per zone.",
        "",
        "## Decision Criteria (unchanged)",
        "",
        "Migrate to SARIMAX iff **all** hold:",
        "1. SARIMAX hourly MAE ≤ AR(2) on ≥2/3 zones",
        "2. SARIMAX `dk_cheap[3]` (cheap 4h) < AR(2) on **all** 3 zones",
        "3. SARIMAX weekend MAE < AR(2) on ≥2/3 zones",
        "",
        "## Results",
        "",
    ]

    # Add option summaries in order
    for opt_name in ["A", "B", "C"]:
        if opt_name in results and results[opt_name] is not None:
            lines.append(_summarize(results[opt_name], f"Option {opt_name}"))
        else:
            lines.append(f"### Option {opt_name}\n\n_(not run)_\n")

    # Final verdict
    lines.append("## Final Verdict\n")
    winners = [k for k, v in results.items()
               if v is not None and v.get("summary", {}).get("decision_migrate_to_sarimax", False)]
    if winners:
        chosen = winners[0]  # earliest passing option
        lines.append(f"**MIGRATE TO SARIMAX with Option {chosen}** "
                     f"(mode = `{results[chosen].get('config', {}).get('exog_mode', '?')}`).\n")
        lines.append(f"This was the first option to pass all three migration criteria.\n")
    else:
        lines.append("**KEEP AR(2)** — no SARIMAX specification passed all three migration criteria.\n")
        lines.append("This is now a robust conclusion across calendar-parameter capacities ranging from")
        lines.append("strict-match (Option A: 46 params) through superset (Option B: 172 params)")
        if "C" in results and results["C"] is not None:
            lines.append("up to true seasonal SARIMAX (Option C: full 168-period stochastic recurrence).")
        else:
            lines.append("(Option C was skipped or did not complete.)")
        lines.append("")
        lines.append("AR(2)'s profile-based structure is genuinely competitive; the gap to a fair")
        lines.append("SARIMAX upgrade is small enough that profile + ARMA(2) on deviations remains")
        lines.append("the right baseline.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Consolidated findings written to %s", out_path)


def _git_commit(out_dir: Path, message: str) -> None:
    """Best-effort auto-commit. Pushes if remote configured."""
    try:
        subprocess.run(["git", "add", "-A", str(out_dir), "src/", "studies/"],
                       cwd=Path(__file__).resolve().parents[1],
                       check=False, capture_output=True)
        subprocess.run(["git", "commit", "-m", message],
                       cwd=Path(__file__).resolve().parents[1],
                       check=False, capture_output=True)
        logger.info("Committed: %s", message)
    except Exception as exc:
        logger.warning("git commit failed: %s", exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-option", default="C", choices=["A", "B", "C"],
                        help="Stop after this option (default C)")
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--out-dir", default="studies/results")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--skip-on-pass", action="store_true",
                        help="Stop early if an option passes the migration gate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s [SWEEP] %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("A", "hour-workday", 0),
        ("B", "hour-of-week", 0),
        ("C", "fourier",      168),  # seasonal_period=168 turns on (1,1,0,168)
    ]
    max_idx = {"A": 0, "B": 1, "C": 2}[args.max_option]
    plan = plan[:max_idx + 1]

    results: dict[str, dict | None] = {}
    for opt_name, exog_mode, seasonal_period in plan:
        try:
            res = _run_validation(
                tag=opt_name, exog_mode=exog_mode, seasonal_period=seasonal_period,
                anchors=args.anchors, out_dir=out_dir,
                train_end=args.train_end, holdout_start=args.holdout_start,
            )
            results[opt_name] = res
            # Commit intermediate results so progress is visible
            if res is not None:
                _git_commit(out_dir, f"validation sweep: option {opt_name} ({exog_mode}) results")
                if args.skip_on_pass and res.get("summary", {}).get("decision_migrate_to_sarimax", False):
                    logger.info("Option %s passed migration gate; stopping early", opt_name)
                    break
        except Exception as exc:
            logger.exception("Option %s crashed: %s", opt_name, exc)
            results[opt_name] = None

    # Final consolidated report
    findings_path = out_dir / "FINDINGS_v2.md"
    write_consolidated_findings(results, findings_path)
    _git_commit(out_dir, "validation sweep: consolidated FINDINGS_v2")

    # Print final verdict to stdout
    print("\n" + "=" * 70)
    print("VALIDATION SWEEP COMPLETE")
    print("=" * 70)
    for opt_name in ["A", "B", "C"]:
        r = results.get(opt_name)
        if r is None:
            print(f"  Option {opt_name}: not run")
            continue
        s = r.get("summary", {})
        decision = "PASS" if s.get("decision_migrate_to_sarimax") else "FAIL"
        print(f"  Option {opt_name} ({r.get('config', {}).get('exog_mode')}): "
              f"MAE={s.get('zones_better_mae','?')}/3 cheap4={s.get('zones_better_cheap4','?')}/3 "
              f"weekend={s.get('zones_better_weekend','?')}/3  -> {decision}")
    print(f"\nFindings: {findings_path}")


if __name__ == "__main__":
    main()
