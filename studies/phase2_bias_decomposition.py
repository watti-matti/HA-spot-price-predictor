"""Phase-2 — post-EMA residual decomposition + evidence for conditional
bias correction.

Question: production already runs a global 14-day-halflife signed-bias
EMA (HourlyBiasCorrector) on the spot point forecast. How much of the
harness-measured bias does it absorb, what bias structure survives it,
and does a CONDITIONAL (per-hour-of-day) corrector beat it?

Method
------
* Point forecasts: the v2.12.0 FRESH_CONS walk-forward predictions from
  the frozen harness (studies/backtest_harness.py) — day-ahead regime.
* Bias correctors are simulated with the REAL production class
  (bias_corrector.OnlineBiasCorrector) under the day-ahead information
  structure: all 24 hours of day D are corrected using EMA state as of
  the end of D-1, then the EMA is updated hourly with (raw forecast,
  actual) pairs, matching Pipeline.update_with_actuals.
* Variants (isolated):
    RAW       no correction
    GLOBAL    production config — one EMA, halflife 14 d, warmup 168 h,
              winsor 5x (exactly what ships today)
    PER_HOUR  24 independent EMAs, one per hour-of-day, halflife 14 d
              (cadence 1/day), warmup 14 daily updates, winsor 5x
* Early eval hours inside each corrector's warmup pass through
  uncorrected — identical treatment across variants.

Output: studies/results/phase2_bias_decomposition.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "studies"))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from bias_corrector import OnlineBiasCorrector  # noqa: E402
from backtest_harness import build_predictions, EVAL_START, _snapshot_id  # noqa: E402
from exp_extra_features import build_dataframe  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS_DIR = REPO / "studies" / "results"


def simulate_global(pred: np.ndarray, actual: np.ndarray,
                    days: np.ndarray) -> np.ndarray:
    """Production HourlyBiasCorrector sim: one EMA, day-ahead structure."""
    bc = OnlineBiasCorrector(halflife_days=14.0, warmup_steps=168,
                             winsor_limit=5.0, cadence_per_day=24)
    out = np.empty_like(pred)
    for d in pd.unique(days):
        m = days == d
        # Correct the whole day with state as of end of previous day.
        out[m] = [bc.correct(p) for p in pred[m]]
        # Then reconcile the day's actuals (arrive after delivery).
        for p, a in zip(pred[m], actual[m]):
            bc.update(p, a)
    return out


def simulate_per_hour(pred: np.ndarray, actual: np.ndarray,
                      days: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Conditional corrector: 24 independent EMAs, one per hour-of-day.
    Same 14-day halflife (one update/day per bin => cadence 1/day),
    warmup 14 daily updates per bin, same winsorisation."""
    bcs = {h: OnlineBiasCorrector(halflife_days=14.0, warmup_steps=14,
                                  winsor_limit=5.0, cadence_per_day=1)
           for h in range(24)}
    out = np.empty_like(pred)
    for d in pd.unique(days):
        m = np.where(days == d)[0]
        for i in m:
            out[i] = bcs[int(hours[i])].correct(pred[i])
        for i in m:
            bcs[int(hours[i])].update(pred[i], actual[i])
    return out


def main() -> None:
    print("Building dataframe + walk-forward predictions…", flush=True)
    df = build_dataframe()
    preds = build_predictions(df)
    raw = preds["FRESH_CONS"]
    ev = np.isfinite(raw) & np.asarray(df.index >= EVAL_START)
    idx = df.index[ev]
    y = df["fi"].values[ev]
    p_raw = raw[ev]
    days = idx.floor("D").values
    hours = idx.hour.values
    mo = idx.month.values

    p_glob = simulate_global(p_raw, y, days)
    p_ph = simulate_per_hour(p_raw, y, days, hours)

    variants = {"RAW": p_raw, "GLOBAL(prod)": p_glob, "PER_HOUR": p_ph}
    p95 = np.percentile(y, 95)
    segs = {
        "ALL": np.ones(len(y), bool),
        "WINTER Dec-Feb": np.isin(mo, [12, 1, 2]),
        "SUMMER May-Jul": np.isin(mo, [5, 6, 7]),
        "midday 8-12 UTC": (hours >= 8) & (hours < 12),
        "evening 15-19 UTC": (hours >= 15) & (hours < 19),
        "tail p95 price": y >= p95,
    }
    res = {"snapshot_id": _snapshot_id(), "n_eval": int(ev.sum()),
           "eval_window": [str(idx[0]), str(idx[-1])], "segments": {}}
    print(f"\nn_eval = {ev.sum():,}   (v2.12.0 FRESH_CONS walk-forward)")
    print(f"{'segment':18s} | " + " | ".join(f"{v:>16s}" for v in variants))
    for sname, m in segs.items():
        row = {}
        cells = []
        for vname, p in variants.items():
            e = p[m] - y[m]
            row[vname] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean())}
            cells.append(f"{row[vname]['mae']:6.2f} ({row[vname]['bias']:+6.1f})")
        res["segments"][sname] = {"n": int(m.sum()), **row}
        print(f"{sname:18s} | " + " | ".join(f"{c:>16s}" for c in cells))

    # Post-GLOBAL residual bias by hour-of-day — the structure a single
    # global EMA cannot express.
    print("\npost-GLOBAL bias by hour-of-day (EUR/MWh):")
    hb = {}
    for h in range(24):
        m = hours == h
        hb[h] = float((p_glob[m] - y[m]).mean())
    res["post_global_bias_by_hour"] = hb
    line = " ".join(f"{h:02d}:{hb[h]:+5.1f}" for h in range(24))
    print("  " + line[:120])
    print("  " + line[120:])

    (RESULTS_DIR / "phase2_bias_decomposition.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    # markdown
    lines = ["# Phase 2 — post-EMA bias decomposition", "",
             f"Snapshot `{res['snapshot_id']}`, eval "
             f"{res['eval_window'][0][:10]} → {res['eval_window'][1][:10]}, "
             f"{res['n_eval']:,} h. Point forecast: v2.12.0 walk-forward "
             f"(frozen harness). Correctors simulated with the production "
             f"OnlineBiasCorrector under day-ahead information structure.",
             "",
             "| segment | " + " | ".join(f"{v} MAE (bias)" for v in variants) + " |",
             "|---|" + "---:|" * len(variants)]
    for sname, seg in res["segments"].items():
        cells = " | ".join(f"{seg[v]['mae']:.2f} ({seg[v]['bias']:+.1f})"
                           for v in variants)
        lines.append(f"| {sname} | {cells} |")
    lines += ["", "Post-GLOBAL residual bias by hour-of-day:",
              "", "| hour | " + " | ".join(f"{h:02d}" for h in range(24)) + " |",
              "|---|" + "---:|" * 24,
              "| bias | " + " | ".join(f"{hb[h]:+.1f}" for h in range(24)) + " |"]
    (RESULTS_DIR / "phase2_bias_decomposition.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote studies/results/phase2_bias_decomposition.{md,json}")


if __name__ == "__main__":
    main()
