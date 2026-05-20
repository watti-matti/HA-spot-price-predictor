"""How quickly does the EMA profile converge from cold-start?

The PV_adjusted_price architecture ships the predictor with a
synthetic fallback profile and learns the user-specific profile
online from observed consumption. Critical question for production:
**how many days of observations are needed before the learned
profile beats the synthetic default?**

This script answers the question empirically by:

1. Taking the full post-PV window (958 days) and computing the
   "ground truth" share_by_rank + baseline.
2. Truncating to the first N days and recomputing.
3. Reporting the deviation from ground truth as a function of N.

If 30 days gets us within ~5 % of the long-run share_by_rank, the
architecture is robust for cold-start. If we need 6+ months, we
need either Fingrid-style bulk-import accelerators or alternative
bootstrap mechanisms (e.g. HDH regression for monthly factor).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))

from exp_share_by_rank import (  # noqa: E402
    build_post_pv_dataset, add_local_calendar,
    baseline_envelope, accumulate_share,
)

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _share_distance(share_a: list[float], share_b: list[float]) -> dict:
    a = np.array(share_a); b = np.array(share_b)
    return {
        "l1":                float(np.abs(a - b).sum()),
        "max_abs_diff":      float(np.abs(a - b).max()),
        "top4_diff":         float(abs(a[:4].sum() - b[:4].sum())),
        "ratio_diff":        float(
            abs(a[:4].sum() / max(a[-4:].sum(), 1e-9)
                 - b[:4].sum() / max(b[-4:].sum(), 1e-9))
        ),
    }


def _baseline_distance(base_a: np.ndarray, base_b: np.ndarray) -> dict:
    diff = base_a - base_b
    valid = np.isfinite(diff)
    return {
        "mae_kwh":  float(np.nanmean(np.abs(diff))),
        "max_abs":  float(np.nanmax(np.abs(diff[valid]))) if valid.any() else float("nan"),
        "rel_mae":  float(
            np.nanmean(np.abs(diff[valid]) / np.maximum(base_a[valid], 0.05))
        ) if valid.any() else float("nan"),
    }


def main() -> None:
    print("Loading full post-PV dataset...")
    df = add_local_calendar(build_post_pv_dataset())
    df = df.sort_index()
    print(f"  total: {len(df):,} hourly rows, "
          f"{df.index[0]} -> {df.index[-1]}")

    # GROUND TRUTH from full window
    baseline_full = baseline_envelope(df)
    share_full = accumulate_share(df, baseline_full)
    print(f"  ground truth: n_days={share_full['n_days']}, "
          f"top4={share_full['share_top4_frac'] * 100:.1f}%, "
          f"ratio={share_full['ratio_top_vs_bot4']:.2f}x")

    # ROLLING WINDOWS from the start
    horizons_days = [7, 14, 30, 60, 90, 180, 365, 720]
    start_ts = df.index[0]
    results: list[dict] = []
    for N in horizons_days:
        end = start_ts + pd.Timedelta(days=N)
        sub = df.loc[df.index < end]
        if len(sub) < 24 * 7:
            print(f"  N={N}d: insufficient data, skipping")
            continue
        baseline_N = baseline_envelope(sub)
        share_N = accumulate_share(sub, baseline_N)
        share_dist = _share_distance(
            share_full["share_by_rank"], share_N["share_by_rank"]
        )
        base_dist = _baseline_distance(baseline_full, baseline_N)
        row = {
            "N_days":       N,
            "n_days_eligible": share_N["n_days"],
            "top4_pct":     share_N["share_top4_frac"] * 100,
            "bot4_pct":     share_N["share_bottom4_frac"] * 100,
            "ratio":        share_N["ratio_top_vs_bot4"],
            "share_l1":     share_dist["l1"],
            "share_max":    share_dist["max_abs_diff"],
            "share_top4_d": share_dist["top4_diff"] * 100,
            "share_ratio_d": share_dist["ratio_diff"],
            "baseline_mae": base_dist["mae_kwh"],
            "baseline_rel_mae": base_dist["rel_mae"],
        }
        results.append(row)
        print(
            f"  N={N:4d}d  n_days={share_N['n_days']:4d}  "
            f"top4={share_N['share_top4_frac'] * 100:5.1f}%  "
            f"ratio={share_N['ratio_top_vs_bot4']:.2f}x  "
            f"share-L1 vs truth={share_dist['l1']:.3f}  "
            f"baseline rel-MAE={base_dist['rel_mae'] * 100:.1f}%"
        )

    # Ground truth row last
    results.append({
        "N_days":          int((df.index[-1] - df.index[0]).total_seconds() / 86400),
        "n_days_eligible": share_full["n_days"],
        "top4_pct":        share_full["share_top4_frac"] * 100,
        "bot4_pct":        share_full["share_bottom4_frac"] * 100,
        "ratio":           share_full["ratio_top_vs_bot4"],
        "share_l1":        0.0,
        "share_max":       0.0,
        "share_top4_d":    0.0,
        "share_ratio_d":   0.0,
        "baseline_mae":    0.0,
        "baseline_rel_mae": 0.0,
        "ground_truth":    True,
    })

    # WRITE OUTPUTS
    out_json = RESULTS_DIR / "exp_bootstrap_learning_curve.json"
    out_md = RESULTS_DIR / "exp_bootstrap_learning_curve.md"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {r['N_days']:4d} | {r['n_days_eligible']:4d} | "
        f"{r['top4_pct']:5.1f} | {r['ratio']:5.2f} | "
        f"{r['share_l1']:5.3f} | "
        f"{r['share_top4_d']:5.2f} | "
        f"{r['baseline_rel_mae'] * 100:5.1f} |"
        for r in results
    )

    truth = results[-1]

    md = f"""# Bootstrap learning curve — when does the EMA become useful?

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_bootstrap_learning_curve.py`](../exp_bootstrap_learning_curve.py).

Tests how quickly the rank-shift profile and baseline envelope
converge to their long-run values as observation history accrues.
The post-PV window from the reference household is used as the
truth-set ({truth['n_days_eligible']} valid days); each row truncates
to the first N days and reports deviation from the full-window estimate.

This answers the **fresh-install bootstrap question**: how many days
of HA recorder observations does the EMA module need before the
learned profile is usefully better than the synthetic fallback?

## Convergence table

| N days | days used | top4 % | ratio | share-L1 vs truth | top4 Δ % | baseline rel-MAE % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{rows}

Columns:

- **share-L1**: total absolute deviation across the 24 ranks,
  normalised. 0 = identical, ~0.5 = uniform-vs-skewed.
- **top4 Δ**: absolute difference in the top-4-rank concentration
  (the headline number for "is the household rank-shifting").
- **baseline rel-MAE**: relative mean absolute error of the
  baseline envelope vs the full-window envelope, weighted by cell.

## Read-out

The first row to inspect is **N=30** — one month of observations.

If 30 days achieves ≤ 5 percentage-points absolute error on the
top4 share and ≤ 20 % relative baseline MAE, the EMA module
becomes a useful upgrade over the synthetic fallback within the
first month of operation on a fresh install. No bulk-import seed
required.

The 90-day row tells us when the profile is "production-grade" —
when the predictor's PV-aware CVaR can be relied on as the
canonical reference signal.

The 365-day row tells us when monthly_factor has been observed
across all 12 months and seasonal extrapolation can stop.

## Practical generalisation for fresh installs

Three bootstrap mechanisms, in order of increasing user friction:

1. **Cold start (default for any HA install)**: synthetic Finnish
   profile scaled to user's `annual_kwh`. data_provenance =
   "synthetic_cold_start". Predictor publishes CVaR with low-
   confidence flag.

2. **Online learning (default everywhere)**: every HA install
   accumulates observations and the EMA module incrementally
   updates baseline + share_by_rank + monthly_factor cells. After
   N days the profile transitions through data_provenance =
   "ema_blended" → "ema_warm" based on the convergence numbers
   above.

3. **Bulk-import accelerator (Finland-only)**: users with Fingrid
   Datahub access can run a one-time
   `extract_household_profile_from_fingrid.py` import to seed the
   monthly_factor and shape from years of metering history. This
   skips the 90-day warm-up directly to "ema_warm". The
   accelerator is **optional convenience**, not a requirement.

The third mechanism is Finland-specific. The first two work for
any HA installation worldwide.

## Open architectural question — monthly_factor bootstrap

The slowest cell to populate is monthly_factor: 12 months of
single-cell observations are needed. The empirical data here
quantifies whether that 12-month wait is actually a problem in
practice (might it be that the shape converges in 90 days, leaving
only monthly_factor as the bottleneck?).

If so, a future addition: regress daily_kwh against
heating-degree-hours per day (HDH = `Σ_h max(0, T_setpoint −
T_outdoor)`). With a few weeks of observations + Open-Meteo
climatology for the user's location, monthly_factor can be
projected onto each month's typical HDH instead of waiting for
each month to be observed individually.

The HDH regression is climate-zone-aware (Open-Meteo provides
historical climatology for any latitude/longitude) and household-
specific (the slope `beta` is learned from local observations).
This converts the 12-month bottleneck into a few-weeks problem
that works anywhere, not just Finland.
"""
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_md}\nWrote {out_json}")


if __name__ == "__main__":
    main()
