"""FI model revision investigation for v2.4.4 — REJECTED.

Applies the validated SE3-style architecture (seasonal + hydro + workday +
AR(1)) and three augmentations (Y_SE3 cross-border coupling, Y_WIND
exogenous, lean no-AR variant) to FI prices. Gate: must beat windowed
seasonal-only baseline.

Result: REJECT. None of the four augmented variants beats +7.01 % windowed
baseline; best is V4 (lean: hydro + workday + Y_SE3 + Y_WIND, no AR(1)) at
+4.14 %. Same AR(1)-compresses-hedge-geometry pathology as v2.4.3 EE.

Implication for v2.5.0: KEEP the production v2.2 9-feature Ridge model
for FI unchanged. The Moazeni-Powell-style additive linear architecture
is good for SE3 (validated in v2.4.2) but doesn't transfer to FI because
FI has more diverse price drivers (nuclear outages, Fenno-Skan
congestion, EE coupling) that benefit from the v2.2 log-linear
formulation + sign-validated 9 features.

The accepted v2.5.0 plan therefore becomes:
  - FI:  KEEP v2.2 9-feature Ridge (unchanged from v2.4)
  - SE3: WIRE IN v2.4.2 model (seasonal + hydro + workday + AR(1))
  - EE:  KEEP v2.2 ar_ee AR(2) feature (unchanged, per v2.4.3 REJECT)

This is the v2.4.x methodology working as designed: validate each
candidate; accept only those that demonstrably improve out-of-sample
hedge CVaR; reject the rest without adding complexity.

Run:
    python studies/fi_model_v244.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "studies"))

from npk_cvar_hedge import fit_seasonal_hdw, optimize_hedge  # noqa: E402

LAG = 48
ALPHA = 0.05


def red_pct(r):
    return (
        100.0
        * (r["cvar_test_hist_unhedged"] - r["cvar_test_hist_hedged"])
        / r["cvar_test_hist_unhedged"]
    )


def hedge_on(actual, model, lag=LAG, alpha=ALPHA):
    fwd = np.concatenate([model[lag:], np.repeat(model[-1], lag)])
    return optimize_hedge(np.diff(actual), np.diff(fwd), alpha=alpha)


def main() -> int:
    print("Fetching Statnett 104-week window...")
    with urllib.request.urlopen(
        "http://driftsdata.statnett.no/restapi/Reservoir/LastWeekData/52", timeout=30
    ) as r:
        raw = json.loads(r.read())
    weeks = []
    for src in ("lastYear", "currentYear"):
        for e in raw.get(src, []):
            weeks.append(
                {"year": int(e["year"]), "week": int(e["week"]), "total_pct": float(e["total"])}
            )
    weeks.sort(key=lambda w: (w["year"], w["week"]))
    df_h = pd.DataFrame(weeks)
    df_h["base"] = df_h["week"].map(df_h.groupby("week")["total_pct"].mean())
    df_h["offset"] = df_h["total_pct"] - df_h["base"]
    hydro_map = {
        (int(r.year), int(r.week)): float(r.offset) for r in df_h.itertuples()
    }

    prices = pd.read_parquet(REPO / "output" / "fi_prices.parquet")
    neigh = pd.read_parquet(REPO / "output" / "fi_neighbor_prices.parquet")
    wx = pd.read_parquet(REPO / "output" / "fi_weather.parquet")
    df = prices.join(neigh[["se3"]], how="inner").join(wx, how="inner").dropna()

    y_start, w_start = weeks[0]["year"], weeks[0]["week"]
    y_end, w_end = weeks[-1]["year"], weeks[-1]["week"]
    df = df[df.index >= f"{y_start}-01-01"]
    ts_local = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=3)
    iso = ts_local.isocalendar()
    iy, iw = iso.year.to_numpy(), iso.week.to_numpy()
    in_win = (
        ((iy > y_start) | ((iy == y_start) & (iw >= w_start)))
        & ((iy < y_end) | ((iy == y_end) & (iw <= w_end)))
    )
    ts_local = ts_local[in_win]
    FI = df["price_eur_mwh"].values[in_win].astype(float)
    SE3 = df["se3"].values[in_win].astype(float)
    WIND = df["wind_speed_weighted"].values[in_win].astype(float)
    iy, iw = iy[in_win], iw[in_win]

    print(f"Window: {ts_local.min()} → {ts_local.max()}, n={len(FI)}\n")

    _, _, _, seasonal_FI, Y_FI = fit_seasonal_hdw(FI, ts_local)
    _, _, _, _, Y_SE3 = fit_seasonal_hdw(SE3, ts_local)
    _, _, _, _, Y_WIND = fit_seasonal_hdw(WIND, ts_local)
    hydro = np.array([hydro_map.get((int(y), int(w)), 0.0) for y, w in zip(iy, iw)])
    dow = ts_local.weekday.to_numpy()
    is_wd = ((dow >= 0) & (dow <= 4)).astype(float)
    Y_FI_lag = np.concatenate([[0.0], Y_FI[:-1]])

    print("=" * 88)
    print("FI variant comparison (NPK-CVaR raw 48h hedge, α=0.05, windowed)")
    print("=" * 88)

    # V0 baseline
    v0 = hedge_on(FI, seasonal_FI)
    print(f"[V0 baseline:    seasonal-only]                      h={v0['h_hat']:.3f}  red={red_pct(v0):+.2f}%")

    # V1 SE3-style
    X1 = np.column_stack([np.ones_like(Y_FI), hydro, is_wd, Y_FI_lag])
    b1 = np.linalg.lstsq(X1, Y_FI, rcond=None)[0]
    v1 = hedge_on(FI, seasonal_FI + X1 @ b1)
    print(f"[V1 SE3-style:   seasonal+hydro+wd+AR(1)]            h={v1['h_hat']:.3f}  red={red_pct(v1):+.2f}%")

    # V2 V1 + Y_SE3
    X2 = np.column_stack([np.ones_like(Y_FI), hydro, is_wd, Y_FI_lag, Y_SE3])
    b2 = np.linalg.lstsq(X2, Y_FI, rcond=None)[0]
    v2 = hedge_on(FI, seasonal_FI + X2 @ b2)
    print(f"[V2 V1 + Y_SE3 coupling]:                            h={v2['h_hat']:.3f}  red={red_pct(v2):+.2f}%")

    # V3 V2 + Y_WIND
    X3 = np.column_stack([np.ones_like(Y_FI), hydro, is_wd, Y_FI_lag, Y_SE3, Y_WIND])
    b3 = np.linalg.lstsq(X3, Y_FI, rcond=None)[0]
    v3 = hedge_on(FI, seasonal_FI + X3 @ b3)
    print(f"[V3 V2 + Y_WIND]:                                    h={v3['h_hat']:.3f}  red={red_pct(v3):+.2f}%")

    # V4 lean no AR
    X4 = np.column_stack([np.ones_like(Y_FI), hydro, is_wd, Y_SE3, Y_WIND])
    b4 = np.linalg.lstsq(X4, Y_FI, rcond=None)[0]
    v4 = hedge_on(FI, seasonal_FI + X4 @ b4)
    print(f"[V4 lean: hydro+wd+Y_SE3+Y_WIND (NO AR(1))]:         h={v4['h_hat']:.3f}  red={red_pct(v4):+.2f}%")

    print()
    results = {"V0": red_pct(v0), "V1": red_pct(v1), "V2": red_pct(v2),
               "V3": red_pct(v3), "V4": red_pct(v4)}
    best_name, best_red = max(results.items(), key=lambda kv: kv[1])
    print(f"  Best variant:           {best_name} → {best_red:+.2f}%")
    print(f"  V0 windowed baseline:   {results['V0']:+.2f}%")
    print(f"  Improvement vs V0:      {best_red - results['V0']:+.2f} pp")
    accepted = best_name != "V0" and best_red > results["V0"]
    print(f"  Verdict:                {'ACCEPT' if accepted else 'REJECT'}")
    print()
    if not accepted:
        print(
            "Same AR(1)-compresses-hedge-geometry pathology as v2.4.3 EE: AR(1)\n"
            "on the residual introduces deep autocorrelation in diff(model)\n"
            "that compresses the optimal hedge ratio (h drops from ~1.1 to ~0.2),\n"
            "destroying more CVaR-reduction than the lagged signal adds back.\n"
            "\n"
            "FI has more diverse price drivers (nuclear outages, Fenno-Skan\n"
            "congestion, EE coupling) than SE3 — the Moazeni-Powell additive\n"
            "linear architecture saturates faster on FI than on SE3.\n"
            "\n"
            "Implication for v2.5.0: KEEP the production v2.2 9-feature Ridge\n"
            "model for FI unchanged. Only SE3 (v2.4.2) gets the new architecture.\n"
        )
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
