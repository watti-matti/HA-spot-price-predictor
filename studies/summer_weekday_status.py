"""Status of the original 2026-07 report: summer weekday morning/evening
peaks mispriced while weekends looked accurate.

Compares the v2.16 model (as production actually behaved: same-hour
neighbour prices unknowable, so zero) against the shipped v2.17.1
leak-free + holiday-aware model, on the honest task (origins 06:00 UTC
before the day-ahead auction; only unpublished hours scored), split by
LOCAL hour and day type — the frame the report was made in.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO/"studies", REPO/"custom_components"/"spot_price_predictor"):
    sys.path.insert(0, str(p))
import seasonal_decomposition as sd                       # noqa: E402
from holidays import build_holiday_set                    # noqa: E402
from exp_extra_features import build_dataframe            # noqa: E402
from backtest_harness import _fit_l1_like_shipped, PHYS_DEPTH, PHYS_SMOOTH  # noqa: E402
from honest_horizon_study import sfit                     # noqa: E402
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")

def main() -> None:
    df = build_dataframe(); idx = df.index; n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    ish = np.array([1. if d.strftime("%Y-%m-%d") in hol else 0. for d in loc])
    wd_plain = (loc.weekday < 5).astype(float)
    wd_hol = ((loc.weekday < 5) & (ish == 0)).astype(float)
    fi = df["fi"].values; wr = df["sigmoid_wind_rho"].values; se = df["solar_effective"].values
    pos = {t: i for i, t in enumerate(idx)}; rows = []
    for m0 in pd.date_range(pd.Timestamp("2025-07-01", tz="UTC"), idx[-1], freq="MS", tz="UTC"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        if tr.sum() < 8000: continue
        seas = sd.compute_seasonal_part(ts, _fit_l1_like_shipped("fi", fi[tr], ts[tr]))
        Yfi = fi - seas
        lag = np.nan_to_num(pd.Series(Yfi, index=idx).shift(168).values)
        Yz = {z: df[z].values - sd.compute_seasonal_part(
                  ts, _fit_l1_like_shipped(z, df[z].values[tr], ts[tr]))
              for z in ("temp", "se1", "se3", "ee")}
        Yw = wr - sd.compute_seasonal_part(ts, sd.fit_components(
            wr[tr], ts[tr], depth=PHYS_DEPTH, smooth=PHYS_SMOOTH))
        Ys = se - sd.compute_seasonal_part(ts, sd.fit_components(
            se[tr], ts[tr], depth=PHYS_DEPTH, smooth=PHYS_SMOOTH))
        nb_now = [np.nan_to_num(Yz[z]) for z in ("se1", "se3", "ee")]
        nb_lag = [np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
                  for z in ("se1", "se3", "ee")]
        variants = {
            "v2.16": (np.column_stack([np.ones(n), lag, wd_plain, Yw, Ys, Yz["temp"]] + nb_now), True),
            "v2.17.1": (np.column_stack([np.ones(n), lag, wd_hol, Yw, Ys, Yz["temp"]] + nb_lag + [ish]), False),
        }
        for k, (X, blank_nb) in variants.items():
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            c = sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0})
            for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D", tz="UTC"):
                o = o.replace(hour=6)
                if o not in pos: continue
                i0 = pos[o]; sl = np.arange(i0 + 1, i0 + 1 + 170)
                if sl[-1] >= n: continue
                Xi = X[sl].copy()
                if blank_nb:
                    Xi[:, 6:9] = 0.0     # production reality for v2.16
                pr = np.maximum(0.0, seas[sl] + Xi @ c)
                if not np.isfinite(pr).all(): continue
                rows.append(pd.DataFrame({
                    "v": k, "err": pr - fi[sl], "act": fi[sl],
                    "h": loc[sl].hour, "month": loc[sl].month,
                    "wd": (loc[sl].weekday < 5)}))
    r = pd.concat(rows, ignore_index=True)
    sm = r[r.month.isin([6, 7, 8])]
    print("SUMMER (Jun-Aug), honest task, by LOCAL hour — mean bias (MAE)\n")
    print(f"{'hr':>3s} | {'actual':>7s} | {'WEEKDAY v2.16':>15s} {'v2.17.1':>15s} "
          f"| {'WEEKEND v2.16':>15s} {'v2.17.1':>15s}")
    for h in range(24):
        cells = []
        for isw in (True, False):
            for v in ("v2.16", "v2.17.1"):
                e = sm[(sm.h == h) & (sm.wd == isw) & (sm.v == v)].err
                cells.append(f"{e.mean():+7.1f}({e.abs().mean():5.1f})" if len(e) else " " * 15)
        a = sm[(sm.h == h) & sm.wd].act.mean()
        print(f"{h:3d} | {a:7.1f} | {cells[0]} {cells[1]} | {cells[2]} {cells[3]}")
    print()
    blocks = {"weekday MORNING 07-11": (True, [7, 8, 9, 10, 11]),
              "weekday EVENING 17-21": (True, [17, 18, 19, 20, 21]),
              "weekday MIDDAY 12-16":  (True, [12, 13, 14, 15, 16]),
              "weekday NIGHT 00-06":   (True, list(range(7))),
              "weekend MORNING 07-11": (False, [7, 8, 9, 10, 11]),
              "weekend EVENING 17-21": (False, [17, 18, 19, 20, 21]),
              "weekend MIDDAY 12-16":  (False, [12, 13, 14, 15, 16]),
              "weekend NIGHT 00-06":   (False, list(range(7)))}
    print(f"{'block':24s} | {'actual':>7s} | {'v2.16 bias  MAE':>17s} | {'v2.17.1 bias  MAE':>19s}")
    for lbl, (isw, hrs) in blocks.items():
        s = sm[(sm.wd == isw) & sm.h.isin(hrs)]
        a = s[s.v == "v2.16"]; b = s[s.v == "v2.17.1"]
        print(f"{lbl:24s} | {a.act.mean():7.1f} | {a.err.mean():+9.1f} {a.err.abs().mean():6.1f} "
              f"| {b.err.mean():+11.1f} {b.err.abs().mean():6.1f}")
    print("\nWeekday-vs-weekend MAE gap (the original report: weekends looked better):")
    for v in ("v2.16", "v2.17.1"):
        s = sm[sm.v == v]
        wdm = s[s.wd].err.abs().mean(); wem = s[~s.wd].err.abs().mean()
        print(f"  {v:8s} weekday MAE {wdm:5.2f}  weekend MAE {wem:5.2f}  gap {wdm-wem:+5.2f}")

if __name__ == "__main__":
    main()
