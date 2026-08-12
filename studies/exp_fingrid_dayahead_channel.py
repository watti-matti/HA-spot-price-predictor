"""WP2.5 — is Fingrid's published day-ahead wind/PV forecast worth adding?

Hypothesis
----------
Fingrid publishes its own day-ahead wind and solar generation forecasts
(datasets 246 / 247). They are NOT reconstructible from the weather we
already fetch: regressing them on our physics proxies leaves an
orthogonal remainder with sd ~660 MW (wind) and ~108 MW (PV). If the
market prices the published number — which it plausibly does, since
bidders read the same feed — that remainder carries information our
weather proxy cannot.

Information set (the whole point)
---------------------------------
Origin 06:00 UTC daily, 170 h horizon, exactly as honest_horizon_study.
Weather is available for the full horizon. **Fingrid's day-ahead
forecast is not** — it covers roughly the next 36-42 h. Evaluating it at
all leads would be an oracle with respect to HORIZON, so the shippable
variants zero or fade it past the publication boundary and the full-
availability variant is reported only as an unreachable ceiling.

Variants
--------
  BASE          shipped nine-feature set, neighbours lagged 168 h
  FG_ALL        + orthogonal remainder at every lead   (CEILING, not shippable)
  FG_D1         + remainder for leads <= DA_HOURS, zero beyond
  FG_D1_FADE    + remainder faded past DA_HOURS with a 48 h half-life
                  (leadtime_fill_study measured hl~48 h as the best
                   bias/variance trade for exactly this boundary)
  FG_SUBST      Fingrid MW replacing the weather proxies rather than
                  added to them — the naive version, measured because it
                  is the obvious thing to try and it is worse

The remainder projection is fitted on TRAIN data only; nothing from the
scored month enters it.

Run:  python studies/exp_fingrid_dayahead_channel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "studies", REPO / "custom_components" / "spot_price_predictor"):
    sys.path.insert(0, str(p))

import seasonal_decomposition as sd                     # noqa: E402
from holidays import build_holiday_set                  # noqa: E402
from exp_extra_features import build_dataframe, SEASONAL_ARTIFACT  # noqa: E402
from honest_horizon_study import sfit                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")
HORIZON = 170
ORIGIN_HOUR = 6          # UTC, before the ~13:00 CET day-ahead auction
DA_HOURS = 42            # how far Fingrid's published DA forecast reaches
FADE_HALFLIFE_H = 48.0


def _remainder(target: np.ndarray, proxy: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Part of `target` that `proxy` cannot explain, projection fitted on
    TRAIN rows only."""
    ok = tr & np.isfinite(target) & np.isfinite(proxy)
    A = np.column_stack([np.ones(ok.sum()), proxy[ok]])
    b, *_ = np.linalg.lstsq(A, target[ok], rcond=None)
    return target - (b[0] + b[1] * proxy)


def main() -> None:
    df = build_dataframe()
    idx = df.index
    n = len(df)
    ts = pd.DatetimeIndex(idx, tz="UTC").values
    pos = {t: i for i, t in enumerate(idx)}

    loc = pd.DatetimeIndex(idx).tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    is_hol = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0 for d in loc])
    is_wd = (loc.weekday < 5).astype(float) * (1.0 - is_hol)

    fi = df["fi"].values
    wind_rho = df["sigmoid_wind_rho"].values
    solar_eff = df["solar_effective"].values
    fg_wind = df["wind_forecast_mw"].values / 1e3      # GW
    fg_pv = df["solar_forecast_mw"].values / 1e3

    rows: list[pd.DataFrame] = []
    months = pd.date_range(EVAL_START, idx[-1], freq="MS", tz="UTC")
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        if tr.sum() < 8000:
            continue

        # L1 refit on train only (same depths as shipped).
        comp_fi = sd.fit_components(
            fi[tr], ts[tr],
            depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                        if k in SEASONAL_ARTIFACT["components"]["fi"]))
        seas = sd.compute_seasonal_part(ts, comp_fi)
        Yfi = fi - seas
        lag_fi = np.nan_to_num(pd.Series(Yfi, index=idx).shift(168).values)

        Yz = {}
        for z in ("temp", "se1", "se3", "ee"):
            shipped = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(
                df[z].values[tr], ts[tr],
                depth=tuple(k for k in ("P_hour", "P_day", "P_week") if k in shipped))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        Zlag = {z: np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
                for z in ("se1", "se3", "ee")}

        pw = sd.fit_components(wind_rho[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        ps = sd.fit_components(solar_eff[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        Yw = wind_rho - sd.compute_seasonal_part(ts, pw)
        Ys = solar_eff - sd.compute_seasonal_part(ts, ps)

        # The channel: what Fingrid's published forecast knows that our
        # weather proxy does not.
        rem_w = _remainder(fg_wind, Yw, tr)
        rem_s = _remainder(fg_pv, Ys, tr)

        base_cols = [np.ones(n), lag_fi, is_wd, Yw, Ys, Yz["temp"],
                     Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol]
        variants = {
            "BASE":       base_cols,
            "FG_ALL":     base_cols + [rem_w, rem_s],
            "FG_D1":      base_cols + [rem_w, rem_s],
            "FG_D1_FADE": base_cols + [rem_w, rem_s],
            "FG_SUBST":   [np.ones(n), lag_fi, is_wd, fg_wind, fg_pv,
                           Yz["temp"], Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol],
        }
        # Sign constraint on the zero-marginal-cost columns (indices 3, 4).
        coefs = {}
        for k, cols in variants.items():
            X = np.column_stack(cols)
            ok = tr & np.isfinite(X).all(1) & np.isfinite(Yfi)
            coefs[k] = (sfit(X[ok], Yfi[ok], 1.0, {3: 0.0, 4: 0.0}), X)

        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D", tz="UTC"):
            o = o.replace(hour=ORIGIN_HOUR)
            if o not in pos:
                continue
            i0 = pos[o]
            sl = np.arange(i0 + 1, i0 + 1 + HORIZON)
            if sl[-1] >= n:
                continue
            lead = np.arange(1, HORIZON + 1)          # hours ahead of origin
            for k, (c, X) in coefs.items():
                Xi = X[sl].copy()
                if k in ("FG_D1", "FG_D1_FADE"):
                    # Fingrid's published forecast runs out at DA_HOURS.
                    w = np.ones(HORIZON)
                    past = lead > DA_HOURS
                    if k == "FG_D1":
                        w[past] = 0.0
                    else:
                        w[past] = 0.5 ** ((lead[past] - DA_HOURS) / FADE_HALFLIFE_H)
                    Xi[:, -2] *= w
                    Xi[:, -1] *= w
                pr = np.maximum(seas[sl] + Xi @ c, -5.0)
                if not np.isfinite(pr).all():
                    continue
                rows.append(pd.DataFrame({
                    "v": k, "err": pr - fi[sl], "act": fi[sl],
                    "lead_d": np.minimum((lead - 1) // 24 + 1, 8),
                    "month": pd.DatetimeIndex(idx[sl]).tz_convert(
                        "Europe/Helsinki").strftime("%Y-%m"),
                    "wd": np.asarray(pd.DatetimeIndex(idx[sl]).tz_convert(
                        "Europe/Helsinki").weekday < 5),
                }))

    r = pd.concat(rows, ignore_index=True)
    order = ["BASE", "FG_ALL", "FG_D1", "FG_D1_FADE", "FG_SUBST"]
    base_mae = r[r.v == "BASE"].err.abs().mean()

    print("=" * 74)
    print("WP2.5 — Fingrid day-ahead channel, origin 06:00 UTC, 170 h horizon")
    print(f"eval {r.month.min()} .. {r.month.max()}   "
          f"{len(r[r.v=='BASE']):,} scored hours per variant")
    print("=" * 74)
    print(f"\n  {'variant':12s} {'MAE':>7s} {'bias':>7s} {'summer':>7s} "
          f"{'wd peak':>8s} {'vs BASE':>8s}")
    for k in order:
        g = r[r.v == k]
        s = g[g.month.str.endswith(("-06", "-07", "-08"))]
        pk = g[g.wd]
        print(f"  {k:12s} {g.err.abs().mean():7.2f} {g.err.mean():+7.2f} "
              f"{s.err.abs().mean():7.2f} {pk.err.abs().mean():8.2f} "
              f"{(1 - g.err.abs().mean() / base_mae) * 100:+7.1f}%")

    print("\n  MAE by lead-time day (the channel only reaches D+1/D+2):")
    print(f"  {'lead':>6s} " + " ".join(f"{k:>11s}" for k in order))
    for L in range(1, 8):
        cells = []
        for k in order:
            g = r[(r.v == k) & (r.lead_d == L)]
            cells.append(f"{g.err.abs().mean():11.2f}" if len(g) else " " * 11)
        print(f"  D+{L:<4d} " + " ".join(cells))

    print("\n  Weekday bias by month (BASE vs best shippable):")
    print(f"  {'month':9s} {'actual':>8s} {'BASE':>9s} {'FG_D1_FADE':>12s}")
    for mo in sorted(r.month.unique()):
        a = r[(r.v == "BASE") & (r.month == mo) & r.wd]
        b = r[(r.v == "FG_D1_FADE") & (r.month == mo) & r.wd]
        if len(a) < 100:
            continue
        print(f"  {mo:9s} {a.act.mean():8.1f} {a.err.mean():+9.2f} "
              f"{b.err.mean():+12.2f}")

    out = {k: {"mae": float(r[r.v == k].err.abs().mean()),
               "bias": float(r[r.v == k].err.mean())} for k in order}
    (RESULTS / "exp_fingrid_dayahead_channel.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote studies/results/exp_fingrid_dayahead_channel.json")


if __name__ == "__main__":
    main()
