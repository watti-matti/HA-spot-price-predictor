"""Does a variance-stabilising transform help this model? (WP2a)

Literature
----------
* Lucia & Schwartz (2002), Rev. Derivatives Research 5(1):5-50 — the
  canonical Nord Pool model works in LOG price with a deterministic
  seasonal function, i.e. seasonality multiplicative in levels. Same
  market, same seasonal structure as ours.
* Koopman, Ooms & Carnero (2007), JASA 102(477):16-27 — periodic
  seasonal Reg-ARFIMA-GARCH on Nord Pool among others: seasonality lives
  in the conditional VARIANCE as well as the mean.
* Uniejewski, Weron & Ziel (2018), IEEE Trans. Power Systems
  33(2):2219-2229 — variance-stabilising transforms for EPF. The log
  transform is standard but INFEASIBLE for series with near-zero or
  negative prices, which is exactly the Finnish summer. They recommend
  asinh after median/MAD standardisation.
* Uniejewski et al. (2025), arXiv:2511.13603 — VSTs are "especially
  valuable in volatile regimes" driven by renewable penetration, and a
  parametrised asinh beats the standard form.

Why asinh and not log here
--------------------------
FI summer prices reach zero and go negative. `log(p + c)` needs an
offset, and a direct log-space refit measured +11.5 % WORSE earlier in
this investigation. asinh is defined on the whole real line and is
asymptotically linear near zero, logarithmic in the tails — the property
that stabilises spike variance without breaking on negatives.

    y = asinh((p - a) / b),      a = median(train), b = 1.4826 * MAD(train)
    p = sinh(y) * b + a

Why no smearing correction
--------------------------
E[f^-1(Y)] != f^-1(E[Y]) in general, so back-transforming a conditional
mean is biased for the conditional mean. But the harness scores MAE, and
the MAE-optimal point forecast is the conditional MEDIAN, which the
back-transform of a symmetric conditional mean approximates directly.
VST + MAE is coherent; VST + RMSE would need Duan smearing.

Variants
--------
  BASE            linear space (what ships)
  ASINH           L1 and L2 both fitted in asinh space, inverted on output
  ASINH_L2        L1 in linear space, only the L2 residual in asinh space
                    (less invasive — no change to the seasonal artifact)
  LOG20           log(p + 20) for reference; expected to lose

Evaluation is origin-based (06:00 UTC daily, 170 h horizon), matching
exp_fingrid_dayahead_channel so the two results are comparable.

Run:  python studies/exp_asinh_vst.py
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

import seasonal_decomposition as sd                      # noqa: E402
from holidays import build_holiday_set                   # noqa: E402
from exp_extra_features import build_dataframe, SEASONAL_ARTIFACT  # noqa: E402
from honest_horizon_study import sfit                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = REPO / "studies" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
EVAL_START = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_FROM = pd.Timestamp("2023-01-01", tz="UTC")
HORIZON = 170
ORIGIN_HOUR = 6
LOG_OFFSET = 20.0
FLOOR = -5.0


def vst_params(x: np.ndarray) -> tuple[float, float]:
    a = float(np.median(x))
    b = float(np.median(np.abs(x - a)) * 1.4826) or 1.0
    return a, b


def fwd(x, kind, a, b):
    if kind == "asinh":
        return np.arcsinh((x - a) / b)
    if kind == "log":
        return np.log(np.maximum(x, -LOG_OFFSET + 1e-6) + LOG_OFFSET)
    return x


def inv(y, kind, a, b):
    if kind == "asinh":
        return np.sinh(y) * b + a
    if kind == "log":
        return np.exp(y) - LOG_OFFSET
    return y


def combined_with_corrector(df, idx, ts, is_wd, is_hol, fi, wind_rho,
                            solar_eff, depth_fi) -> dict:
    """Non-overlapping hourly replay of BASE and ASINH, each fed through
    the SHIPPED PerHourBiasCorrector exactly as the coordinator does."""
    import importlib.util
    import types
    spp = REPO / "custom_components" / "spot_price_predictor"
    pkg = types.ModuleType("spp_vst")
    pkg.__path__ = [str(spp)]
    sys.modules["spp_vst"] = pkg
    for m in ("bias_corrector", "dtaci", "hourly_calibration"):
        sp = importlib.util.spec_from_file_location(f"spp_vst.{m}", spp / f"{m}.py")
        mod = importlib.util.module_from_spec(sp)
        sys.modules[f"spp_vst.{m}"] = mod
        sp.loader.exec_module(mod)
    hc = sys.modules["spp_vst.hourly_calibration"]

    n = len(idx)
    pred = {k: np.full(n, np.nan) for k in ("BASE", "ASINH")}
    for m0 in pd.date_range(EVAL_START, idx[-1], freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        blk = np.asarray((idx >= m0) & (idx < m1))
        if tr.sum() < 8000 or not blk.any():
            continue
        Yz = {}
        for z in ("temp", "se1", "se3", "ee"):
            sh = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(df[z].values[tr], ts[tr],
                                  depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                                              if k in sh))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        Zl = {z: np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
              for z in ("se1", "se3", "ee")}
        pw = sd.fit_components(wind_rho[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        ps = sd.fit_components(solar_eff[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        Yw = wind_rho - sd.compute_seasonal_part(ts, pw)
        Ys = solar_eff - sd.compute_seasonal_part(ts, ps)
        for k in ("BASE", "ASINH"):
            if k == "ASINH":
                a, b = vst_params(fi[tr])
                tgt_full = fwd(fi, "asinh", a, b)
            else:
                a, b, tgt_full = 0.0, 1.0, fi
            comp = sd.fit_components(tgt_full[tr], ts[tr], depth=depth_fi)
            seas = sd.compute_seasonal_part(ts, comp)
            tgt = tgt_full - seas
            lag = np.nan_to_num(pd.Series(tgt, index=idx).shift(168).values)
            X = np.column_stack([np.ones(n), lag, is_wd, Yw, Ys, Yz["temp"],
                                 Zl["se1"], Zl["se3"], Zl["ee"], is_hol])
            ok = tr & np.isfinite(X).all(1) & np.isfinite(tgt)
            c = sfit(X[ok], tgt[ok], 1.0, {3: 0.0, 4: 0.0})
            yh = seas[blk] + X[blk] @ c
            pred[k][blk] = np.maximum(
                inv(yh, "asinh", a, b) if k == "ASINH" else yh, FLOOR)

    m = np.isfinite(pred["BASE"]) & np.isfinite(pred["ASINH"])
    sub, act = idx[m], fi[m]
    lt = sub.tz_convert("Europe/Helsinki")
    res = {}
    for k in ("BASE", "ASINH"):
        for corr in (False, True):
            p = pred[k][m]
            if corr:
                bc = hc.PerHourBiasCorrector()
                o = np.empty(len(p))
                for i in range(len(p)):
                    o[i] = bc.correct(float(p[i]), int(sub[i].hour))
                    bc.update(float(p[i]), float(act[i]), int(sub[i].hour))
                p = o
            e = pd.Series(p - act, index=sub)
            res[k + (" + corrector" if corr else " raw")] = {
                "mae": float(e.abs().mean()), "bias": float(e.mean()),
                "monthly_bias": float(
                    e.groupby(lt.strftime("%Y-%m")).mean().abs().mean()),
                "jul_wd": float(e[(lt.year == 2026) & (lt.month == 7)
                                  & (lt.weekday < 5)].mean()),
            }
    return res


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
    depth_fi = tuple(k for k in ("P_hour", "P_day", "P_week")
                     if k in SEASONAL_ARTIFACT["components"]["fi"])

    variants = {"BASE": ("none", False), "ASINH": ("asinh", False),
                "ASINH_L2": ("asinh", True), "LOG20": ("log", False)}
    rows: list[pd.DataFrame] = []

    for m0 in pd.date_range(EVAL_START, idx[-1], freq="MS", tz="UTC"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        tr = np.asarray((idx < m0) & (idx >= TRAIN_FROM))
        if tr.sum() < 8000:
            continue

        # Exogenous block is identical across variants — only the target
        # space differs, so the delta is the transform and nothing else.
        Yz = {}
        for z in ("temp", "se1", "se3", "ee"):
            shipped = SEASONAL_ARTIFACT["components"][z]
            c = sd.fit_components(df[z].values[tr], ts[tr],
                                  depth=tuple(k for k in ("P_hour", "P_day", "P_week")
                                              if k in shipped))
            Yz[z] = df[z].values - sd.compute_seasonal_part(ts, c)
        Zlag = {z: np.nan_to_num(pd.Series(Yz[z], index=idx).shift(168).values)
                for z in ("se1", "se3", "ee")}
        pw = sd.fit_components(wind_rho[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        ps = sd.fit_components(solar_eff[tr], ts[tr], depth=("P_hour", "P_week"),
                               smooth={"P_week": 7})
        Yw = wind_rho - sd.compute_seasonal_part(ts, pw)
        Ys = solar_eff - sd.compute_seasonal_part(ts, ps)

        fitted = {}
        for name, (kind, l2_only) in variants.items():
            a, b = vst_params(fi[tr]) if kind != "none" else (0.0, 1.0)
            if l2_only:
                # L1 stays linear; only the residual is transformed.
                comp = sd.fit_components(fi[tr], ts[tr], depth=depth_fi)
                seas_lin = sd.compute_seasonal_part(ts, comp)
                resid = fi - seas_lin
                ar, br = vst_params(resid[tr])
                target = fwd(resid, kind, ar, br)
                seas_t = np.zeros(n)
                back = ("resid", kind, ar, br, seas_lin)
            else:
                z = fwd(fi, kind, a, b)
                comp = sd.fit_components(z[tr], ts[tr], depth=depth_fi)
                seas_t = sd.compute_seasonal_part(ts, comp)
                target = z - seas_t
                back = ("full", kind, a, b, None)
            lag = np.nan_to_num(pd.Series(target, index=idx).shift(168).values)
            X = np.column_stack([np.ones(n), lag, is_wd, Yw, Ys, Yz["temp"],
                                 Zlag["se1"], Zlag["se3"], Zlag["ee"], is_hol])
            ok = tr & np.isfinite(X).all(1) & np.isfinite(target)
            c = sfit(X[ok], target[ok], 1.0, {3: 0.0, 4: 0.0})
            fitted[name] = (c, X, seas_t, back)

        for o in pd.date_range(m0, m1 - pd.Timedelta(hours=1), freq="D", tz="UTC"):
            o = o.replace(hour=ORIGIN_HOUR)
            if o not in pos:
                continue
            i0 = pos[o]
            sl = np.arange(i0 + 1, i0 + 1 + HORIZON)
            if sl[-1] >= n:
                continue
            for name, (c, X, seas_t, back) in fitted.items():
                mode, kind, aa, bb, seas_lin = back
                yhat = seas_t[sl] + X[sl] @ c
                if mode == "resid":
                    pr = seas_lin[sl] + inv(yhat, kind, aa, bb)
                else:
                    pr = inv(yhat, kind, aa, bb)
                pr = np.maximum(pr, FLOOR)
                if not np.isfinite(pr).all():
                    continue
                lt = pd.DatetimeIndex(idx[sl]).tz_convert("Europe/Helsinki")
                rows.append(pd.DataFrame({
                    "v": name, "err": pr - fi[sl], "act": fi[sl],
                    "pred": pr, "day": lt.date,
                    "month": lt.strftime("%Y-%m"),
                    "wd": np.asarray(lt.weekday < 5),
                    "lh": lt.hour,
                }))

    r = pd.concat(rows, ignore_index=True)
    order = [k for k in variants if (r.v == k).any()]
    base = r[r.v == "BASE"].err.abs().mean()

    print("=" * 76)
    print("WP2a — variance-stabilising transform, origin 06:00 UTC, 170 h horizon")
    print(f"eval {r.month.min()} .. {r.month.max()}   "
          f"{len(r[r.v == 'BASE']):,} scored hours per variant")
    print("=" * 76)
    print(f"\n  {'variant':10s} {'MAE':>7s} {'bias':>8s} {'summer':>7s} "
          f"{'wd peak':>8s} {'|mth bias|':>11s} {'vs BASE':>8s}")
    for k in order:
        g = r[r.v == k]
        s = g[g.month.str.endswith(("-06", "-07", "-08"))]
        pk = g[g.wd & g.lh.isin([7, 8, 9, 10, 17, 18, 19, 20, 21])]
        mb = g[g.wd].groupby("month").err.mean().abs().mean()
        print(f"  {k:10s} {g.err.abs().mean():7.2f} {g.err.mean():+8.2f} "
              f"{s.err.abs().mean():7.2f} {pk.err.abs().mean():8.2f} "
              f"{mb:11.2f} {(1 - g.err.abs().mean() / base) * 100:+7.1f}%")

    print("\n  Weekday bias by month:")
    print(f"  {'month':9s} {'actual':>8s} " + " ".join(f"{k:>10s}" for k in order))
    for mo in sorted(r.month.unique()):
        a = r[(r.v == "BASE") & (r.month == mo) & r.wd]
        if len(a) < 100:
            continue
        cells = " ".join(
            f"{r[(r.v == k) & (r.month == mo) & r.wd].err.mean():+10.2f}"
            for k in order)
        print(f"  {mo:9s} {a.act.mean():8.1f} " + cells)

    # Koopman's point: does a global transform fix the CONDITIONAL spread?
    print("\n  Daily amplitude law  (actual vs each variant):  amp = a + b x level")
    for k in ["ACTUAL"] + order:
        src = r[r.v == order[0]] if k == "ACTUAL" else r[r.v == k]
        col = "act" if k == "ACTUAL" else "pred"
        d = src.groupby("day").agg(lvl=("act", "mean"),
                                   amp=(col, lambda s: s.max() - s.min()))
        A = np.column_stack([np.ones(len(d)), d.lvl.values])
        b, *_ = np.linalg.lstsq(A, d.amp.values, rcond=None)
        print(f"    {k:10s} {b[0]:7.2f} + {b[1]:6.3f} x level")

    print()
    print("=" * 76)
    print("With the shipped per-hour bias corrector on top")
    print("=" * 76)
    print("  The VST predicts a conditional MEDIAN, which is MAE-optimal but")
    print("  sits below the mean of a right-skewed price. That shows up as a")
    print("  raw low bias. The question that decides shipping is whether the")
    print("  v2.18.0 corrector (3-day half-life) absorbs it.")
    print()
    combined = combined_with_corrector(df, idx, ts, is_wd, is_hol, fi,
                                       wind_rho, solar_eff, depth_fi)
    print(f"  {'config':26s} {'MAE':>7s} {'bias':>8s} {'|mth bias|':>11s} "
          f"{'2026-07 wd':>11s}")
    for lbl, v in combined.items():
        print(f"  {lbl:26s} {v['mae']:7.2f} {v['bias']:+8.2f} "
              f"{v['monthly_bias']:11.2f} {v['jul_wd']:+11.2f}")

    out = {k: {"mae": float(r[r.v == k].err.abs().mean()),
               "bias": float(r[r.v == k].err.mean())} for k in order}
    out["with_corrector"] = combined
    (RESULTS / "exp_asinh_vst.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")
    print("\nWrote studies/results/exp_asinh_vst.json")


if __name__ == "__main__":
    main()
