"""v2.18.0 — what the bias-corrector retune and the calendar fix are worth.

Rebuilds the production-equivalent forecast (frozen shipped L1 + L2,
`Y_fi_lag168` zeroed as the coordinator does, neighbours lagged 168 h)
from the data store, then replays the SHIPPED `PerHourBiasCorrector`
class under the old and new tuning.

Two questions, measured separately so the release notes can attribute
them:

  1. Bias corrector: half-life 14 -> 3, warm-up gate 14 -> 2, and the
     CMA -> EMA hand-over (`adaptive_init`) replacing the zero start.
  2. `is_workday` on the local calendar instead of UTC.

Run:  python studies/bias_corrector_warmup_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SPP = REPO / "custom_components" / "spot_price_predictor"
sys.path.insert(0, str(SPP))

import seasonal_decomposition as sd                      # noqa: E402
from holidays import build_holiday_set                   # noqa: E402

# hourly_calibration uses `from . import`; load it as a package member.
import importlib.util
import types

_pkg = types.ModuleType("spp")
_pkg.__path__ = [str(SPP)]
sys.modules["spp"] = _pkg
for _m in ("bias_corrector", "dtaci", "hourly_calibration"):
    _s = importlib.util.spec_from_file_location(f"spp.{_m}", SPP / f"{_m}.py")
    _mod = importlib.util.module_from_spec(_s)
    sys.modules[f"spp.{_m}"] = _mod
    _s.loader.exec_module(_mod)
hc = sys.modules["spp.hourly_calibration"]
bias_mod = sys.modules["spp.bias_corrector"]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STORE = REPO / "data_store"
TRAIN_FROM = "2023-01-01"


def _sigmoid_turbine_rho(wind, temp_c, v_mid=7.5, k_steep=1.5, rho_ref=1.225):
    wind = np.asarray(wind, dtype=float)
    sigmoid = 1.0 / (1.0 + np.exp(-(wind - v_mid) / k_steep))
    rho = 101_325.0 / (287.05 * (np.asarray(temp_c, dtype=float) + 273.15))
    return sigmoid * (rho / rho_ref)


def _solar_effective(ghi, temp_c, coeff_per_C=0.004, noct_coeff=0.03):
    ghi = np.asarray(ghi, dtype=float)
    cell = np.asarray(temp_c, dtype=float) + noct_coeff * ghi
    return ghi * (1.0 - coeff_per_C * np.maximum(0.0, cell - 25.0))


def build_production_forecast(local_workday: bool) -> pd.DataFrame:
    """Reproduce what the deployed pipeline computes, hour by hour."""
    S = json.loads((SPP / "data" / "seasonal_components_default.json")
                   .read_text(encoding="utf-8"))
    M = json.loads((SPP / "data" / "spike_model_default.json")
                   .read_text(encoding="utf-8"))
    co = dict(zip(["intercept"] + M["ridge_features"], M["ridge_coef"]))

    pr = pd.read_parquet(STORE / "fi_prices.parquet")
    we = pd.read_parquet(STORE / "fi_weather.parquet")
    nb = pd.read_parquet(STORE / "fi_neighbor_prices.parquet")
    for d in (pr, we, nb):
        d.index = pd.DatetimeIndex(d.index)
    nb = nb[~nb.index.duplicated()].resample("1h").mean()
    df = pr.join(we, how="inner").join(nb, how="left")
    df = df[df.index >= TRAIN_FROM]
    ts = df.index.values

    wr = _sigmoid_turbine_rho(df.wind_speed_weighted.values,
                              df.temperature_weighted.values)
    se = _solar_effective(df.solar_irradiance_weighted.values,
                          df.temperature_weighted.values)
    ph = M["physics_seasonal"]
    Y_wr = wr - sd.compute_seasonal_part(ts, ph["sigmoid_wind_rho"])
    Y_se = se - sd.compute_seasonal_part(ts, ph["solar_effective"])
    Y_t = (df.temperature_weighted.values
           - sd.compute_seasonal_part(ts, S["components"]["temp"]))

    loc = df.index.tz_convert("Europe/Helsinki")
    hol = build_holiday_set(2022, 2028)
    ish = np.array([1.0 if d.strftime("%Y-%m-%d") in hol else 0.0
                    for d in loc])
    wd_raw = ((loc.weekday if local_workday else df.index.weekday) < 5)
    is_wd = wd_raw.astype(float) * (1.0 - ish)

    Z = {}
    for z in ("se1", "se3", "ee"):
        raw = df[z].values.astype(float)
        raw = np.where(np.isfinite(raw), raw, np.nanmean(raw))
        Z[z] = (pd.Series(raw - sd.compute_seasonal_part(ts, S["components"][z]),
                          index=df.index).shift(168).fillna(0.0).values)

    seasonal = sd.compute_seasonal_part(ts, S["components"]["fi"])
    ridge = (co["intercept"]
             + co["is_workday"] * is_wd
             + co["Y_sigmoid_wind_rho"] * Y_wr
             + co["Y_solar_effective"] * Y_se
             + co["Y_temp"] * Y_t
             + co["Y_se1_lag168"] * Z["se1"]
             + co["Y_se3_lag168"] * Z["se3"]
             + co["Y_ee_lag168"] * Z["ee"]
             + co["is_holiday"] * ish)
    out = pd.DataFrame({
        "pred": np.maximum(seasonal + ridge, -5.0),
        "act": df.price_eur_mwh.values,
        "wd": np.asarray(loc.weekday < 5),
        "lh": loc.hour,
    }, index=df.index)
    return out.dropna()


def replay(frame: pd.DataFrame, corrector) -> np.ndarray:
    """Feed (forecast, actual) pairs through the corrector as the
    coordinator's reconciliation loop does: correct first, then learn."""
    out = np.empty(len(frame))
    p = frame.pred.values
    a = frame.act.values
    h = frame.index.hour.values
    for i in range(len(frame)):
        out[i] = corrector.correct(float(p[i]), int(h[i]))
        corrector.update(float(p[i]), float(a[i]), int(h[i]))
    return out


def _legacy() -> "hc.PerHourBiasCorrector":
    """The v2.17.3 tuning: 14-day half-life, 14-update gate, zero init."""
    obj = hc.PerHourBiasCorrector(halflife_days=14.0, warmup_updates=14,
                                  _inner={})
    obj._inner = {
        h: bias_mod.OnlineBiasCorrector(halflife_days=14.0, warmup_steps=14,
                                        winsor_limit=5.0, cadence_per_day=1,
                                        adaptive_init=False)
        for h in range(24)
    }
    return obj


def _report(name: str, frame: pd.DataFrame, corrected: np.ndarray) -> dict:
    e = pd.Series(corrected - frame.act.values, index=frame.index)
    wd = e[frame.wd.values]
    peak = e[frame.wd.values & frame.lh.isin([7, 8, 9, 10, 17, 18, 19, 20, 21]).values]
    monthly = e.groupby([e.index.year, e.index.month]).mean().abs().mean()
    jul = e[(e.index.year == 2026) & (e.index.month == 7)
            & frame.wd.values
            & frame.lh.isin([7, 8, 9, 10, 17, 18, 19, 20, 21]).values]
    return {"name": name, "mae": e.abs().mean(), "wd_mae": wd.abs().mean(),
            "peak_mae": peak.abs().mean(), "mth_bias": monthly,
            "jul_peak_bias": jul.mean()}


def main() -> None:
    utc_frame = build_production_forecast(local_workday=False)
    loc_frame = build_production_forecast(local_workday=True)

    print("=" * 78)
    print("v2.18.0 verification — production-equivalent replay, "
          f"{utc_frame.index[0].date()} .. {utc_frame.index[-1].date()}")
    print("=" * 78)

    rows = [
        _report("v2.17.3 (UTC workday, halflife 14, gate 14)",
                utc_frame, replay(utc_frame, _legacy())),
        _report("  + local workday only",
                loc_frame, replay(loc_frame, _legacy())),
        _report("  + bias retune only",
                utc_frame, replay(utc_frame, hc.PerHourBiasCorrector())),
        _report("v2.18.0 (both)",
                loc_frame, replay(loc_frame, hc.PerHourBiasCorrector())),
        _report("reference: no correction at all",
                loc_frame, loc_frame.pred.values),
    ]
    print(f"\n  {'configuration':44s} {'MAE':>7s} {'weekday':>8s} {'wd peak':>8s} "
          f"{'|mth bias|':>11s} {'2026-07 wd peak bias':>21s}")
    base = rows[0]["mae"]
    for r in rows:
        print(f"  {r['name']:44s} {r['mae']:7.2f} {r['wd_mae']:8.2f} "
              f"{r['peak_mae']:8.2f} {r['mth_bias']:11.2f} "
              f"{r['jul_peak_bias']:+21.2f}")
    print(f"\n  v2.18.0 vs v2.17.3: MAE {(1 - rows[3]['mae'] / base) * 100:+.1f} %, "
          f"weekday {(1 - rows[3]['wd_mae'] / rows[0]['wd_mae']) * 100:+.1f} %, "
          f"monthly bias {(1 - rows[3]['mth_bias'] / rows[0]['mth_bias']) * 100:+.1f} %")

    # Post-install behaviour: the state wipe on every model change.
    print("\n  Three weeks after a state wipe (the post-install case):")
    print(f"  {'configuration':44s} {'bias':>8s} {'MAE':>8s}")
    for label, corr in (("v2.17.3", _legacy()), ("v2.18.0", hc.PerHourBiasCorrector())):
        biases, maes = [], []
        for start in ("2026-05-01", "2026-06-01", "2026-07-01"):
            sub = loc_frame[loc_frame.index >= start]
            c = (_legacy() if label == "v2.17.3"
                 else hc.PerHourBiasCorrector())
            got = replay(sub, c)
            win = sub.index < pd.Timestamp(start, tz="UTC") + pd.Timedelta("21D")
            e = got[win] - sub.act.values[win]
            biases.append(e.mean())
            maes.append(np.abs(e).mean())
        print(f"  {label:44s} {np.mean(biases):+8.2f} {np.mean(maes):8.2f}")


if __name__ == "__main__":
    main()
