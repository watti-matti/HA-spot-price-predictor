"""
Decision-gate validation: AR(2) vs SARIMAX for SE1/SE3/EE neighbor prices.

Trains both models on 2022-04 -> 2025-12-31 and walks forward through
2026 holdout, scoring 168h forecasts against actual prices.

Decision criteria for SARIMAX migration:
  1. SARIMAX hourly MAE <= AR(2) MAE on at least 2 of 3 zones
  2. SARIMAX dk_cheap[3] (cheap 4h) MAE < AR(2) on all 3 zones
  3. SARIMAX weekend MAE < AR(2) weekend MAE on at least 2 of 3 zones

Usage:
    python -m studies.validate_neighbor_models [--anchors 4] [--horizon 168]
                                                [--quick] [--zones se1,se3,ee]

  --quick       Use 1 anchor per zone (smoke-test); ~5 min runtime
  --anchors N   Number of walk-forward anchor points spread through holdout (default 4)
  --horizon H   Forecast horizon in hours (default 168 = 1 week)
  --zones LIST  Comma-separated zones to evaluate (default: se1,se3,ee)
  --train-end YYYY-MM-DD   End of training window (default 2025-12-31)
  --holdout-start YYYY-MM-DD   Start of holdout (default 2026-01-01)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project root on path for `import src.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dk_utils import compute_dk_cheap_peak  # noqa: E402

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# --- Holiday sets per country (cached for efficiency) -----------------------------------

_HOLIDAY_SETS: dict[str, set] = {}


def _holiday_set(country: str, years: list[int]) -> set:
    key = f"{country}:{min(years)}-{max(years)}"
    if key not in _HOLIDAY_SETS:
        import holidays as _holidays
        country_map = {"SE": _holidays.Sweden, "EE": _holidays.Estonia, "FI": _holidays.Finland}
        cls = country_map.get(country.upper(), _holidays.Sweden)
        _HOLIDAY_SETS[key] = set(cls(years=years))
    return _HOLIDAY_SETS[key]


# --- Forecast helpers -----------------------------------------------------------------


def _ar2_forecast(
    train_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
    fi_holidays: set[str],
) -> pd.Series:
    """Train AR(2) per src/features.py:build_ar_models, then forecast."""
    from src.features import build_ar_models

    # Build models (this re-implements the calling convention)
    ar_models = build_ar_models(
        neighbor_prices={"x": train_series},
        df_index=train_series.index,
        holidays=fi_holidays,
    )
    if "x" not in ar_models:
        return pd.Series([np.nan] * len(forecast_index), index=forecast_index)
    m = ar_models["x"]
    profile_wd = m["profile_wd"]
    profile_we = m["profile_we"]
    ar_c = m["ar_coefs"]

    # Local time for hour-of-day and workday flags
    try:
        from zoneinfo import ZoneInfo
        local_idx = forecast_index.tz_convert(ZoneInfo("Europe/Helsinki"))
    except Exception:
        local_idx = forecast_index + pd.Timedelta(hours=3)

    # Initialize deviation from last 2 train values vs their profile
    last_two = train_series.dropna().iloc[-2:].values
    if len(last_two) < 2:
        dev_t1 = dev_t2 = 0.0
    else:
        last_local = train_series.dropna().index[-1]
        try:
            ll = pd.DatetimeIndex([last_local]).tz_convert(ZoneInfo("Europe/Helsinki"))[0]
        except Exception:
            ll = last_local + pd.Timedelta(hours=3)
        h = ll.hour
        wd = (ll.dayofweek < 5) and (ll.strftime("%Y-%m-%d") not in fi_holidays)
        prof = profile_wd[h] if wd else profile_we[h]
        dev_t1 = float(last_two[-1]) - prof
        dev_t2 = float(last_two[-2]) - prof

    preds = []
    for ts in local_idx:
        h = ts.hour
        wd = (ts.dayofweek < 5) and (ts.strftime("%Y-%m-%d") not in fi_holidays)
        profile = profile_wd[h] if wd else profile_we[h]
        dev_new = ar_c[0] * dev_t1 + ar_c[1] * dev_t2
        preds.append(max(0.0, profile + dev_new))
        dev_t2 = dev_t1
        dev_t1 = dev_new

    return pd.Series(preds, index=forecast_index, name="ar2")


def _sarimax_forecast(
    train_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
    country: str,
    *,
    order: tuple = (2, 0, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
    exog_mode: str = "fourier",
) -> pd.Series:
    """Train HourlyNordPoolSARIMAX, then forecast."""
    from src.sarimax_neighbor import HourlyNordPoolSARIMAX

    m = HourlyNordPoolSARIMAX(
        country=country,
        order=order,
        seasonal_order=seasonal_order,
        diurnal_K=2,
        annual_K=2,
        include_weekend_interaction=True,
        exog_mode=exog_mode,
    )
    m.fit(train_series)
    fc = m.forecast(horizon=len(forecast_index))
    fc.index = forecast_index
    return fc


# --- Metric computation ---------------------------------------------------------------


def _per_day_dk_metrics(actuals: pd.Series, preds: pd.Series) -> dict[str, list[float]]:
    """Compute dk_cheap and dk_peak MAE per k for each forecast day.

    Both series should be hourly. Returns:
      cheap_mae[k=0..11], peak_mae[k=0..11], cheap_bias, peak_bias
    """
    # Group by local date
    try:
        from zoneinfo import ZoneInfo
        local = actuals.index.tz_convert(ZoneInfo("Europe/Helsinki"))
    except Exception:
        local = actuals.index + pd.Timedelta(hours=3)
    dates = local.strftime("%Y-%m-%d")

    cheap_errs: list[list[float]] = [[] for _ in range(12)]
    peak_errs:  list[list[float]] = [[] for _ in range(12)]

    for d in pd.unique(dates):
        mask = (dates == d)
        if mask.sum() != 24:
            continue
        a = actuals.values[mask]
        p = preds.values[mask]
        if np.any(np.isnan(a)) or np.any(np.isnan(p)):
            continue
        a_cheap, a_peak = compute_dk_cheap_peak(list(a))
        p_cheap, p_peak = compute_dk_cheap_peak(list(p))
        for k in range(12):
            cheap_errs[k].append(p_cheap[k] - a_cheap[k])
            peak_errs[k].append(p_peak[k] - a_peak[k])

    cheap_mae = [float(np.mean(np.abs(e))) if e else float("nan") for e in cheap_errs]
    peak_mae = [float(np.mean(np.abs(e))) if e else float("nan") for e in peak_errs]
    cheap_bias = [float(np.mean(e)) if e else float("nan") for e in cheap_errs]
    peak_bias = [float(np.mean(e)) if e else float("nan") for e in peak_errs]
    return {
        "cheap_mae": cheap_mae,
        "peak_mae": peak_mae,
        "cheap_bias": cheap_bias,
        "peak_bias": peak_bias,
        "n_days": len(cheap_errs[0]),
    }


def _scalar_metrics(
    actuals: pd.Series,
    preds: pd.Series,
    country: str,
) -> dict[str, float]:
    """Compute hourly-level MAE/RMSE plus weekend / holiday isolation."""
    a = actuals.values
    p = preds.values
    err = p - a
    mask_finite = np.isfinite(a) & np.isfinite(p)
    a, p, err = a[mask_finite], p[mask_finite], err[mask_finite]

    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "spearman_rho": float(_spearman(a, p)),
    }

    try:
        from zoneinfo import ZoneInfo
        local = actuals.index[mask_finite].tz_convert(ZoneInfo("Europe/Helsinki"))
    except Exception:
        local = actuals.index[mask_finite] + pd.Timedelta(hours=3)

    # Weekend mask (Sat=5, Sun=6)
    we_mask = local.dayofweek.values >= 5
    if we_mask.sum() > 0:
        out["mae_weekend"] = float(np.mean(np.abs(err[we_mask])))
        out["mae_weekday"] = float(np.mean(np.abs(err[~we_mask]))) if (~we_mask).sum() > 0 else float("nan")

    # Holiday mask
    years = sorted(set(local.year))
    hol = _holiday_set(country, years)
    hol_iso = {h.strftime("%Y-%m-%d") if hasattr(h, "strftime") else str(h) for h in hol}
    hol_mask = np.array([d.strftime("%Y-%m-%d") in hol_iso for d in local])
    if hol_mask.sum() > 0:
        out["mae_holiday"] = float(np.mean(np.abs(err[hol_mask])))
        out["n_holiday_hours"] = int(hol_mask.sum())
    else:
        out["mae_holiday"] = float("nan")
        out["n_holiday_hours"] = 0

    out["n_obs"] = int(mask_finite.sum())
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy dependency."""
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    return float((rx_c * ry_c).sum() / denom) if denom > 0 else float("nan")


# --- Main validation loop -------------------------------------------------------------


def validate(
    zones: list[str],
    train_end: pd.Timestamp,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
    n_anchors: int,
    horizon: int,
    out_dir: Path,
    *,
    exog_mode: str = "fourier",
    seasonal_order: tuple = (0, 0, 0, 0),
) -> dict[str, Any]:
    """Run head-to-head AR(2) vs SARIMAX validation."""
    # Load cached neighbor prices
    np_path = Path("output/fi_neighbor_prices.parquet")
    if not np_path.exists():
        raise FileNotFoundError(
            f"{np_path} not found. Run training pipeline first to populate cache."
        )
    df = pd.read_parquet(np_path)

    # Resample to hourly (some sources have 15-min granularity at the tail)
    df = df.resample("1h").mean()
    df = df.dropna(how="all")

    # Country mapping
    country_map = {"se1": "SE", "se3": "SE", "ee": "EE"}

    # FI holidays for AR(2) workday flag (matches existing code path)
    fi_years = list(range(2022, 2027))
    fi_hol_set = {h.strftime("%Y-%m-%d") for h in _holiday_set("FI", fi_years)}

    # Pick anchor timestamps spread through holdout
    holdout_idx = df.index[(df.index >= holdout_start) & (df.index <= holdout_end)]
    if len(holdout_idx) == 0:
        raise ValueError(f"No data in holdout window {holdout_start} -> {holdout_end}")
    n_anchors = max(1, min(n_anchors, len(holdout_idx) // (horizon + 24)))

    # Anchors: spread through holdout, leaving room for `horizon` hours of forecast
    last_anchor_idx = max(0, len(holdout_idx) - horizon)
    if n_anchors == 1:
        anchor_positions = [last_anchor_idx // 2]
    else:
        anchor_positions = list(np.linspace(0, last_anchor_idx, n_anchors).astype(int))
    anchors = [holdout_idx[p] for p in anchor_positions]
    logger.info("Walk-forward anchors: %s", [a.strftime("%Y-%m-%d %H:%M") for a in anchors])

    results: dict[str, Any] = {"zones": {}, "summary": {}}

    for zone in zones:
        if zone not in df.columns:
            logger.warning("Zone %s not in cached data, skipping", zone)
            continue
        country = country_map.get(zone.lower(), "SE")
        zone_series = df[zone].dropna()
        logger.info(
            "=== Zone %s (country=%s) — %d hourly observations 2022-04 to %s ===",
            zone.upper(), country, len(zone_series),
            zone_series.index.max().strftime("%Y-%m-%d"),
        )

        per_anchor_ar = []
        per_anchor_sx = []

        for ai, anchor in enumerate(anchors):
            logger.info("[%s] anchor %d/%d at %s",
                        zone, ai + 1, len(anchors),
                        anchor.strftime("%Y-%m-%d %H:%M"))

            # Train slice: from start to anchor (exclusive)
            train = zone_series.loc[:anchor].iloc[:-1]
            if len(train) < 168 * 4:  # need at least 4 weeks
                logger.warning("  too little train data (%d), skipping", len(train))
                continue

            # Holdout slice: anchor to anchor + horizon
            holdout_end_anchor = anchor + pd.Timedelta(hours=horizon - 1)
            actual = zone_series.loc[anchor:holdout_end_anchor]
            if len(actual) < horizon:
                logger.warning("  truncated holdout (%d/%d), skipping", len(actual), horizon)
                continue

            # AR(2) forecast
            try:
                ar_pred = _ar2_forecast(train, actual.index, fi_hol_set)
                ar_metrics = _scalar_metrics(actual, ar_pred, country)
                ar_dk = _per_day_dk_metrics(actual, ar_pred)
                logger.info("  AR(2)   MAE=%.2f weekend=%.2f cheap[3]=%.2f peak[0]=%.2f",
                            ar_metrics["mae"], ar_metrics.get("mae_weekend", float("nan")),
                            ar_dk["cheap_mae"][3], ar_dk["peak_mae"][0])
            except Exception as e:
                logger.exception("  AR(2) failed: %s", e)
                continue

            # SARIMAX forecast
            try:
                sx_pred = _sarimax_forecast(train, actual.index, country,
                                            exog_mode=exog_mode,
                                            seasonal_order=seasonal_order)
                sx_metrics = _scalar_metrics(actual, sx_pred, country)
                sx_dk = _per_day_dk_metrics(actual, sx_pred)
                logger.info("  SARIMAX MAE=%.2f weekend=%.2f cheap[3]=%.2f peak[0]=%.2f",
                            sx_metrics["mae"], sx_metrics.get("mae_weekend", float("nan")),
                            sx_dk["cheap_mae"][3], sx_dk["peak_mae"][0])
            except Exception as e:
                logger.exception("  SARIMAX failed: %s", e)
                sx_metrics = {"mae": float("nan"), "rmse": float("nan"),
                              "bias": float("nan"), "spearman_rho": float("nan")}
                sx_dk = {"cheap_mae": [float("nan")] * 12, "peak_mae": [float("nan")] * 12,
                         "cheap_bias": [float("nan")] * 12, "peak_bias": [float("nan")] * 12,
                         "n_days": 0}

            per_anchor_ar.append({"anchor": anchor.isoformat(), **ar_metrics, **{f"dk_{k}": v for k, v in ar_dk.items()}})
            per_anchor_sx.append({"anchor": anchor.isoformat(), **sx_metrics, **{f"dk_{k}": v for k, v in sx_dk.items()}})

        if not per_anchor_ar:
            continue

        # Aggregate across anchors
        def agg(metrics_list, key, op="mean"):
            vals = [m[key] for m in metrics_list if not np.isnan(m.get(key, float("nan")))]
            if not vals:
                return float("nan")
            return float(np.mean(vals)) if op == "mean" else float(np.median(vals))

        def agg_list(metrics_list, key):
            vals = [m[key] for m in metrics_list]
            return [float(np.nanmean([v[k] for v in vals])) for k in range(12)]

        results["zones"][zone] = {
            "country": country,
            "n_anchors": len(per_anchor_ar),
            "ar2": {
                "mae": agg(per_anchor_ar, "mae"),
                "rmse": agg(per_anchor_ar, "rmse"),
                "bias": agg(per_anchor_ar, "bias"),
                "spearman": agg(per_anchor_ar, "spearman_rho"),
                "mae_weekend": agg(per_anchor_ar, "mae_weekend"),
                "mae_weekday": agg(per_anchor_ar, "mae_weekday"),
                "mae_holiday": agg(per_anchor_ar, "mae_holiday"),
                "dk_cheap_mae": agg_list(per_anchor_ar, "dk_cheap_mae"),
                "dk_peak_mae":  agg_list(per_anchor_ar, "dk_peak_mae"),
                "dk_cheap_bias": agg_list(per_anchor_ar, "dk_cheap_bias"),
                "dk_peak_bias":  agg_list(per_anchor_ar, "dk_peak_bias"),
            },
            "sarimax": {
                "mae": agg(per_anchor_sx, "mae"),
                "rmse": agg(per_anchor_sx, "rmse"),
                "bias": agg(per_anchor_sx, "bias"),
                "spearman": agg(per_anchor_sx, "spearman_rho"),
                "mae_weekend": agg(per_anchor_sx, "mae_weekend"),
                "mae_weekday": agg(per_anchor_sx, "mae_weekday"),
                "mae_holiday": agg(per_anchor_sx, "mae_holiday"),
                "dk_cheap_mae": agg_list(per_anchor_sx, "dk_cheap_mae"),
                "dk_peak_mae":  agg_list(per_anchor_sx, "dk_peak_mae"),
                "dk_cheap_bias": agg_list(per_anchor_sx, "dk_cheap_bias"),
                "dk_peak_bias":  agg_list(per_anchor_sx, "dk_peak_bias"),
            },
            "per_anchor_ar": per_anchor_ar,
            "per_anchor_sx": per_anchor_sx,
        }

    # Decision criteria
    n_zones = len(results["zones"])
    if n_zones > 0:
        zones_with_better_mae = sum(
            1 for z in results["zones"].values()
            if z["sarimax"]["mae"] <= z["ar2"]["mae"]
        )
        zones_with_better_cheap4 = sum(
            1 for z in results["zones"].values()
            if z["sarimax"]["dk_cheap_mae"][3] < z["ar2"]["dk_cheap_mae"][3]
        )
        zones_with_better_weekend = sum(
            1 for z in results["zones"].values()
            if (z["sarimax"]["mae_weekend"] < z["ar2"]["mae_weekend"])
        )
        decision_migrate = (
            zones_with_better_mae >= 2
            and zones_with_better_cheap4 == n_zones
            and zones_with_better_weekend >= 2
        )
        results["summary"] = {
            "n_zones_evaluated": n_zones,
            "zones_better_mae": zones_with_better_mae,
            "zones_better_cheap4": zones_with_better_cheap4,
            "zones_better_weekend": zones_with_better_weekend,
            "decision_migrate_to_sarimax": decision_migrate,
        }

    return results


def write_report(results: dict[str, Any], out_path: Path) -> None:
    """Render results as a markdown report."""
    lines: list[str] = []
    lines.append(f"# AR(2) vs SARIMAX Neighbor Model Validation\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    summary = results.get("summary", {})
    if summary:
        lines.append("## Decision Summary\n")
        lines.append(f"- Zones evaluated: {summary['n_zones_evaluated']}")
        lines.append(f"- Zones where SARIMAX beats AR(2) hourly MAE: "
                     f"{summary['zones_better_mae']}/{summary['n_zones_evaluated']} "
                     f"(criterion: ≥2)")
        lines.append(f"- Zones where SARIMAX beats AR(2) `dk_cheap[3]` (cheap 4h): "
                     f"{summary['zones_better_cheap4']}/{summary['n_zones_evaluated']} "
                     f"(criterion: all)")
        lines.append(f"- Zones where SARIMAX beats AR(2) weekend MAE: "
                     f"{summary['zones_better_weekend']}/{summary['n_zones_evaluated']} "
                     f"(criterion: ≥2)")
        decision = summary["decision_migrate_to_sarimax"]
        marker = "✅ MIGRATE" if decision else "❌ KEEP AR(2)"
        lines.append(f"\n**Decision: {marker}**\n")

    for zone, z in results.get("zones", {}).items():
        lines.append(f"\n## Zone: {zone.upper()} (country={z['country']}, "
                     f"n_anchors={z['n_anchors']})\n")

        lines.append("### Hourly forecast metrics\n")
        lines.append("| Metric | AR(2) | SARIMAX | Δ (sx-ar2) |")
        lines.append("|--------|-------|---------|------------|")
        for key, label in [
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("bias", "Bias"),
            ("spearman", "Spearman ρ"),
            ("mae_weekend", "MAE weekend"),
            ("mae_weekday", "MAE weekday"),
            ("mae_holiday", "MAE holiday"),
        ]:
            ar = z["ar2"].get(key, float("nan"))
            sx = z["sarimax"].get(key, float("nan"))
            delta = sx - ar if (not np.isnan(ar) and not np.isnan(sx)) else float("nan")
            lines.append(f"| {label} | {ar:.3f} | {sx:.3f} | {delta:+.3f} |")

        lines.append("\n### dk_cheap MAE (cheapest k hours)\n")
        lines.append("| k | AR(2) | SARIMAX | Δ |")
        lines.append("|---|-------|---------|---|")
        for k in range(12):
            ar = z["ar2"]["dk_cheap_mae"][k]
            sx = z["sarimax"]["dk_cheap_mae"][k]
            d = sx - ar
            lines.append(f"| {k+1} | {ar:.2f} | {sx:.2f} | {d:+.2f} |")

        lines.append("\n### dk_peak MAE (priciest k hours)\n")
        lines.append("| k | AR(2) | SARIMAX | Δ |")
        lines.append("|---|-------|---------|---|")
        for k in range(12):
            ar = z["ar2"]["dk_peak_mae"][k]
            sx = z["sarimax"]["dk_peak_mae"][k]
            d = sx - ar
            lines.append(f"| {k+1} | {ar:.2f} | {sx:.2f} | {d:+.2f} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", default="se1,se3,ee",
                        help="Comma-separated zones (default: se1,se3,ee)")
    parser.add_argument("--train-end", default="2025-12-31",
                        help="End of training window (default 2025-12-31)")
    parser.add_argument("--holdout-start", default="2026-01-01",
                        help="Start of holdout (default 2026-01-01)")
    parser.add_argument("--holdout-end", default=None,
                        help="End of holdout (default: today)")
    parser.add_argument("--anchors", type=int, default=4,
                        help="Walk-forward anchor points (default 4)")
    parser.add_argument("--horizon", type=int, default=168,
                        help="Forecast horizon in hours (default 168)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 anchor per zone (smoke test)")
    parser.add_argument("--out-dir", default="studies/results",
                        help="Output directory for report")
    parser.add_argument("--exog-mode", default="fourier",
                        choices=["fourier", "hour-workday", "hour-of-week"],
                        help="SARIMAX calendar exog mode (default: fourier)")
    parser.add_argument("--seasonal-period", type=int, default=0,
                        help="Seasonal period in hours (0=no seasonal state, 168=weekly)")
    parser.add_argument("--tag", default="",
                        help="Tag appended to output filenames for run identification")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if args.quick:
        args.anchors = 1

    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    holdout_start = pd.Timestamp(args.holdout_start, tz="UTC")
    if args.holdout_end:
        holdout_end = pd.Timestamp(args.holdout_end, tz="UTC")
    else:
        holdout_end = pd.Timestamp.now(tz="UTC")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Validation: train ending %s, holdout %s -> %s",
                train_end.strftime("%Y-%m-%d"),
                holdout_start.strftime("%Y-%m-%d"),
                holdout_end.strftime("%Y-%m-%d"))
    logger.info("Zones: %s, anchors: %d, horizon: %d hours",
                zones, args.anchors, args.horizon)
    logger.info("SARIMAX exog_mode: %s, seasonal_period: %d",
                args.exog_mode, args.seasonal_period)

    seasonal_order = (1, 1, 0, args.seasonal_period) if args.seasonal_period > 0 else (0, 0, 0, 0)

    results = validate(
        zones=zones,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        n_anchors=args.anchors,
        horizon=args.horizon,
        out_dir=out_dir,
        exog_mode=args.exog_mode,
        seasonal_order=seasonal_order,
    )
    results["config"] = {
        "exog_mode": args.exog_mode,
        "seasonal_order": list(seasonal_order),
        "anchors": args.anchors,
        "horizon": args.horizon,
        "train_end": str(train_end),
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
    }

    # Write JSON dump
    today_str = datetime.now().strftime("%Y%m%d_%H%M")
    tag_str = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"validation_{today_str}{tag_str}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Raw results written to %s", json_path)

    # Write markdown report
    md_path = out_dir / f"validation_{today_str}{tag_str}.md"
    write_report(results, md_path)

    # Print decision to console
    print("\n" + "=" * 60)
    summary = results.get("summary", {})
    if summary:
        decision = summary["decision_migrate_to_sarimax"]
        print(f"DECISION: {'MIGRATE TO SARIMAX' if decision else 'KEEP AR(2)'}")
        print(f"  zones better MAE:     {summary['zones_better_mae']}/{summary['n_zones_evaluated']}  (need >=2)")
        print(f"  zones better cheap4:  {summary['zones_better_cheap4']}/{summary['n_zones_evaluated']}  (need all)")
        print(f"  zones better weekend: {summary['zones_better_weekend']}/{summary['n_zones_evaluated']}  (need >=2)")
    print("=" * 60)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
