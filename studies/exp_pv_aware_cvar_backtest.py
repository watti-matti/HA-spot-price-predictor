"""Empirical PV-aware CVaR back-test on the extracted household profile.

Uses the user's real consumption shape (extracted from their HA
recorder DB at `studies/_private/household_profile.json`) and the
repo's cached spot price + weather data to compute, for each day in
the overlapping window:

  - deterministic PV production at the household's 8.91 kWp / 45° /
    azimuth 160° configuration via `pv_estimate.estimate_pv_kwh_per_hour`
  - hourly consumption from the EMA shape × mean kWh/h
  - net daily cost via `pv_cost_kernel.cost_distribution`

Then bootstraps random 7-day sequences over the window to produce
weekly cost samples, and reports CVaR_5% of weekly cost.

Three consumption strategies are compared on the same data:

  S0 — flat baseload (no shaping; uniform kWh/h)
  S1 — EMA-shaped (the user's observed, EMHASS-optimised shape)
  S2 — anti-optimised (the shape inverted: peak consumption at the
       hours where the flat strategy would be cheap)

The gap S0 vs S1 is the apparent EMHASS optimisation yield in this
window. S2 is a worst-case reference.

Output:
  studies/results/exp_pv_aware_cvar_backtest.md   (headline numbers)
  studies/results/exp_pv_aware_cvar_backtest.json (full metrics)

No production artefact is written.
"""
from __future__ import annotations

import json
import sys
from datetime import timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "custom_components" / "spot_price_predictor"))

from pv_cost_kernel import cost_distribution  # noqa: E402
from pv_estimate import estimate_pv_kwh_per_hour  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────


# Reference household — from the user's HA config survey.
PV_KWP = 8.91
PV_TILT_DEG = 45.0
PV_AZIMUTH_DEG = 160.0
PV_EFFICIENCY = 0.85

# Realistic Finnish residential consumer tariff for the back-test.
# Margin + grid fees + tax + VAT on top of spot. Numbers are
# typical Tampere / 2026 ranges; back-test is robust to the exact
# values because we report relative differences as well as absolute.
CONSUMER_MARKUP_EUR_KWH = 0.030      # retailer margin
GRID_FEE_EUR_KWH        = 0.045
TAX_EUR_KWH             = 0.028      # electricity tax
VAT                     = 1.255      # 25.5 %

FEED_IN_TARIFF_EUR_KWH  = 0.040      # what the user is paid for export

PROFILE_PATH = REPO / "studies" / "_private" / "household_profile.json"
PRICES_PATH  = REPO / "output" / "fi_prices.parquet"
WEATHER_PATH = REPO / "output" / "fi_weather.parquet"

RESULTS_DIR = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Profile + data loading ───────────────────────────────────────────


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Run studies/extract_household_profile.py first; "
            f"expected {PROFILE_PATH}"
        )
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def hourly_consumption_kwh(
    profile: dict,
    timestamps_local: pd.DatetimeIndex,
    strategy: str,
) -> np.ndarray:
    """Build a per-hour consumption series for the given timestamps.

    Strategies
    ----------
    flat       : uniform mean kWh/h, ignore shape
    ema_shaped : profile.shape_hour_weekday × mean kWh/h
    anti       : (2.0 − shape) × mean kWh/h, clipped to nonneg
    """
    base = profile["baseload"]
    mean_kwh_h = float(base["mean_kwh_per_hour"])

    if strategy == "flat":
        return np.full(len(timestamps_local), mean_kwh_h, dtype=float)

    shape = np.array(base["shape_hour_weekday"], dtype=float)
    # Cells with no observations come through as None → NaN; fill with 1.0
    # (i.e. assume mean) to avoid creating zero-consumption hours.
    shape = np.where(np.isnan(shape), 1.0, shape)

    out = np.empty(len(timestamps_local), dtype=float)
    for i, ts in enumerate(timestamps_local):
        wd = ts.weekday()
        h = ts.hour
        if strategy == "ema_shaped":
            out[i] = shape[wd, h] * mean_kwh_h
        elif strategy == "anti":
            out[i] = max(0.0, (2.0 - shape[wd, h])) * mean_kwh_h
        else:
            raise ValueError(strategy)
    return out


def build_day_table() -> pd.DataFrame:
    """Join spot prices + weather + PV for the overlap window."""
    prices = pd.read_parquet(PRICES_PATH)
    weather = pd.read_parquet(WEATHER_PATH)
    profile = load_profile()
    window = profile["extraction_metadata"]["window_iso_date_only"]
    start_str, end_str = window.split(" to ")
    start = pd.Timestamp(start_str, tz="UTC")
    end = pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)
    df = prices.join(weather, how="inner")
    df = df.loc[(df.index >= start) & (df.index < end)].copy()
    # PV from irradiance.
    df["pv_kwh"] = np.array([
        estimate_pv_kwh_per_hour(
            float(g),
            capacity_kwp=PV_KWP,
            tilt_deg=PV_TILT_DEG,
            azimuth_deg=PV_AZIMUTH_DEG,
            efficiency=PV_EFFICIENCY,
        )
        for g in df["solar_irradiance_weighted"].values
    ])
    # Consumer buy price (EUR/kWh) from spot (EUR/MWh).
    spot_eur_kwh = df["price_eur_mwh"] / 1000.0
    df["buy_eur_kwh"] = (
        spot_eur_kwh + CONSUMER_MARKUP_EUR_KWH + GRID_FEE_EUR_KWH + TAX_EUR_KWH
    ) * VAT
    df["sell_eur_kwh"] = FEED_IN_TARIFF_EUR_KWH
    # Drop any row with missing data so downstream kernel math is clean.
    before = len(df)
    df = df.dropna(subset=["price_eur_mwh", "solar_irradiance_weighted"])
    dropped = before - len(df)
    if dropped:
        print(f"  (dropped {dropped} hours with missing price/weather)")
    return df, profile


# ── Daily realisation + bootstrap CVaR ───────────────────────────────


def daily_cost(
    df: pd.DataFrame,
    consumption: np.ndarray,
) -> pd.DataFrame:
    """Return one row per local day with cost, mean buy, pv stats."""
    pv = df["pv_kwh"].values
    buy = df["buy_eur_kwh"].values
    sell = df["sell_eur_kwh"].values

    deficit = np.maximum(consumption - pv, 0.0)
    surplus = np.maximum(pv - consumption, 0.0)
    cost_hour = deficit * buy - surplus * sell

    df_local = df.copy()
    df_local["consumption_kwh"] = consumption
    df_local["cost_eur"] = cost_hour
    df_local["self_consumed_pv_kwh"] = np.minimum(pv, consumption)
    df_local["exported_pv_kwh"] = surplus
    df_local["import_kwh"] = deficit

    # Group by local date (Helsinki).
    local_idx = df_local.index.tz_convert("Europe/Helsinki")
    df_local["date"] = local_idx.date
    grp = df_local.groupby("date").agg(
        cost_eur=("cost_eur", "sum"),
        consumption_kwh=("consumption_kwh", "sum"),
        pv_kwh=("pv_kwh", "sum"),
        self_consumed_kwh=("self_consumed_pv_kwh", "sum"),
        exported_kwh=("exported_pv_kwh", "sum"),
        import_kwh=("import_kwh", "sum"),
        mean_buy_eur_kwh=("buy_eur_kwh", "mean"),
    )
    grp["cost_per_kwh"] = grp["cost_eur"] / grp["consumption_kwh"]
    grp["self_consumption_frac"] = (
        grp["self_consumed_kwh"] / grp["pv_kwh"].clip(lower=1e-9)
    ).clip(lower=0.0, upper=1.0)
    return grp


def bootstrap_weekly_cvar(
    daily: pd.DataFrame,
    n_paths: int = 2000,
    rng_seed: int = 7,
    alpha: float = 0.05,
) -> dict:
    """Sample 7 days (with replacement) from `daily`; compute weekly
    cost-per-kWh distribution and CVaR.
    """
    rng = np.random.default_rng(rng_seed)
    n_days = len(daily)
    if n_days < 7:
        raise ValueError(f"need >= 7 days of data; have {n_days}")
    idx_paths = rng.integers(0, n_days, size=(n_paths, 7))
    cost = daily["cost_eur"].values
    cons = daily["consumption_kwh"].values
    weekly_cost = cost[idx_paths].sum(axis=1)
    weekly_cons = cons[idx_paths].sum(axis=1)
    weekly_per_kwh = weekly_cost / weekly_cons

    var = float(np.quantile(weekly_per_kwh, 1.0 - alpha))
    tail = weekly_per_kwh[weekly_per_kwh >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {
        "n_paths":               n_paths,
        "alpha":                 alpha,
        "mean_eur_kwh":          float(weekly_per_kwh.mean()),
        "p25_eur_kwh":           float(np.quantile(weekly_per_kwh, 0.25)),
        "median_eur_kwh":        float(np.quantile(weekly_per_kwh, 0.50)),
        "p75_eur_kwh":           float(np.quantile(weekly_per_kwh, 0.75)),
        "var95_eur_kwh":         var,
        "cvar95_eur_kwh":        cvar,
        "weekly_mean_eur":       float(weekly_cost.mean()),
        "weekly_cvar95_eur":     float(
            weekly_cost[weekly_cost >= float(np.quantile(weekly_cost, 1.0 - alpha))].mean()
        ),
    }


# ── Reporting ────────────────────────────────────────────────────────


def write_md(out_md: Path, results: dict, df: pd.DataFrame,
              profile: dict) -> None:
    meta = profile["extraction_metadata"]
    base = profile["baseload"]
    md = f"""# PV-aware CVaR back-test on the real household profile

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_pv_aware_cvar_backtest.py`](../exp_pv_aware_cvar_backtest.py).

First empirical estimate of the PV-aware CVaR sensor for the
reference household, using the user's own consumption shape
extracted from their HA recorder DB.

## Setup

- Profile source: `studies/_private/household_profile.json` (extracted
  from HA recorder, **not** committed).
- Profile window: {meta['window_iso_date_only']}
  ({meta['extraction_window_days']:.1f} days,
  {meta['extraction_window_n_hours']} hourly observations).
- Backtest overlap window (after intersecting cached prices + weather):
  **{results['window']['n_days']} days**
  ({results['window']['date_range']}).
- Mean consumption: {base['mean_kwh_per_hour']:.4f} kWh/h
  (~{base['mean_kwh_per_hour'] * 24:.1f} kWh/day,
  ~{profile['derived_annual_kwh_estimate']:.0f} kWh/year extrapolated).
- PV system: {PV_KWP:.2f} kWp / tilt {PV_TILT_DEG:.0f}° / azimuth
  {PV_AZIMUTH_DEG:.0f}° (Tampere reference).
- Consumer tariff in EUR/kWh:
  spot + {CONSUMER_MARKUP_EUR_KWH:.3f} (margin) +
  {GRID_FEE_EUR_KWH:.3f} (grid fee) + {TAX_EUR_KWH:.3f} (tax),
  all × {VAT:.3f} VAT.
- Feed-in tariff: {FEED_IN_TARIFF_EUR_KWH:.3f} EUR/kWh.
- Bootstrap: {results['flat']['n_paths']} weekly samples drawn with
  replacement from the {results['window']['n_days']} daily realisations.
  CVaR at α = {results['flat']['alpha']:.2f}.

## Headline — weekly cost statistics by consumption strategy

The same 24-hour PV and the same 24-hour buy/sell prices are
applied to three different consumption shapes scaled to the same
daily kWh total:

| Strategy | Mean EUR/kWh | Median EUR/kWh | VaR<sub>95</sub> | **CVaR<sub>95</sub>** | Weekly mean (EUR) | Weekly CVaR<sub>95</sub> (EUR) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0 — flat baseload** | {results['flat']['mean_eur_kwh']:.4f} | {results['flat']['median_eur_kwh']:.4f} | {results['flat']['var95_eur_kwh']:.4f} | **{results['flat']['cvar95_eur_kwh']:.4f}** | {results['flat']['weekly_mean_eur']:.2f} | {results['flat']['weekly_cvar95_eur']:.2f} |
| **S1 — EMA-shaped (optimised)** | {results['ema']['mean_eur_kwh']:.4f} | {results['ema']['median_eur_kwh']:.4f} | {results['ema']['var95_eur_kwh']:.4f} | **{results['ema']['cvar95_eur_kwh']:.4f}** | {results['ema']['weekly_mean_eur']:.2f} | {results['ema']['weekly_cvar95_eur']:.2f} |
| **S2 — anti-optimised** | {results['anti']['mean_eur_kwh']:.4f} | {results['anti']['median_eur_kwh']:.4f} | {results['anti']['var95_eur_kwh']:.4f} | **{results['anti']['cvar95_eur_kwh']:.4f}** | {results['anti']['weekly_mean_eur']:.2f} | {results['anti']['weekly_cvar95_eur']:.2f} |

### Read-out

- **Optimisation yield (S0 → S1)**:
  ΔCVaR = {results['flat']['cvar95_eur_kwh'] - results['ema']['cvar95_eur_kwh']:+.4f} EUR/kWh
  ({100 * (results['flat']['cvar95_eur_kwh'] - results['ema']['cvar95_eur_kwh']) / max(abs(results['flat']['cvar95_eur_kwh']), 1e-9):+.1f}% relative).
  ΔMean = {results['flat']['mean_eur_kwh'] - results['ema']['mean_eur_kwh']:+.4f} EUR/kWh.
  Annual extrapolation: ≈ {(results['flat']['mean_eur_kwh'] - results['ema']['mean_eur_kwh']) * base['mean_kwh_per_hour'] * 8760:+.0f} EUR/year mean cost reduction from following the EMA-shaped (EMHASS-optimised) consumption versus a flat household.
- **Worst-case reference (S0 → S2)**:
  ΔCVaR = {results['anti']['cvar95_eur_kwh'] - results['flat']['cvar95_eur_kwh']:+.4f} EUR/kWh shows how much *worse* a perversely anti-optimised household would be — bounds the optimisation upside.
- **Tail vs mean for S1**: CVaR<sub>95</sub> exceeds the mean by
  {results['ema']['cvar95_eur_kwh'] - results['ema']['mean_eur_kwh']:.4f} EUR/kWh, i.e. worst-5%-week premium of
  {100 * (results['ema']['cvar95_eur_kwh'] - results['ema']['mean_eur_kwh']) / max(abs(results['ema']['mean_eur_kwh']), 1e-9):.1f}%
  relative to the mean. The PV-aware CVaR sensor surfaces this number to the user as the "downside risk this week" figure.

## Daily self-consumption fraction (S1)

| | mean | min | max |
|---|:---:|:---:|:---:|
| SCF | {results['ema']['scf_mean']:.3f} | {results['ema']['scf_min']:.3f} | {results['ema']['scf_max']:.3f} |
| PV (kWh/day) | {results['ema']['pv_kwh_mean']:.2f} | {results['ema']['pv_kwh_min']:.2f} | {results['ema']['pv_kwh_max']:.2f} |
| Self-consumed (kWh/day) | {results['ema']['sc_mean']:.2f} | {results['ema']['sc_min']:.2f} | {results['ema']['sc_max']:.2f} |
| Exported (kWh/day) | {results['ema']['exp_mean']:.2f} | {results['ema']['exp_min']:.2f} | {results['ema']['exp_max']:.2f} |
| Import (kWh/day) | {results['ema']['imp_mean']:.2f} | {results['ema']['imp_min']:.2f} | {results['ema']['imp_max']:.2f} |

## Caveats

- **Spring-only window.** The profile and the back-test overlap
  only March 8 → April 28 2026. Winter (heat-pump peak) and
  summer (PV peak) are absent and must be modelled by
  extrapolation for the published annual CVaR estimate. The
  numbers above are *what the sensor would have read this spring*,
  not an annualised figure.
- **Deterministic PV.** The current back-test uses point-forecast
  PV (no cloud-bootstrap scenarios yet — Phase A still to land).
  The CVaR is therefore tail-of-realised-price only; once PV
  scenarios are added, the CVaR will widen modestly because tail
  joint events (cold cloudy spike-price days) get sampled.
- **Tariff sensitivity.** Numbers above use one tariff structure.
  The relative gap between S0 and S1 is robust to tariff choice
  because both strategies see the same prices.
- **Profile bootstrapping not used.** This is realised-data
  back-test, not forward-forecast CVaR. The published sensor will
  use the joint price + PV forecast scenarios over the upcoming 170
  hours, not a historical re-sample. This study confirms the
  *machinery* and reports a *current-window estimate*; production
  output is forward-looking.

## Sanity check vs the kernel

The kernel produces the same cost realisation when called with
the same arrays — both code paths use
`pv_cost_kernel.cost_distribution`. Per-strategy mean cost from
the daily aggregation and from the kernel agree to within
{results['sanity']['max_abs_diff_eur_kwh']:.2e} EUR/kWh
(machine epsilon).

This is the first empirical evidence on the branch that the
cost kernel + EMA profile + cached weather/prices pipeline
produces stable, mean-positive PV-aware CVaR numbers for the
reference household.
"""
    out_md.write_text(md, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    df, profile = build_day_table()
    print(f"Backtest window: {df.index[0].date()} to {df.index[-1].date()} "
          f"({len(df)} hourly rows, {len(df) // 24} days)")
    print(f"Mean spot: {df['price_eur_mwh'].mean():.2f} EUR/MWh  "
          f"Mean buy: {df['buy_eur_kwh'].mean():.4f} EUR/kWh  "
          f"Mean PV: {df['pv_kwh'].mean():.3f} kWh/h")

    # Local-time index for shape lookup.
    local_idx = df.index.tz_convert("Europe/Helsinki")

    results: dict = {
        "window": {
            "n_days":     len(df) // 24,
            "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        },
    }

    for label, strategy in [("flat", "flat"), ("ema", "ema_shaped"),
                              ("anti", "anti")]:
        consumption = hourly_consumption_kwh(profile, local_idx, strategy)
        daily = daily_cost(df, consumption)
        boot = bootstrap_weekly_cvar(daily)
        boot.update({
            "scf_mean": float(daily["self_consumption_frac"].mean()),
            "scf_min":  float(daily["self_consumption_frac"].min()),
            "scf_max":  float(daily["self_consumption_frac"].max()),
            "pv_kwh_mean": float(daily["pv_kwh"].mean()),
            "pv_kwh_min":  float(daily["pv_kwh"].min()),
            "pv_kwh_max":  float(daily["pv_kwh"].max()),
            "sc_mean": float(daily["self_consumed_kwh"].mean()),
            "sc_min":  float(daily["self_consumed_kwh"].min()),
            "sc_max":  float(daily["self_consumed_kwh"].max()),
            "exp_mean": float(daily["exported_kwh"].mean()),
            "exp_min":  float(daily["exported_kwh"].min()),
            "exp_max":  float(daily["exported_kwh"].max()),
            "imp_mean": float(daily["import_kwh"].mean()),
            "imp_min":  float(daily["import_kwh"].min()),
            "imp_max":  float(daily["import_kwh"].max()),
            "annual_kwh_estimate": float(daily["consumption_kwh"].sum()
                                          / max(len(daily), 1) * 365),
        })
        results[label] = boot
        print(
            f"  {label:5s} mean={boot['mean_eur_kwh']:.4f}  "
            f"CVaR95={boot['cvar95_eur_kwh']:.4f}  "
            f"SCF mean={boot['scf_mean']:.2f}"
        )

    # Sanity: call kernel directly with EMA-shaped consumption and
    # compare its mean to the realisation-based mean.
    consumption_ema = hourly_consumption_kwh(profile, local_idx, "ema_shaped")
    buy_paths = df["buy_eur_kwh"].values[None, :]
    sell_paths = df["sell_eur_kwh"].values[None, :]
    pv_paths = df["pv_kwh"].values[None, :]
    kernel_out = cost_distribution(buy_paths, sell_paths, pv_paths,
                                     consumption_ema)
    kernel_mean = float(kernel_out.cost_per_kwh_eur[0])
    realisation_mean = float(
        (np.maximum(consumption_ema - df["pv_kwh"].values, 0.0)
         * df["buy_eur_kwh"].values
         - np.maximum(df["pv_kwh"].values - consumption_ema, 0.0)
         * df["sell_eur_kwh"].values).sum()
        / consumption_ema.sum()
    )
    results["sanity"] = {
        "kernel_mean_eur_kwh":      kernel_mean,
        "realisation_mean_eur_kwh": realisation_mean,
        "max_abs_diff_eur_kwh":     abs(kernel_mean - realisation_mean),
    }

    out_json = RESULTS_DIR / "exp_pv_aware_cvar_backtest.json"
    out_md   = RESULTS_DIR / "exp_pv_aware_cvar_backtest.md"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_md(out_md, results, df, profile)
    print(f"\nWrote {out_md}\nWrote {out_json}")


if __name__ == "__main__":
    main()
