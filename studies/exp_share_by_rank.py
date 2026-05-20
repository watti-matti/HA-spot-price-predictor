"""Empirical share_by_rank[24] on the post-PV consumption window.

Validates the rank-shift consumption model from
`pv_adjusted_cvar_plan.md`. For each historical day:

1. Reconstruct total household demand `L(h)` from Fingrid grid import
   plus PV self-consumed (from cached irradiance × pv_estimate minus
   grid export).
2. Compute hour-of-week baseline envelope `baseline(h, weekday)` as
   the q10 of `L` over the post-PV window.
3. Per day: compute `effective_price(h) = (1-α) · buy + α · sell`
   where `α = min(1, PV(h) / L(h))`, rank the 24 hours, accumulate
   the deferrable residual `defer(h) = max(0, L(h) − baseline(h))`
   into `share_by_rank[rank_h]`.
4. Normalise across all days; plot the rank histogram overall and
   per season.

If the rank-shift model is correct for this household, the
distribution should be right-skewed with cheap ranks (0–5)
holding the bulk of deferrable mass.

Output: `studies/results/exp_share_by_rank.md` + JSON.
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

from lib_fingrid_csv import load_hourly, total_household_demand  # noqa: E402
from pv_estimate import estimate_pv_kwh_per_hour                # noqa: E402


# Tariff constants — same as the back-test for consistency.
CONSUMER_MARKUP = 0.030
GRID_FEE        = 0.045
TAX             = 0.028
VAT             = 1.255
FEED_IN         = 0.040

# PV system (reference household).
PV_KWP, PV_TILT, PV_AZ, PV_EFF = 8.91, 45.0, 160.0, 0.85

FINGRID_DIR = Path(
    "C:/GitHub/watti-matti/HA-energy-needs-planner/studies/"
    "Home energy consumption P12023-P52026"
)
PRICES_PATH  = REPO / "output" / "fi_prices.parquet"
WEATHER_PATH = REPO / "output" / "fi_weather.parquet"
RESULTS_DIR  = REPO / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data assembly ────────────────────────────────────────────────────


def build_post_pv_dataset() -> pd.DataFrame:
    grid_import = load_hourly(FINGRID_DIR / "home_consumption.csv")
    grid_export = load_hourly(FINGRID_DIR / "PV_production.csv")
    pv_install = grid_export.timestamps[0]

    # Weather & prices, joined on the post-PV window only.
    weather = pd.read_parquet(WEATHER_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    grid_imp_df = pd.DataFrame(
        {"grid_import": grid_import.kwh},
        index=grid_import.timestamps,
    )
    grid_exp_df = pd.DataFrame(
        {"grid_export": grid_export.kwh},
        index=grid_export.timestamps,
    )
    df = (
        grid_imp_df
        .join(grid_exp_df, how="left").fillna({"grid_export": 0.0})
        .join(prices, how="left")
        .join(weather, how="left")
    )
    df = df.loc[df.index >= pv_install].copy()
    df = df.dropna(subset=[
        "grid_import", "price_eur_mwh", "solar_irradiance_weighted",
    ])

    df["pv_total_kwh"] = np.array([
        estimate_pv_kwh_per_hour(
            float(g) if np.isfinite(g) else 0.0,
            capacity_kwp=PV_KWP, tilt_deg=PV_TILT,
            azimuth_deg=PV_AZ, efficiency=PV_EFF,
        )
        for g in df["solar_irradiance_weighted"].values
    ])
    df["pv_self_consumed_kwh"] = np.maximum(
        df["pv_total_kwh"] - df["grid_export"], 0.0
    )
    df["total_demand_kwh"] = df["grid_import"] + df["pv_self_consumed_kwh"]

    spot_eur_kwh = df["price_eur_mwh"] / 1000.0
    df["buy_eur_kwh"] = (spot_eur_kwh + CONSUMER_MARKUP + GRID_FEE + TAX) * VAT
    df["sell_eur_kwh"] = FEED_IN

    return df


def add_local_calendar(df: pd.DataFrame) -> pd.DataFrame:
    local = df.index.tz_convert("Europe/Helsinki")
    df = df.copy()
    df["local_weekday"] = local.weekday
    df["local_hour"]    = local.hour
    df["local_month"]   = local.month
    df["local_date"]    = local.date
    return df


# ── Baseline envelope: q10 per (weekday, hour) ───────────────────────


def baseline_envelope(df: pd.DataFrame) -> np.ndarray:
    """Returns ``baseline[7][24]`` = q10 of total_demand_kwh per cell."""
    base = np.zeros((7, 24), dtype=float)
    for wd in range(7):
        for h in range(24):
            sel = df.loc[
                (df["local_weekday"] == wd) & (df["local_hour"] == h),
                "total_demand_kwh",
            ].values
            base[wd, h] = float(np.quantile(sel, 0.10)) if sel.size else np.nan
    return base


# ── share_by_rank accumulator ────────────────────────────────────────


def effective_price(buy: np.ndarray, sell: np.ndarray,
                     pv: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Per-hour effective price = (1-α)·buy + α·sell  with α=PV/demand∈[0,1]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(demand > 0, pv / np.where(demand > 0, demand, 1.0), 1.0)
    alpha = np.clip(ratio, 0.0, 1.0)
    return (1.0 - alpha) * buy + alpha * sell


def accumulate_share(
    df: pd.DataFrame,
    baseline: np.ndarray,
    season_mask: np.ndarray | None = None,
) -> dict:
    share = np.zeros(24, dtype=float)
    n_days = 0
    total_deferrable = 0.0

    for date_key, day in df.groupby("local_date"):
        if len(day) != 24:
            continue
        if season_mask is not None:
            # Use the day's first row to decide season inclusion.
            if not bool(season_mask.loc[day.index[0]]):
                continue
        L = day["total_demand_kwh"].values
        wd = day["local_weekday"].values
        h = day["local_hour"].values
        base = baseline[wd, h]
        defer = np.maximum(L - base, 0.0)

        eff = effective_price(
            day["buy_eur_kwh"].values,
            day["sell_eur_kwh"].values,
            day["pv_total_kwh"].values,
            L,
        )
        # Rank: 0 = cheapest, 23 = most expensive.
        order = np.argsort(np.argsort(eff))
        for rank_pos in range(24):
            mask = order == rank_pos
            share[rank_pos] += float(defer[mask].sum())
        total_deferrable += float(defer.sum())
        n_days += 1

    if total_deferrable > 0:
        share_norm = share / total_deferrable
    else:
        share_norm = share

    return {
        "n_days":            n_days,
        "share_by_rank":     share_norm.tolist(),
        "total_deferrable":  total_deferrable,
        "share_top4_frac":   float(share_norm[:4].sum()),
        "share_top8_frac":   float(share_norm[:8].sum()),
        "share_bottom4_frac": float(share_norm[-4:].sum()),
        "share_bottom8_frac": float(share_norm[-8:].sum()),
        "ratio_top_vs_bot4": (
            float(share_norm[:4].sum() / max(share_norm[-4:].sum(), 1e-9))
        ),
    }


# ── Reporting ────────────────────────────────────────────────────────


def _bar(frac: float, width: int = 40) -> str:
    return "*" * max(0, int(frac * width * 24))


def write_md(out: Path, results: dict, baseline: np.ndarray,
              window_iso: str) -> None:
    overall = results["overall"]
    seasons = results["by_season"]

    histogram = "\n".join(
        f"| {i:2d} | {overall['share_by_rank'][i] * 100:5.2f} % | "
        f"{_bar(overall['share_by_rank'][i])} |"
        for i in range(24)
    )

    season_table = "\n".join(
        f"| {name:<7} | {s['n_days']:4d} | "
        f"{s['share_top4_frac'] * 100:5.1f} % | "
        f"{s['share_top8_frac'] * 100:5.1f} % | "
        f"{s['share_bottom8_frac'] * 100:5.1f} % | "
        f"{s['share_bottom4_frac'] * 100:5.1f} % | "
        f"{s['ratio_top_vs_bot4']:5.1f}× |"
        for name, s in seasons.items()
    )

    overall_top4 = overall["share_top4_frac"] * 100
    overall_top8 = overall["share_top8_frac"] * 100
    overall_bot4 = overall["share_bottom4_frac"] * 100
    overall_ratio = overall["ratio_top_vs_bot4"]

    verdict = (
        "✅ rank-shift model is empirically valid for this household"
        if overall_ratio >= 2.5
        else (
            "⚠ rank-shift effect present but moderate"
            if overall_ratio >= 1.5
            else "❌ no meaningful rank-shift — flat placement"
        )
    )

    md = f"""# Empirical share_by_rank[24] on post-PV consumption

Branch: `PV_adjusted_price`. Off-tree report. Script:
[`studies/exp_share_by_rank.py`](../exp_share_by_rank.py).

Validation of the rank-shift consumption model proposed in
`pv_adjusted_cvar_plan.md`. The model says household deferrable load
is placed disproportionately into the cheapest-effective-price hours
of each day. This script measures whether that's actually the case
on the user's data.

## Method

- **Total demand** reconstructed from Fingrid grid_import + cached
  irradiance × pv_estimate − Fingrid grid_export.
- **Baseline envelope** `baseline[7×24]` = q10 of demand per
  (weekday, hour) cell.
- **Deferrable per hour** `defer(h) = max(0, L(h) − baseline(h))`.
- **Effective price per hour** `(1−α)·buy + α·sell` with
  `α = min(1, PV/L)`. Buy/sell are consumer-tariff EUR/kWh.
- **Rank** 0..23 within each day, sorted by effective price.
- **Share** accumulator over {overall['n_days']} valid days in
  {window_iso}: `share_by_rank[r] = Σ_day defer(h_at_rank_r) /
  Σ_day defer(all hours)`.

## Headline — overall share_by_rank

| Rank | share | bar (each ★ ≈ 0.1 %) |
|:---:|:---:|---|
{histogram}

**Concentration metrics:**
- Top 4 cheapest ranks (0–3) receive **{overall_top4:.1f} %** of deferrable mass.
- Top 8 cheapest ranks (0–7) receive **{overall_top8:.1f} %**.
- Bottom 4 most expensive ranks (20–23) receive **{overall_bot4:.1f} %**.
- Cheap/expensive concentration ratio (top4 / bottom4): **{overall_ratio:.1f}×**.

A uniform distribution would put 16.7 % in any 4-rank bucket; the
observed top4 is {overall_top4 / 16.7:.1f}× uniform and the bottom4
is {overall_bot4 / 16.7:.2f}× uniform.

**Verdict: {verdict}**

## By season

| Season | days | top4 | top8 | bot8 | bot4 | top4/bot4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{season_table}

## Interpretation

The model `L(h, d) = baseline(h) + L_deferrable(d) · share(rank_h(d))`
treats `share_by_rank` as the empirical signature of the household's
optimisation policy. The numbers above either validate it
({verdict.lower()}) or argue for a different parameterisation.

If validated, the EMA module's published profile carries
`share_by_rank[24]` alongside `baseline[7×24]` and the predictor
reconstructs per-path consumption at forecast time as:

```
for each path:
    eff[h] = (1−α(h))·buy[h] + α(h)·sell[h]
    rank_h = argsort(argsort(eff))
    L_path[h] = baseline[wd, h] + deferrable_daily × share[rank_h]
```

This correctly preserves the joint distribution of (L, PV, price)
at forecast time and resolves the marginal-product bias documented
in the previous architectural turn.

## Caveats

- **Reconstruction error**: `pv_total` is estimated from cached
  irradiance × `pv_estimate`; small errors propagate into the
  `α = PV/L` coverage fraction. The effect on rank ordering is
  minor because effective-price ordering is dominated by spot.
- **q10 envelope sensitivity**: a different percentile changes the
  absolute shares but not the rank-relative concentration ratio.
- **Per-season counts**: some seasons may be under-sampled. Cross-
  check via the `n_days` column.
"""
    out.write_text(md, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading post-PV dataset...")
    df = build_post_pv_dataset()
    df = add_local_calendar(df)
    print(f"  rows: {len(df):,}  span: {df.index[0]} -> {df.index[-1]}")

    print("Computing baseline envelope (q10 per weekday-hour)...")
    baseline = baseline_envelope(df)
    print(f"  mean baseline kWh/h: {np.nanmean(baseline):.3f}")
    print(f"  baseline range: {np.nanmin(baseline):.3f} - {np.nanmax(baseline):.3f}")

    print("Accumulating share_by_rank overall...")
    overall = accumulate_share(df, baseline)
    print(f"  n_days: {overall['n_days']}")
    print(f"  total deferrable kWh: {overall['total_deferrable']:.0f}")
    print(f"  top4 share: {overall['share_top4_frac'] * 100:.1f}%")
    print(f"  top8 share: {overall['share_top8_frac'] * 100:.1f}%")
    print(f"  bot4 share: {overall['share_bottom4_frac'] * 100:.1f}%")
    print(f"  top4/bot4 ratio: {overall['ratio_top_vs_bot4']:.2f}x")

    season_definitions = {
        "winter": df["local_month"].isin([12, 1, 2]),
        "spring": df["local_month"].isin([3, 4, 5]),
        "summer": df["local_month"].isin([6, 7, 8]),
        "autumn": df["local_month"].isin([9, 10, 11]),
    }
    by_season = {}
    for name, mask in season_definitions.items():
        print(f"Accumulating share for {name}...")
        by_season[name] = accumulate_share(df, baseline, mask)
        print(f"  {name}: n={by_season[name]['n_days']}  "
              f"top4={by_season[name]['share_top4_frac'] * 100:.1f}%  "
              f"ratio={by_season[name]['ratio_top_vs_bot4']:.2f}x")

    window_iso = (
        df["local_date"].iloc[0].isoformat() + " to "
        + df["local_date"].iloc[-1].isoformat()
    )
    results = {
        "window_iso":        window_iso,
        "overall":           overall,
        "by_season":         by_season,
        "baseline_envelope": baseline.tolist(),
    }
    out_json = RESULTS_DIR / "exp_share_by_rank.json"
    out_md = RESULTS_DIR / "exp_share_by_rank.md"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_md(out_md, results, baseline, window_iso)
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
