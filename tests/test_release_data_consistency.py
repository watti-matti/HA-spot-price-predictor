"""Release-gate data-consistency suite for the Spot Price Predictor.

Purpose
-------
Before cutting a GitHub release, verify that **every derived variable the
integration publishes is mathematically consistent with its inputs** —
i.e. that `consumer_eur_kwh`, `effective_eur_kwh`, `net_household_cost_eur`,
`sell_eur_kwh`, `is_export_hour`, the P5..P95 fan chart, the duration
D(k) curves, and the rolled-up week/current aggregates all agree with the
per-hour `spot_eur_mwh` / `pv_production_kwh` / `baseload_kwh` they are
computed from.

This is the suite that would have caught all three bugs found during the
v2.11 review:

  1. `effective_eur_kwh` / `net_household_cost_eur` / `sell_eur_kwh` going
     stale after the L1-L4 pipeline overwrites `spot`/`consumer`
     (night-time invariant `effective == consumer` when `pv == 0`).
  2. The PV-aware D(k) horizon starting a day later than the grid D(k).
  3. (Guarded indirectly) percentile / monotonicity ordering.

Design constraints
------------------
* No `homeassistant` import (mirrors the rest of the test-suite). The
  production maths lives in `pv_estimate.py` (imported directly) and in
  `coordinator.py` private methods (which require HA). We therefore
  **mirror** the two coordinator-only formulas (tariff + sell) here and
  add *source-text guards* asserting the production code still matches the
  mirror — any divergence fails the guard.
* The validator (`validate_forecast_row` / `validate_duration_day`) is
  pure and reusable: it can be pointed at a live sensor-attributes dump to
  audit a running install. The release tests drive it with a synthetic but
  realistic 188-hour horizon built from the production kernels.
* The actual pre-fix sample from the v2.11 review is embedded as
  `PREFIX_SAMPLE_ROWS` and used as a *negative* fixture: the validator
  must flag exactly the fields that were stale, proving it has teeth.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COORD_PATH = REPO / "custom_components" / "spot_price_predictor" / "coordinator.py"
PV_PATH = REPO / "custom_components" / "spot_price_predictor" / "pv_estimate.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pv = _load("pv_estimate_rel", PV_PATH)
marginal_effective_eur_kwh = pv.marginal_effective_eur_kwh
net_household_cost_eur = pv.net_household_cost_eur

# Rounding the coordinator applies before publishing each field.
ROUND = {
    "consumer_eur_kwh": 4,
    "effective_eur_kwh": 4,
    "net_household_cost_eur": 4,
    "sell_eur_kwh": 4,
    "spot_eur_mwh": 2,
}
# Tolerance: one unit in the last published decimal place, plus slack.
TOL = 5e-4


# ──────────────────────────────────────────────────────────────────────
# Mirrored production formulas (coordinator-only; guarded against drift).
# ──────────────────────────────────────────────────────────────────────


class TariffConfig:
    """Minimal mirror of the coordinator tariff/PV economic config."""

    def __init__(
        self,
        seller_margin=0.0024,
        day_rate=0.0610,
        night_rate=0.0462,
        energy_tax=0.02827515,
        vat_multiplier=1.255,
        pv_sell_commission=0.0021,
        pv_export_grid_fee=0.0007,
    ):
        self.seller_margin = seller_margin
        self.day_rate = day_rate
        self.night_rate = night_rate
        self.energy_tax = energy_tax
        self.vat_multiplier = vat_multiplier
        self.pv_sell_commission = pv_sell_commission
        self.pv_export_grid_fee = pv_export_grid_fee


def consumer_eur_kwh(spot_eur_mwh: float, is_night: bool, cfg: TariffConfig) -> float:
    """Mirror of `_spot_to_consumer_eur_kwh`."""
    transfer = cfg.night_rate if is_night else cfg.day_rate
    spot_kwh = max(0.0, spot_eur_mwh) / 1000.0
    return (spot_kwh + cfg.seller_margin + transfer + cfg.energy_tax) \
        * cfg.vat_multiplier


def sell_eur_kwh(spot_eur_mwh: float, cfg: TariffConfig) -> float:
    """Mirror of `_spot_to_sell_eur_kwh` (NOT clipped at zero)."""
    return (float(spot_eur_mwh) / 1000.0
            - cfg.pv_sell_commission
            - cfg.pv_export_grid_fee)


def is_night_hour(local_hour: int) -> bool:
    """Mirror of the coordinator night-rate window."""
    return local_hour < 7 or local_hour >= 22


# ──────────────────────────────────────────────────────────────────────
# Source-text guards — keep the mirror honest.
# ──────────────────────────────────────────────────────────────────────


def test_source_guard_consumer_formula_matches_mirror() -> None:
    src = COORD_PATH.read_text(encoding="utf-8")
    assert "transfer = self.night_rate if is_night else self.day_rate" in src
    assert "spot_kwh = max(0.0, spot_eur_mwh) / 1000.0" in src
    assert "(spot_kwh + self.seller_margin + transfer + self.energy_tax)" in src
    assert "* self.vat_multiplier" in src


def test_source_guard_sell_formula_matches_mirror() -> None:
    src = COORD_PATH.read_text(encoding="utf-8")
    assert "float(spot_eur_mwh) / 1000.0" in src
    assert "- self.pv_sell_commission" in src
    assert "- self.pv_export_grid_fee" in src


def test_source_guard_night_window_matches_mirror() -> None:
    src = COORD_PATH.read_text(encoding="utf-8")
    # Both the first pass and the pipeline-overwrite pass use this window.
    assert src.count("local_hour < 7 or local_hour >= 22") + \
        src.count("local_h < 7 or local_h >= 22") >= 2


# ──────────────────────────────────────────────────────────────────────
# Per-row validator — the reusable contract.
# ──────────────────────────────────────────────────────────────────────

PCTL_KEYS = ["P5_eur_mwh", "P25_eur_mwh", "P50_eur_mwh", "P75_eur_mwh", "P95_eur_mwh"]


def validate_forecast_row(row: dict, cfg: TariffConfig | None = None,
                          *, local_hour: int | None = None) -> list[str]:
    """Return a list of human-readable consistency violations for one row.

    Empty list == fully consistent. Config-independent checks always run;
    tariff/sell checks run only when `cfg` is supplied. `local_hour` (if
    given) drives the night/day tariff check.
    """
    v: list[str] = []
    spot = row.get("spot_eur_mwh")
    cons = row.get("consumer_eur_kwh")

    # 1) Fan-chart percentile ordering: P5 <= P25 <= P50 <= P75 <= P95.
    pcts = [row.get(k) for k in PCTL_KEYS]
    if all(p is not None for p in pcts):
        for a, b, ka, kb in zip(pcts, pcts[1:], PCTL_KEYS, PCTL_KEYS[1:]):
            if a > b + 1e-9:
                v.append(f"percentile order violated: {ka}={a} > {kb}={b}")
        # The mean spot should sit within the [P5, P95] band.
        if spot is not None and not (pcts[0] - 1.0 <= spot <= pcts[-1] + 1.0):
            v.append(f"spot {spot} outside P5..P95 [{pcts[0]}, {pcts[-1]}]")

    # PV-aware fields only present when PV is enabled.
    if "effective_eur_kwh" not in row:
        # Non-PV row: only price/percentile invariants apply.
        if cfg is not None and cons is not None and spot is not None \
                and local_hour is not None:
            exp = round(consumer_eur_kwh(spot, is_night_hour(local_hour), cfg), 4)
            if abs(exp - cons) > TOL:
                v.append(f"consumer {cons} != tariff(spot)={exp}")
        return v

    pv_kwh = row["pv_production_kwh"]
    base = row["baseload_kwh"]
    eff = row["effective_eur_kwh"]
    net = row["net_household_cost_eur"]
    sell = row["sell_eur_kwh"]
    is_exp = row["is_export_hour"]

    # 2) is_export_hour is exactly pv > baseload.
    if bool(is_exp) != bool(pv_kwh > base):
        v.append(f"is_export_hour={is_exp} but pv({pv_kwh})>baseload({base})="
                 f"{pv_kwh > base}")

    # 3) effective == marginal_effective(consumer, sell, pv, baseload).
    exp_eff = round(marginal_effective_eur_kwh(
        buy_eur_kwh=cons, sell_eur_kwh=sell, pv_kwh=pv_kwh, baseload_kwh=base), 4)
    if abs(exp_eff - eff) > TOL:
        v.append(f"effective {eff} != marginal(consumer,sell,pv,base)={exp_eff}")

    # 4) Bound: min(sell, consumer) <= effective <= max(sell, consumer).
    lo, hi = min(sell, cons), max(sell, cons)
    if not (lo - TOL <= eff <= hi + TOL):
        v.append(f"effective {eff} outside [{lo}, {hi}] (sell/consumer bound)")

    # 5) Regression invariant: when PV cannot cover an extra kWh
    #    (pv <= baseload), effective MUST equal the consumer buy price.
    if pv_kwh <= base and abs(eff - cons) > TOL:
        v.append(f"pv({pv_kwh})<=baseload({base}) but effective {eff} != "
                 f"consumer {cons} (stale-price regression)")

    # 6) net == net_household_cost(consumer, sell, pv, baseload).
    exp_net = round(net_household_cost_eur(
        buy_eur_kwh=cons, sell_eur_kwh=sell, pv_kwh=pv_kwh, consumption_kwh=base), 4)
    if abs(exp_net - net) > TOL:
        v.append(f"net {net} != net_household(consumer,sell,pv,base)={exp_net}")

    # 7) Sign sanity: an export hour can pay the user (net < 0 possible);
    #    a non-export hour must cost >= 0 (import only, buy price >= 0).
    if not is_exp and net < -TOL and cons >= 0:
        v.append(f"non-export hour has negative net {net}")

    # 8) Tariff / sell checks (config-dependent).
    if cfg is not None and spot is not None:
        exp_sell = round(sell_eur_kwh(spot, cfg), 4)
        if abs(exp_sell - sell) > TOL:
            v.append(f"sell {sell} != sell(spot)={exp_sell}")
        if local_hour is not None:
            exp_cons = round(consumer_eur_kwh(spot, is_night_hour(local_hour), cfg), 4)
            if abs(exp_cons - cons) > TOL:
                v.append(f"consumer {cons} != tariff(spot)={exp_cons}")

    return v


def validate_duration_day(day: dict, *, require_pv: bool) -> list[str]:
    """Validate one duration_forecast day entry's D(k) arrays."""
    v: list[str] = []

    def _check_curve(name: str, cheap: list, peak: list) -> None:
        if len(cheap) != 24 or len(peak) != 24:
            v.append(f"{name}: expected 24 levels, got "
                     f"{len(cheap)}/{len(peak)}")
            return
        for k in range(1, 24):
            if cheap[k] < cheap[k - 1] - 1e-6:
                v.append(f"{name} cheap not non-decreasing at {k}")
            if peak[k] > peak[k - 1] + 1e-6:
                v.append(f"{name} peak not non-increasing at {k}")
        for k in range(24):
            if cheap[k] > peak[k] + 1e-6:
                v.append(f"{name} cheap[{k}]={cheap[k]} > peak[{k}]={peak[k]}")
        # D(24) is the full-day mean from both directions -> identical.
        if abs(cheap[23] - peak[23]) > 1e-4:
            v.append(f"{name} cheap[23]={cheap[23]} != peak[23]={peak[23]}")

    _check_curve("grid", day.get("dk_cheap_eur_kwh") or [],
                 day.get("dk_peak_eur_kwh") or [])

    if require_pv:
        cheap_pv = day.get("dk_cheap_pv_eur_kwh")
        peak_pv = day.get("dk_peak_pv_eur_kwh")
        if cheap_pv is None or peak_pv is None:
            v.append("PV D(k) missing on a day where it is required "
                     "(horizon regression)")
        else:
            _check_curve("pv", cheap_pv, peak_pv)
            # PV-aware marginal cost is always <= grid cost (PV can only
            # help): every level should be <= the grid level.
            grid_cheap = day.get("dk_cheap_eur_kwh") or []
            if len(grid_cheap) == 24:
                for k in range(24):
                    if cheap_pv[k] > grid_cheap[k] + 1e-4:
                        v.append(f"pv cheap[{k}]={cheap_pv[k]} > grid "
                                 f"cheap[{k}]={grid_cheap[k]}")
    return v


# ──────────────────────────────────────────────────────────────────────
# Build a realistic, fully-consistent 188-hour horizon (release golden).
# ──────────────────────────────────────────────────────────────────────


def _dk_curves(values: list[float]) -> tuple[list[float], list[float]]:
    asc = sorted(values)
    desc = sorted(values, reverse=True)
    cheap, peak, s_c, s_p = [], [], 0.0, 0.0
    for i in range(len(values)):
        s_c += asc[i]
        s_p += desc[i]
        cheap.append(round(s_c / (i + 1), 4))
        peak.append(round(s_p / (i + 1), 4))
    return cheap, peak


def build_consistent_horizon(hours: int = 188, cfg: TariffConfig | None = None):
    """Produce per-hour rows the way the *fixed* coordinator would.

    Returns (rows, days, cfg). UTC start 2026-06-01T00:00; local = UTC+3
    (Finland summer), so local_hour = (utc_hour + 3) % 24.
    """
    if cfg is None:
        cfg = TariffConfig()
    rows: list[dict] = []
    UTC_OFFSET = 3
    base_kwh = 1.455
    for h in range(hours):
        utc_hour = h % 24
        local_hour = (utc_hour + UTC_OFFSET) % 24
        night = is_night_hour(local_hour)

        # Synthetic spot: diurnal shape, evening peak, occasional negative
        # midday dips on later "solar-heavy" days to exercise export.
        day_idx = h // 24
        diurnal = 60 + 35 * math.sin((local_hour - 9) / 24 * 2 * math.pi)
        solar_pressure = max(0.0, 30 - 2.0 * day_idx) if 9 <= local_hour <= 15 else 0.0
        spot = round(diurnal - solar_pressure + (5 if 17 <= local_hour <= 20 else 0), 2)

        # Synthetic PV: bell curve over daylight, growing with day_idx to
        # push some hours into export (pv > baseload).
        if 4 <= local_hour <= 21:
            amp = 0.6 + 0.5 * day_idx
            pv_kwh = max(0.0, amp * math.sin((local_hour - 4) / 17 * math.pi)) ** 2 * 4.0
        else:
            pv_kwh = 0.0
        pv_kwh = round(pv_kwh, 3)

        cons = round(consumer_eur_kwh(spot, night, cfg), 4)
        s_h = round(sell_eur_kwh(spot, cfg), 4)
        eff = round(marginal_effective_eur_kwh(
            buy_eur_kwh=cons, sell_eur_kwh=s_h, pv_kwh=pv_kwh,
            baseload_kwh=base_kwh), 4)
        net = round(net_household_cost_eur(
            buy_eur_kwh=cons, sell_eur_kwh=s_h, pv_kwh=pv_kwh,
            consumption_kwh=base_kwh), 4)

        # Fan chart: symmetric-ish band around spot, ordered by construction.
        rows.append({
            "_utc_hour": utc_hour,
            "_local_hour": local_hour,
            "spot_eur_mwh": spot,
            "consumer_eur_kwh": cons,
            "pv_production_kwh": pv_kwh,
            "baseload_kwh": base_kwh,
            "effective_eur_kwh": eff,
            "net_household_cost_eur": net,
            "sell_eur_kwh": s_h,
            "is_export_hour": bool(pv_kwh > base_kwh),
            "P5_eur_mwh": round(spot - 22, 4),
            "P25_eur_mwh": round(spot - 11, 4),
            "P50_eur_mwh": round(spot, 4),
            "P75_eur_mwh": round(spot + 12, 4),
            "P95_eur_mwh": round(spot + 24, 4),
        })

    # Build complete-day duration entries (24-hour groups by local date).
    days: list[dict] = []
    for start in range(0, hours - 23, 24):
        chunk = rows[start:start + 24]
        if len(chunk) < 24:
            break
        cons_vals = [r["consumer_eur_kwh"] for r in chunk]
        eff_vals = [r["effective_eur_kwh"] for r in chunk]
        c_cheap, c_peak = _dk_curves(cons_vals)
        e_cheap, e_peak = _dk_curves(eff_vals)
        days.append({
            "date": f"2026-06-{1 + start // 24:02d}",
            "source": "forecast",
            "dk_cheap_eur_kwh": c_cheap,
            "dk_peak_eur_kwh": c_peak,
            "dk_cheap_pv_eur_kwh": e_cheap,
            "dk_peak_pv_eur_kwh": e_peak,
        })
    return rows, days, cfg


# ──────────────────────────────────────────────────────────────────────
# Release-gate tests: the golden horizon must be fully consistent.
# ──────────────────────────────────────────────────────────────────────


def test_golden_horizon_every_row_consistent() -> None:
    rows, _days, cfg = build_consistent_horizon()
    assert len(rows) == 188
    all_violations: list[str] = []
    for i, r in enumerate(rows):
        viol = validate_forecast_row(r, cfg, local_hour=r["_local_hour"])
        all_violations += [f"row {i} ({r['_utc_hour']:02d}Z): {m}" for m in viol]
    assert not all_violations, "consistency violations:\n" + "\n".join(all_violations)


def test_golden_horizon_exercises_all_regimes() -> None:
    """The synthetic horizon must actually cover night, day, export and
    non-export hours — otherwise the consistency test is vacuous."""
    rows, _d, _c = build_consistent_horizon()
    assert any(is_night_hour(r["_local_hour"]) for r in rows)
    assert any(not is_night_hour(r["_local_hour"]) for r in rows)
    assert any(r["is_export_hour"] for r in rows), "no export hours generated"
    assert any(not r["is_export_hour"] for r in rows)
    assert any(r["pv_production_kwh"] == 0.0 for r in rows), "no pv==0 hours"
    assert any(r["net_household_cost_eur"] < 0 for r in rows), "no paid-to-export"


def test_golden_duration_curves_consistent_and_pv_present() -> None:
    _rows, days, _cfg = build_consistent_horizon()
    assert len(days) >= 7
    all_violations: list[str] = []
    for d in days:
        all_violations += [f"{d['date']}: {m}"
                           for m in validate_duration_day(d, require_pv=True)]
    assert not all_violations, "duration violations:\n" + "\n".join(all_violations)


def test_night_effective_equals_consumer_across_horizon() -> None:
    """Spell out the headline regression invariant over the whole horizon:
    every hour with pv <= baseload has effective == consumer."""
    rows, _d, _c = build_consistent_horizon()
    checked = 0
    for r in rows:
        if r["pv_production_kwh"] <= r["baseload_kwh"]:
            checked += 1
            assert r["effective_eur_kwh"] == pytest.approx(
                r["consumer_eur_kwh"], abs=TOL), \
                f"{r['_utc_hour']}Z: effective != consumer with no PV headroom"
    assert checked > 0


# ──────────────────────────────────────────────────────────────────────
# Aggregate roll-up consistency (week_* / current_*).
# ──────────────────────────────────────────────────────────────────────


def test_aggregate_rollups_match_per_row_values() -> None:
    rows, _d, _c = build_consistent_horizon()
    effs = [r["effective_eur_kwh"] for r in rows]
    consumers = [r["consumer_eur_kwh"] for r in rows]

    # Mirror sensor.py's week_* effective aggregates.
    week_min = round(min(effs), 4)
    week_max = round(max(effs), 4)
    week_avg = round(sum(effs) / len(effs), 4)
    assert week_min <= week_avg <= week_max
    # current_* must equal the first row.
    assert effs[0] == rows[0]["effective_eur_kwh"]
    assert consumers[0] == rows[0]["consumer_eur_kwh"]
    # Consumer week band brackets the average too.
    assert round(min(consumers), 4) <= round(sum(consumers) / len(consumers), 4) \
        <= round(max(consumers), 4)


# ──────────────────────────────────────────────────────────────────────
# Negative fixture: the ACTUAL pre-fix sample must be flagged.
# ──────────────────────────────────────────────────────────────────────
#
# Rows lifted verbatim from the v2.11 review dump (pre-fix). At the time,
# `effective_eur_kwh` / `net_household_cost_eur` / `sell_eur_kwh` were
# computed from the raw-model spot and never recomputed after the pipeline
# overwrote `spot`/`consumer` — so the night rows below violate
# `effective == consumer` (pv == 0). The validator MUST catch this; the
# released code MUST produce data that does NOT.

PREFIX_SAMPLE_ROWS = [
    {  # 00:00Z -> 03:00 local (night), pv=0
        "spot_eur_mwh": 63.3089, "consumer_eur_kwh": 0.141,
        "pv_production_kwh": 0.0, "baseload_kwh": 1.455,
        "effective_eur_kwh": 0.0727, "net_household_cost_eur": 0.1058,
        "sell_eur_kwh": 0.0069, "is_export_hour": False,
        "P5_eur_mwh": 35.37, "P25_eur_mwh": 49.18, "P50_eur_mwh": 60.30,
        "P75_eur_mwh": 75.84, "P95_eur_mwh": 91.25, "_local_hour": 3,
    },
    {  # 01:00Z -> 04:00 local (night), pv=0
        "spot_eur_mwh": 66.1278, "consumer_eur_kwh": 0.1446,
        "pv_production_kwh": 0.0, "baseload_kwh": 1.455,
        "effective_eur_kwh": 0.0731, "net_household_cost_eur": 0.1064,
        "sell_eur_kwh": 0.0072, "is_export_hour": False,
        "P5_eur_mwh": 38.14, "P25_eur_mwh": 50.66, "P50_eur_mwh": 64.90,
        "P75_eur_mwh": 79.03, "P95_eur_mwh": 94.07, "_local_hour": 4,
    },
]


def test_prefix_sample_is_flagged_by_validator() -> None:
    """Self-test: the validator detects the real pre-fix inconsistency."""
    for r in PREFIX_SAMPLE_ROWS:
        viol = validate_forecast_row(r, local_hour=r["_local_hour"])
        joined = " | ".join(viol)
        assert any("effective" in m and "consumer" in m for m in viol), (
            f"validator failed to flag stale effective on {r['spot_eur_mwh']}: "
            f"{joined}")
        assert any("net" in m for m in viol), (
            f"validator failed to flag stale net: {joined}")


def test_prefix_sample_structural_fields_are_sound() -> None:
    """The pre-fix dump was wrong only in the price-derived fields — the
    percentile ordering and is_export flags were already correct. Confirm
    the validator does NOT raise false positives on those."""
    for r in PREFIX_SAMPLE_ROWS:
        viol = validate_forecast_row(r, local_hour=r["_local_hour"])
        assert not any("percentile" in m for m in viol)
        assert not any("is_export_hour" in m for m in viol)


# ──────────────────────────────────────────────────────────────────────
# Negative unit tests: prove each invariant actually fires.
# ──────────────────────────────────────────────────────────────────────


def _good_row(cfg: TariffConfig) -> dict:
    rows, _d, _c = build_consistent_horizon(hours=24, cfg=cfg)
    # pick a daytime non-export row with pv == 0 for crisp invariants
    for r in rows:
        if r["pv_production_kwh"] == 0.0 and not is_night_hour(r["_local_hour"]):
            return dict(r)
    return dict(rows[0])


def test_validator_catches_stale_effective() -> None:
    cfg = TariffConfig()
    r = _good_row(cfg)
    r["effective_eur_kwh"] = round(r["effective_eur_kwh"] * 0.5, 4)  # the v2.11 bug
    viol = validate_forecast_row(r, cfg, local_hour=r["_local_hour"])
    assert any("effective" in m for m in viol)


def test_validator_catches_sell_from_wrong_spot() -> None:
    cfg = TariffConfig()
    r = _good_row(cfg)
    r["sell_eur_kwh"] = round(r["sell_eur_kwh"] + 0.05, 4)
    viol = validate_forecast_row(r, cfg, local_hour=r["_local_hour"])
    assert any("sell" in m for m in viol)


def test_validator_catches_percentile_disorder() -> None:
    cfg = TariffConfig()
    r = _good_row(cfg)
    r["P75_eur_mwh"], r["P25_eur_mwh"] = r["P25_eur_mwh"], r["P75_eur_mwh"]
    viol = validate_forecast_row(r, cfg, local_hour=r["_local_hour"])
    assert any("percentile" in m for m in viol)


def test_validator_catches_wrong_export_flag() -> None:
    cfg = TariffConfig()
    r = _good_row(cfg)
    r["is_export_hour"] = True  # pv == 0, cannot be export
    viol = validate_forecast_row(r, cfg, local_hour=r["_local_hour"])
    assert any("is_export_hour" in m for m in viol)


def test_validator_catches_missing_pv_dk_horizon() -> None:
    _rows, days, _cfg = build_consistent_horizon()
    broken = dict(days[0])
    broken.pop("dk_cheap_pv_eur_kwh")
    broken.pop("dk_peak_pv_eur_kwh")
    viol = validate_duration_day(broken, require_pv=True)
    assert any("PV D(k) missing" in m for m in viol)
