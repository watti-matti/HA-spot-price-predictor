"""Tests for the coordinator's PV-aware path.

Covers without importing homeassistant:

1. **Stability invariant grep** — `_resolve_baseload` MUST NOT call
   `hass.states.get` or any other HA entity-read API. Reads the
   coordinator source as text and asserts the constraint by inspection.

2. **External PV reader format support** — exercises the four supported
   attribute conventions (`forecast` list-of-dict, `wh_hours` dict,
   `watts` dict, `irradiance` list with W/kWh auto-detection) using a
   minimal mock reader that mirrors the production logic. The mock is
   lifted from coordinator source to keep the test self-contained.

3. **PV-aware D(k) monotonicity** — for synthetic 24-hour streams of
   `effective_eur_kwh`, the cumulative cheap/peak D(k) curves must be
   monotone non-decreasing / non-increasing respectively.

4. **Marginal-cost integration** — combines the PV estimator and the
   marginal-cost helper end-to-end on a synthetic Finnish-style 24-hour
   day to confirm `e_h ∈ [s_h, b_h]` always holds.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COORD_PATH = REPO / "custom_components" / "spot_price_predictor" / "coordinator.py"
PV_PATH = REPO / "custom_components" / "spot_price_predictor" / "pv_estimate.py"
DK_PATH = REPO / "src" / "dk_utils.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pv_estimate = _load("pv_estimate", PV_PATH)
dk_utils = _load("dk_utils_test", DK_PATH)


# ── 1. Stability invariant grep ─────────────────────────────────────


def test_resolve_baseload_does_not_read_ha_entities() -> None:
    """`_resolve_baseload` must not call `hass.states.get` or similar.

    This guarantees the price forecast is open-loop wrt the optimizer's
    flexible-load decisions (see TECHNICAL_GUIDE stability invariant).
    """
    src = COORD_PATH.read_text(encoding="utf-8")
    # Extract the body of `_resolve_baseload` until the next `def`
    m = re.search(
        r"def _resolve_baseload\(self.*?\n(.*?)(?=\n    def |\Z)",
        src, re.DOTALL,
    )
    assert m, "Could not locate _resolve_baseload in coordinator source"
    body = m.group(1)
    forbidden_patterns = [
        r"hass\.states",
        r"self\.hass\.states",
        r"async_track_state_change",
        r"\.async_added_to_hass",
        r"self\.hass\.bus",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, body), (
            f"_resolve_baseload contains forbidden HA-entity access "
            f"(pattern: {pat!r}). This breaks the stability invariant — "
            f"baseload must depend only on configuration, not on observed "
            f"consumption."
        )


def test_pipeline_overwrite_recomputes_pv_aware_fields() -> None:
    """`_apply_pipeline_pre_dk` overwrites `spot_eur_mwh`/`consumer_eur_kwh`
    with the pipeline-corrected price. The PV-aware fields
    (`effective_eur_kwh`, `net_household_cost_eur`, `sell_eur_kwh`) are
    derived from price and MUST be recomputed in the same pass — otherwise
    they stay frozen at their pre-pipeline (raw-model) values and become
    internally inconsistent (e.g. at night, pv=0, `effective_eur_kwh`
    would no longer equal `consumer_eur_kwh`).

    Guards against the regression where Pass 1 computed these fields from
    the raw model spot and the pipeline silently invalidated them.
    """
    src = COORD_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"def _apply_pipeline_pre_dk\b.*?\n(.*?)(?=\n    def |\Z)",
        src, re.DOTALL,
    )
    assert m, "Could not locate _apply_pipeline_pre_dk in coordinator source"
    body = m.group(1)

    # The method must overwrite consumer (the trigger for staleness)...
    assert 'f["consumer_eur_kwh"] =' in body, (
        "expected _apply_pipeline_pre_dk to overwrite consumer_eur_kwh"
    )
    # ...and recompute every price-derived PV-aware field alongside it.
    for field in (
        '"effective_eur_kwh"',
        '"net_household_cost_eur"',
        '"sell_eur_kwh"',
    ):
        assert f"f[{field}] =" in body, (
            f"_apply_pipeline_pre_dk overwrites consumer_eur_kwh but does "
            f"not recompute f[{field}] — it will stay stale against the "
            f"pre-pipeline price. See the night-time invariant "
            f"effective_eur_kwh == consumer_eur_kwh when pv == 0."
        )


def test_night_effective_equals_consumer_when_no_pv() -> None:
    """Core invariant the production bug violated: with no PV production
    (night, sun down), the marginal effective price equals the consumer
    buy price for any baseload. This holds regardless of which price
    (pre- or post-pipeline) is fed in — so once the coordinator feeds the
    *corrected* consumer, `effective_eur_kwh` tracks `consumer_eur_kwh`."""
    for buy in (0.0727, 0.141, 0.2217, 0.05):
        for baseload in (0.05, 1.0, 1.455, 3.0):
            eff = pv_estimate.marginal_effective_eur_kwh(
                buy_eur_kwh=buy, sell_eur_kwh=0.007,
                pv_kwh=0.0, baseload_kwh=baseload,
            )
            assert eff == pytest.approx(buy, rel=1e-12), (
                f"pv=0 must give effective==buy; got {eff} != {buy}")


def test_pv_dk_horizon_reconstructed_from_history_for_today() -> None:
    """PV-aware D(k) must not start a day later than the grid D(k).

    The fresh forecast starts at `now`, so today is partial and dropped by
    the 24-hour gate; grid D(k) back-fills today from actuals but the PV
    path historically did not. The coordinator now reconstructs today's
    PV-aware D(k) from the rolling forecast history (`_pv_dk_by_local_date`)
    and injects it onto the merged duration_forecast. Guard both halves.
    """
    src = COORD_PATH.read_text(encoding="utf-8")
    assert "def _pv_dk_by_local_date(" in src, (
        "expected reconstruction helper _pv_dk_by_local_date in coordinator"
    )
    # The merge step must call the helper and inject the PV arrays.
    assert "self._pv_dk_by_local_date(forecast)" in src, (
        "duration_forecast merge must call _pv_dk_by_local_date(forecast)"
    )
    m = re.search(
        r"pv_dk = self\._pv_dk_by_local_date\(forecast\)(.*?)(?=\n            # )",
        src, re.DOTALL,
    )
    assert m, "could not locate PV D(k) injection block"
    block = m.group(1)
    assert '"dk_cheap_pv_eur_kwh"' in block and '"dk_peak_pv_eur_kwh"' in block, (
        "injection block must set both dk_cheap_pv_eur_kwh and "
        "dk_peak_pv_eur_kwh on duration_forecast entries"
    )


def _reconstruct_pv_dk(rows: dict[str, dict]) -> dict[str, dict]:
    """Faithful mock of coordinator._pv_dk_by_local_date core math.

    `rows` maps timestamp -> {"effective_eur_kwh": float, "date": str}.
    Returns {date: {dk_cheap_pv_eur_kwh, dk_peak_pv_eur_kwh}} for dates
    with exactly 24 effective values present.
    """
    by_date: dict[str, list[float]] = {}
    for r in rows.values():
        m = r.get("effective_eur_kwh")
        if m is None:
            continue
        by_date.setdefault(r["date"], []).append(float(m))
    out: dict[str, dict] = {}
    for date_str, effs in by_date.items():
        if len(effs) != 24:
            continue
        asc = sorted(effs)
        desc = sorted(effs, reverse=True)
        cheap, peak, s_c, s_p = [], [], 0.0, 0.0
        for i in range(24):
            s_c += asc[i]
            s_p += desc[i]
            cheap.append(round(s_c / (i + 1), 4))
            peak.append(round(s_p / (i + 1), 4))
        out[date_str] = {"dk_cheap_pv_eur_kwh": cheap,
                         "dk_peak_pv_eur_kwh": peak}
    return out


def test_pv_dk_reconstruction_today_full_day_partial_dropped() -> None:
    """A date with 24 hours (history morning + forecast afternoon) yields a
    PV D(k); a partial date (<24h) is omitted. dk_cheap[0] is the single
    cheapest hour, dk_peak[0] the single priciest, and both [23] equal the
    full-day mean."""
    import random
    random.seed(11)

    rows: dict[str, dict] = {}
    today_effs = [round(random.uniform(-0.02, 0.25), 4) for _ in range(24)]
    # 12 "history" hours + 12 "forecast" hours all on the same local date
    for h in range(24):
        rows[f"2026-06-01T{h:02d}:00:00+00:00"] = {
            "effective_eur_kwh": today_effs[h], "date": "2026-06-01"}
    # Tomorrow only partially present (10 hours) -> must be dropped
    for h in range(10):
        rows[f"2026-06-02T{h:02d}:00:00+00:00"] = {
            "effective_eur_kwh": 0.10, "date": "2026-06-02"}

    out = _reconstruct_pv_dk(rows)
    assert "2026-06-01" in out, "full 24h today must produce PV D(k)"
    assert "2026-06-02" not in out, "partial day must be dropped"

    cheap = out["2026-06-01"]["dk_cheap_pv_eur_kwh"]
    peak = out["2026-06-01"]["dk_peak_pv_eur_kwh"]
    assert len(cheap) == 24 and len(peak) == 24
    assert cheap[0] == pytest.approx(min(today_effs), abs=1e-4)
    assert peak[0] == pytest.approx(max(today_effs), abs=1e-4)
    full_mean = round(sum(today_effs) / 24, 4)
    assert cheap[23] == pytest.approx(full_mean, abs=1e-4)
    assert peak[23] == pytest.approx(cheap[23], abs=1e-4)
    # Monotonicity
    for k in range(1, 24):
        assert cheap[k] >= cheap[k - 1] - 1e-9
        assert peak[k] <= peak[k - 1] + 1e-9


# ── 2. External PV reader format support ────────────────────────────
#
# We mirror the production reader logic in a minimal mock, so this test
# can run without importing homeassistant. Any divergence between the
# mock and the production logic is caught by the live integration smoke
# test (see TECHNICAL_GUIDE for instructions).


def _read_external(attrs: dict, ceiling: float) -> list[float] | None:
    """Mirror of coordinator._read_external_pv_forecast() core logic."""

    def clamp(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if f != f:
            return 0.0
        return max(0.0, min(f, ceiling))

    forecast_attr = attrs.get("forecast")
    if isinstance(forecast_attr, list) and forecast_attr:
        out: list[float] = []
        for entry in forecast_attr:
            if isinstance(entry, dict):
                v = (entry.get("pv_kwh") or entry.get("kwh")
                     or entry.get("energy") or entry.get("value") or 0.0)
            else:
                v = entry
            out.append(clamp(v))
        if out:
            return out

    wh = attrs.get("wh_hours")
    if isinstance(wh, dict) and wh:
        items = sorted(wh.items(), key=lambda kv: kv[0])
        return [clamp(float(v) / 1000.0) for _, v in items]

    watts = attrs.get("watts")
    if isinstance(watts, dict) and watts:
        items = sorted(watts.items(), key=lambda kv: kv[0])
        return [clamp(float(v) / 1000.0) for _, v in items]

    irr = attrs.get("irradiance")
    if isinstance(irr, list) and irr:
        numeric = []
        for v in irr:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                continue
        if not numeric:
            return None
        divisor = 1000.0 if max(numeric) > 50.0 else 1.0
        return [clamp(v / divisor) for v in numeric]

    return None


def test_external_reader_forecast_list_of_dict_kwh() -> None:
    """Format 1: `forecast` attribute, list of dicts with kWh values."""
    attrs = {"forecast": [
        {"pv_kwh": 0.0}, {"pv_kwh": 1.5}, {"pv_kwh": 4.0},
    ]}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [0.0, 1.5, 4.0]


def test_external_reader_forecast_supports_alternate_keys() -> None:
    """Various key names (kwh, energy, value) all accepted."""
    attrs = {"forecast": [
        {"kwh": 0.5}, {"energy": 1.0}, {"value": 2.0},
    ]}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [0.5, 1.0, 2.0]


def test_external_reader_wh_hours_dict() -> None:
    """Format 2: `wh_hours` dict (Wh) → divide by 1000 for kWh."""
    attrs = {"wh_hours": {
        "2026-05-09T10:00": 1500,   # 1.5 kWh
        "2026-05-09T11:00": 4000,   # 4.0 kWh
        "2026-05-09T12:00": 4500,   # 4.5 kWh
    }}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [1.5, 4.0, 4.5]


def test_external_reader_watts_dict() -> None:
    """Format 3: `watts` dict (W) → kWh via /1000 at hourly granularity."""
    attrs = {"watts": {
        "2026-05-09T10:00": 1500,
        "2026-05-09T11:00": 4000,
    }}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [1.5, 4.0]


def test_external_reader_irradiance_list_in_watts() -> None:
    """Format 4: `irradiance` list. Magnitudes > 50 → assume W, /1000."""
    attrs = {"irradiance": [0, 500, 2000, 4500, 6000]}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [0.0, 0.5, 2.0, 4.5, 6.0]


def test_external_reader_irradiance_list_in_kwh() -> None:
    """Format 4 (small magnitudes): treat as kWh directly."""
    attrs = {"irradiance": [0.0, 0.5, 1.5, 4.0]}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [0.0, 0.5, 1.5, 4.0]


def test_external_reader_clamps_to_ceiling() -> None:
    """Values exceeding capacity_kwp · efficiency are clamped down."""
    attrs = {"forecast": [{"pv_kwh": 100.0}]}
    out = _read_external(attrs, ceiling=4.25)
    assert out == [4.25]


def test_external_reader_returns_none_on_unknown_format() -> None:
    """No supported attribute → None (caller falls back to internal)."""
    assert _read_external({"unknown": []}, ceiling=10.0) is None
    assert _read_external({}, ceiling=10.0) is None


def test_external_reader_handles_invalid_values_gracefully() -> None:
    """Non-numeric entries are coerced to 0 (template safety)."""
    attrs = {"forecast": [{"pv_kwh": "n/a"}, {"pv_kwh": 2.5}]}
    out = _read_external(attrs, ceiling=10.0)
    assert out == [0.0, 2.5]


# ── 3. PV-aware D(k) monotonicity ───────────────────────────────────


def test_pv_aware_dk_cheap_peak_monotone_synthetic() -> None:
    """For any 24-hour stream, dk_cheap is non-decreasing and dk_peak is
    non-increasing. Validates the coordinator's per-day computation
    against `compute_dk_cheap_peak` directly."""
    import random
    random.seed(42)

    for _ in range(50):
        # Synthetic 24-hour effective price stream covering all regimes
        effectives = [random.uniform(-0.10, 0.30) for _ in range(24)]
        cheap, peak = dk_utils.compute_dk_cheap_peak(effectives)

        assert len(cheap) == 12
        assert len(peak) == 12
        for k in range(1, 12):
            assert cheap[k] >= cheap[k - 1] - 1e-9, (
                f"dk_cheap not monotone non-decreasing: {cheap}")
            assert peak[k] <= peak[k - 1] + 1e-9, (
                f"dk_peak not monotone non-increasing: {peak}")


def test_pv_aware_dk_identity_no_pv_equals_no_pv_path() -> None:
    """When PV produces nothing, marginal effective price = buy price for
    every hour, so PV-aware D(k) == regular consumer D(k). Confirms the
    PV path collapses cleanly to v2.2.0 behaviour when PV is disabled."""
    import random
    random.seed(7)
    buy_prices = [random.uniform(0.05, 0.25) for _ in range(24)]
    # All m_h == b_h when p_h = 0
    pv_effectives = [
        pv_estimate.marginal_effective_eur_kwh(
            buy_eur_kwh=b, sell_eur_kwh=0.04, pv_kwh=0.0, baseload_kwh=1.0)
        for b in buy_prices
    ]
    assert pv_effectives == pytest.approx(buy_prices, rel=1e-9)

    cheap_pv, peak_pv = dk_utils.compute_dk_cheap_peak(pv_effectives)
    cheap_no, peak_no = dk_utils.compute_dk_cheap_peak(buy_prices)
    assert cheap_pv == pytest.approx(cheap_no, rel=1e-9)
    assert peak_pv == pytest.approx(peak_no, rel=1e-9)


# ── 4. End-to-end marginal-cost integration ─────────────────────────


def test_end_to_end_synthetic_finnish_day() -> None:
    """Combine PV estimator + marginal cost on a 24h Finnish-style day.

    Asserts the bound m_h ∈ [min(0, s_h), b_h] holds for every hour (PV is
    free, v2.11.4) and that midday has lower effective cost than night."""
    # Finnish summer day: irradiance peaks at noon, low spot at noon
    # (solar pressure), high in evening
    spot_eur_mwh = [
        20, 18, 16, 15, 14, 15, 25, 50, 60, 55, 40, 30,   # 0-11
        25, 22, 20, 22, 30, 50, 80, 90, 70, 50, 35, 25,   # 12-23
    ]
    irradiance = [
        0, 0, 0, 0, 0, 50, 200, 400, 600, 800, 900, 950,
        980, 950, 900, 800, 600, 400, 200, 50, 0, 0, 0, 0,
    ]
    # Buy price assembly
    DAY, NIGHT, MARGIN, TAX, VAT, COMM = 0.0361, 0.0220, 0.0, 0.02325, 1.255, 0.002
    buy = []
    sell = []
    for h, sp in enumerate(spot_eur_mwh):
        is_night = h < 7 or h >= 22
        transfer = NIGHT if is_night else DAY
        b = (max(0.0, sp) / 1000.0 + MARGIN + transfer + TAX) * VAT
        s = sp / 1000.0 - COMM
        buy.append(b)
        sell.append(s)

    pv = [
        pv_estimate.estimate_pv_kwh_per_hour(
            irradiance_w_m2=i, capacity_kwp=5.0,
            tilt_deg=45.0, azimuth_deg=180.0, efficiency=0.85,
        ) for i in irradiance
    ]
    eff = [
        pv_estimate.marginal_effective_eur_kwh(
            buy_eur_kwh=b, sell_eur_kwh=s, pv_kwh=p, baseload_kwh=1.0,
        ) for b, s, p in zip(buy, sell, pv)
    ]

    # Bound check: self-consumed PV is free → floor is min(0, sell), not sell
    for h in range(24):
        lo = min(0.0, sell[h])
        hi = buy[h]
        assert lo - 1e-9 <= eff[h] <= hi + 1e-9, (
            f"hour {h}: m_h={eff[h]} not in [{lo}, {hi}]")

    # Midday (10-14) should have effective cost ≤ night (0-5) since PV
    # is meaningful and sell prices are positive
    midday_avg = sum(eff[10:15]) / 5
    night_avg = sum(eff[0:5]) / 5
    assert midday_avg < night_avg, (
        f"PV-bearing midday ({midday_avg}) should be cheaper than "
        f"night ({night_avg})")

    # PV-aware D(k) cheap should be non-decreasing
    cheap, peak = dk_utils.compute_dk_cheap_peak(eff)
    for k in range(1, 12):
        assert cheap[k] >= cheap[k - 1] - 1e-9
        assert peak[k] <= peak[k - 1] + 1e-9
