"""Time-zone alignment guard for Open-Meteo irradiance → PV.

Motivated by a field report that PV-aware prices looked "suspiciously
late" — high PV power after local sunset — the classic fingerprint of a
UTC↔local (Finland +2/+3 h) offset leaking into the solar series.

The pipeline keeps everything in UTC and only converts to local for
display, so it is *correct* — but the correctness rests on one fragile,
previously-unguarded assumption: Open-Meteo is queried with
``timezone=UTC`` and ``fetch_weather`` therefore tags the naive returned
timestamps as ``+00:00``. If that request param ever changed to ``auto``
or a named zone, every irradiance value would be silently mislabelled by
the Finland offset and PV would bleed past sunset.

These tests:
  1. pin the ``timezone=UTC`` request param (with the failure mode in the
     message so nobody "helpfully" changes it);
  2. exercise the REAL ``fetch_weather`` against a canned Open-Meteo
     payload and assert each irradiance value keeps its own UTC timestamp;
  3. convert the fetched series to **Finnish local time** and assert there
     is NO PV after sunset / before sunrise and the peak sits at local
     solar noon — the user's exact symptom, asserted away;
  4. confirm the local-hour conversion is DST-aware (summer +3, winter +2)
     so the timing does not drift by an hour across seasons.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "custom_components" / "spot_price_predictor"
COORD = PKG / "coordinator.py"
HELSINKI = ZoneInfo("Europe/Helsinki")


# ── Module loading without HomeAssistant / aiohttp ──────────────────


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def api_client_mod():
    """Load api_client.py with .const rewritten and aiohttp/HA stubbed."""
    _stub("homeassistant")
    _stub("homeassistant.const",
          Platform=types.SimpleNamespace(SENSOR="sensor"),
          UnitOfSpeed=types.SimpleNamespace(METERS_PER_SECOND="m/s"))
    _stub("aiohttp", ClientSession=object)

    _load("_spp_const", PKG / "const.py")

    src = (PKG / "api_client.py").read_text(encoding="utf-8")
    src = src.replace("from .const import", "from _spp_const import")
    mod = types.ModuleType("_spp_api_client")
    exec(compile(src, str(PKG / "api_client.py"), "exec"), mod.__dict__)
    sys.modules["_spp_api_client"] = mod
    return mod


@pytest.fixture(scope="module")
def pv_estimate_mod():
    return _load("_spp_pv_estimate", PKG / "pv_estimate.py")


# ── Fake aiohttp session returning a canned Open-Meteo payload ───────


def _solar_bell_w_m2(utc_hour: int) -> float:
    """Diurnal irradiance bell peaking at 10:00 UTC (~13:00 Helsinki
    summer), exactly zero outside roughly 02:00-18:00 UTC."""
    return max(0.0, 1000.0 - abs(utc_hour - 10) * 120.0)


def _openmeteo_payload(start="2026-06-01T00:00", hours=48) -> dict:
    base = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    times, solar, wind, temp = [], [], [], []
    for i in range(hours):
        ts = base + timedelta(hours=i)
        # Open-Meteo returns NAIVE local-to-requested-tz strings, no offset.
        times.append(ts.strftime("%Y-%m-%dT%H:%M"))
        solar.append(_solar_bell_w_m2(ts.hour))
        wind.append(4.0)
        temp.append(12.0)
    return {"hourly": {
        "time": times,
        "global_tilted_irradiance_instant": solar,
        "wind_speed_120m": wind,
        "temperature_2m": temp,
    }}


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeGetCtx:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return _FakeResp(self._payload)

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Returns the SAME payload for every location query, so the weighted
    average over locations equals the single shared series."""

    def __init__(self, payload):
        self._payload = payload
        self.calls: list[dict] = []

    def get(self, url, params=None):
        self.calls.append(dict(params or {}))
        return _FakeGetCtx(self._payload)


def _fetch(api_client_mod, payload):
    session = _FakeSession(payload)
    client = api_client_mod.SpotPriceApiClient(session)
    return asyncio.run(client.fetch_weather()), session


# ── 1. Request param guard ──────────────────────────────────────────


def test_openmeteo_request_uses_utc_timezone(api_client_mod):
    """fetch_weather MUST request timezone=UTC.

    The naive Open-Meteo timestamps are tagged +00:00 downstream; that is
    only valid if the API returned UTC. Any other value (auto / named
    zone) would shift every irradiance value by the Finland offset and put
    PV after sunset — the reported bug.
    """
    payload = _openmeteo_payload()
    _, session = _fetch(api_client_mod, payload)
    assert session.calls, "fetch_weather made no HTTP calls"
    for params in session.calls:
        assert params.get("timezone") == "UTC", (
            "Open-Meteo must be queried with timezone=UTC so naive "
            "timestamps can be tagged as UTC; got "
            f"{params.get('timezone')!r}")


def test_openmeteo_requests_instant_irradiance(api_client_mod):
    """The instantaneous GTI variable is timestamped AT the hour, so it
    aligns cleanly to the forecast clock (period-averaged variants are
    labelled at the interval edge and would skew the bell)."""
    payload = _openmeteo_payload()
    _, session = _fetch(api_client_mod, payload)
    for params in session.calls:
        assert "global_tilted_irradiance_instant" in params.get("hourly", "")


# ── 2. Real fetch_weather: timestamps stay paired with their values ──


def test_fetch_weather_tags_naive_times_as_utc(api_client_mod):
    payload = _openmeteo_payload(hours=30)
    rows, _ = _fetch(api_client_mod, payload)
    times = payload["hourly"]["time"]
    solar = payload["hourly"]["global_tilted_irradiance_instant"]

    assert rows, "no weather rows returned"
    for i, row in enumerate(rows):
        # Naive Open-Meteo string is tagged as UTC, exactly once.
        assert row["timestamp"] == times[i] + "+00:00", (
            f"row {i} timestamp {row['timestamp']!r} != {times[i]}+00:00")
        assert row["timestamp"].count("+") == 1
        # The value travels with its own timestamp — no positional drift.
        assert row["solar_weighted"] == pytest.approx(solar[i], abs=1e-6)


def test_fetch_weather_solar_peak_at_utc_10(api_client_mod):
    payload = _openmeteo_payload()
    rows, _ = _fetch(api_client_mod, payload)
    peak = max(rows, key=lambda r: r["solar_weighted"])
    peak_utc = datetime.fromisoformat(peak["timestamp"])
    assert peak_utc.hour == 10, (
        f"solar peak at {peak_utc.hour}:00 UTC, expected 10:00 UTC")


# ── 3. THE symptom test: no PV after sunset in LOCAL Finnish time ────


def test_no_pv_after_local_sunset(api_client_mod, pv_estimate_mod):
    """Convert each fetched (UTC) hour to Europe/Helsinki and assert PV is
    zero through the night and peaks at local solar noon. A UTC↔local
    offset bug would light up PV at 22:00-04:00 local — the field report."""
    payload = _openmeteo_payload()
    rows, _ = _fetch(api_client_mod, payload)

    NIGHT = {22, 23, 0, 1, 2, 3, 4}
    peak_local_hour = None
    peak_pv = -1.0
    for row in rows:
        ts_utc = datetime.fromisoformat(row["timestamp"])
        local_hour = ts_utc.astimezone(HELSINKI).hour
        pv = pv_estimate_mod.estimate_pv_kwh_per_hour(
            irradiance_w_m2=float(row["solar_weighted"]),
            capacity_kwp=8.91, tilt_deg=45.0,
            azimuth_deg=160.0, efficiency=0.85,
        )
        if local_hour in NIGHT:
            assert row["solar_weighted"] == 0.0, (
                f"irradiance {row['solar_weighted']} at LOCAL {local_hour}:00 "
                f"(UTC {ts_utc.hour}:00) — sun is down, expected 0")
            assert pv == 0.0, (
                f"PV {pv} kWh at LOCAL {local_hour}:00 — PV after sunset "
                f"indicates a UTC↔local offset in the irradiance series")
        if pv > peak_pv:
            peak_pv = pv
            peak_local_hour = local_hour

    assert peak_pv > 0.0, "no PV produced at all — bell mis-built"
    assert 12 <= peak_local_hour <= 14, (
        f"PV peaks at local {peak_local_hour}:00; Finnish solar noon is "
        f"~13:00 — a peak in the evening means a timezone shift")


# ── 4. Local-hour conversion is DST-aware ───────────────────────────


def test_local_hour_conversion_is_dst_aware():
    """Helsinki is UTC+3 in summer (DST) and UTC+2 in winter. A fixed
    offset would drift PV timing by an hour between seasons."""
    summer = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc).astimezone(HELSINKI)
    winter = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc).astimezone(HELSINKI)
    assert summer.hour == 13, "summer 10:00 UTC should be 13:00 Helsinki (UTC+3)"
    assert winter.hour == 12, "winter 10:00 UTC should be 12:00 Helsinki (UTC+2)"


# ── 5. Source guards: production stays wired to local-time alignment ─


def test_coordinator_night_detection_uses_local_hour():
    """The forecast's night flag (which drives night transfer tariff and
    sanity of effective price) must be derived from the LOCAL hour, not
    the raw UTC hour."""
    src = (PKG / "coordinator.py").read_text(encoding="utf-8")
    assert "local_hour = self._get_local_hour(ts)" in src, (
        "forecast loop must compute local_hour from the UTC timestamp")
    assert "is_night = local_hour" in src, (
        "night detection must use local_hour, not the UTC hour")


def test_get_local_hour_uses_zoneinfo_astimezone():
    """_get_local_hour must convert via the configured zone (DST-aware),
    not by adding a constant offset on the happy path."""
    src = (PKG / "coordinator.py").read_text(encoding="utf-8")
    assert "astimezone(self._tz).hour" in src, (
        "_get_local_hour must use astimezone(self._tz) for DST-correct "
        "local hours")


# ── 6. The "Oma hinta" (effective_eur_kwh) card contract ────────────
#
# That card plots `new Date(f.timestamp).getTime()` vs effective_eur_kwh.
# Two things must hold for it to render PV at the right local time:
#   (a) f.timestamp carries a UTC offset, else `new Date()` parses the
#       string as BROWSER-LOCAL and shifts every point;
#   (b) at local night PV is 0, so effective == consumer (no green
#       discount after sunset — the user's exact symptom).


def test_forecast_timestamp_carries_utc_offset():
    """Forecast rows must timestamp from a tz-aware UTC `now`, so
    `ts.isoformat()` keeps `+00:00` and the card's `new Date(f.timestamp)`
    parses as UTC instead of browser-local."""
    src = COORD.read_text(encoding="utf-8")
    assert "now = datetime.now(timezone.utc).replace(minute=0" in src, (
        "forecast `now` must be tz-aware UTC")
    assert '"timestamp": ts.isoformat()' in src, (
        "forecast rows must store ts.isoformat() (keeps the +00:00 offset "
        "that makes `new Date(f.timestamp)` UTC-safe in the dashboard)")


# ── 7. External-PV irradiance gate (phantom night PV) ───────────────
#
# Field report on 2.11.9-beta.1 (pv_source=external): pv_production_kwh of
# 2-5 kWh at 02:00-03:00 LOCAL while the model's own `solar` irradiance was
# 0 — physically impossible. _compute_pv_forecast now gates the external
# series by the (correctly aligned) irradiance: sun down → PV 0.


def _compute_pv_external_gated(external, weather, n_hours, start_utc):
    """Mirror of the gated external branch of _compute_pv_forecast."""
    sun_down = 5.0

    def sun_up(i):
        if not weather or i >= len(weather):
            return True  # no irradiance to judge against → don't gate
        return float(weather[i].get("solar_weighted", 0.0) or 0.0) > sun_down

    if isinstance(external, dict):
        return [(float(external.get(start_utc + timedelta(hours=i), 0.0))
                 if sun_up(i) else 0.0) for i in range(n_hours)]
    out = list(external[:n_hours])
    while len(out) < n_hours:
        out.append(0.0)
    return [v if sun_up(i) else 0.0 for i, v in enumerate(out)]


def test_external_pv_zeroed_when_sun_is_down():
    """A misaligned external source claiming 4 kWh at a night hour (where
    aligned irradiance is 0) must be gated to 0; daytime passes through."""
    start = datetime(2026, 6, 24, 0, 0, tzinfo=timezone.utc)
    # Aligned irradiance: bell peaking at 10:00 UTC, 0 at night.
    weather = [{"solar_weighted": _solar_bell_w_m2((start + timedelta(hours=i)).hour)}
               for i in range(24)]
    # External source wrongly reports production at 23:00/00:00 UTC (night).
    external = {start + timedelta(hours=h): 4.5 for h in range(24)}
    out = _compute_pv_external_gated(external, weather, 24, start)
    for i in range(24):
        utc_hour = (start + timedelta(hours=i)).hour
        if _solar_bell_w_m2(utc_hour) <= 5.0:
            assert out[i] == 0.0, (
                f"external PV not gated at UTC {utc_hour}:00 (sun down)")
        else:
            assert out[i] == 4.5, f"daytime external PV altered at {utc_hour}:00"


def test_external_pv_positional_list_also_gated():
    """The positional fallback (source without recognised timestamps) is
    gated too — that path was the likely origin of the night production."""
    start = datetime(2026, 6, 24, 0, 0, tzinfo=timezone.utc)
    weather = [{"solar_weighted": _solar_bell_w_m2((start + timedelta(hours=i)).hour)}
               for i in range(24)]
    external = [4.0] * 24  # flat list, no timestamps
    out = _compute_pv_external_gated(external, weather, 24, start)
    night = [out[i] for i in range(24)
             if _solar_bell_w_m2((start + timedelta(hours=i)).hour) <= 5.0]
    assert night and all(v == 0.0 for v in night), "night hours must be 0"


def test_compute_pv_forecast_gates_external_by_irradiance():
    """Source guard: the external branch of _compute_pv_forecast must zero
    PV when the aligned irradiance says the sun is down."""
    src = COORD.read_text(encoding="utf-8")
    import re
    m = re.search(r"def _compute_pv_forecast\(.*?\n(.*?)(?=\n    def )",
                  src, re.DOTALL)
    assert m, "could not locate _compute_pv_forecast"
    body = m.group(1)
    assert "_sun_is_up" in body and "solar_weighted" in body, (
        "_compute_pv_forecast must gate external PV against solar_weighted "
        "irradiance (sun-down → 0) to prevent phantom night production")


# ── 8. irradiance + iso_time alignment (the meteo_7day source) ──────
#
# sensor.meteo_7day_forecast_total publishes `irradiance` (PV power, W) and
# a naive-LOCAL `iso_time` axis. The reader previously consumed the list
# POSITIONALLY (ignoring iso_time), dropping the local-13:00 peak into the
# small hours. It now aligns by iso_time via _parse_ts (naive == local).


def _parse_ts_local(s, tz=HELSINKI):
    """Mirror of _read_external_pv_forecast._parse_ts: naive == local zone."""
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def test_irradiance_iso_time_alignment_peak_and_night():
    """Aligning by the naive-local iso_time axis puts the local-13:00 peak
    at 10:00 UTC and leaves every local-night hour at 0."""
    iso_time = [f"2026-06-24T{h:02d}:00" for h in range(24)]
    # Field-dump morning values; strictly-decreasing afternoon (unique peak).
    irr = [0, 0, 0, 0, 0, 16, 64, 96, 337, 1047, 2465, 4765, 6112, 6897,
           6500, 5500, 4000, 2400, 1000, 300, 80, 0, 0, 0]
    scaled = [min(v / 1000.0, 10.0) for v in irr]   # W→kWh, generous ceiling
    dated = {}
    for ts_raw, val in zip(iso_time, scaled):
        ts = _parse_ts_local(ts_raw)
        if ts is not None:
            dated[ts] = val

    peak_ts = max(dated, key=dated.get)
    assert peak_ts.hour == 10, f"peak keyed at {peak_ts.hour}:00 UTC, expected 10"
    assert peak_ts.astimezone(HELSINKI).hour == 13, "peak should be 13:00 local"
    for ts, val in dated.items():
        lh = ts.astimezone(HELSINKI).hour
        if lh in {22, 23, 0, 1, 2, 3, 4}:
            assert val == 0.0, f"PV {val} at local {lh}:00 after alignment"


def test_irradiance_branch_aligns_by_time_axis():
    """Source guard: the irradiance branch must align by its iso_time/time
    axis via _parse_ts, not consume the list positionally."""
    src = COORD.read_text(encoding="utf-8")
    i = src.index('irr_attr = attrs.get("irradiance")')
    branch = src[i:i + 1400]
    assert 'attrs.get("iso_time")' in branch, (
        "irradiance branch must read the iso_time axis")
    assert "_parse_ts" in branch, (
        "irradiance branch must align timestamps via _parse_ts")


def test_effective_price_has_no_pv_discount_at_local_night(
        api_client_mod, pv_estimate_mod):
    """End-to-end on the real fetch + estimators: the effective price the
    'Oma hinta' card plots must equal the plain consumer buy price at local
    night (PV=0). A discount (green) after sunset means PV is mis-timed."""
    payload = _openmeteo_payload()
    rows, _ = _fetch(api_client_mod, payload)
    buy, sell = 0.15, 0.04   # PV is the only thing that could lower effective
    for row in rows:
        ts_utc = datetime.fromisoformat(row["timestamp"])
        local_hour = ts_utc.astimezone(HELSINKI).hour
        pv = pv_estimate_mod.estimate_pv_kwh_per_hour(
            irradiance_w_m2=float(row["solar_weighted"]),
            capacity_kwp=8.91, tilt_deg=45.0,
            azimuth_deg=160.0, efficiency=0.85)
        eff = pv_estimate_mod.marginal_effective_eur_kwh(
            buy_eur_kwh=buy, sell_eur_kwh=sell, pv_kwh=pv, baseload_kwh=1.0)
        if local_hour in {23, 0, 1, 2, 3}:
            assert eff == pytest.approx(buy, rel=1e-9), (
                f"effective {eff} != consumer {buy} at LOCAL {local_hour}:00 "
                f"(UTC {ts_utc.hour}:00) — a PV discount after sunset means "
                f"the irradiance series is shifted")
