"""Tests for the Nordpool-compatible spot-price-forecast sensor.

Validates the shape contract — state in EUR/kWh, raw_today /
raw_tomorrow / raw_extended attributes — without instantiating the
full HomeAssistant runtime. Loads sensor.py directly via importlib
spec_from_file_location with stubbed HA modules.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_PKG = Path(__file__).parent.parent / "custom_components" / "spot_price_predictor"


@pytest.fixture(scope="module", autouse=True)
def _stub_ha_and_load_sensor():
    """Inject minimal HA stubs and load sensor.py as a standalone module."""

    def _fake(name: str, **attrs):
        mod = SimpleNamespace(**attrs)
        sys.modules[name] = mod
        return mod

    _fake("homeassistant")
    _fake("homeassistant.components")
    _fake("homeassistant.components.sensor",
            SensorDeviceClass=SimpleNamespace(
                MONETARY="monetary", WIND_SPEED="wind_speed"),
            SensorEntity=object,
            SensorStateClass=SimpleNamespace(
                MEASUREMENT="measurement", TOTAL="total"),
            )
    _fake("homeassistant.config_entries", ConfigEntry=object)
    _fake("homeassistant.core", HomeAssistant=object)
    _fake("homeassistant.helpers")
    _fake("homeassistant.helpers.entity_platform",
            AddEntitiesCallback=object)

    class _FakeCoordinatorEntity:
        def __init__(self, coordinator):
            self.coordinator = coordinator

    _fake("homeassistant.helpers.update_coordinator",
            CoordinatorEntity=_FakeCoordinatorEntity)

    # const.py imports homeassistant.const for Platform; sensor.py imports
    # UnitOfSpeed. Stub both.
    class _Platform:
        SENSOR = "sensor"

    class _UnitOfSpeed:
        METERS_PER_SECOND = "m/s"
    _fake("homeassistant.const", Platform=_Platform, UnitOfSpeed=_UnitOfSpeed)

    # const.py is plain Python — load directly.
    spec = importlib.util.spec_from_file_location(
        "_spp_const", _PKG / "const.py")
    const_mod = importlib.util.module_from_spec(spec)
    sys.modules["_spp_const"] = const_mod
    spec.loader.exec_module(const_mod)

    # coordinator.py is heavy — only sensor.py imports SpotPriceCoordinator
    # type annotation. Stub it.
    coord_stub = SimpleNamespace(SpotPriceCoordinator=object)
    sys.modules["_spp_coordinator"] = coord_stub

    # Read sensor.py, transform relative imports to use our stubs, exec.
    src = (_PKG / "sensor.py").read_text(encoding="utf-8")
    src = src.replace("from .const import", "from _spp_const import")
    src = src.replace("from .coordinator import",
                       "from _spp_coordinator import")
    sensor_mod = type(sys)("_spp_sensor")
    exec(compile(src, str(_PKG / "sensor.py"), "exec"), sensor_mod.__dict__)
    sys.modules["_spp_sensor"] = sensor_mod
    yield sensor_mod


@pytest.fixture
def sensor_module(_stub_ha_and_load_sensor):
    return _stub_ha_and_load_sensor


def _build_forecast(n_hours: int = 48) -> list[dict]:
    start = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)
    rows = []
    for h in range(n_hours):
        ts = start + timedelta(hours=h)
        rows.append({
            "timestamp":      ts.isoformat(),
            "spot_eur_mwh":   50.0 + h * 0.5,
            "P5_eur_mwh":     40.0 + h * 0.5,
            "P95_eur_mwh":    60.0 + h * 0.5,
            "consumer_eur_kwh": 0.18 + h * 0.001,
        })
    return rows


def _make_entry():
    e = MagicMock()
    e.entry_id = "test_entry"
    e.data = {}
    return e


def test_state_is_current_spot_in_eur_per_kwh(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 87.5,
        "forecast": _build_forecast(),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    assert s.native_value == pytest.approx(0.0875, abs=1e-6)


def test_state_returns_none_when_no_data(sensor_module):
    coord = MagicMock()
    coord.data = None
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    assert s.native_value is None


def test_attributes_have_required_keys(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 50.0,
        "forecast": _build_forecast(),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    for k in ("raw_today", "raw_tomorrow", "raw_extended",
                "today_min", "today_avg", "today_max",
                "tomorrow_min", "tomorrow_avg", "tomorrow_max",
                "forecast_horizon_h", "currency", "unit",
                "source", "last_updated"):
        assert k in attrs, f"missing attribute: {k}"


def test_raw_extended_length_matches_forecast(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 50.0,
        "forecast": _build_forecast(72),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    assert attrs["forecast_horizon_h"] == 72
    assert len(attrs["raw_extended"]) == 72


def test_raw_extended_entries_have_start_end_value(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 50.0,
        "forecast": _build_forecast(),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    for item in attrs["raw_extended"][:5]:
        assert "start" in item
        assert "end" in item
        assert "value" in item
        # Nordpool reports in the configured unit; here EUR/kWh.
        assert 0 < item["value"] < 10


def test_values_converted_from_mwh_to_kwh(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 100.0,
        "forecast": [{
            "timestamp":    "2026-04-15T00:00:00+00:00",
            "spot_eur_mwh": 123.0,
            "P5_eur_mwh":   100.0,
            "P95_eur_mwh":  150.0,
        }],
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    assert attrs["raw_extended"][0]["value"] == pytest.approx(0.123, abs=1e-6)


def test_confidence_band_present_when_fan_chart_attached(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 50.0,
        "forecast": _build_forecast(),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    assert "confidence_band" in attrs
    band = attrs["confidence_band"]
    assert "p5" in band and "p95" in band
    assert len(band["p5"]) == len(attrs["raw_extended"])


def test_today_avg_within_band(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_spot_eur_mwh": 50.0,
        "forecast": _build_forecast(),
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    attrs = s.extra_state_attributes
    if attrs["today_min"] is not None:
        assert attrs["today_min"] <= attrs["today_avg"] <= attrs["today_max"]


def test_entity_id_is_spot_price_forecast_fi(sensor_module):
    coord = MagicMock()
    coord.data = None
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    assert s.entity_id == "sensor.spot_price_forecast_fi"


def test_device_info_returns_dict(sensor_module):
    coord = MagicMock()
    coord.data = None
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    info = s.device_info
    assert isinstance(info, dict)
    assert "name" in info
    assert "identifiers" in info


def test_attributes_empty_when_forecast_missing(sensor_module):
    coord = MagicMock()
    coord.data = {"forecast": []}
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    assert s.extra_state_attributes == {}


def test_native_value_none_when_current_spot_missing(sensor_module):
    coord = MagicMock()
    coord.data = {"forecast": [{"timestamp": "2026-04-15T00:00:00+00:00",
                                  "spot_eur_mwh": 50.0}]}
    s = sensor_module.SpotPriceForecastSensor(coord, _make_entry())
    # current_spot_eur_mwh missing → native_value is None
    assert s.native_value is None


# ── Effective Wind Speed sensor ─────────────────────────────────────

def test_wind_sensor_state_and_forecast(sensor_module):
    coord = MagicMock()
    coord.data = {
        "current_wind": 5.4,
        "forecast": [
            {"timestamp": "2026-04-15T00:00:00+00:00", "wind": 5.4},
            {"timestamp": "2026-04-15T01:00:00+00:00", "wind": 6.1},
            {"timestamp": "2026-04-15T02:00:00+00:00", "spot_eur_mwh": 50.0},  # no wind
        ],
        "last_update": "2026-04-15T00:00:00+00:00",
    }
    s = sensor_module.EffectiveWindSpeedSensor(coord, _make_entry())
    assert s.native_value == pytest.approx(5.4)
    attrs = s.extra_state_attributes
    # Only entries carrying wind are surfaced.
    assert len(attrs["forecast"]) == 2
    assert attrs["forecast"][0] == {"timestamp": "2026-04-15T00:00:00+00:00", "wind": 5.4}
    assert attrs["forecast_hours"] == 2
    assert attrs["height_m"] == 120
    assert s._attr_native_unit_of_measurement == "m/s"
    assert s._attr_device_class == "wind_speed"


def test_wind_sensor_none_without_data(sensor_module):
    coord = MagicMock()
    coord.data = None
    s = sensor_module.EffectiveWindSpeedSensor(coord, _make_entry())
    assert s.native_value is None
    assert s.extra_state_attributes == {}


def test_wind_sensor_registered_in_setup(sensor_module):
    """Defensive: the sensor must be in the always-created entities list."""
    import inspect
    src = inspect.getsource(sensor_module.async_setup_entry)
    assert "EffectiveWindSpeedSensor(coordinator, entry)" in src
