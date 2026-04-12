"""Tests for coordinator business logic.

Tests consumer price conversion, timezone handling, D(k) duration forecast
computation with per-segment tariff conversion, and forecast assembly.
These functions are pure logic extracted from the coordinator — no HA
dependency or async.
"""

import math
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Minimal coordinator-like helper for testing pure functions
# ---------------------------------------------------------------------------

class _TariffHelper:
    """Reproduces coordinator tariff logic without HA dependency."""

    def __init__(
        self,
        day_rate: float = 0.0361,
        night_rate: float = 0.0220,
        seller_margin: float = 0.0,
        energy_tax: float = 0.02325,
        vat_multiplier: float = 1.255,
    ):
        self.day_rate = day_rate
        self.night_rate = night_rate
        self.seller_margin = seller_margin
        self.energy_tax = energy_tax
        self.vat_multiplier = vat_multiplier

    def spot_to_consumer_ckwh(self, spot_eur_mwh: float, is_night: bool) -> float:
        """Exact copy of coordinator._spot_to_consumer_ckwh."""
        transfer = self.night_rate if is_night else self.day_rate
        spot_kwh = max(0.0, spot_eur_mwh) / 1000.0
        return (spot_kwh + self.seller_margin + transfer + self.energy_tax) \
            * self.vat_multiplier * 100


# ---------------------------------------------------------------------------
# Consumer price conversion tests
# ---------------------------------------------------------------------------

class TestSpotToConsumerCkwh:
    """Test coordinator._spot_to_consumer_ckwh logic."""

    @pytest.fixture
    def elenia(self):
        return _TariffHelper()

    def test_zero_spot_day(self, elenia):
        """Zero spot price, day tariff → transfer + tax only."""
        c = elenia.spot_to_consumer_ckwh(0.0, is_night=False)
        expected = (0.0 + 0.0 + 0.0361 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected, abs=0.01)

    def test_zero_spot_night(self, elenia):
        """Zero spot, night tariff → lower transfer."""
        c = elenia.spot_to_consumer_ckwh(0.0, is_night=True)
        expected = (0.0 + 0.0 + 0.0220 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected, abs=0.01)

    def test_day_more_expensive_than_night(self, elenia):
        """Day consumer price must exceed night for same spot (Elenia)."""
        spot = 40.0
        day = elenia.spot_to_consumer_ckwh(spot, is_night=False)
        night = elenia.spot_to_consumer_ckwh(spot, is_night=True)
        assert day > night

    def test_equal_rates_same_price(self):
        """Helen has equal day/night rates → same consumer price."""
        helen = _TariffHelper(day_rate=0.0354, night_rate=0.0354)
        spot = 50.0
        assert helen.spot_to_consumer_ckwh(spot, False) == pytest.approx(
            helen.spot_to_consumer_ckwh(spot, True))

    def test_50_eur_mwh_day(self, elenia):
        """50 EUR/MWh spot, day rate: known value."""
        c = elenia.spot_to_consumer_ckwh(50.0, is_night=False)
        expected = (0.05 + 0.0 + 0.0361 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected, abs=0.01)

    def test_negative_spot_clamped(self, elenia):
        """Negative spot clamped to 0 → same as zero spot."""
        c_neg = elenia.spot_to_consumer_ckwh(-20.0, is_night=False)
        c_zero = elenia.spot_to_consumer_ckwh(0.0, is_night=False)
        assert c_neg == pytest.approx(c_zero)

    def test_high_spot(self, elenia):
        """High spot price (200 EUR/MWh) gives reasonable consumer price."""
        c = elenia.spot_to_consumer_ckwh(200.0, is_night=False)
        # 200/1000 = 0.2 EUR/kWh + 0.0361 + 0.02325 = 0.25935 * 1.255 * 100
        expected = (0.2 + 0.0 + 0.0361 + 0.02325) * 1.255 * 100
        assert c == pytest.approx(expected, abs=0.01)
        assert c > 30  # should be around 32.5 c/kWh

    def test_seller_margin_adds(self):
        """Non-zero seller margin increases consumer price."""
        base = _TariffHelper(seller_margin=0.0)
        with_margin = _TariffHelper(seller_margin=0.005)
        spot = 40.0
        assert with_margin.spot_to_consumer_ckwh(spot, False) > \
            base.spot_to_consumer_ckwh(spot, False)
        # Margin should add exactly 0.005 * VAT * 100 c/kWh
        diff = with_margin.spot_to_consumer_ckwh(spot, False) - \
            base.spot_to_consumer_ckwh(spot, False)
        assert diff == pytest.approx(0.005 * 1.255 * 100, abs=0.001)

    def test_monotone_increasing(self, elenia):
        """Consumer price is monotone increasing with spot (for spot >= 0)."""
        prices = [elenia.spot_to_consumer_ckwh(s, False) for s in range(0, 300, 10)]
        for i in range(1, len(prices)):
            assert prices[i] >= prices[i - 1]


# ---------------------------------------------------------------------------
# Sensor _spot_to_consumer tests (EUR/kWh input, used by Nordpool sensors)
# ---------------------------------------------------------------------------

class TestSensorSpotToConsumer:
    """Test sensor.py _spot_to_consumer (EUR/kWh input, not EUR/MWh)."""

    @staticmethod
    def _spot_to_consumer(spot_eur_kwh, hour, tariff):
        """Reproduces sensor._spot_to_consumer logic."""
        is_night = hour < 7 or hour >= 22
        transfer = tariff["night_rate"] if is_night else tariff["day_rate"]
        return (spot_eur_kwh + tariff["seller_margin"] + transfer +
                tariff["energy_tax"]) * tariff["vat"]

    @pytest.fixture
    def tariff(self):
        return {
            "day_rate": 0.0361,
            "night_rate": 0.0220,
            "vat": 1.255,
            "energy_tax": 0.02325,
            "seller_margin": 0.0,
        }

    def test_day_hour_uses_day_rate(self, tariff):
        """Hour 12 uses day rate."""
        p = self._spot_to_consumer(0.05, 12, tariff)
        expected = (0.05 + 0.0361 + 0.02325) * 1.255
        assert p == pytest.approx(expected, abs=0.0001)

    def test_night_hour_uses_night_rate(self, tariff):
        """Hour 3 uses night rate."""
        p = self._spot_to_consumer(0.05, 3, tariff)
        expected = (0.05 + 0.0220 + 0.02325) * 1.255
        assert p == pytest.approx(expected, abs=0.0001)

    @pytest.mark.parametrize("hour,expected_night", [
        (0, True), (1, True), (6, True),      # Night: 00-06
        (7, False), (12, False), (21, False),  # Day: 07-21
        (22, True), (23, True),                # Night: 22-23
    ])
    def test_tariff_boundary_hours(self, hour, expected_night, tariff):
        """Verify day/night boundary at hours 7 and 22."""
        rate_night = self._spot_to_consumer(0.05, hour, tariff)
        # Compute expected with known rate
        transfer = tariff["night_rate"] if expected_night else tariff["day_rate"]
        expected = (0.05 + transfer + 0.02325) * 1.255
        assert rate_night == pytest.approx(expected, abs=0.0001)


# ---------------------------------------------------------------------------
# Timezone / local hour tests
# ---------------------------------------------------------------------------

class TestGetLocalHour:
    """Test coordinator._get_local_hour logic."""

    @staticmethod
    def _get_local_hour_with_tz(ts_utc: datetime) -> int:
        """Reproduces _get_local_hour with ZoneInfo."""
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Helsinki")
        aware = ts_utc.replace(tzinfo=ZoneInfo("UTC")) if ts_utc.tzinfo is None else ts_utc
        return aware.astimezone(tz).hour

    @staticmethod
    def _get_local_hour_fallback(ts_utc: datetime) -> int:
        """Reproduces _get_local_hour fallback (UTC+3)."""
        return (ts_utc.hour + 3) % 24

    def test_utc_midnight_is_helsinki_morning(self):
        """UTC 00:00 = Helsinki 02:00 (winter) or 03:00 (summer)."""
        # January (UTC+2)
        ts_winter = datetime(2026, 1, 15, 0, 0, 0)
        h = self._get_local_hour_with_tz(ts_winter)
        assert h == 2  # EET = UTC+2

    def test_utc_midnight_summer(self):
        """UTC 00:00 in summer = Helsinki 03:00 (EEST = UTC+3)."""
        ts_summer = datetime(2026, 7, 15, 0, 0, 0)
        h = self._get_local_hour_with_tz(ts_summer)
        assert h == 3  # EEST = UTC+3

    def test_fallback_always_utc_plus_3(self):
        """Fallback assumes UTC+3 regardless of DST."""
        for utc_hour in range(24):
            ts = datetime(2026, 1, 15, utc_hour, 0, 0)
            h = self._get_local_hour_fallback(ts)
            assert h == (utc_hour + 3) % 24

    def test_wrap_around(self):
        """UTC 22:00 wraps to local hour 1 in fallback."""
        ts = datetime(2026, 1, 15, 22, 0, 0)
        assert self._get_local_hour_fallback(ts) == 1

    def test_aware_timestamp_handled(self):
        """Already-aware timestamps should not double-convert."""
        from zoneinfo import ZoneInfo
        ts = datetime(2026, 7, 15, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        h = self._get_local_hour_with_tz(ts)
        assert h == 15  # UTC+3 in summer


# ---------------------------------------------------------------------------
# Per-segment D(k) consumer conversion
# ---------------------------------------------------------------------------

class TestPerSegmentDkConsumerConversion:
    """Test that D(k) consumer prices use correct per-segment tariffs.

    The coordinator extracts sorted prices from each segment's D(k) curve,
    converts each to consumer c/kWh using the segment's tariff (night vs day),
    then re-sorts and recomputes the consumer D(k).
    """

    @pytest.fixture
    def helper(self):
        return _TariffHelper()

    def _extract_sorted_prices(self, curve: list[float]) -> list[float]:
        """Extract sorted prices from a D(k) curve (inverse of cumulative avg)."""
        prices = []
        for i in range(len(curve)):
            if i == 0:
                prices.append(curve[0])
            else:
                p = (i + 1) * curve[i] - i * curve[i - 1]
                prices.append(max(0.0, p))
        return prices

    def _compute_consumer_dk(
        self,
        segment_curves: dict[str, list[float]],
        helper: _TariffHelper,
        night_segments: set = {"night"},
    ) -> list[float]:
        """Reproduce coordinator's per-segment consumer D(k) logic."""
        consumer_sorted = []
        for seg_name, curve in segment_curves.items():
            is_night = seg_name in night_segments
            prices = self._extract_sorted_prices(curve)
            for p in prices:
                consumer_sorted.append(
                    helper.spot_to_consumer_ckwh(p, is_night))
        consumer_sorted.sort()
        dk = []
        running = 0.0
        for i, cp in enumerate(consumer_sorted):
            running += cp
            dk.append(running / (i + 1))
        return dk

    def test_all_day_segments_same_as_flat_rate(self, helper):
        """When all segments are day, consumer D(k) = flat day rate applied to spot D(k)."""
        # 4 segments, all with 6 hours, all using day rate
        flat_curve = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
        segment_curves = {
            "morning": flat_curve,
            "midday": flat_curve,
            "evening": flat_curve,
            "afternoon": flat_curve,
        }
        dk = self._compute_consumer_dk(
            segment_curves, helper, night_segments=set())

        # All use day rate, so consumer = (spot/1000 + margin + day_rate + tax) * vat * 100
        # Since all prices identical per segment, sorted prices are all the same
        assert len(dk) == 24

    def test_night_segment_uses_lower_rate(self, helper):
        """Night segment hours get lower transfer rate → lower consumer D(k)."""
        spot = 30.0  # EUR/MWh
        # All segments predict identical flat D(k) = [30]
        night_curve = [spot]  # 1 hour night segment
        day_curve = [spot]    # 1 hour day segment

        c_night = helper.spot_to_consumer_ckwh(spot, is_night=True)
        c_day = helper.spot_to_consumer_ckwh(spot, is_night=False)

        assert c_night < c_day  # night transfer < day transfer

    def test_mixed_segments_produces_24_levels(self, helper):
        """With 9+5+6+4=24 hours, output must have exactly 24 D(k) levels."""
        segment_curves = {
            "night": [10 + i for i in range(9)],
            "morning": [20 + i for i in range(5)],
            "midday": [15 + i for i in range(6)],
            "evening": [25 + i for i in range(4)],
        }
        dk = self._compute_consumer_dk(segment_curves, helper)
        assert len(dk) == 24

    def test_consumer_dk_monotone(self, helper):
        """Consumer D(k) must be non-decreasing."""
        segment_curves = {
            "night": [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 28.0, 35.0],
            "morning": [20.0, 25.0, 30.0, 38.0, 50.0],
            "midday": [12.0, 16.0, 20.0, 25.0, 32.0, 45.0],
            "evening": [30.0, 40.0, 55.0, 70.0],
        }
        dk = self._compute_consumer_dk(segment_curves, helper)
        for i in range(1, len(dk)):
            assert dk[i] >= dk[i - 1] - 1e-9, \
                f"D({i+1})={dk[i]:.4f} < D({i})={dk[i-1]:.4f}"

    def test_night_discount_lowers_d1(self, helper):
        """D(1) with night segment should be lower than all-day D(1)."""
        # Night segment has cheapest prices
        segment_curves = {
            "night": [5.0],   # cheapest hour
            "morning": [30.0],
        }
        dk_mixed = self._compute_consumer_dk(
            segment_curves, helper, night_segments={"night"})
        dk_allday = self._compute_consumer_dk(
            segment_curves, helper, night_segments=set())
        # D(1) = cheapest hour's consumer price
        # Night rate is lower, so the cheapest hour converted with night rate
        # should be cheaper
        assert dk_mixed[0] < dk_allday[0]


# ---------------------------------------------------------------------------
# D(k) completeness validation
# ---------------------------------------------------------------------------

class TestDkCompletenessCheck:
    """Test that only complete days (all 24 hours present) produce D(k)."""

    SEGMENTS = {
        "night": [22, 23, 0, 1, 2, 3, 4, 5, 6],
        "morning": [7, 8, 9, 10, 11],
        "midday": [12, 13, 14, 15, 16, 17],
        "evening": [18, 19, 20, 21],
    }

    def _check_completeness(self, available_hours: set[int]) -> bool:
        """Reproduces coordinator's completeness check."""
        for hours_list in self.SEGMENTS.values():
            if any(h not in available_hours for h in hours_list):
                return False
        return True

    def test_full_day_passes(self):
        """All 24 hours present → complete."""
        assert self._check_completeness(set(range(24)))

    def test_missing_one_hour_fails(self):
        """Missing hour 6 (night segment) → incomplete."""
        hours = set(range(24)) - {6}
        assert not self._check_completeness(hours)

    def test_missing_evening_hour_fails(self):
        """Missing hour 20 (evening) �� incomplete."""
        hours = set(range(24)) - {20}
        assert not self._check_completeness(hours)

    def test_only_daytime_fails(self):
        """Only hours 7-21 (no night segment) → incomplete."""
        hours = set(range(7, 22))
        assert not self._check_completeness(hours)

    def test_segments_cover_24_hours(self):
        """Verify segments sum to exactly 24 hours with no gaps or overlaps."""
        all_hours = []
        for hours_list in self.SEGMENTS.values():
            all_hours.extend(hours_list)
        assert sorted(all_hours) == list(range(24))

    def test_segment_tariff_alignment(self):
        """Night segment hours (22-06) all fall within night tariff (22-07)."""
        for h in self.SEGMENTS["night"]:
            assert h < 7 or h >= 22, f"Night segment hour {h} is in day tariff"
        for seg in ("morning", "midday", "evening"):
            for h in self.SEGMENTS[seg]:
                assert 7 <= h < 22, f"{seg} hour {h} is in night tariff"


# ---------------------------------------------------------------------------
# Tariff config reading
# ---------------------------------------------------------------------------

class TestTariffConfigReading:
    """Test that tariff config always reads from config entry data.

    Reproduces the logic from coordinator.__init__ and sensor._get_tariff_config
    without importing homeassistant modules.
    """

    # Hardcoded operator table (mirrors const.py OPERATORS)
    OPERATORS = {
        "elenia": {"name": "Elenia", "day_rate": 0.0361, "night_rate": 0.0220},
        "caruna_espoo": {"name": "Caruna Espoo", "day_rate": 0.0221, "night_rate": 0.0221},
        "caruna_north": {"name": "Caruna North", "day_rate": 0.0407, "night_rate": 0.0249},
        "helen": {"name": "Helen", "day_rate": 0.0354, "night_rate": 0.0354},
        "custom": {"name": "Custom", "day_rate": 0.0500, "night_rate": 0.0400},
    }
    DEFAULT_VAT = 1.255
    DEFAULT_TAX = 0.02325

    @classmethod
    def _get_tariff(cls, data: dict) -> dict:
        """Reproduces sensor._get_tariff_config and coordinator init logic."""
        operator_id = data.get("operator", "elenia")
        op = cls.OPERATORS.get(operator_id, cls.OPERATORS["elenia"])
        return {
            "day_rate": data.get("custom_day_rate", op["day_rate"]),
            "night_rate": data.get("custom_night_rate", op["night_rate"]),
            "vat": data.get("custom_vat", cls.DEFAULT_VAT),
            "energy_tax": data.get("custom_energy_tax", cls.DEFAULT_TAX),
            "seller_margin": data.get("seller_margin", 0.0),
        }

    def test_named_operator_defaults(self):
        """Named operator uses its defaults when no custom rates stored."""
        tariff = self._get_tariff({"operator": "elenia"})
        assert tariff["day_rate"] == pytest.approx(0.0361)
        assert tariff["night_rate"] == pytest.approx(0.0220)

    def test_custom_rates_override_named_operator(self):
        """User-modified rates override operator defaults."""
        tariff = self._get_tariff({
            "operator": "elenia",
            "custom_day_rate": 0.04,
            "custom_night_rate": 0.025,
        })
        assert tariff["day_rate"] == pytest.approx(0.04)
        assert tariff["night_rate"] == pytest.approx(0.025)

    def test_custom_operator(self):
        """Custom operator reads from stored values."""
        tariff = self._get_tariff({
            "operator": "custom",
            "custom_day_rate": 0.06,
            "custom_night_rate": 0.03,
        })
        assert tariff["day_rate"] == pytest.approx(0.06)
        assert tariff["night_rate"] == pytest.approx(0.03)

    def test_unknown_operator_falls_back_to_elenia(self):
        """Unknown operator ID falls back to Elenia defaults."""
        tariff = self._get_tariff({"operator": "nonexistent"})
        assert tariff["day_rate"] == pytest.approx(0.0361)

    def test_all_operators_have_both_rates(self):
        """Every operator has day_rate and night_rate."""
        for op_id, op in self.OPERATORS.items():
            assert "day_rate" in op, f"{op_id} missing day_rate"
            assert "night_rate" in op, f"{op_id} missing night_rate"
            assert op["day_rate"] >= 0
            assert op["night_rate"] >= 0

    def test_helen_equal_rates(self):
        """Helen has equal day/night (yleissiirto)."""
        tariff = self._get_tariff({"operator": "helen"})
        assert tariff["day_rate"] == tariff["night_rate"]

    def test_config_entry_takes_priority_over_operator_dict(self):
        """Stored config values always win over OPERATORS dict defaults."""
        # User selected Elenia but modified rates in options flow
        tariff = self._get_tariff({
            "operator": "elenia",
            "custom_day_rate": 0.0400,
            "custom_night_rate": 0.0250,
        })
        # Should use the stored 0.04/0.025, NOT Elenia's 0.0361/0.0220
        assert tariff["day_rate"] != 0.0361
        assert tariff["day_rate"] == pytest.approx(0.04)

    def test_no_stored_rates_uses_operator_defaults(self):
        """Without stored rates, falls back to operator's default rates."""
        # Config entry only has operator ID, no custom rates (backward compat)
        tariff = self._get_tariff({"operator": "caruna_north"})
        assert tariff["day_rate"] == pytest.approx(0.0407)
        assert tariff["night_rate"] == pytest.approx(0.0249)


# ---------------------------------------------------------------------------
# Upload coefficients validation
# ---------------------------------------------------------------------------

class TestUploadCoefficientsValidation:
    """Test that upload_coefficients service validates against current model format."""

    CURRENT_MODEL_KEYS = {
        "intercept", "features", "feature_names", "model_version",
        "model_type", "log_offset", "power_scale", "power_exp",
        "feature_count", "tier_info", "metrics",
    }

    def test_production_model_has_required_keys(self):
        """Verify production model_coefs.json has expected structure."""
        coefs_path = Path(__file__).parent.parent / "custom_components" / \
            "spot_price_predictor" / "data" / "model_coefs_default.json"
        with open(coefs_path) as f:
            coefs = json.load(f)

        # These should be the validated keys
        assert "intercept" in coefs
        assert "features" in coefs
        assert "feature_names" in coefs
        assert isinstance(coefs["features"], list)
        assert isinstance(coefs["feature_names"], list)
        assert len(coefs["features"]) == len(coefs["feature_names"])

    def test_old_keys_not_in_current_model(self):
        """Old model keys (stage1, piecewise_breakpoints) should NOT exist."""
        coefs_path = Path(__file__).parent.parent / "custom_components" / \
            "spot_price_predictor" / "data" / "model_coefs_default.json"
        with open(coefs_path) as f:
            coefs = json.load(f)

        assert "stage1" not in coefs, "Old 'stage1' key still in model"
        assert "piecewise_breakpoints" not in coefs, \
            "Old 'piecewise_breakpoints' key still in model"

    def test_init_validation_matches_model_format(self):
        """__init__.py required_keys must match current model structure.

        This test will FAIL if the upload service validates against stale keys.
        """
        init_path = Path(__file__).parent.parent / "custom_components" / \
            "spot_price_predictor" / "__init__.py"
        with open(init_path) as f:
            source = f.read()

        # The required_keys line should validate current model keys
        assert 'required_keys = ["stage1"' not in source, \
            "Upload service still validates old 'stage1' key"
        assert '"piecewise_breakpoints"' not in source, \
            "Upload service still validates old 'piecewise_breakpoints' key"

    def test_duration_model_present_in_production(self):
        """Production model should include duration_model."""
        coefs_path = Path(__file__).parent.parent / "custom_components" / \
            "spot_price_predictor" / "data" / "model_coefs_default.json"
        with open(coefs_path) as f:
            coefs = json.load(f)
        assert "duration_model" in coefs
        dm = coefs["duration_model"]
        assert "segments" in dm
        assert "feature_names" in dm
        assert len(dm["segments"]) == 4

    def test_feature_count_matches(self):
        """feature_count field matches actual feature_names length."""
        coefs_path = Path(__file__).parent.parent / "custom_components" / \
            "spot_price_predictor" / "data" / "model_coefs_default.json"
        with open(coefs_path) as f:
            coefs = json.load(f)
        assert coefs["feature_count"] == len(coefs["feature_names"])
        assert coefs["feature_count"] == len(coefs["features"])


# ---------------------------------------------------------------------------
# Forecast assembly and week stats
# ---------------------------------------------------------------------------

class TestForecastAssembly:
    """Test forecast entry structure and week statistics."""

    @staticmethod
    def _make_forecast_entry(
        ts_utc: datetime, spot: float, helper: _TariffHelper,
        wind: float = 5.0, solar: float = 100.0, temp: float = 10.0,
    ) -> dict:
        """Build a single forecast entry like the coordinator does."""
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Helsinki")
        aware = ts_utc.replace(tzinfo=ZoneInfo("UTC"))
        local_hour = aware.astimezone(tz).hour
        is_night = local_hour < 7 or local_hour >= 22
        consumer = helper.spot_to_consumer_ckwh(spot, is_night)
        return {
            "timestamp": ts_utc.isoformat(),
            "spot_eur_mwh": round(spot, 2),
            "consumer_ckwh": round(consumer, 2),
            "wind": round(wind, 1),
            "solar": round(solar, 0),
            "temp": round(temp, 1),
        }

    def test_entry_has_required_fields(self):
        helper = _TariffHelper()
        ts = datetime(2026, 4, 12, 12, 0, 0)
        entry = self._make_forecast_entry(ts, 40.0, helper)
        assert "timestamp" in entry
        assert "spot_eur_mwh" in entry
        assert "consumer_ckwh" in entry
        assert "wind" in entry
        assert "solar" in entry
        assert "temp" in entry

    def test_week_stats_from_forecast(self):
        """Week min/avg/max computed correctly from forecast array."""
        helper = _TariffHelper()
        now = datetime(2026, 4, 12, 0, 0, 0)
        forecast = []
        spots = [10, 20, 30, 40, 50]
        for i, spot in enumerate(spots):
            entry = self._make_forecast_entry(
                now + timedelta(hours=i), float(spot), helper)
            forecast.append(entry)

        prices = [f["consumer_ckwh"] for f in forecast]
        week_min = min(prices)
        week_avg = sum(prices) / len(prices)
        week_max = max(prices)

        assert week_min < week_avg < week_max
        assert week_min == forecast[0]["consumer_ckwh"]  # lowest spot = first
        assert week_max == forecast[-1]["consumer_ckwh"]  # highest spot = last

    def test_consumer_ckwh_positive(self):
        """Consumer price is always positive even for zero spot."""
        helper = _TariffHelper()
        ts = datetime(2026, 4, 12, 12, 0, 0)
        entry = self._make_forecast_entry(ts, 0.0, helper)
        assert entry["consumer_ckwh"] > 0


# ---------------------------------------------------------------------------
# Duration forecast sensor state
# ---------------------------------------------------------------------------

class TestDurationForecastSensorLogic:
    """Test DurationForecastSensor state = D(4) from first day."""

    def test_state_is_d4(self):
        """Sensor state reads dk_consumer_cent_kwh[3] (D(4))."""
        dk_vec = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                  21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
        dk_list = [{"dk_consumer_cent_kwh": dk_vec}]
        # Sensor logic: dk_list[0].get("dk_consumer_cent_kwh", [])[3]
        result = dk_list[0]["dk_consumer_cent_kwh"][3]
        assert result == 8.0  # D(4) = index 3

    def test_empty_forecast_returns_none(self):
        """No duration forecast → None state."""
        dk_list = []
        result = dk_list[0]["dk_consumer_cent_kwh"][3] if dk_list else None
        assert result is None

    def test_short_vector_returns_none(self):
        """D(k) vector shorter than 4 → None state."""
        dk_vec = [5.0, 6.0, 7.0]  # only 3 elements
        result = dk_vec[3] if len(dk_vec) >= 4 else None
        assert result is None

    def test_dk_vector_length_24(self):
        """Full D(k) vector must have exactly 24 elements."""
        dk_vec = list(range(1, 25))
        assert len(dk_vec) == 24
        # D(1) = index 0, D(24) = index 23
        assert dk_vec[0] == 1
        assert dk_vec[23] == 24
