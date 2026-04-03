"""
Holiday calculator driven by region config.

Computes public holidays from rules defined in the region YAML file.
Supports fixed dates, Easter-relative dates, and Finnish special rules
(Midsummer, All Saints).

Usage:
    from src.holidays import build_holiday_set
    holidays = build_holiday_set(region_config, start_year=2020, end_year=2030)
    # holidays is a set of "YYYY-MM-DD" strings
"""

from datetime import date, timedelta
from typing import Any


def _easter_sunday(year: int) -> date:
    """Compute Easter Sunday using the Gregorian (Anonymous) algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _midsummer_eve(year: int) -> date:
    """Finnish Midsummer Eve: Friday between June 19-25."""
    for d in range(19, 26):
        candidate = date(year, 6, d)
        if candidate.weekday() == 4:  # Friday
            return candidate
    raise ValueError(f"No Friday found between Jun 19-25 for year {year}")


def _midsummer_day(year: int) -> date:
    """Finnish Midsummer Day: Saturday between June 20-26."""
    return _midsummer_eve(year) + timedelta(days=1)


def _all_saints(year: int) -> date:
    """Finnish All Saints' Day: Saturday between Oct 31 - Nov 6."""
    for offset in range(7):
        d = 31 + offset
        month = 10 if d <= 31 else 11
        day = d if d <= 31 else d - 31
        candidate = date(year, month, day)
        if candidate.weekday() == 5:  # Saturday
            return candidate
    raise ValueError(f"No Saturday found between Oct 31 - Nov 6 for year {year}")


SPECIAL_RULE_HANDLERS = {
    "midsummer": _midsummer_eve,
    "midsummer_day": _midsummer_day,
    "all_saints": _all_saints,
}


def holidays_for_year(config: dict[str, Any], year: int) -> set[date]:
    """Compute all holidays for a given year from region config rules."""
    holidays_config = config.get("holidays", {})
    result: set[date] = set()

    # Fixed holidays
    for entry in holidays_config.get("fixed", []):
        try:
            result.add(date(year, entry["month"], entry["day"]))
        except (ValueError, KeyError):
            pass  # Skip invalid dates (e.g., Feb 30)

    # Easter-based holidays
    easter = _easter_sunday(year)
    for entry in holidays_config.get("easter_based", []):
        offset = entry.get("offset", 0)
        result.add(easter + timedelta(days=offset))

    # Special rules (Midsummer, All Saints, etc.)
    for entry in holidays_config.get("special_rules", []):
        rule_type = entry.get("type", "")
        handler = SPECIAL_RULE_HANDLERS.get(rule_type)
        if handler:
            try:
                result.add(handler(year))
            except ValueError:
                pass

    return result


def build_holiday_set(
    config: dict[str, Any],
    start_year: int = 2018,
    end_year: int = 2031,
) -> set[str]:
    """Build a set of ISO date strings for all holidays in the given year range.

    Args:
        config: Region config dict (must contain 'holidays' section).
        start_year: First year to compute holidays for (inclusive).
        end_year: Last year to compute holidays for (exclusive).

    Returns:
        Set of "YYYY-MM-DD" strings.
    """
    result: set[str] = set()
    for year in range(start_year, end_year):
        for d in holidays_for_year(config, year):
            result.add(d.isoformat())
    return result
