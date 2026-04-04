"""Pure Python Finnish holiday calculator. No external dependencies."""

from __future__ import annotations

from datetime import date, timedelta


def _easter_sunday(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _midsummer_eve(year: int) -> date:
    """Finnish Midsummer Eve: Friday between June 19-25."""
    for d in range(19, 26):
        candidate = date(year, 6, d)
        if candidate.weekday() == 4:
            return candidate
    raise ValueError(f"No midsummer eve found for {year}")


def _all_saints(year: int) -> date:
    """Finnish All Saints' Day: Saturday between Oct 31 - Nov 6."""
    for offset in range(7):
        d = 31 + offset
        month = 10 if d <= 31 else 11
        day = d if d <= 31 else d - 31
        candidate = date(year, month, day)
        if candidate.weekday() == 5:
            return candidate
    raise ValueError(f"No all saints found for {year}")


# Finnish fixed holidays: (month, day)
_FIXED_HOLIDAYS = [
    (1, 1),   # New Year
    (1, 6),   # Epiphany
    (5, 1),   # May Day
    (12, 6),  # Independence Day
    (12, 24), # Christmas Eve
    (12, 25), # Christmas Day
    (12, 26), # St. Stephen's Day
]

# Easter-relative offsets
_EASTER_OFFSETS = [
    -2,   # Good Friday
    0,    # Easter Sunday
    1,    # Easter Monday
    39,   # Ascension Day
    49,   # Whit Sunday
]


def finnish_holidays_for_year(year: int) -> set[date]:
    """Return all Finnish public holidays for a given year."""
    result: set[date] = set()

    for month, day in _FIXED_HOLIDAYS:
        result.add(date(year, month, day))

    easter = _easter_sunday(year)
    for offset in _EASTER_OFFSETS:
        result.add(easter + timedelta(days=offset))

    result.add(_midsummer_eve(year))
    result.add(_midsummer_eve(year) + timedelta(days=1))  # Midsummer Day
    result.add(_all_saints(year))

    return result


def build_holiday_set(start_year: int, end_year: int) -> set[str]:
    """Build set of ISO date strings for Finnish holidays in [start_year, end_year)."""
    result: set[str] = set()
    for year in range(start_year, end_year):
        for d in finnish_holidays_for_year(year):
            result.add(d.isoformat())
    return result


def is_holiday(dt: date, holidays: set[str]) -> bool:
    """Check if a date is a Finnish public holiday."""
    return dt.isoformat() in holidays
