"""Calendar-aware temporal arithmetic + bilingual human formatting."""

from datetime import datetime, timedelta

import pytest

from memory_core.temporal import (
    Interval,
    days_between,
    duration_between,
    format_human,
    months_between,
    weeks_between,
    years_between,
)


# --------------------------------------------------------------------- #
# duration_between / days_between / weeks_between                       #
# --------------------------------------------------------------------- #


def test_duration_between_datetimes_is_absolute():
    a = datetime(2024, 6, 1)
    b = datetime(2024, 6, 5)
    assert duration_between(a, b) == timedelta(days=4)
    assert duration_between(b, a) == timedelta(days=4)


def test_days_between_floors_partial_days():
    a = datetime(2024, 6, 1, 0, 0)
    b = datetime(2024, 6, 3, 23, 0)
    # 2 days 23 hours → floor to 2 whole days
    assert days_between(a, b) == 2


def test_weeks_between_only_counts_full_weeks():
    a = datetime(2024, 1, 1)
    # Jan 1 → Jan 15 is exactly 14 days = 2 full weeks
    assert weeks_between(a, datetime(2024, 1, 15)) == 2
    # Jan 1 → Jan 14 is 13 days = 1 full week
    assert weeks_between(a, datetime(2024, 1, 14)) == 1


def test_duration_accepts_intervals_and_uses_start():
    iv1 = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1))
    iv2 = Interval(datetime(2024, 1, 5), datetime(2024, 1, 10))
    # anchored on starts: Jan 5 - Jan 1 = 4 days
    assert duration_between(iv1, iv2) == timedelta(days=4)


def test_duration_mixed_datetime_and_interval():
    iv = Interval(datetime(2024, 1, 1), datetime(2024, 2, 1))
    pt = datetime(2024, 1, 8)
    assert days_between(iv, pt) == 7


def test_duration_rejects_unsupported_type():
    with pytest.raises(TypeError):
        duration_between("2024-01-01", datetime(2024, 1, 5))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# months_between — calendar-aware, the headline behaviour               #
# --------------------------------------------------------------------- #


def test_months_jan31_to_feb28_nonleap_is_one():
    """The classic case: end-of-Jan to end-of-Feb in a non-leap year is
    one whole month, not 28 days masquerading as zero."""
    assert months_between(datetime(2023, 1, 31), datetime(2023, 2, 28)) == 1


def test_months_jan31_to_feb27_nonleap_is_zero():
    """One day short of the clamped anniversary → still 0 months."""
    assert months_between(datetime(2023, 1, 31), datetime(2023, 2, 27)) == 0


def test_months_jan31_to_feb29_leap_is_one():
    assert months_between(datetime(2024, 1, 31), datetime(2024, 2, 29)) == 1


def test_months_jan31_to_feb28_leap_is_zero():
    """In a leap year the anchor is Feb 29; Feb 28 is one day short."""
    assert months_between(datetime(2024, 1, 31), datetime(2024, 2, 28)) == 0


def test_months_zero_for_same_month():
    assert months_between(datetime(2024, 6, 1), datetime(2024, 6, 30)) == 0


def test_months_full_year_span():
    assert months_between(datetime(2023, 6, 11), datetime(2024, 6, 11)) == 12


def test_months_honours_time_of_day():
    a = datetime(2024, 1, 15, 12, 0)
    b_short = datetime(2024, 2, 15, 11, 59)
    b_exact = datetime(2024, 2, 15, 12, 0)
    assert months_between(a, b_short) == 0
    assert months_between(a, b_exact) == 1


def test_months_negative_direction_returns_positive():
    """Sign is stripped — duration semantics."""
    assert months_between(datetime(2024, 6, 1), datetime(2023, 6, 1)) == 12


def test_months_through_year_boundary():
    assert months_between(datetime(2023, 11, 15), datetime(2024, 2, 15)) == 3


# --------------------------------------------------------------------- #
# years_between                                                         #
# --------------------------------------------------------------------- #


def test_years_feb29_to_feb28_next_year_is_one():
    assert years_between(datetime(2020, 2, 29), datetime(2021, 2, 28)) == 1


def test_years_feb29_to_feb27_next_year_is_zero():
    assert years_between(datetime(2020, 2, 29), datetime(2021, 2, 27)) == 0


def test_years_full_decade():
    assert years_between(datetime(2010, 5, 1), datetime(2020, 5, 1)) == 10


def test_years_one_day_short_returns_zero():
    assert years_between(datetime(2010, 5, 1), datetime(2011, 4, 30)) == 0


# --------------------------------------------------------------------- #
# format_human — English                                                #
# --------------------------------------------------------------------- #


def test_format_zero_en():
    assert format_human(timedelta(0), "en") == "0 seconds"


def test_format_singular_day_en():
    assert format_human(timedelta(days=1), "en") == "1 day"


def test_format_plural_days_en():
    assert format_human(timedelta(days=5), "en") == "5 days"


def test_format_weeks_with_remainder_en():
    # 16 days = 2 weeks + 2 days
    assert format_human(timedelta(days=16), "en") == "2 weeks 2 days"


def test_format_one_month_four_days_en():
    """Spec example: '1 month 4 days' — using a 30.4375 day-per-month
    average, 34 days = 1 month + (34 - int(30.4375)) = 1 month 4 days.
    """
    assert format_human(timedelta(days=34), "en") == "1 month 4 days"


def test_format_hours_and_minutes_en():
    assert format_human(timedelta(hours=2, minutes=30), "en") == "2 hours 30 minutes"


def test_format_only_minutes_en():
    assert format_human(timedelta(minutes=15), "en") == "15 minutes"


def test_format_seconds_only_en():
    assert format_human(timedelta(seconds=42), "en") == "42 seconds"


# --------------------------------------------------------------------- #
# format_human — Russian, three-form pluralisation                      #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1 день"),     # one
        (2, "2 дня"),      # few
        (3, "3 дня"),
        (4, "4 дня"),
        (5, "5 дней"),     # many
        (6, "6 дней"),
    ],
)
def test_format_ru_day_plural_forms(n, expected):
    """Days-only test: at 7+ days the formatter promotes to weeks."""
    assert format_human(timedelta(days=n), "ru") == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1 час"),       # one
        (2, "2 часа"),      # few
        (5, "5 часов"),     # many
        (11, "11 часов"),   # 11..14 → many even when ending in 1
        (12, "12 часов"),
        (14, "14 часов"),
        (21, "21 час"),     # ends in 1 (not 11) → one
        (22, "22 часа"),    # ends in 2 → few
        (23, "23 часа"),    # ends in 3 → few
    ],
)
def test_format_ru_hour_plural_forms(n, expected):
    """Hours stay sub-day (≤23) so the formatter doesn't promote to the
    'days' bucket — lets us test the full Russian three-form rule."""
    assert format_human(timedelta(hours=n), "ru") == expected


def test_format_ru_hours_and_minutes():
    assert format_human(timedelta(hours=1, minutes=21), "ru") == "1 час 21 минута"


def test_format_ru_two_hours_two_minutes():
    """Few/few combination."""
    assert format_human(timedelta(hours=2, minutes=2), "ru") == "2 часа 2 минуты"


def test_format_ru_zero():
    assert format_human(timedelta(0), "ru") == "0 секунд"


def test_format_unsupported_language_raises():
    with pytest.raises(ValueError):
        format_human(timedelta(days=1), "fr")


def test_format_negative_duration_uses_absolute():
    """Direction is the caller's job to format, not ours."""
    assert format_human(timedelta(days=-3), "en") == "3 days"
